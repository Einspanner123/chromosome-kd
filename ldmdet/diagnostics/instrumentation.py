"""运行时探针 (Probe) — 非侵入式模型插桩核心.

在模型前向/反向过程中收集监控数据, 由 TrainingDiagnosticsHook 定期上传到 SwanLab.
设计为单例, 全局可访问, 零配置安全 (未启用时所有方法为 no-op).

监控维度 (按用户需求):
- 梯度统计: per-module L2 范数, mean/std/zero_ratio (backward hook 自动采集)
- 激活值统计: time_emb, step_emb, per-head cls_logits/pred_bboxes, solver x_t/x0_pred
- 损失/Box 统计: loss_cls/loss_bbox/loss_giou 分解, L1/GIoU 详细统计
- 诊断量: eta_str, eta_3rd, box_renewal 率, coupling 熵, proposal 置信度分布

数据形式:
- 标量: mean/std/min/max/norm
- 直方图: SwanLab Histogram (分布可视化)
- 详细统计: box_renewal_rate, coupling_entropy, proposal_score_hist

频率控制:
- 训练: 每 train_interval 步采集一次 (默认 100)
- 推理: 每次推理都采集 (可关闭)

使用方式 (在模型代码中):
    from ldmdet.diagnostics.instrumentation import probe

    # 记录标量
    probe.record_scalar('solver/eta_str', eta_str)

    # 记录张量统计 (标量 + 直方图)
    probe.record_tensor_stats('cascade/time_emb', time_emb)

    # 记录详细统计 (自定义 dict)
    probe.record_detail('box_renewal', {'rate': 0.3, 'n_renewed': 50})

未启用时 (probe.enabled == False), 所有 record_* 方法立即返回, 零开销.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# SwanLab 直方图桶数 (2^10 = 1024, 自动降采样)
_HIST_BINS = 64
# 直方图最大元素数 (超过则随机下采样, 避免 OOM)
_HIST_MAX_ELEMENTS = 100_000


# ──────────────────────────────────────────────
# 直方图工具
# ──────────────────────────────────────────────

def _compute_histogram(tensor: Tensor, bins: int = _HIST_BINS) -> Optional[Dict[str, Any]]:
    """计算张量直方图统计.

    返回 SwanLab 兼容的直方图格式:
        {
            'min': float, 'max': float, 'mean': float, 'std': float,
            'hist': list[float],  # 每个桶的计数 (归一化到 0-1)
            'bin_edges': list[float],  # 桶边界
        }
    """
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return None

    # 下采样 (避免大张量 OOM)
    if flat.numel() > _HIST_MAX_ELEMENTS:
        perm = torch.randperm(flat.numel(), device=flat.device)[:_HIST_MAX_ELEMENTS]
        flat = flat[perm]

    flat = flat.float().cpu()

    t_min = float(flat.min().item())
    t_max = float(flat.max().item())
    t_mean = float(flat.mean().item())
    t_std = float(flat.std().item()) if flat.numel() > 1 else 0.0

    # 处理常数张量 (min == max)
    if t_max - t_min < 1e-12:
        hist = [0.0] * bins
        hist[0] = 1.0
        bin_edges = [t_min] * (bins + 1)
    else:
        hist_tensor = torch.histc(flat, bins=bins, min=t_min, max=t_max)
        total = hist_tensor.sum().item()
        hist = (hist_tensor / max(total, 1)).tolist()
        bin_edges = torch.linspace(t_min, t_max, bins + 1).tolist()

    return {
        'min': t_min,
        'max': t_max,
        'mean': t_mean,
        'std': t_std,
        'hist': hist,
        'bin_edges': bin_edges,
    }


def _compute_tensor_stats(tensor: Tensor) -> Dict[str, float]:
    """计算张量标量统计 (轻量, 不含直方图)."""
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'norm': 0.0}

    flat = flat.float()
    return {
        'mean': float(flat.mean().item()),
        'std': float(flat.std().item()) if flat.numel() > 1 else 0.0,
        'min': float(flat.min().item()),
        'max': float(flat.max().item()),
        'norm': float(flat.norm(2).item()),
    }


# ──────────────────────────────────────────────
# Probe 单例
# ──────────────────────────────────────────────

class _Probe:
    """运行时探针单例.

    线程安全, 未启用时所有方法为 no-op.
    """

    def __init__(self):
        self._enabled = False
        self._inference_enabled = False
        self._train_interval = 100
        self._train_step = 0
        self._should_collect = False  # 当前 step 是否应采集
        self._lock = threading.Lock()

        # 缓冲区: {scope: {key: value}}
        self._scalars: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._histograms: Dict[str, Dict[str, Dict]] = defaultdict(dict)
        self._details: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # 推理缓冲区 (每次推理单独 flush)
        self._inference_scalars: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._inference_histograms: Dict[str, Dict[str, Dict]] = defaultdict(dict)
        self._inference_details: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # 梯度 hook 句柄
        self._grad_hooks: list = []
        self._grad_buffer: Dict[str, float] = {}

    # ── 生命周期 ───────────────────────────────

    def enable(
        self,
        train_interval: int = 100,
        inference_enabled: bool = True,
    ) -> None:
        """启用探针.

        Args:
            train_interval: 训练时采集间隔 (步), 默认 100
            inference_enabled: 是否在推理时采集, 默认 True
        """
        self._enabled = True
        self._inference_enabled = inference_enabled
        self._train_interval = train_interval
        logger.info(
            f'[Probe] Enabled: train_interval={train_interval}, '
            f'inference={inference_enabled}'
        )

    def disable(self) -> None:
        """禁用探针并清理."""
        self._enabled = False
        self._inference_enabled = False
        self._clear_buffers()
        self._remove_grad_hooks()
        logger.info('[Probe] Disabled')

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def inference_enabled(self) -> bool:
        return self._enabled and self._inference_enabled

    # ── 训练步控制 ─────────────────────────────

    def on_train_iter_begin(self, step: int) -> None:
        """训练 iter 开始时调用 (由 Hook 触发)."""
        if not self._enabled:
            return
        self._train_step = step
        self._should_collect = (step % self._train_interval == 0)

    def on_train_iter_end(self, step: int) -> None:
        """训练 iter 结束时调用 (由 Hook 触发), flush 训练缓冲区."""
        if not self._enabled or not self._should_collect:
            return
        self._flush_train(step)
        self._should_collect = False

    # ── 推理控制 ───────────────────────────────

    def on_inference_begin(self) -> None:
        """推理开始时调用."""
        if not self.inference_enabled:
            return
        self._clear_inference_buffers()

    def on_inference_end(self) -> None:
        """推理结束时调用, flush 推理缓冲区."""
        if not self.inference_enabled:
            return
        self._flush_inference()

    # ── 数据记录 (训练) ───────────────────────

    def record_scalar(self, key: str, value: float) -> None:
        """记录标量 (训练缓冲区)."""
        if not self._enabled or not self._should_collect:
            return
        with self._lock:
            self._scalars['train'][key] = float(value)

    def record_tensor_stats(
        self,
        key: str,
        tensor: Tensor,
        histogram: bool = True,
    ) -> None:
        """记录张量统计 (标量 + 可选直方图).

        Args:
            key: 指标名 (如 'cascade/time_emb')
            tensor: 待统计的张量
            histogram: 是否记录直方图 (默认 True)
        """
        if not self._enabled or not self._should_collect:
            return
        if not isinstance(tensor, Tensor) or tensor.numel() == 0:
            return
        with self._lock:
            stats = _compute_tensor_stats(tensor)
            for k, v in stats.items():
                self._scalars['train'][f'{key}/{k}'] = v
            if histogram:
                hist = _compute_histogram(tensor)
                if hist is not None:
                    self._histograms['train'][key] = hist

    def record_detail(self, key: str, data: Dict[str, Any]) -> None:
        """记录详细统计 (自定义 dict, 标量值提取到 SwanLab)."""
        if not self._enabled or not self._should_collect:
            return
        with self._lock:
            self._details['train'][key] = data
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    self._scalars['train'][f'{key}/{k}'] = float(v)

    # ── 数据记录 (推理) ───────────────────────

    def record_inference_scalar(self, key: str, value: float) -> None:
        """记录推理标量."""
        if not self.inference_enabled:
            return
        with self._lock:
            self._inference_scalars['inference'][key] = float(value)

    def record_inference_tensor_stats(
        self,
        key: str,
        tensor: Tensor,
        histogram: bool = True,
    ) -> None:
        """记录推理张量统计."""
        if not self.inference_enabled:
            return
        if not isinstance(tensor, Tensor) or tensor.numel() == 0:
            return
        with self._lock:
            stats = _compute_tensor_stats(tensor)
            for k, v in stats.items():
                self._inference_scalars['inference'][f'{key}/{k}'] = v
            if histogram:
                hist = _compute_histogram(tensor)
                if hist is not None:
                    self._inference_histograms['inference'][key] = hist

    def record_inference_detail(self, key: str, data: Dict[str, Any]) -> None:
        """记录推理详细统计."""
        if not self.inference_enabled:
            return
        with self._lock:
            self._inference_details['inference'][key] = data
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    self._inference_scalars['inference'][f'{key}/{k}'] = float(v)

    # ── 梯度采集 ───────────────────────────────

    def register_grad_hooks(self, model: torch.nn.Module) -> None:
        """注册 per-module 梯度 hook.

        在指定模型的子模块上注册 full_backward_hook, 采集梯度 L2 范数.
        仅在 _should_collect 时实际采集.
        """
        if not self._enabled:
            return

        self._remove_grad_hooks()

        # 按模块组注册 (避免每个叶子模块都注册, 太多)
        # 只注册命名模块 (跳过容器)
        target_modules = [
            ('backbone', getattr(model, 'backbone', None)),
            ('neck', getattr(model, 'neck', None)),
            ('time_mlp', getattr(getattr(model, 'bbox_head', None), 'time_mlp', None)),
            ('step_mlp', getattr(getattr(model, 'bbox_head', None), 'step_mlp', None)),
        ]

        # cascade head 逐个注册
        bbox_head = getattr(model, 'bbox_head', None)
        if bbox_head is not None and hasattr(bbox_head, 'head_series'):
            for i, h in enumerate(bbox_head.head_series):
                target_modules.append((f'cascade_head_{i}', h))

        for name, module in target_modules:
            if module is None:
                continue
            hook = module.register_full_backward_hook(
                self._make_grad_hook(name)
            )
            self._grad_hooks.append(hook)

    def _make_grad_hook(self, module_name: str):
        """创建梯度采集 hook."""
        def hook(module, grad_input, grad_output):
            if not self._enabled or not self._should_collect:
                return
            # grad_output 是 tuple, 取第一个非 None
            grads = [g for g in grad_output if g is not None]
            if not grads:
                return
            # 合并所有梯度输出
            all_g = torch.cat([g.detach().reshape(-1) for g in grads])
            if all_g.numel() == 0:
                return
            with self._lock:
                self._scalars['train'][f'grads/{module_name}/norm'] = float(all_g.norm(2).item())
                self._scalars['train'][f'grads/{module_name}/mean'] = float(all_g.mean().item())
                self._scalars['train'][f'grads/{module_name}/std'] = (
                    float(all_g.std().item()) if all_g.numel() > 1 else 0.0
                )
                self._scalars['train'][f'grads/{module_name}/max'] = (
                    float(all_g.abs().max().item())
                )
        return hook

    def _remove_grad_hooks(self) -> None:
        for h in self._grad_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._grad_hooks.clear()

    # ── Flush (上传到 SwanLab) ─────────────────

    def _flush_train(self, step: int) -> None:
        """上传训练缓冲区到 SwanLab."""
        with self._lock:
            scalars = dict(self._scalars.get('train', {}))
            histograms = dict(self._histograms.get('train', {}))

        if not scalars and not histograms:
            return

        # 上传标量
        if scalars:
            self._swanlab_log(scalars, step)

        # 上传直方图 (SwanLab 支持 swanlab.Histogram)
        for key, hist_data in histograms.items():
            self._swanlab_log_histogram(key, hist_data, step)

        # 清空训练缓冲区
        with self._lock:
            self._scalars['train'].clear()
            self._histograms['train'].clear()
            self._details['train'].clear()

    def _flush_inference(self) -> None:
        """上传推理缓冲区到 SwanLab."""
        with self._lock:
            scalars = dict(self._inference_scalars.get('inference', {}))
            histograms = dict(self._inference_histograms.get('inference', {}))

        if not scalars and not histograms:
            return

        # 推理用 train_step 作为 step (与训练对齐)
        step = self._train_step

        if scalars:
            self._swanlab_log(scalars, step)

        for key, hist_data in histograms.items():
            self._swanlab_log_histogram(key, hist_data, step)

        with self._lock:
            self._inference_scalars['inference'].clear()
            self._inference_histograms['inference'].clear()
            self._inference_details['inference'].clear()

    def _swanlab_log(self, data: Dict[str, float], step: int) -> None:
        """上传标量到 SwanLab (静默失败)."""
        try:
            import swanlab
            swanlab.log(data, step=step)
        except Exception as e:
            logger.debug(f'[Probe] swanlab.log failed: {e}')

    def _swanlab_log_histogram(
        self, key: str, hist_data: Dict, step: int
    ) -> None:
        """上传直方图到 SwanLab (静默失败)."""
        try:
            import swanlab
            # SwanLab 支持 swanlab.Histogram 或直接传 dict
            # 使用 swanlab.Histogram 包装
            hist = swanlab.Histogram(
                hist_data['hist'],
                bin_edges=hist_data['bin_edges'],
            )
            swanlab.log({f'hist/{key}': hist}, step=step)
        except Exception:
            # 降级: 上传直方图统计标量
            try:
                import swanlab
                swanlab.log({
                    f'hist/{key}/min': hist_data['min'],
                    f'hist/{key}/max': hist_data['max'],
                    f'hist/{key}/mean': hist_data['mean'],
                    f'hist/{key}/std': hist_data['std'],
                }, step=step)
            except Exception as e:
                logger.debug(f'[Probe] histogram log failed: {e}')

    # ── 清理 ───────────────────────────────────

    def _clear_buffers(self) -> None:
        with self._lock:
            self._scalars.clear()
            self._histograms.clear()
            self._details.clear()
            self._clear_inference_buffers()

    def _clear_inference_buffers(self) -> None:
        self._inference_scalars.clear()
        self._inference_histograms.clear()
        self._inference_details.clear()


# ──────────────────────────────────────────────
# 全局单例
# ──────────────────────────────────────────────

probe = _Probe()
