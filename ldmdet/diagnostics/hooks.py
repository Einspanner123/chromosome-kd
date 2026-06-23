"""TrainingDiagnosticsHook — 训练诊断 Hook.

提供权重/梯度/激活统计、数值健康度检查、训练动态监控,
按模块组聚合, 上传到 SwanLab.

监控维度:
- 权重统计: mean/std/norm/dead_ratio/has_nan/has_inf (每 weight_grad_interval iter)
- 梯度统计: mean/std/norm/zero_ratio/grad_to_weight_ratio (同上)
- 激活统计: mean/std/dead_ratio/saturation (每 activation_interval iter)
- 数值健康度: NaN/Inf/极端值检测
- 训练动态: 损失分解 (每 iter)

设计原则:
- 低开销: 采样频率可配, 默认权重/梯度 100 iter, 激活 500 iter
- 模块化: 按模块组聚合, 避免指标爆炸
- 可扩展: 方向特定诊断通过 diagnostics_callback 注入
"""

import logging
from typing import Dict, List, Optional, Callable, Any

import torch
import torch.nn as nn
from torch import Tensor

from mmengine.hooks import Hook
from mmdet.registry import HOOKS

from ldmdet.diagnostics.stats import (
    compute_weight_stats,
    compute_grad_stats,
    compute_activation_stats,
    check_numerical_health,
    group_modules_by_prefix,
    ModuleGroup,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# SwanLab 上传抽象 (便于测试 mock)
# ──────────────────────────────────────────────

def swanlab_log(data: Dict[str, float], step: int) -> None:
    """上传标量到 SwanLab.

    生产环境调用 swanlab.log, 测试环境可 mock.
    """
    try:
        import swanlab
        swanlab.log(data, step=step)
    except Exception:
        pass  # 静默失败, 不影响训练


# ──────────────────────────────────────────────
# TrainingDiagnosticsHook
# ──────────────────────────────────────────────

@HOOKS.register_module(force=True)
class TrainingDiagnosticsHook(Hook):
    """训练诊断 Hook.

    Args:
        weight_grad_interval: 权重/梯度统计采样间隔 (iter), 默认 100
        activation_interval: 激活统计采样间隔 (iter), 默认 500
        module_prefixes: {模块名前缀: 组名} 映射, 如 {'backbone.': 'backbone'}
        activation_layers: 要采集激活的层名列表, 如 ['backbone.0', 'head.0']
        log_weights: 是否记录权重统计
        log_grads: 是否记录梯度统计
        log_activations: 是否记录激活统计
        log_numerical_health: 是否记录数值健康度
        log_loss_breakdown: 是否记录损失分解
        diagnostics_callback: 方向特定诊断回调 (可选)
            签名: callback(runner, outputs, step) -> Dict[str, float]
    """

    priority = 'LOW'

    def __init__(
        self,
        weight_grad_interval: int = 100,
        activation_interval: int = 500,
        module_prefixes: Optional[Dict[str, str]] = None,
        activation_layers: Optional[List[str]] = None,
        log_weights: bool = True,
        log_grads: bool = True,
        log_activations: bool = True,
        log_numerical_health: bool = True,
        log_loss_breakdown: bool = True,
        diagnostics_callback: Optional[Callable] = None,
    ):
        self.weight_grad_interval = weight_grad_interval
        self.activation_interval = activation_interval
        self.module_prefixes = module_prefixes or {
            'backbone.': 'backbone',
            'neck.': 'neck',
            'bbox_head.head_series.': 'head',
            'bbox_head.time_mlp': 'time_mlp',
            'bbox_head.criterion': 'criterion',
        }
        self.activation_layers = activation_layers or []
        self.log_weights = log_weights
        self.log_grads = log_grads
        self.log_activations = log_activations
        self.log_numerical_health = log_numerical_health
        self.log_loss_breakdown = log_loss_breakdown
        self.diagnostics_callback = diagnostics_callback

        # 运行时状态
        self._activation_hooks: List[Any] = []
        self._activation_cache: Dict[str, Tensor] = {}
        self._module_groups: Dict[str, ModuleGroup] = {}

    # ── 生命周期 ────────────────────────────────

    def before_run(self, runner) -> None:
        """注册前向 hook 采集激活."""
        model = self._get_model(runner)

        # 分组模块
        self._module_groups = group_modules_by_prefix(model, self.module_prefixes)

        # 注册激活 hook
        if self.log_activations and self.activation_layers:
            self._register_activation_hooks(model)

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch=None,
        outputs=None,
    ) -> None:
        """训练 iter 后采集诊断数据."""
        step = runner.iter

        # 1. 训练动态 (每 iter, 若有 outputs)
        if self.log_loss_breakdown and outputs is not None:
            self._log_loss_breakdown(outputs, step)

        # 2. 权重/梯度统计 (按 interval)
        if self._should_collect_weight_grad(runner):
            self._collect_weight_grad_stats(runner, step)

        # 3. 激活统计 (按 interval)
        if self.log_activations and self._should_collect_activation(runner):
            self._collect_activation_stats(step)

        # 4. 方向特定诊断回调
        if self.diagnostics_callback is not None:
            try:
                diag_data = self.diagnostics_callback(runner, outputs, step)
                if diag_data:
                    swanlab_log(diag_data, step)
            except Exception as e:
                logger.warning(f'diagnostics_callback failed: {e}')

    # ── 权重/梯度统计 ───────────────────────────

    def _should_collect_weight_grad(self, runner) -> bool:
        """采样频率控制: iter % interval == 0 且 iter > 0."""
        return runner.iter > 0 and runner.iter % self.weight_grad_interval == 0

    def _should_collect_activation(self, runner) -> bool:
        """采样频率控制: iter % interval == 0 且 iter > 0."""
        return runner.iter > 0 and runner.iter % self.activation_interval == 0

    def _collect_weight_grad_stats(self, runner, step: int) -> None:
        """采集权重和梯度统计, 按模块组聚合."""
        model = self._get_model(runner)

        # 懒初始化模块组 (若 before_run 未调用)
        if not self._module_groups:
            self._module_groups = group_modules_by_prefix(model, self.module_prefixes)

        log_data: Dict[str, float] = {}

        for group_name, group in self._module_groups.items():
            if not group.modules:
                continue

            # 合并组内所有模块的参数
            all_params: List[nn.Parameter] = []
            for mod in group.modules:
                all_params.extend(list(mod.parameters()))

            if not all_params:
                continue

            # 权重统计
            if self.log_weights:
                all_w = torch.cat([p.data.reshape(-1) for p in all_params])
                w_stats = self._compute_combined_weight_stats(all_params)
                for k, v in w_stats.items():
                    log_data[f'weights/{group_name}/{k}'] = v

            # 梯度统计
            if self.log_grads:
                g_stats = self._compute_combined_grad_stats(all_params)
                for k, v in g_stats.items():
                    log_data[f'grads/{group_name}/{k}'] = v

        if log_data:
            swanlab_log(log_data, step)

    def _compute_combined_weight_stats(self, params: List[nn.Parameter]) -> Dict[str, float]:
        """合并多个参数的权重统计."""
        all_w = torch.cat([p.data.reshape(-1) for p in params])
        has_nan = bool(torch.isnan(all_w).any().item())
        has_inf = bool(torch.isinf(all_w).any().item())

        # 死神经元检测 (仅 weight, dim>=2)
        dead_count = 0
        total_out_dims = 0
        for p in params:
            if p.dim() >= 2:
                for i in range(p.shape[0]):
                    if p.data[i].abs().sum().item() == 0.0:
                        dead_count += 1
                    total_out_dims += 1

        return {
            'mean': float(all_w.mean().item()),
            'std': float(all_w.std().item()) if all_w.numel() > 1 else 0.0,
            'norm': float(all_w.norm(2).item()),
            'dead_ratio': dead_count / max(total_out_dims, 1),
            'has_nan': float(has_nan),
            'has_inf': float(has_inf),
        }

    def _compute_combined_grad_stats(self, params: List[nn.Parameter]) -> Dict[str, float]:
        """合并多个参数的梯度统计."""
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return {
                'mean': 0.0, 'std': 0.0, 'norm': 0.0,
                'zero_ratio': 1.0, 'has_nan': 0.0, 'has_inf': 0.0,
            }

        all_g = torch.cat([g.reshape(-1) for g in grads])
        zero_ratio = float((all_g == 0.0).float().mean().item())
        has_nan = float(torch.isnan(all_g).any().item())
        has_inf = float(torch.isinf(all_g).any().item())

        return {
            'mean': float(all_g.mean().item()),
            'std': float(all_g.std().item()) if all_g.numel() > 1 else 0.0,
            'norm': float(all_g.norm(2).item()),
            'zero_ratio': zero_ratio,
            'has_nan': has_nan,
            'has_inf': has_inf,
        }

    # ── 激活统计 ─────────────────────────────────

    def _register_activation_hooks(self, model: nn.Module) -> None:
        """给指定层注册前向 hook."""
        for name, module in model.named_modules():
            if name in self.activation_layers:
                hook = module.register_forward_hook(self._make_activation_hook(name))
                self._activation_hooks.append(hook)

    def _make_activation_hook(self, layer_name: str) -> Callable:
        """创建激活采集 hook."""
        def hook(module, input, output):
            if isinstance(output, Tensor):
                self._activation_cache[layer_name] = output.detach()
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                if isinstance(output[0], Tensor):
                    self._activation_cache[layer_name] = output[0].detach()
        return hook

    def _collect_activation_stats(self, step: int) -> None:
        """采集激活统计."""
        if not self._activation_cache:
            return

        log_data: Dict[str, float] = {}
        for layer_name, activation in self._activation_cache.items():
            # 用层名推导组名
            group_name = self._infer_group_name(layer_name)
            stats = compute_activation_stats(activation)
            for k, v in stats.items():
                log_data[f'acts/{group_name}/{k}'] = v

        if log_data:
            swanlab_log(log_data, step)

        # 清理缓存
        self._activation_cache.clear()

    def _infer_group_name(self, layer_name: str) -> str:
        """从层名推导组名."""
        for prefix, group_name in self.module_prefixes.items():
            if layer_name.startswith(prefix) or layer_name == prefix.rstrip('.'):
                return group_name
        return layer_name.split('.')[0]

    def _remove_hooks(self) -> None:
        """清理前向 hook."""
        for hook in self._activation_hooks:
            hook.remove()
        self._activation_hooks.clear()

    # ── 训练动态 ─────────────────────────────────

    def _log_loss_breakdown(self, outputs: Dict, step: int) -> None:
        """记录损失分解."""
        log_data: Dict[str, float] = {}
        total = 0.0
        for name, val in outputs.items():
            if name.startswith('loss') and isinstance(val, Tensor):
                v = float(val.item())
                log_data[f'dynamics/{name}'] = v
                total += v
        if total > 0:
            log_data['dynamics/loss_total'] = total
        if log_data:
            swanlab_log(log_data, step)

    # ── 工具 ─────────────────────────────────────

    def _get_model(self, runner) -> nn.Module:
        """获取模型 (处理 DataParallel)."""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        return model
