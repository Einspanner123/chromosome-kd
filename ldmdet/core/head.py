"""DiffusionDetHead — 核心迭代去噪检测头

直接移植自 LDMDet/mods/diffusiondet_head.py (已验证工作),
仅更新 import 路径到 ldmdet 纯 PyTorch 库。
"""

import copy
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldmdet.coupling import build_coupling
from ldmdet.data.structures import (
    InstanceData,
    ModelOutput,
)
from ldmdet.diffusion.embeddings import SinusoidalPositionEmbeddings
from ldmdet.diffusion.noise_schedule import cosine_noise_schedule
from ldmdet.diffusion.rectified_flow import RectifiedFlow
from ldmdet.diffusion.sampling import DiffusionSampler, _get_img_shape
from ldmdet.utils.box_ops import bbox_xyxy_to_cxcywh


class DiffusionDetHead(nn.Module):
    """扩散检测头。支持 DDPM 和 Rectified Flow，Euler/Heun/DPM-Solver++。"""

    def __init__(
        self,
        num_classes: int = 80,
        feat_channels: int = 256,
        num_proposals: int = 500,
        num_heads: int = 6,
        prior_prob: float = 0.01,
        snr_scale: float = 2.0,
        timesteps: int = 1000,
        sampling_timesteps: int = 1,
        solver_type: str = 'euler',
        box_renewal: bool = True,
        use_ensemble: bool = True,
        deep_supervision: bool = True,
        cascade_detach: bool = True,
        ddim_sampling_eta: float = 1.0,
        diffusion_type: str = 'ddpm',
        rf_schedule: str = 'linear',
        rf_power: float = 1.0,
        rf_shift: float = 1.0,
        single_head: nn.Module = None,
        roi_extractor: nn.Module = None,
        criterion: nn.Module = None,
        use_nms: bool = True,
        nms_thr: float = 0.5,
        score_thr: float = 0.05,
        min_keep: int = 10,
        filter_unknown: bool = True,
        gt_reweight: bool = True,
        use_checkpoint: bool = False,
        coupling: Optional[nn.Module] = None,
        pre_noise_layer: int = 2,
        loss_aux: Optional[Dict] = None,
        torch_compile: bool = False,
        amp_dtype: Optional[torch.dtype] = None,
        topk_pruning_enabled: bool = False,
        topk_k: int = 100,
        topk_pruning_step: int = 0,
        # IO4: 级联头提前退出 (推理优化)
        head_early_exit_enabled: bool = False,
        head_exit_box_threshold: float = 0.005,
        head_exit_cls_threshold: float = 0.98,
        head_exit_min_heads: int = 3,
        head_exit_max_heads: int = 6,
        head_exit_time_aware: bool = True,
        # IO1: 自适应步数提前终止 (推理优化)
        step_early_exit_enabled: bool = False,
        step_exit_threshold: float = 0.01,
        step_exit_cls_threshold: float = 0.95,
        step_exit_min_steps: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_proposals = num_proposals
        self.num_heads = num_heads
        self.snr_scale = snr_scale
        self.diffusion_type = diffusion_type
        self.deep_supervision = deep_supervision
        self.cascade_detach = cascade_detach
        self.filter_unknown = filter_unknown
        self.gt_reweight = gt_reweight
        self.timesteps = timesteps
        self.use_checkpoint = use_checkpoint
        self.rf_schedule = rf_schedule
        self.rf_shift = rf_shift
        self.rf_power = rf_power
        self.box_renewal = box_renewal
        self.use_ensemble = use_ensemble
        self.solver_type = solver_type

        self.loss_aux = loss_aux

        # 扩散组件
        self.rf = RectifiedFlow(snr_scale=snr_scale)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(feat_channels),
            nn.Linear(feat_channels, feat_channels * 4),
            nn.SiLU(),
            nn.Linear(feat_channels * 4, feat_channels * 4),
        )

        # DDPM 缓存
        if diffusion_type == 'ddpm':
            betas = cosine_noise_schedule(timesteps).float()
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            self.register_buffer('alphas_cumprod', alphas_cumprod)
        else:
            self.alphas_cumprod = None

        # 级联 Head
        self.head_series = nn.ModuleList(
            [copy.deepcopy(single_head) for _ in range(num_heads)]
        )
        self.roi_extractor = roi_extractor
        self.criterion = criterion
        self.pre_noise_layer = pre_noise_layer

        # 耦合策略
        self.ot_coupling = coupling is not None and not isinstance(
            coupling, bool
        )
        self.ot_module = (
            coupling if coupling is not None else build_coupling('random')
        )

        # 采样器
        self._sampler = DiffusionSampler(
            diffusion_type=diffusion_type,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            solver_type=solver_type,
            ddim_sampling_eta=ddim_sampling_eta,
            rf_schedule=rf_schedule,
            rf_power=rf_power,
            rf_shift=rf_shift,
            snr_scale=snr_scale,
            box_renewal=box_renewal,
            use_ensemble=use_ensemble,
            use_nms=use_nms,
            nms_thr=nms_thr,
            score_thr=score_thr,
            min_keep=min_keep,
        )

        self._init_weights(prior_prob)

        # AMP: 仅模型前向使用半精度，criterion 始终 FP32
        # 推荐值: torch.bfloat16 (同动态范围，无需 GradScaler)
        self.amp_dtype = amp_dtype

        if torch_compile and hasattr(torch, 'compile'):
            self.forward = torch.compile(self.forward, dynamic=True)

        # IO3: Top-K 框剪枝 (推理优化, 不影响训练)
        self.topk_pruning_enabled = topk_pruning_enabled
        self.topk_k = topk_k
        self.topk_pruning_step = topk_pruning_step
        self._pruning_stats = {}

        # IO4: 级联头提前退出 (推理优化, 不影响训练)
        self.head_early_exit_enabled = head_early_exit_enabled
        self.head_exit_box_threshold = head_exit_box_threshold
        self.head_exit_cls_threshold = head_exit_cls_threshold
        self.head_exit_min_heads = head_exit_min_heads
        self.head_exit_max_heads = min(head_exit_max_heads, num_heads)
        self.head_exit_time_aware = head_exit_time_aware
        self._exit_stats = {}

        # IO1: 自适应步数提前终止 (推理优化, 不影响训练)
        # RF 直线路径: t→0 时 x_t ≈ x_0, x0_pred 趋于稳定
        # 检测相邻步 x0_pred 的相对 L2 变化 + argmax 一致率, 收敛则提前终止
        self.step_early_exit_enabled = step_early_exit_enabled
        self.step_exit_threshold = step_exit_threshold
        self.step_exit_cls_threshold = step_exit_cls_threshold
        self.step_exit_min_steps = step_exit_min_steps
        self._step_exit_stats = {}

    def _init_weights(self, prior_prob):
        for head in self.head_series:
            if hasattr(head, 'cls_head'):
                last_layer = head.cls_head[-1]
                if hasattr(last_layer, 'bias') and last_layer.bias is not None:
                    bias_value = -(math.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(last_layer.bias, bias_value)

    # ================================================================
    # 前向传播
    # ================================================================

    def forward(self, features, bboxes, t):
        time_emb = self.time_mlp(t)
        # IO4: 每次推理清空退出统计 (仅本次 forward 记录)
        if not self.training and self.head_early_exit_enabled:
            self._exit_stats = {}
        inter_cls_logits = []
        inter_pred_bboxes = []
        inter_curr_proposals = []
        curr_bboxes = bboxes
        curr_proposals = None
        prev_bboxes = None
        prev_logits = None

        for i, head in enumerate(self.head_series):
            result = head(
                features, curr_bboxes, curr_proposals,
                self.roi_extractor, time_emb,
            )
            if len(result) == 4:
                cls_logits, pred_bboxes, curr_proposals, _ = result
            else:
                cls_logits, pred_bboxes, curr_proposals = result
            inter_cls_logits.append(cls_logits)
            inter_pred_bboxes.append(pred_bboxes)
            inter_curr_proposals.append(curr_proposals)

            # IO4: 级联头提前退出 (仅推理时启用)
            # 从 min_heads 头开始检测收敛, 不在最后一个头检测 (无意义)
            if (
                not self.training
                and self.head_early_exit_enabled
                and prev_bboxes is not None
                and i >= self.head_exit_min_heads - 1
                and i < len(self.head_series) - 1
            ):
                converged, stats = self._check_head_convergence(
                    pred_bboxes, prev_bboxes,
                    cls_logits, prev_logits, t,
                )
                if converged:
                    # 用当前头结果填充剩余头, 保持输出形状
                    for _ in range(len(self.head_series) - i - 1):
                        inter_cls_logits.append(cls_logits.clone())
                        inter_pred_bboxes.append(pred_bboxes.clone())
                        inter_curr_proposals.append(curr_proposals)
                    self._exit_stats = {
                        'exit_head_idx': i,
                        'total_heads': len(self.head_series),
                        **stats,
                    }
                    break

            curr_bboxes = (
                pred_bboxes.detach()
                if self.cascade_detach
                else pred_bboxes
            )
            prev_bboxes = pred_bboxes
            prev_logits = cls_logits

        if self.deep_supervision:
            return (
                torch.stack(inter_cls_logits),
                torch.stack(inter_pred_bboxes),
                inter_curr_proposals,
            )
        return (
            torch.stack(inter_cls_logits[-1:]),
            torch.stack(inter_pred_bboxes[-1:]),
            inter_curr_proposals[-1:],
        )

    # ================================================================
    # IO4: 级联头提前退出辅助方法
    # ================================================================

    def _get_exit_threshold(self, t_norm: float) -> Tuple[float, float]:
        """根据归一化时间步 [0,1] 返回 (box_threshold, cls_threshold)。

        早期步 (t_norm > 0.5): 框还在大幅调整, 用严格阈值 (少退出)
        中期步 (0.2 < t_norm ≤ 0.5): 适中阈值
        后期步 (t_norm ≤ 0.2): 框已稳定, 用宽松阈值 (多退出)
        """
        if t_norm > 0.5:
            return 0.002, 0.99
        elif t_norm > 0.2:
            return 0.005, 0.98
        else:
            return 0.01, 0.95

    def _check_head_convergence(
        self,
        curr_bboxes: torch.Tensor,
        prev_bboxes: torch.Tensor,
        curr_logits: torch.Tensor,
        prev_logits: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[bool, Dict]:
        """检测级联头是否收敛。

        判据:
        1. 框位置: 框对角线归一化的 L2 变化 < box_threshold
        2. 分类: 高置信度框 (sigmoid > 0.3) 的 argmax 一致率 > cls_threshold

        Args:
            curr_bboxes: 当前头预测框 [bs, N, 4] xyxy
            prev_bboxes: 上一头预测框 [bs, N, 4] xyxy
            curr_logits: 当前头分类 logits [bs, N, C]
            prev_logits: 上一头分类 logits [bs, N, C]
            t: 时间步 [bs], 范围 [0, timesteps]

        Returns:
            (converged, stats): 是否收敛, 统计信息字典
        """
        # 归一化时间步
        if torch.is_tensor(t):
            t_norm = (t.mean() / self.timesteps).item()
        else:
            t_norm = float(t) / self.timesteps

        # 获取阈值
        if self.head_exit_time_aware:
            box_thr, cls_thr = self._get_exit_threshold(t_norm)
        else:
            box_thr = self.head_exit_box_threshold
            cls_thr = self.head_exit_cls_threshold

        # 框位置收敛: 用框对角线长度归一化
        w = (curr_bboxes[..., 2] - curr_bboxes[..., 0]).clamp(min=1.0)
        h = (curr_bboxes[..., 3] - curr_bboxes[..., 1]).clamp(min=1.0)
        box_scale = torch.sqrt(w * w + h * h)
        delta = (curr_bboxes - prev_bboxes).norm(dim=-1)
        relative_delta = (delta / box_scale).mean()

        # 分类收敛: 高置信度框的 argmax 一致率
        curr_scores = torch.sigmoid(curr_logits).max(dim=-1)[0]
        valid_mask = curr_scores > 0.3
        if valid_mask.any():
            curr_label = curr_logits.argmax(dim=-1)
            prev_label = prev_logits.argmax(dim=-1)
            consistency = (curr_label == prev_label).float()
            mean_consistency = (
                (consistency * valid_mask.float()).sum()
                / valid_mask.float().sum()
            )
        else:
            # 无高置信度框, 默认分类收敛
            mean_consistency = torch.tensor(
                1.0, device=curr_bboxes.device
            )

        converged = bool(
            relative_delta < box_thr and mean_consistency > cls_thr
        )

        stats = {
            'mean_relative_delta': relative_delta.item(),
            'mean_consistency': mean_consistency.item(),
            'box_thr': box_thr,
            'cls_thr': cls_thr,
            't_norm': t_norm,
            'converged': converged,
        }
        return converged, stats

    # ================================================================
    # IO1: 自适应步数提前终止辅助方法
    # ================================================================

    def _check_step_convergence(
        self,
        x0_curr: torch.Tensor,
        x0_prev: torch.Tensor,
        cls_curr: torch.Tensor,
        cls_prev: torch.Tensor,
        t_curr: float,
    ) -> Tuple[bool, Dict]:
        """检测相邻采样步的 x0 预测是否收敛。

        RF 直线路径理论: t→0 时 x_t = (1-t)x_0 + t·noise → x_0,
        模型预测的 x0_pred 在小 t 时趋于稳定。

        判据 (batch 级别, 标量比较):
        1. 框位置: ‖x0_curr - x0_prev‖ / ‖x0_curr‖ < step_exit_threshold
        2. 分类: 高置信度框 (sigmoid > 0.3) 的 argmax 一致率 > step_exit_cls_threshold

        Args:
            x0_curr: 当前步 x0 预测 [bs, N, 4]
            x0_prev: 上一步 x0 预测 [bs, N, 4]
            cls_curr: 当前步分类 logits [bs, N, C]
            cls_prev: 上一步分类 logits [bs, N, C]
            t_curr: 当前归一化时间步 [0, 1]

        Returns:
            (converged, stats): 是否收敛 (batch 标量), 统计信息字典
        """
        # 框位置收敛: 相对 L2 变化
        # 用 x0_curr 的范数归一化, 避免 scale 依赖
        delta = (x0_curr - x0_prev).norm(dim=-1)  # [bs, N]
        magnitude = x0_curr.norm(dim=-1).clamp(min=1e-6)  # [bs, N]
        relative_delta = (delta / magnitude).mean()  # 标量

        # 分类收敛: 高置信度框的 argmax 一致率
        curr_scores = torch.sigmoid(cls_curr).max(dim=-1)[0]  # [bs, N]
        valid_mask = curr_scores > 0.3
        if valid_mask.any():
            curr_label = cls_curr.argmax(dim=-1)  # [bs, N]
            prev_label = cls_prev.argmax(dim=-1)  # [bs, N]
            consistency = (curr_label == prev_label).float()
            mean_consistency = (
                (consistency * valid_mask.float()).sum()
                / valid_mask.float().sum()
            )
        else:
            # 无高置信度框, 默认分类收敛
            mean_consistency = torch.tensor(
                1.0, device=x0_curr.device
            )

        converged = bool(
            relative_delta < self.step_exit_threshold
            and mean_consistency > self.step_exit_cls_threshold
        )

        stats = {
            'mean_relative_delta': relative_delta.item(),
            'mean_consistency': mean_consistency.item(),
            'box_thr': self.step_exit_threshold,
            'cls_thr': self.step_exit_cls_threshold,
            't_curr': float(t_curr),
            'converged': converged,
        }
        return converged, stats

    # ================================================================
    # 训练损失
    # ================================================================

    def loss(self, features, img_metas, gt_bboxes, gt_labels, x_raw_shared=None):
        device = features[0].device
        bs = len(img_metas)

        targets = self._normalize_targets(gt_bboxes, gt_labels, img_metas, bs)
        t = self._sample_t(bs, device)
        x_boxes, x_starts, x_noises, matched_gt_indices = (
            self._build_training_targets(
                bs, device, t, targets, gt_bboxes,
                external_noise=x_raw_shared,
            )
        )
        x_noisy_batch = torch.stack(x_boxes)
        curr_bboxes = self._sampler.raw_to_xyxy(x_noisy_batch, img_metas)

        t_input = t if self.diffusion_type == 'ddpm' else t * self.timesteps

        # 模型前向：若启用 AMP，在 autocast 下执行（线性层/attention 用半精度加速）
        if self.amp_dtype is not None:
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                all_cls_logits, all_pred_bboxes, all_curr_proposals = self(
                    features, curr_bboxes, t_input
                )
            # autocast 输出可能为半精度，criterion 需 FP32（如 cdist 不支持 BF16）
            all_cls_logits = all_cls_logits.float()
            all_pred_bboxes = all_pred_bboxes.float()
        else:
            all_cls_logits, all_pred_bboxes, all_curr_proposals = self(
                features, curr_bboxes, t_input
            )

        norm_pred_bboxes = self._normalize_pred_bboxes(
            all_pred_bboxes, img_metas
        )
        outputs = self._build_outputs(all_cls_logits, norm_pred_bboxes)
        # 方向三: 传 t 给 criterion (若 criterion 不支持 t 则被忽略, 向后兼容)
        # t 是 [bs] 的扩散时间, 用于 SNR 感知匹配和损失加权
        losses = self.criterion(outputs, targets, t=t)

        return losses

    # ================================================================
    # 推理
    # ================================================================

    @torch.no_grad()
    def predict(
        self, features, img_metas, rescale=True, return_trajectory=False
    ):
        device = features[0].device
        bs = len(img_metas)
        time_pairs = self._sampler.build_time_pairs(device)
        x_raw = torch.randn(bs, self.num_proposals, 4, device=device)

        ensemble_results = []
        trajectory = []
        dpm_solver = self._sampler.create_dpm_solver()
        if dpm_solver is not None:
            dpm_solver.reset()

        # IO4: 收集每步提前退出统计
        exit_head_indices = []

        # IO1: 步级收敛检测状态
        x0_prev = None
        cls_prev = None
        step_exit_idx = None  # 收敛退出的步索引 (None 表示未退出)

        for step_idx, (t_curr, t_next) in enumerate(time_pairs):
            cls_logits, pred_bboxes, x0_raw = self._forward_at_t(
                features, x_raw, t_curr, img_metas
            )

            # IO4: 记录本步退出头索引
            if (
                self.head_early_exit_enabled
                and self._exit_stats
                and 'exit_head_idx' in self._exit_stats
            ):
                exit_head_indices.append(
                    self._exit_stats['exit_head_idx']
                )

            # IO3: Top-K 框剪枝 — 在指定步后保留 Top-K 高置信框
            if (
                self.topk_pruning_enabled
                and step_idx == self.topk_pruning_step
                and x_raw.shape[1] > self.topk_k
            ):
                x_raw, cls_logits, pred_bboxes, x0_raw, _ = (
                    self._sampler.apply_topk_pruning(
                        x_raw, cls_logits, pred_bboxes, x0_raw,
                        k=self.topk_k,
                    )
                )
                # 剪枝后 DPM-Solver history 维度不匹配, 必须重置
                if dpm_solver is not None:
                    dpm_solver.reset()
                # 记录剪枝统计 (供 SwanLab 插桩)
                scores = torch.sigmoid(cls_logits).max(-1)[0]
                self._pruning_stats = {
                    'pruning_step': step_idx,
                    'n_before': self.num_proposals,
                    'n_after': self.topk_k,
                    'kept_mean_score': scores.mean().item(),
                }
                # 剪枝改变了 proposal 集合, 清空 IO1 收敛历史
                x0_prev = None
                cls_prev = None

            if return_trajectory:
                trajectory.append((cls_logits.detach(), pred_bboxes.detach()))
            if self.use_ensemble:
                ensemble_results.append((cls_logits, pred_bboxes))

            # Always keep last result for non-ensemble (DDPM single-step)
            if not ensemble_results:
                ensemble_results.append((cls_logits, pred_bboxes))

            # IO1: 自适应步数提前终止 (仅推理时启用)
            # 检测当前步与上一步的 x0_pred 收敛, 收敛则填充剩余步并退出
            if (
                self.step_early_exit_enabled
                and x0_prev is not None
                and step_idx >= self.step_exit_min_steps
                and step_idx < len(time_pairs) - 1
            ):
                converged, stats = self._check_step_convergence(
                    x0_raw, x0_prev, cls_logits, cls_prev, t_curr,
                )
                if converged:
                    # 用当前步结果填充剩余 ensemble 步, 保持多步投票一致性
                    for _ in range(len(time_pairs) - step_idx - 1):
                        ensemble_results.append(
                            (cls_logits.clone(), pred_bboxes.clone())
                        )
                    step_exit_idx = step_idx
                    self._step_exit_stats = {
                        'exit_step_idx': step_idx,
                        'total_steps': len(time_pairs),
                        **stats,
                    }
                    break

            # 训练时不触发 IO1, 但 predict 仅推理调用, 上述分支已足够

            if self.diffusion_type == 'ddpm':
                curr_bboxes_xyxy, x_raw = self._sampler.ddim_step(
                    t_curr,
                    t_next,
                    x_raw,
                    cls_logits,
                    pred_bboxes,
                    img_metas,
                    self.alphas_cumprod,
                )
                if t_next < 0:
                    break
            else:
                if dpm_solver is not None:
                    x_raw = dpm_solver.step(x_raw, x0_raw, t_curr, step_idx)
                elif self.solver_type == 'heun' and t_next > 0:
                    def model_fn(x_tmp, t_tmp):
                        _, _, x0_tmp = self._forward_at_t(
                            features, x_tmp, t_tmp, img_metas
                        )
                        return x0_tmp, None
                    x_raw = self.rf.heun_step(
                        x_raw, x0_raw, t_curr, t_next, model_fn
                    )
                else:
                    x_raw = self.rf.step(
                        x_raw, x0_raw, t_curr, t_next
                    )

                if self.box_renewal:
                    x_raw = self._sampler.apply_box_renewal(x_raw, cls_logits)
                if t_next <= 0:
                    break

            # IO1: 更新收敛检测历史 (仅当本步未触发提前退出时)
            # 若已 break, 此行不执行
            x0_prev = x0_raw
            cls_prev = cls_logits

        results = self._sampler.post_process(
            ensemble_results, img_metas, rescale
        )

        # IO3: SwanLab 插桩 — 上传剪枝统计
        if self.topk_pruning_enabled and self._pruning_stats:
            try:
                import swanlab
                swanlab.log({
                    'inference/pruning_n_before': self._pruning_stats['n_before'],
                    'inference/pruning_n_after': self._pruning_stats['n_after'],
                    'inference/pruning_kept_mean_score': self._pruning_stats['kept_mean_score'],
                    'inference/pruning_step': self._pruning_stats['pruning_step'],
                })
            except Exception:
                pass

        # IO4: SwanLab 插桩 — 上传级联头提前退出统计
        if self.head_early_exit_enabled and exit_head_indices:
            try:
                import swanlab
                # 实际执行头数 = exit_idx + 1 (exit_idx 从 0 开始)
                active_heads_list = [i + 1 for i in exit_head_indices]
                avg_active = sum(active_heads_list) / len(active_heads_list)
                swanlab.log({
                    'inference/early_exit_avg_active_heads': avg_active,
                    'inference/early_exit_total_heads': self.num_heads,
                    'inference/early_exit_exit_steps': len(exit_head_indices),
                    'inference/early_exit_saving_ratio': 1.0 - avg_active / self.num_heads,
                })
            except Exception:
                pass

        # IO1: SwanLab 插桩 — 上传步级提前终止统计
        if self.step_early_exit_enabled and step_exit_idx is not None:
            try:
                import swanlab
                total_steps = len(time_pairs)
                # 实际执行步数 = exit_idx + 1 (exit_idx 从 0 开始)
                # Heun 每步 2 NFE, exit 后省略 (total - exit - 1) 步 = 2*(total-exit-1) NFE
                active_steps = step_exit_idx + 1
                swanlab.log({
                    'inference/step_exit_active_steps': active_steps,
                    'inference/step_exit_total_steps': total_steps,
                    'inference/step_exit_saving_ratio': 1.0 - active_steps / total_steps,
                    'inference/step_exit_relative_delta': self._step_exit_stats.get('mean_relative_delta', 0.0),
                    'inference/step_exit_consistency': self._step_exit_stats.get('mean_consistency', 0.0),
                })
            except Exception:
                pass

        if return_trajectory:
            return results, trajectory
        return results

    # ================================================================
    # 训练辅助
    # ================================================================

    def _normalize_targets(self, gt_bboxes, gt_labels, img_metas, bs):
        targets = []
        for i in range(bs):
            h, w = _get_img_shape(img_metas[i])[:2]
            scale = gt_bboxes[i].new_tensor([w, h, w, h])
            targets.append(
                InstanceData(
                    labels=gt_labels[i],
                    bboxes=gt_bboxes[i] / scale,
                    img_shape=(h, w),
                )
            )
        return targets

    def _sample_t(self, bs, device):
        if self.diffusion_type == 'ddpm':
            return torch.randint(
                0, self.timesteps, (bs,), device=device
            ).long()
        t = torch.rand((bs,), device=device)
        if self.rf_schedule == 'shifted':
            t = self.rf_shift * t / (1 + (self.rf_shift - 1) * t)
        return t

    def _build_training_targets(self, bs, device, t, targets, gt_bboxes, external_noise=None):
        x_boxes, x_starts, x_noises, matched_gt_indices = [], [], [], []
        for i in range(bs):
            num_gt = gt_bboxes[i].shape[0]
            if num_gt == 0:
                if external_noise is not None:
                    noise = external_noise[i]
                else:
                    noise = torch.randn(self.num_proposals, 4, device=device)
                x_boxes.append(noise)
                x_starts.append(torch.zeros_like(noise))
                x_noises.append(noise)
                matched_gt_indices.append(
                    torch.zeros(
                        self.num_proposals, dtype=torch.long, device=device
                    )
                )
                continue
            norm_gt_cxcywh = bbox_xyxy_to_cxcywh(targets[i].bboxes)
            gt_diffusion = (norm_gt_cxcywh * 2 - 1) * self.snr_scale
            if external_noise is not None:
                noise = external_noise[i]
            else:
                noise = torch.randn(self.num_proposals, 4, device=device)
            x_start, matched_idx = self._couple_single_image(
                noise, gt_diffusion, targets[i].labels, device
            )
            matched_gt_indices.append(matched_idx)
            x_noisy, x_noise = self._forward_diffusion(
                x_start, noise, t[i : i + 1]
            )
            x_starts.append(x_start)
            x_noises.append(x_noise)
            x_boxes.append(x_noisy)
        return x_boxes, x_starts, x_noises, matched_gt_indices

    def _couple_single_image(self, noise, gt_diffusion, gt_labels, device):
        if self.ot_coupling and self.diffusion_type == 'rectified_flow':
            return self.ot_module.couple(
                noise, gt_diffusion, gt_labels, device
            )
        num_gt = gt_diffusion.shape[0]
        idx = torch.randint(0, num_gt, (self.num_proposals,), device=device)
        return gt_diffusion[idx], idx

    def _forward_diffusion(self, x_start, noise, t):
        if self.diffusion_type == 'ddpm':
            return self.q_sample(x_start, t), torch.zeros_like(x_start)
        x_noisy, _ = self.rf.q_sample(x_start, x_noise=noise, t=t)
        return x_noisy, noise

    def _forward_at_t(self, features, x_raw, t, img_metas):
        bs, device = x_raw.shape[0], x_raw.device
        curr_bboxes = self._sampler.raw_to_xyxy(x_raw, img_metas)
        t_input = torch.full((bs,), t * self.timesteps, device=device)
        # AMP: 推理前向用半精度加速 GEMM (FFN 占 42.4% 瓶颈),
        # 输出转回 fp32 以保证后续 box_renewal / NMS / solver 的数值精度
        if self.amp_dtype is not None:
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                cls_logits_seq, pred_bboxes_seq, _ = self(
                    features, curr_bboxes, t_input
                )
            cls_logits_seq = cls_logits_seq.float()
            pred_bboxes_seq = pred_bboxes_seq.float()
        else:
            cls_logits_seq, pred_bboxes_seq, _ = self(
                features, curr_bboxes, t_input
            )
        cls_logits_last = cls_logits_seq[-1]
        pred_bboxes_last = pred_bboxes_seq[-1]
        x0 = self._sampler.xyxy_to_raw(pred_bboxes_last, img_metas)

        return cls_logits_last, pred_bboxes_last, x0

    def _normalize_pred_bboxes(self, all_pred_bboxes, img_metas):
        # 构建 scale 张量并广播除法，消除逐 head 逐 image 的双重循环
        # all_pred_bboxes: [num_heads, bs, num_proposals, 4]
        num_heads = all_pred_bboxes.shape[0]
        bs = len(img_metas)
        # 构建 [bs, 4] 的 scale 张量
        scales = all_pred_bboxes.new_zeros(bs, 4)
        for i in range(bs):
            h, w = _get_img_shape(img_metas[i])[:2]
            scales[i] = all_pred_bboxes.new_tensor([w, h, w, h])
        # [num_heads, bs, 1, 4] / [1, bs, 1, 4] → [num_heads, bs, num_proposals, 4]
        return all_pred_bboxes / scales.unsqueeze(0).unsqueeze(2)

    def _build_outputs(self, all_cls_logits, norm_pred_bboxes):
        main_logits = all_cls_logits[-1]
        main_bboxes = norm_pred_bboxes[-1]
        aux_outputs = None
        if self.deep_supervision and all_cls_logits.shape[0] > 1:
            aux_outputs = [
                ModelOutput(
                    pred_logits=all_cls_logits[i],
                    pred_boxes=norm_pred_bboxes[i],
                )
                for i in range(all_cls_logits.shape[0] - 1)
            ]
        return ModelOutput(
            pred_logits=main_logits,
            pred_boxes=main_bboxes,
            aux_outputs=aux_outputs,
        )

    # ================================================================
    # DDPM 基线 (用于与 RF 对比实验)
    # ================================================================

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # 先按 t 索引取 bs 个值, 再 sqrt (bs << timesteps, 避免对全长向量开方)
        alpha_t = self.alphas_cumprod.gather(-1, t.long())
        reshape = (-1,) + (1,) * (x_start.dim() - 1)
        sqrt_alpha = alpha_t.sqrt().reshape(reshape)
        sqrt_one_minus = (1.0 - alpha_t).sqrt().reshape(reshape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise
