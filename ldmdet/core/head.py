"""DiffusionDetHead — 核心迭代去噪检测头

直接移植自 LDMDet/mods/diffusiondet_head.py (已验证工作),
仅更新 import 路径到 ldmdet 纯 PyTorch 库。
"""

import copy
import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldmdet.coupling import build_coupling
from ldmdet.data.structures import (
    InstanceData,
    ModelOutput,
)
from ldmdet.diagnostics.instrumentation import probe
from ldmdet.diffusion.box_chart import ValidBoxChart
from ldmdet.diffusion.embeddings import SinusoidalPositionEmbeddings
from ldmdet.diffusion.noise_schedule import cosine_noise_schedule
from ldmdet.diffusion.rectified_flow import RectifiedFlow
from ldmdet.diffusion.sampling import DiffusionSampler, _get_img_shape
from ldmdet.utils.box_ops import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

logger = logging.getLogger(__name__)


def calibrate_class_logits(cls_logits, quality_logits, beta=2.0):
    """Fuse class probability and IoU quality, returning stable logits."""
    probability = cls_logits.sigmoid()
    quality = quality_logits.sigmoid()
    calibrated = probability * quality.pow(beta)
    calibrated = calibrated.clamp(min=1e-6, max=1.0 - 1e-6)
    return torch.logit(calibrated)


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
        box_parameterization: str = 'linear_cxcywh',
        box_chart_eps: float = 1e-6,
        box_chart_mean: Optional[list] = None,
        box_chart_covariance: Optional[list] = None,
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
        # IQC: IoU Quality Calibration. The single head predicts q(IoU), and
        # inference ranks detections by p(class) * q ** beta.
        quality_score_beta: float = 2.0,
        quality_calibration_mode: str = 'solver_coupled',
        quality_only_training: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_proposals = num_proposals
        self.num_heads = num_heads
        self.snr_scale = snr_scale
        self.box_parameterization = box_parameterization
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
        self.quality_score_beta = float(quality_score_beta)
        self.quality_calibration_mode = quality_calibration_mode
        self.quality_only_training = bool(quality_only_training)
        self._last_quality_logits = None
        if self.quality_score_beta < 0:
            raise ValueError('quality_score_beta must be non-negative')
        if self.quality_calibration_mode not in ('solver_coupled', 'final_only'):
            raise ValueError(
                'quality_calibration_mode must be solver_coupled or final_only')

        if box_parameterization not in ('linear_cxcywh', 'gap_ilr'):
            raise ValueError(
                'box_parameterization must be linear_cxcywh or gap_ilr, '
                f'got {box_parameterization!r}')
        if box_parameterization == 'gap_ilr':
            if diffusion_type != 'rectified_flow':
                raise ValueError('gap_ilr is currently implemented for RF only')
            if box_chart_mean is None or box_chart_covariance is None:
                raise ValueError(
                    'gap_ilr requires box_chart_mean and box_chart_covariance')
            self.box_chart = ValidBoxChart(
                eps=box_chart_eps,
                mean=torch.as_tensor(box_chart_mean, dtype=torch.float64),
                covariance=torch.as_tensor(
                    box_chart_covariance, dtype=torch.float64),
            )
        else:
            self.box_chart = None

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
        # C2 is deliberately a last-stage-only intervention. Removing the
        # cloned quality modules from earlier cascade heads also avoids unused
        # parameters under DistributedDataParallel.
        if getattr(single_head, 'quality_head', None) is not None:
            for head in self.head_series[:-1]:
                head.quality_head = None
                head.predict_iou_quality = False
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
            box_chart=self.box_chart,
        )

        self._init_weights(prior_prob)

        if self.quality_only_training:
            quality_head = getattr(self.head_series[-1], 'quality_head', None)
            if quality_head is None:
                raise ValueError(
                    'quality_only_training requires predict_iou_quality=True')
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for parameter in quality_head.parameters():
                parameter.requires_grad_(True)

        # AMP: 仅模型前向使用半精度，criterion 始终 FP32
        # 推荐值: torch.bfloat16 (同动态范围，无需 GradScaler)
        # 支持字符串 (配置文件无需 import torch, 避免 mmengine lazy_import 冲突)
        if isinstance(amp_dtype, str):
            amp_dtype = getattr(torch, amp_dtype)
        self.amp_dtype = amp_dtype

        if torch_compile and hasattr(torch, 'compile'):
            self.forward = torch.compile(self.forward, dynamic=True)

        # IO3: Top-K 框剪枝 (推理优化, 不影响训练)
        self.topk_pruning_enabled = topk_pruning_enabled
        self.topk_k = topk_k
        self.topk_pruning_step = topk_pruning_step
        self._pruning_stats = {}

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

    def forward(self, features, bboxes, t, img_metas=None):
        time_emb = self.time_mlp(t)
        # 探针: time_emb 激活统计 (训练时每 100 步, 推理时每次)
        if self.training:
            probe.record_tensor_stats('cascade/time_emb', time_emb)
        else:
            probe.record_inference_tensor_stats('cascade/time_emb', time_emb)
        inter_cls_logits = []
        inter_pred_bboxes = []
        inter_curr_proposals = []
        last_quality_logits = None
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
                cls_logits, pred_bboxes, curr_proposals, quality_logits = result
                last_quality_logits = quality_logits
            else:
                cls_logits, pred_bboxes, curr_proposals = result
            inter_cls_logits.append(cls_logits)
            inter_pred_bboxes.append(pred_bboxes)
            inter_curr_proposals.append(curr_proposals)

            # 探针: per-head 激活统计 (cls_logits + pred_bboxes)
            if self.training:
                probe.record_tensor_stats(f'cascade/head{i}/cls_logits', cls_logits)
                probe.record_tensor_stats(f'cascade/head{i}/pred_bboxes', pred_bboxes)
            else:
                probe.record_inference_tensor_stats(f'cascade/head{i}/cls_logits', cls_logits)
                probe.record_inference_tensor_stats(f'cascade/head{i}/pred_bboxes', pred_bboxes)

            # 级联: 将当前 head 输出作为下一 head 的输入
            curr_bboxes = (
                pred_bboxes.detach()
                if self.cascade_detach
                else pred_bboxes
            )
            prev_bboxes = pred_bboxes
            prev_logits = cls_logits

        self._last_quality_logits = last_quality_logits

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
    # 训练损失
    # ================================================================

    def loss(self, features, img_metas, gt_bboxes, gt_labels, x_raw_shared=None):
        if self.quality_only_training:
            # The detector may still build backbone/neck features normally,
            # but detaching here makes C2 a strict ranking-only intervention:
            # no existing representation, classifier, or regressor can drift.
            features = tuple(feature.detach() for feature in features)
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
                    features, curr_bboxes, t_input, img_metas
                )
            # autocast 输出可能为半精度，criterion 需 FP32（如 cdist 不支持 BF16）
            all_cls_logits = all_cls_logits.float()
            all_pred_bboxes = all_pred_bboxes.float()
        else:
            all_cls_logits, all_pred_bboxes, all_curr_proposals = self(
                features, curr_bboxes, t_input, img_metas
            )

        norm_pred_bboxes = self._normalize_pred_bboxes(
            all_pred_bboxes, img_metas
        )
        outputs = self._build_outputs(all_cls_logits, norm_pred_bboxes)
        if self._last_quality_logits is not None:
            # C2 isolates the final cascade stage. Auxiliary heads retain the
            # original losses so the experiment has a single causal change.
            outputs.pred_quality = self._last_quality_logits.float()
        losses = self.criterion(outputs, targets, t=t, box_targets=None)

        # 探针: 训练时 t 分布 + 损失分解
        probe.record_tensor_stats('train/t', t)
        for name, val in losses.items():
            if isinstance(val, torch.Tensor):
                probe.record_scalar(f'train/loss/{name}', val.item())

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

        # 探针: 推理开始标记 (清空推理缓冲区)
        probe.on_inference_begin()

        time_pairs = self._sampler.build_time_pairs(device)
        x_raw = torch.randn(bs, self.num_proposals, 4, device=device)

        ensemble_results = []
        trajectory = []
        dpm_solver = self._sampler.create_dpm_solver()
        # 显式校验: 防止 solver_type 拼写错误或未知值时静默降级到 Euler.
        # create_dpm_solver() 仅对 euler/heun 返回 None (合法), 其他 solver_type
        # 必须返回有效实例, 否则下方 if-elif-else 会静默走 Euler 分支.
        if dpm_solver is None:
            assert self.solver_type in ('euler', 'heun', 'ddim'), (
                f"solver_type='{self.solver_type}' 不被 create_dpm_solver() 支持, "
                f"且不属于 euler/heun/ddim. 请检查配置或扩展 create_dpm_solver()."
            )
        if dpm_solver is not None:
            dpm_solver.reset()

        # D3 化解路径 A: 记录上一步被 box_renewal 重置的 proposal mask
        # 在下一步 DPM-Solver++ step() 中传入, 对被 renewal 的 proposal
        # 置零 D1 校正项, 避免 renewal 噪声污染 x0_history 导致 D1 失效
        _renewal_mask: Optional[torch.Tensor] = None  # [bs, N] bool
        for step_idx, (t_curr, t_next) in enumerate(time_pairs):
            cls_logits, pred_bboxes, x0_raw = self._forward_at_t(
                features, x_raw, t_curr, img_metas
            )
            output_logits = (
                self._quality_ranking_logits(cls_logits)
                if self.quality_calibration_mode == 'final_only'
                else cls_logits
            )
            # 探针: 推理时 per-step 激活统计 (x0_pred + cls_logits + pred_bboxes)
            probe.record_inference_tensor_stats(
                f'inference/step{step_idx}/x0_pred', x0_raw
            )
            probe.record_inference_tensor_stats(
                f'inference/step{step_idx}/cls_logits', cls_logits
            )
            probe.record_inference_tensor_stats(
                f'inference/step{step_idx}/pred_bboxes', pred_bboxes
            )
            probe.record_inference_scalar(
                f'inference/step{step_idx}/t_curr', float(t_curr)
            )

            # IO3: Top-K 框剪枝 — 在指定步后保留 Top-K 高置信框
            if (
                self.topk_pruning_enabled
                and step_idx == self.topk_pruning_step
                and x_raw.shape[1] > self.topk_k
            ):
                x_raw, cls_logits, pred_bboxes, x0_raw, topk_indices = (
                    self._sampler.apply_topk_pruning(
                        x_raw, cls_logits, pred_bboxes, x0_raw,
                        k=self.topk_k,
                    )
                )
                output_logits = output_logits.gather(
                    1, topk_indices.unsqueeze(-1).expand(
                        -1, -1, output_logits.shape[-1]))
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
            if return_trajectory:
                trajectory.append((cls_logits.detach(), pred_bboxes.detach()))
            if self.use_ensemble:
                ensemble_results.append((output_logits, pred_bboxes))
            else:
                # Non-ensemble inference must return the latest solver result.
                # The previous ``if not ensemble_results`` implementation kept
                # step 0 forever, silently making later NFEs ineffective.
                ensemble_results[:] = [(output_logits, pred_bboxes)]

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
                    x_raw = dpm_solver.step(x_raw, x0_raw, t_curr, step_idx,
                                             renewal_mask=_renewal_mask)
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
                    # 探针: 记录 box_renewal 前的 x_raw (用于计算重置率)
                    x_raw_before = x_raw.clone()
                    n_before = x_raw.shape[1]
                    x_raw = self._sampler.apply_box_renewal(x_raw, cls_logits)
                    # D3 化解路径 A: 记录被 renewal 的 proposal mask
                    # 比较新旧 x_raw, 不一致的 proposal 即被 renewal
                    _renewal_mask = ~torch.isclose(
                        x_raw, x_raw_before, atol=1e-6
                    ).all(dim=-1)  # [bs, N]
                    # 探针: box_renewal 统计 (重置率 + 置信度分布)
                    n_after = x_raw.shape[1]
                    if n_before > 0:
                        renewal_rate = 1.0 - min(n_after, n_before) / n_before
                        probe.record_inference_scalar(
                            f'inference/step{step_idx}/box_renewal_rate', renewal_rate
                        )
                    # proposal 置信度分布 (max sigmoid score)
                    scores = torch.sigmoid(cls_logits).max(-1)[0]
                    probe.record_inference_tensor_stats(
                        f'inference/step{step_idx}/proposal_scores', scores
                    )
                # 探针: solver 推进后的 x_raw 统计
                probe.record_inference_tensor_stats(
                    f'inference/step{step_idx}/x_raw_after', x_raw
                )
                if t_next <= 0:
                    break

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
            except Exception as e:
                logger.warning(f'[SwanLab] inference metrics log failed: {e}')

        # R1 诊断: 收集本次推理的 eta_str 历史 (DPM-Solver++ 直线度指标)
        # dpm_solver.eta_str_history 在每个 step 后 append 一个值;
        # 4 步推理下通常 length=3 (step 0 是 linear, 无 D1; step 1/2/3 记录)
        # 保存到 self._last_eta_str_log 供实验脚本读取 (不写入 SwanLab, 避免污染训练指标)
        # 注意: 不用 try-except 吞错 — 如果 dpm_solver 缺少诊断字段, 应显式报错而非静默回退
        if dpm_solver is not None:
            self._last_eta_str_log = list(dpm_solver.eta_str_history)
            # 方向 A/D 诊断: 同时收集 per-dim eta_str 和 eta_3rd
            self._last_eta_str_per_dim_log = [
                list(x) for x in dpm_solver.eta_str_per_dim_history
            ]
            self._last_eta_3rd_log = list(dpm_solver.eta_3rd_history)
        else:
            self._last_eta_str_log = []
            self._last_eta_str_per_dim_log = []
            self._last_eta_3rd_log = []

        # 探针: 上传 solver 诊断量到推理缓冲区
        for i, eta in enumerate(self._last_eta_str_log):
            probe.record_inference_scalar(f'inference/eta_str/step{i}', eta)
        for i, eta in enumerate(self._last_eta_3rd_log):
            probe.record_inference_scalar(f'inference/eta_3rd/step{i}', eta)
        # per-dim eta_str (cx, cy, w, h)
        for i, eta_dim in enumerate(self._last_eta_str_per_dim_log):
            for j, dim_name in enumerate(['cx', 'cy', 'w', 'h']):
                if j < len(eta_dim):
                    probe.record_inference_scalar(
                        f'inference/eta_str_per_dim/step{i}/{dim_name}', eta_dim[j]
                    )

        # 探针: 推理结束, flush 推理缓冲区到 SwanLab
        probe.on_inference_end()

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

    def _build_training_targets(
        self, bs, device, t, targets, gt_bboxes,
        external_noise=None,
    ):
        """构建训练 targets (x_noisy, x_start, x_noise, matched_gt_idx)。"""
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
            gt_diffusion = self._sampler.normalized_xyxy_to_raw(
                targets[i].bboxes)
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
                    features, curr_bboxes, t_input, img_metas
                )
            cls_logits_seq = cls_logits_seq.float()
            pred_bboxes_seq = pred_bboxes_seq.float()
        else:
            cls_logits_seq, pred_bboxes_seq, _ = self(
                features, curr_bboxes, t_input, img_metas
            )
        cls_logits_last = cls_logits_seq[-1]
        pred_bboxes_last = pred_bboxes_seq[-1]
        if (
            self._last_quality_logits is not None
            and self.quality_calibration_mode == 'solver_coupled'
        ):
            cls_logits_last = calibrate_class_logits(
                cls_logits_last,
                self._last_quality_logits.float(),
                self.quality_score_beta,
            )
        x0 = self._sampler.xyxy_to_raw(pred_bboxes_last, img_metas)

        return cls_logits_last, pred_bboxes_last, x0

    def _quality_ranking_logits(self, cls_logits):
        """Apply quality only to emitted detection scores.

        In ``final_only`` mode the caller keeps ``cls_logits`` for Top-K,
        renewal and solver updates, and uses this result only in post-processing.
        """
        if self._last_quality_logits is None:
            return cls_logits
        return calibrate_class_logits(
            cls_logits,
            self._last_quality_logits.float(),
            self.quality_score_beta,
        )

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
