"""DiT 架构的扩散检测头

使用 Deformable Cross-Attention DiT Block 替代 RoIAlign + DynamicConv，
实现框-图像的双向全局交互。

共享模块:
- OTCoupling: OT 耦合 (Sinkhorn / Stochastic / Group-Hierarchical / KCEC)
- DiffusionSampler: 推理采样 (Heun / DPM-Solver++)、坐标转换、后处理
- RectifiedFlow: RF 扩散过程
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from .box_tokenizer import BoxTokenizer
from .deformable_attn import flatten_fpn_features
from .dit_single_head import DiTSingleHead
from .embeddings import SinusoidalPositionEmbeddings
from .noise_schedule import cosine_noise_schedule, load_buffer
from .ot_coupling import OTCoupling
from .rectified_flow import RectifiedFlow
from .sampling import DiffusionSampler, _get_img_shape
from .structures import InstanceData, ModelOutput
from .utils import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh


class DiTDiffusionDetHead(nn.Module):
    """DiT 架构的扩散检测头

    替代 DiffusionDetHead，使用 Deformable Cross-Attention DiT Block
    替代 RoIAlign + DynamicConv，实现框-图像的双向全局交互。

    与 DiffusionDetHead 共享:
    - RF/DDPM 扩散流程
    - OT 耦合 (通过 OTCoupling 模块)
    - 推理采样 (通过 DiffusionSampler 模块)
    - 检测损失 (Focal + L1 + GIoU)
    - Deep Supervision

    DiT 特有:
    - BoxTokenizer: 将框坐标编码为 token
    - Deformable Cross-Attention: 全局框-图像交互
    - 归一化坐标空间: 预测框在 [0,1] 归一化空间
    """

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
        ddim_sampling_eta: float = 1.0,
        diffusion_type: str = 'rectified_flow',
        rf_schedule: str = 'linear',
        rf_power: float = 1.0,
        rf_shift: float = 1.0,
        single_head: nn.Module = None,
        criterion: nn.Module = None,
        use_nms: bool = True,
        nms_thr: float = 0.5,
        score_thr: float = 0.05,
        min_keep: int = 60,
        prediction_mode: str = 'x0',
        velocity_loss_weight: float = 1.0,
        # OT 耦合参数
        ot_coupling: bool = False,
        ot_matcher: str = 'nearest',
        ot_epsilon: float = 1.0,
        ot_num_iters: int = 20,
        ot_sample: bool = False,
        ot_sample_seed: Optional[int] = None,
        # LSAS 参数
        use_lsas: bool = False,
        lsas_num_bins: int = 100,
        lsas_temp: float = 1.0,
        # 训练稳定化参数
        t_sampling: str = 'uniform',
        t_sampling_bins: int = 8,
        # DiT 特有参数
        num_fpn_levels: int = 4,
        num_ref_points: int = 8,
        box_init_mode: str = 'zero',
        adaln_params: int = 9,
        regression_mode: str = 'direct',
        use_adaln_zero: bool = True,
        num_blocks: int = 1,
        share_heads: bool = True,
    ):
        super().__init__()

        # ---- 基础配置 ----
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_proposals = num_proposals
        self.num_heads = num_heads
        self.snr_scale = snr_scale
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.box_renewal = box_renewal
        self.use_ensemble = use_ensemble
        self.deep_supervision = deep_supervision
        self.ddim_sampling_eta = ddim_sampling_eta
        self.diffusion_type = diffusion_type
        self.rf_schedule = rf_schedule
        self.rf_power = rf_power
        self.rf_shift = rf_shift
        self.solver_type = solver_type
        self.prediction_mode = prediction_mode
        self.velocity_loss_weight = velocity_loss_weight
        self.ot_coupling = ot_coupling
        self.use_lsas = use_lsas
        self.lsas_num_bins = lsas_num_bins
        self.lsas_temp = lsas_temp
        self.t_sampling = t_sampling
        self.t_sampling_bins = t_sampling_bins
        self.num_fpn_levels = num_fpn_levels
        self.num_ref_points = num_ref_points
        self.box_init_mode = box_init_mode
        self.adaln_params = adaln_params
        self.use_adaln_zero = use_adaln_zero

        # 测试配置
        self.use_nms = use_nms
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.min_keep = min_keep

        # ---- 子模块 ----
        self.criterion = criterion

        # OT 耦合模块
        self.ot_module = OTCoupling(
            ot_coupling=ot_coupling,
            ot_matcher=ot_matcher,
            ot_epsilon=ot_epsilon,
            ot_num_iters=ot_num_iters,
            ot_sample=ot_sample,
            ot_sample_seed=ot_sample_seed,
        )

        # 采样器 (非 nn.Module)
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

        # ---- 扩散过程参数 ----
        if self.diffusion_type == 'ddpm':
            self._build_diffusion_buffers()
        elif self.diffusion_type == 'rectified_flow':
            self.rf = RectifiedFlow(snr_scale=snr_scale)

        # ---- DiT 特有组件 ----
        self.box_tokenizer = BoxTokenizer(
            feat_channels=feat_channels,
            num_fpn_levels=num_fpn_levels,
            init_mode=box_init_mode,
            num_proposals=num_proposals,
        )

        # ---- 检测头序列 ----
        if single_head is not None:
            if isinstance(single_head, dict):
                if 'num_blocks' not in single_head:
                    single_head['num_blocks'] = num_blocks
                if share_heads:
                    shared_module = MODELS.build(single_head)
                    self.head_series = nn.ModuleList(
                        [shared_module for _ in range(num_heads)]
                    )
                else:
                    self.head_series = nn.ModuleList(
                        [MODELS.build(single_head) for _ in range(num_heads)]
                    )
            else:
                if share_heads:
                    self.head_series = nn.ModuleList(
                        [single_head for _ in range(num_heads)]
                    )
                else:
                    import copy

                    self.head_series = nn.ModuleList(
                        [copy.deepcopy(single_head) for _ in range(num_heads)]
                    )
        else:
            self.head_series = nn.ModuleList(
                [
                    DiTSingleHead(
                        num_classes=num_classes,
                        feat_channels=feat_channels,
                        num_heads=8,
                        num_fpn_levels=num_fpn_levels,
                        num_ref_points=num_ref_points,
                        prediction_mode=prediction_mode,
                        adaln_params=adaln_params,
                        regression_mode=regression_mode,
                        use_adaln_zero=use_adaln_zero,
                        num_blocks=num_blocks,
                    )
                    for _ in range(num_heads)
                ]
            )

        # ---- 时间步嵌入 MLP ----
        time_dim = feat_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(feat_channels),
            nn.Linear(feat_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # LSAS: 可学习时间分布
        if self.use_lsas and self.diffusion_type == 'rectified_flow':
            self.lsas_logits = nn.Parameter(torch.zeros(lsas_num_bins))
            self.lsas_bin_edges = torch.linspace(0, 1, lsas_num_bins + 1)

        self.prior_prob = prior_prob
        self._init_weights()

    # ================================================================
    # 初始化
    # ================================================================

    def _init_weights(self):
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if m.out_features in [
                        self.num_classes,
                        self.num_classes + 1,
                    ]:
                        nn.init.constant_(m.bias, bias_value)
                    else:
                        nn.init.constant_(m.bias, 0)

    # ================================================================
    # DDPM 扩散过程 (与 DiffusionDetHead 一致)
    # ================================================================

    def _build_diffusion_buffers(self):
        betas = cosine_noise_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        alphas_cumprod = alphas_cumprod.float()
        alphas_cumprod_prev = alphas_cumprod_prev.float()

        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            'sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            'sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        posterior_variance = (
            betas.float()
            * (1.0 - alphas_cumprod_prev)
            / (1.0 - alphas_cumprod)
        )
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(posterior_variance.clamp(min=1e-20)),
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = load_buffer(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = load_buffer(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    # ================================================================
    # DiT 特有: 坐标转换
    # ================================================================

    def _xyxy_to_raw(self, bboxes, img_metas):
        """xyxy 图像坐标 → raw diffusion 空间"""
        x0 = bboxes.clone()
        for i, meta in enumerate(img_metas):
            h, w = _get_img_shape(meta)[:2]
            scale = x0.new_tensor([w, h, w, h])
            x0[i] /= scale
        x0 = bbox_xyxy_to_cxcywh(x0)
        x0 = (x0 * 2 - 1) * self.snr_scale
        return x0

    def _raw_to_xyxy(self, raw_bboxes, img_metas):
        """raw diffusion 空间 → xyxy 图像坐标"""
        raw_bboxes = torch.nan_to_num(
            raw_bboxes, nan=0.0, posinf=0.0, neginf=0.0
        )
        bboxes = (
            raw_bboxes.clamp(-self.snr_scale, self.snr_scale) / self.snr_scale
            + 1
        ) / 2
        bboxes = bbox_cxcywh_to_xyxy(bboxes)
        bboxes = torch.nan_to_num(bboxes, nan=0.5, posinf=1.0, neginf=0.0)
        bboxes = bboxes.clamp(0, 1)
        bboxes[..., 0] = torch.min(bboxes[..., 0], bboxes[..., 2] - 1e-4)
        bboxes[..., 1] = torch.min(bboxes[..., 1], bboxes[..., 3] - 1e-4)
        for i, meta in enumerate(img_metas):
            h, w = _get_img_shape(meta)[:2]
            scale = bboxes.new_tensor([w, h, w, h])
            bboxes[i] *= scale
        return bboxes

    def _init_inference_boxes(self, bs, device):
        """生成推理时的初始框

        spatial_prior 模式: 使用 anchor_boxes (均匀网格锚点) 作为初始框，
        避免 randn 经 clamp 后 width/height 退化为 0 导致 delta regression 失效。
        其他模式: 使用 randn 采样。
        """
        if self.box_init_mode == 'spatial_prior' and hasattr(
            self.box_tokenizer, 'anchor_boxes'
        ):
            anchors = self.box_tokenizer.anchor_boxes.to(device)
            x_raw = (anchors * 2 - 1) * self.snr_scale
            x_raw = x_raw.unsqueeze(0).expand(bs, -1, -1).clone()
            return x_raw
        else:
            return torch.randn(bs, self.num_proposals, 4, device=device)

    def _normalize_bboxes_for_tokenizer(self, bboxes, img_metas):
        """将 xyxy 图像坐标转为 [0,1] 归一化坐标供 BoxTokenizer 使用"""
        normed = bboxes.clone()
        for i, meta in enumerate(img_metas):
            h, w = _get_img_shape(meta)[:2]
            scale = normed.new_tensor([w, h, w, h])
            normed[i] /= scale
        normed = normed.clamp(0, 1)
        return normed

    def _normed_to_img(self, normed_bboxes, img_metas):
        """归一化 [0,1] 坐标 → xyxy 图像坐标"""
        img_bboxes = normed_bboxes.clone()
        for i, meta in enumerate(img_metas):
            h, w = _get_img_shape(meta)[:2]
            scale = img_bboxes.new_tensor([w, h, w, h])
            img_bboxes[i] *= scale
        return img_bboxes

    # ================================================================
    # 前向传播
    # ================================================================

    def forward(
        self,
        features: Tuple[Tensor],
        bboxes: Tensor,
        t: Tensor,
        proposals: Optional[Tensor] = None,
        img_metas: Optional[List] = None,
    ) -> Tuple:
        """DiT 前向传播，迭代去噪

        Args:
            features: FPN 特征元组
            bboxes: 当前边界框 [bs, num_proposals, 4] (xyxy, 图像坐标)
            t: 当前时间步 [bs]
            proposals: 未使用 (保留接口兼容)
            img_metas: 图像元信息列表，用于坐标归一化

        Returns:
            all_cls_logits, all_pred_bboxes, all_objectness, all_velocity, all_curr_proposals
        """
        bs, num_boxes = bboxes.shape[:2]
        device = bboxes.device

        time_emb = self.time_mlp(t)

        fpn_list = list(features)[: self.num_fpn_levels]
        while len(fpn_list) < self.num_fpn_levels:
            fpn_list.append(fpn_list[-1])
        fpn_flattened, spatial_shapes, level_start_index = (
            flatten_fpn_features(fpn_list)
        )

        if img_metas is not None:
            normed_bboxes = self._normalize_bboxes_for_tokenizer(
                bboxes, img_metas
            )
        else:
            normed_bboxes = bboxes.clamp(min=0)
            h, w = fpn_list[0].shape[2], fpn_list[0].shape[3]
            stride = 4
            normed_bboxes[..., [0, 2]] = normed_bboxes[..., [0, 2]] / (
                w * stride
            )
            normed_bboxes[..., [1, 3]] = normed_bboxes[..., [1, 3]] / (
                h * stride
            )
            normed_bboxes = normed_bboxes.clamp(0, 1)

        box_tokens, _ = self.box_tokenizer(normed_bboxes, fpn_list)

        inter_cls_logits = []
        inter_pred_bboxes = []
        inter_objectness = []
        inter_velocity = []
        inter_curr_proposals = []

        curr_tokens = box_tokens
        curr_normed = normed_bboxes

        for head in self.head_series:
            cls_logits, pred_bboxes, updated_tokens, objectness, velocity = (
                head(
                    curr_tokens,
                    fpn_flattened,
                    spatial_shapes,
                    level_start_index,
                    time_emb,
                    curr_normed,
                )
            )

            inter_cls_logits.append(cls_logits)
            inter_pred_bboxes.append(pred_bboxes)
            inter_objectness.append(objectness)
            inter_velocity.append(velocity)
            inter_curr_proposals.append(
                updated_tokens.unsqueeze(0)
                if updated_tokens.dim() == 3
                else updated_tokens
            )

            curr_tokens = updated_tokens
            curr_normed = pred_bboxes.detach().clamp(0, 1)

        if self.deep_supervision:
            return (
                torch.stack(inter_cls_logits),
                torch.stack(inter_pred_bboxes),
                inter_objectness,
                inter_velocity,
                inter_curr_proposals,
            )
        else:
            return (
                torch.stack(inter_cls_logits[-1:]),
                torch.stack(inter_pred_bboxes[-1:]),
                inter_objectness[-1:],
                inter_velocity[-1:],
                inter_curr_proposals[-1:],
            )

    # ================================================================
    # 训练
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

    def _build_training_targets(
        self, bs, device, t, targets, gt_bboxes, img_metas
    ):
        x_boxes = []
        x_starts = []
        x_noises = []
        matched_gt_indices = []

        for i in range(bs):
            num_gt = gt_bboxes[i].shape[0]
            if num_gt == 0:
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
            noise = torch.randn(self.num_proposals, 4, device=device)

            # OT 耦合或随机配对
            if self.ot_coupling and self.diffusion_type == 'rectified_flow':
                x_start, _, matched_idx = self.ot_module.couple(
                    noise,
                    gt_diffusion,
                    targets[i].labels,
                    device,
                )
                matched_gt_indices.append(matched_idx)
            else:
                idx = torch.randint(
                    0, num_gt, (self.num_proposals,), device=device
                )
                sample_bboxes = bbox_xyxy_to_cxcywh(targets[i].bboxes[idx])
                x_start = (sample_bboxes * 2 - 1) * self.snr_scale
                matched_gt_indices.append(idx)

            # 前向扩散
            if self.diffusion_type == 'ddpm':
                x_noisy = self.q_sample(x_start, t[i : i + 1])
                x_starts.append(x_start)
                x_noises.append(torch.zeros_like(x_start))
            else:
                x_noisy, _ = self.rf.q_sample(
                    x_start, x_noise=noise, t=t[i : i + 1]
                )
                x_starts.append(x_start)
                x_noises.append(noise)
            x_boxes.append(x_noisy)

        return x_boxes, x_starts, x_noises, matched_gt_indices

    def _sample_t(self, bs, device):
        """采样训练时间步"""
        if self.diffusion_type == 'ddpm':
            return torch.randint(
                0, self.timesteps, (bs,), device=device
            ).long(), None
        if self.use_lsas:
            return self._sample_time_lsas(bs, device)
        if self.t_sampling == 'stratified':
            n_bins = self.t_sampling_bins
            bin_size = bs // n_bins
            remainder = bs % n_bins
            parts = []
            for i in range(n_bins):
                n = bin_size + (1 if i < remainder else 0)
                t_bin = (i + torch.rand(n, device=device)) / n_bins
                parts.append(t_bin)
            t = torch.cat(parts).clamp(1e-5, 1.0 - 1e-5)
            t = t[torch.randperm(bs, device=device)]
        else:
            t = torch.rand((bs,), device=device)
        if self.rf_schedule == 'shifted':
            t = self.rf_shift * t / (1 + (self.rf_shift - 1) * t)
        return t, None

    def _sample_time_lsas(self, bs, device):
        probs = F.softmax(self.lsas_logits / self.lsas_temp, dim=0)
        bin_idx = torch.multinomial(probs, bs, replacement=True)
        bin_width = 1.0 / self.lsas_num_bins
        t = (bin_idx.float() + torch.rand(bs, device=device)) * bin_width
        t = t.clamp(1e-5, 1.0 - 1e-5)
        if self.rf_schedule == 'shifted':
            t = self.rf_shift * t / (1 + (self.rf_shift - 1) * t)
        log_probs = torch.log(probs[bin_idx] + 1e-10)
        return t, log_probs

    def loss(self, features, img_metas, gt_bboxes, gt_labels):
        device = features[0].device
        bs = len(img_metas)

        targets = self._normalize_targets(gt_bboxes, gt_labels, img_metas, bs)
        t, lsas_log_probs = self._sample_t(bs, device)

        x_boxes, x_starts, x_noises, matched_gt_indices = (
            self._build_training_targets(
                bs, device, t, targets, gt_bboxes, img_metas
            )
        )
        x_noisy_batch = torch.stack(x_boxes)
        curr_bboxes = self._raw_to_xyxy(x_noisy_batch, img_metas)

        t_input = t if self.diffusion_type == 'ddpm' else t * self.timesteps

        (
            all_cls_logits,
            all_pred_bboxes,
            all_objectness,
            all_velocity,
            all_curr_proposals,
        ) = self(features, curr_bboxes, t_input, img_metas=img_metas)

        # DiT: pred_bboxes 已在归一化空间，无需额外归一化
        norm_pred_bboxes = all_pred_bboxes

        outputs = ModelOutput(
            pred_logits=all_cls_logits[-1],
            pred_boxes=norm_pred_bboxes[-1],
            pred_objectness=all_objectness[-1]
            if all_objectness[0] is not None
            else None,
        )
        if self.deep_supervision and self.num_heads > 1:
            outputs.aux_outputs = [
                ModelOutput(
                    pred_logits=all_cls_logits[i],
                    pred_boxes=norm_pred_bboxes[i],
                    pred_objectness=all_objectness[i]
                    if all_objectness[0] is not None
                    else None,
                )
                for i in range(self.num_heads - 1)
            ]

        losses = self.criterion(outputs, targets)

        self._add_velocity_loss(
            losses, all_velocity, x_starts, x_noises, device
        )
        self._add_lsas_loss(losses, lsas_log_probs)

        return losses

    def _add_velocity_loss(
        self, losses, all_velocity, x_starts, x_noises, device
    ):
        if not (
            self.prediction_mode == 'velocity'
            and all_velocity[0] is not None
            and self.diffusion_type == 'rectified_flow'
        ):
            return
        v_target = torch.stack(x_noises) - torch.stack(x_starts)
        vel_weight = self.velocity_loss_weight
        last_v = all_velocity[-1]
        if last_v is not None:
            losses['loss_velocity'] = F.mse_loss(last_v, v_target) * vel_weight
        if self.deep_supervision and self.num_heads > 1:
            aux = torch.tensor(0.0, device=device)
            n = 0
            for hi in range(self.num_heads - 1):
                if all_velocity[hi] is not None:
                    aux = aux + F.mse_loss(all_velocity[hi], v_target)
                    n += 1
            if n > 0:
                losses['loss_velocity_aux'] = aux / n * vel_weight * 0.5

    @staticmethod
    def _add_lsas_loss(losses, lsas_log_probs):
        if lsas_log_probs is None:
            return
        with torch.no_grad():
            total = sum(losses.values())
        losses['loss_lsas'] = total.detach() * (-lsas_log_probs).mean() * 0.01

    # ================================================================
    # 推理
    # ================================================================

    def _forward_at_t(self, features, x_raw, t, img_metas):
        """在指定时间步 t 进行前向预测"""
        bs, device = x_raw.shape[0], x_raw.device
        curr_bboxes = self._raw_to_xyxy(x_raw, img_metas)
        t_input = torch.full((bs,), t * self.timesteps, device=device)
        cls_logits_seq, pred_bboxes_seq, _, _, _ = self(
            features, curr_bboxes, t_input, img_metas=img_metas
        )
        last_cls_logits = cls_logits_seq[-1]
        last_pred_bboxes = pred_bboxes_seq[-1]
        # DiT: pred_bboxes 在归一化空间，需先转图像坐标再转 raw
        last_pred_bboxes_img = self._normed_to_img(last_pred_bboxes, img_metas)
        x0_raw = self._xyxy_to_raw(last_pred_bboxes_img, img_metas)
        return last_cls_logits, last_pred_bboxes_img, x0_raw

    @torch.no_grad()
    def predict(
        self, features, img_metas, rescale=True, return_trajectory=False
    ):
        """推理模式的前向传播"""
        device = features[0].device
        bs = len(img_metas)

        # 1. 准备采样时间序列
        time_pairs = self._sampler.build_time_pairs(device)

        # 初始框
        x_raw = self._init_inference_boxes(bs, device)
        x0_prev = None
        ensemble_results = []
        trajectory = []

        dpm_solver = self._sampler.create_dpm_solver()

        # 2. 迭代采样
        for step_idx, (t_curr, t_next) in enumerate(time_pairs):
            cls_logits, pred_bboxes, x0_raw = self._forward_at_t(
                features, x_raw, t_curr, img_metas
            )

            x0_prev = x0_raw.detach()

            if return_trajectory:
                trajectory.append((cls_logits.detach(), pred_bboxes.detach()))

            if self.use_ensemble:
                ensemble_results.append((cls_logits, pred_bboxes))

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
                    x_raw = self.rf.step(x_raw, x0_raw, t_curr, t_next)

                if self.box_renewal:
                    x_raw = self._apply_box_renewal(x_raw, cls_logits)

                if t_next <= 0:
                    break

        # 3. 后处理
        results = self._sampler.post_process(
            ensemble_results, img_metas, rescale
        )

        if return_trajectory:
            return results, trajectory
        return results

    def _apply_box_renewal(self, x_raw, cls_logits):
        """DiT 特有的 box renewal: 支持 spatial_prior 模式用 anchor 替换低分框"""
        bs, device = x_raw.shape[0], x_raw.device
        scores = torch.sigmoid(cls_logits).max(-1)[0]
        x_raw_new = x_raw.clone()

        for i in range(bs):
            conf = scores[i]
            keep = conf > self.score_thr
            num_renew = (~keep).sum().item()
            if num_renew > 0:
                if self.box_init_mode == 'spatial_prior' and hasattr(
                    self.box_tokenizer, 'anchor_boxes'
                ):
                    anchors = self.box_tokenizer.anchor_boxes.to(device)
                    anchor_raw = (anchors * 2 - 1) * self.snr_scale
                    replace_idx = torch.randint(
                        0, anchor_raw.shape[0], (num_renew,), device=device
                    )
                    x_raw_new[i, ~keep] = anchor_raw[replace_idx]
                else:
                    x_raw_new[i, ~keep] = torch.randn(
                        num_renew, 4, device=device
                    )
        return x_raw_new
