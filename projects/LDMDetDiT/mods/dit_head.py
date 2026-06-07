import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import batched_nms

from .box_tokenizer import (
    BoxTokenizer,
)
from .deformable_attn import flatten_fpn_features
from .dit_single_head import DiTSingleHead
from .modules import (
    SinusoidalPositionEmbeddings,
    cosine_noise_schedule,
    load_buffer,
)
from .rectified_flow import RectifiedFlow
from .structures import DetectionResult, InstanceData, ModelOutput
from .utils import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh


class DiTDiffusionDetHead(nn.Module):
    """DiT 架构的扩散检测头

    替代 DiffusionDetHead，使用 Deformable Cross-Attention DiT Block
    替代 RoIAlign + DynamicConv，实现框-图像的双向全局交互。

    保留与 DiffusionDetHead 相同的:
    - RF/DDPM 扩散流程
    - OT 耦合 (Sinkhorn / Stochastic / Group-Hierarchical / KCEC)
    - 检测损失 (Focal + L1 + GIoU)
    - 推理采样 (Heun / DPM-Solver++)
    - Deep Supervision
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
        ot_coupling: bool = False,
        ot_matcher: str = 'nearest',
        ot_epsilon: float = 1.0,
        ot_num_iters: int = 20,
        ot_sample: bool = False,
        ot_sample_seed: Optional[int] = None,
        use_lsas: bool = False,
        lsas_num_bins: int = 100,
        lsas_temp: float = 1.0,
        t_sampling: str = 'uniform',
        t_sampling_bins: int = 8,
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
        self.regression_mode = regression_mode
        self.velocity_loss_weight = velocity_loss_weight
        self.ot_coupling = ot_coupling
        self.ot_matcher = ot_matcher
        self.ot_epsilon = ot_epsilon
        self.ot_num_iters = ot_num_iters
        self.ot_sample = ot_sample
        self.ot_sample_seed = ot_sample_seed
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

        self.use_nms = use_nms
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.min_keep = min_keep

        self.criterion = criterion

        if self.diffusion_type == 'ddpm':
            self._build_diffusion_buffers()
        elif self.diffusion_type == 'rectified_flow':
            self.rf = RectifiedFlow(snr_scale=snr_scale)

        self.box_tokenizer = BoxTokenizer(
            feat_channels=feat_channels,
            num_fpn_levels=num_fpn_levels,
            init_mode=box_init_mode,
            num_proposals=num_proposals,
        )

        if single_head is not None:
            if isinstance(single_head, dict):
                # 注入 num_blocks 参数到 single_head 配置中
                if 'num_blocks' not in single_head:
                    single_head['num_blocks'] = num_blocks
                if share_heads:
                    shared_module = DiTSingleHead(**single_head)
                    self.head_series = nn.ModuleList(
                        [shared_module for _ in range(num_heads)]
                    )
                else:
                    self.head_series = nn.ModuleList(
                        [DiTSingleHead(**single_head) for _ in range(num_heads)]
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

        time_dim = feat_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(feat_channels),
            nn.Linear(feat_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        if self.use_lsas and self.diffusion_type == 'rectified_flow':
            self.lsas_logits = nn.Parameter(torch.zeros(lsas_num_bins))
            self.lsas_bin_edges = torch.linspace(0, 1, lsas_num_bins + 1)

        self.prior_prob = prior_prob

        self._init_weights()

    def _init_weights(self):
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for name, m in self.named_modules():
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

        for head in self.head_series:
            for dit_block in head.dit_blocks:
                if hasattr(dit_block, 'adaln_mlp') and getattr(
                    dit_block, 'use_adaln_zero', True
                ):
                    nn.init.zeros_(dit_block.adaln_mlp[-1].weight)
                    nn.init.zeros_(dit_block.adaln_mlp[-1].bias)
                elif hasattr(dit_block, 'adaln_mlp'):
                    nn.init.xavier_uniform_(
                        dit_block.adaln_mlp[-1].weight, gain=0.01
                    )
                    nn.init.constant_(dit_block.adaln_mlp[-1].bias, 0.1)

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

    # ---- 坐标转换 (与 DiffusionDetHead 一致) ----

    @staticmethod
    def _get_img_shape(meta):
        if isinstance(meta, dict):
            return meta.get('pad_shape') or meta.get('img_shape')
        return getattr(meta, 'pad_shape', None) or meta.img_shape

    @staticmethod
    def _get_scale_factor(meta):
        if isinstance(meta, dict):
            return meta.get('scale_factor')
        return meta.scale_factor

    def _xyxy_to_raw(self, bboxes, img_metas):
        x0 = bboxes.clone()
        for i, meta in enumerate(img_metas):
            h, w = self._get_img_shape(meta)[:2]
            scale = x0.new_tensor([w, h, w, h])
            x0[i] /= scale
        x0 = bbox_xyxy_to_cxcywh(x0)
        x0 = (x0 * 2 - 1) * self.snr_scale
        return x0

    def _init_inference_boxes(self, bs, device):
        """生成推理时的初始框。

        RF 模式: 从纯噪声 (标准正态分布) 开始, 与训练时的 x_noise 分布一致。
        DDPM 模式: spatial_prior 使用 anchor_boxes 提供位置先验。
        """
        if self.diffusion_type == 'ddpm' and self.box_init_mode == 'spatial_prior' and hasattr(
            self.box_tokenizer, 'anchor_boxes'
        ):
            anchors = self.box_tokenizer.anchor_boxes.to(device)
            # anchors: (num_proposals, 4) cxcywh [0,1]
            # 转换到 raw 空间: (cxcywh * 2 - 1) * snr_scale
            x_raw = (anchors * 2 - 1) * self.snr_scale
            x_raw = x_raw.unsqueeze(0).expand(bs, -1, -1).clone()
            return x_raw
        else:
            # RF 模式: 纯噪声, 与训练时 q_sample 的 x_noise 一致
            return torch.randn(bs, self.num_proposals, 4, device=device)

    def _raw_to_xyxy(self, raw_bboxes, img_metas):
        # 入口防护: 清洗上游可能传入的 NaN/Inf
        raw_bboxes = torch.nan_to_num(
            raw_bboxes, nan=0.0, posinf=0.0, neginf=0.0
        )
        bboxes = (
            raw_bboxes.clamp(-self.snr_scale, self.snr_scale) / self.snr_scale
            + 1
        ) / 2
        bboxes = bbox_cxcywh_to_xyxy(bboxes)
        # 二次清洗: cxcywh→xyxy 转换可能产生 NaN (w/h 为 0 时除零)
        bboxes = torch.nan_to_num(bboxes, nan=0.5, posinf=1.0, neginf=0.0)
        bboxes = bboxes.clamp(0, 1)
        bboxes[..., 0] = torch.min(bboxes[..., 0], bboxes[..., 2] - 1e-4)
        bboxes[..., 1] = torch.min(bboxes[..., 1], bboxes[..., 3] - 1e-4)
        for i, meta in enumerate(img_metas):
            h, w = self._get_img_shape(meta)[:2]
            scale = bboxes.new_tensor([w, h, w, h])
            bboxes[i] *= scale
        return bboxes

    def _normalize_bboxes_for_tokenizer(self, bboxes, img_metas):
        """将 xyxy 图像坐标转为 [0,1] 归一化坐标供 BoxTokenizer 使用

        归一化后 clamp 到 [0,1] 防止异常坐标传播到 reference_points。
        """
        normed = bboxes.clone()
        for i, meta in enumerate(img_metas):
            h, w = self._get_img_shape(meta)[:2]
            scale = normed.new_tensor([w, h, w, h])
            normed[i] /= scale
        normed = normed.clamp(0, 1)
        return normed

    def _raw_cxcywh_to_normed_xyxy(self, raw_bboxes):
        """将 raw 空间 cxcywh 预测转换为归一化 xyxy [0,1]

        raw = (cxcywh_normed * 2 - 1) * snr_scale
        逆变换: cxcywh_normed = (raw / snr_scale + 1) / 2
        """
        cxcywh = (
            raw_bboxes.clamp(-self.snr_scale, self.snr_scale) / self.snr_scale
            + 1
        ) / 2
        xyxy = bbox_cxcywh_to_xyxy(cxcywh)
        # 二次清洗: cxcywh→xyxy 转换可能产生 NaN (w/h 为 0 时除零)
        xyxy = torch.nan_to_num(xyxy, nan=0.5, posinf=1.0, neginf=0.0)
        return xyxy.clamp(0, 1)

    def _normed_xyxy_to_raw_cxcywh(self, normed_xyxy):
        """将归一化 xyxy [0,1] 转换为 raw 空间 cxcywh

        _raw_cxcywh_to_normed_xyxy 的逆变换。
        raw = (cxcywh_normed * 2 - 1) * snr_scale
        """
        cxcywh = bbox_xyxy_to_cxcywh(normed_xyxy)
        raw = (cxcywh * 2 - 1) * self.snr_scale
        return raw

    # ---- OT 耦合 (与 DiffusionDetHead 一致) ----

    def _ot_multinomial(self, row_probs):
        if self.ot_sample_seed is None:
            return torch.multinomial(row_probs, 1).squeeze(-1)
        if not hasattr(self, '_ot_sample_generators'):
            self._ot_sample_generators = {}
        device = row_probs.device
        key = str(device)
        if key not in self._ot_sample_generators:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(self.ot_sample_seed))
            self._ot_sample_generators[key] = gen
        return torch.multinomial(
            row_probs, 1, generator=self._ot_sample_generators[key]
        ).squeeze(-1)

    def _sinkhorn_transport(self, cost, row_mass=None, col_mass=None):
        N, K = cost.shape
        device = cost.device
        if row_mass is None:
            row_mass = torch.ones(N, device=device) / max(N, 1)
        if col_mass is None:
            proposals_per_gt = max(N // max(K, 1), 1)
            col_mass = torch.full((K,), proposals_per_gt / N, device=device)
            col_mass = col_mass / col_mass.sum()
        else:
            col_mass = col_mass / col_mass.sum().clamp_min(1e-10)
        log_K_mat = -cost / max(self.ot_epsilon, 1e-6)
        log_u = torch.zeros(N, device=device)
        log_v = torch.zeros(K, device=device)
        for _ in range(self.ot_num_iters):
            log_u = torch.log(row_mass + 1e-10) - torch.logsumexp(
                log_K_mat + log_v.unsqueeze(0), dim=1
            )
            log_v = torch.log(col_mass + 1e-10) - torch.logsumexp(
                log_K_mat + log_u.unsqueeze(1), dim=0
            )
        return torch.exp(log_u.unsqueeze(1) + log_K_mat + log_v.unsqueeze(0))

    def _sinkhorn_match(self, noise, gt_diffusion, device):
        cost = torch.cdist(noise, gt_diffusion, p=2)
        transport = self._sinkhorn_transport(cost)
        if self.ot_sample:
            row_probs = transport / transport.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-10)
            return self._ot_multinomial(row_probs)
        return transport.argmax(dim=1)

    def _couple_ot(self, noise, gt_diffusion, gt_labels, device):
        if gt_diffusion.shape[0] == 0:
            return noise, torch.zeros(
                noise.shape[0], dtype=torch.long, device=device
            )
        if self.ot_matcher == 'sinkhorn':
            matched_gt_idx = self._sinkhorn_match(noise, gt_diffusion, device)
        else:
            cost = torch.cdist(noise, gt_diffusion, p=2)
            matched_gt_idx = cost.argmin(dim=1)
        x_start = gt_diffusion[matched_gt_idx]
        return x_start, matched_gt_idx

    # ---- 时间采样 ----

    def _sample_t(self, bs, device):
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

    # ---- 前向传播 ----

    def forward(
        self,
        features: Tuple[Tensor],
        bboxes: Tensor,
        t: Tensor,
        proposals: Optional[Tensor] = None,
        img_metas: Optional[List] = None,
        x_noisy_raw: Optional[Tensor] = None,
    ) -> Tuple:
        """DiT 前向传播，迭代去噪

        Args:
            features: FPN 特征元组
            bboxes: 当前边界框 [bs, num_proposals, 4] (xyxy, 图像坐标)
            t: 当前时间步 [bs]
            proposals: 未使用 (保留接口兼容)
            img_metas: 图像元信息列表，用于坐标归一化
            x_noisy_raw: 训练时传入的真实 x_t raw 空间坐标 (无 clamp 污染)，
                         用于 v-prediction 模式下精确反推 x0

        Returns:
            all_cls_logits, all_pred_bboxes, all_pred_bboxes_raw, all_objectness, all_velocity, all_curr_proposals
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
        inter_pred_bboxes_raw = []
        inter_x0_raw = []  # x0 预测 (raw 空间, 无 clamp 畸变)
        inter_objectness = []
        inter_velocity = []
        inter_curr_proposals = []

        curr_tokens = box_tokens
        curr_normed = normed_bboxes
        # 跟踪 raw 空间坐标，避免从 clamped normed 反推时的信息丢失
        # curr_x_raw 始终保持为 x_noisy_raw (原始 x_t)，因为 v-prediction 的
        # 数学定义 v = (x_t - x_0) / t 要求起点始终是 x_t
        # curr_normed 则更新为 x0 预测 (cascade)，让后续 head 看到更准的位置
        curr_x_raw = x_noisy_raw

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

            if self.regression_mode == 'direct':
                # v-prediction: pred_bboxes 是 velocity v
                v_pred = pred_bboxes
                inter_pred_bboxes_raw.append(v_pred)

                # 从 v 反推 x0
                # x0 = x_t - v * t
                t_continuous = (t / self.timesteps).view(-1, 1, 1)
                if curr_x_raw is not None:
                    # 有 raw 空间坐标 (训练或推理), 直接用
                    x0_raw = curr_x_raw - v_pred * t_continuous
                else:
                    # 无 raw 空间坐标, 从 curr_normed 反推 (有 clamp 误差)
                    x_t_raw = self._normed_xyxy_to_raw_cxcywh(curr_normed)
                    x0_raw = x_t_raw - v_pred * t_continuous
                pred_bboxes_normed = self._raw_cxcywh_to_normed_xyxy(x0_raw)
                inter_x0_raw.append(x0_raw)
            else:
                pred_bboxes_normed = pred_bboxes  # delta 模式已是归一化 xyxy
                inter_pred_bboxes_raw.append(None)
                inter_x0_raw.append(None)

            inter_cls_logits.append(cls_logits)
            inter_pred_bboxes.append(pred_bboxes_normed)
            inter_objectness.append(objectness)
            inter_velocity.append(velocity)
            inter_curr_proposals.append(
                updated_tokens.unsqueeze(0)
                if updated_tokens.dim() == 3
                else updated_tokens
            )

            curr_tokens = updated_tokens
            # Cascade 设计: 后续 head 的 Reference Points 跟着 x0 预测更新
            # Deformable Attention 的 Reference Points 仅决定"看哪里提取特征"
            # 不影响 v-prediction 的数学正确性: 网络可以学习
            # [在 x0_hat 提取的清晰特征] + [时间步 t] → [全局速度 v_t]
            curr_normed = pred_bboxes_normed.detach().clamp(0, 1)

        if self.deep_supervision:
            return (
                torch.stack(inter_cls_logits),
                torch.stack(inter_pred_bboxes),
                inter_pred_bboxes_raw,
                inter_x0_raw,
                inter_objectness,
                inter_velocity,
                inter_curr_proposals,
            )
        else:
            return (
                torch.stack(inter_cls_logits[-1:]),
                torch.stack(inter_pred_bboxes[-1:]),
                inter_pred_bboxes_raw[-1:],
                inter_x0_raw[-1:],
                inter_objectness[-1:],
                inter_velocity[-1:],
                inter_curr_proposals[-1:],
            )

    # ---- 训练 ----

    def _normalize_targets(self, gt_bboxes, gt_labels, img_metas, bs):
        targets = []
        for i in range(bs):
            h, w = self._get_img_shape(img_metas[i])[:2]
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

            if self.ot_coupling and self.diffusion_type == 'rectified_flow':
                x_start, matched_idx = self._couple_ot(
                    noise, gt_diffusion, targets[i].labels, device
                )
                matched_gt_indices.append(matched_idx)
            else:
                idx = torch.randint(
                    0, num_gt, (self.num_proposals,), device=device
                )
                sample_bboxes = bbox_xyxy_to_cxcywh(targets[i].bboxes[idx])
                x_start = (sample_bboxes * 2 - 1) * self.snr_scale
                matched_gt_indices.append(idx)

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

    def loss(self, features, img_metas, gt_bboxes, gt_labels):
        device = features[0].device
        bs = len(img_metas)

        targets = self._normalize_targets(gt_bboxes, gt_labels, img_metas, bs)
        t, lsas_log_probs = self._sample_t(bs, device)

        x_boxes, x_starts, x_noises, matched_gt_indices = (
            self._build_training_targets(
                bs,
                device,
                t,
                targets,
                gt_bboxes,
                img_metas,
            )
        )
        x_noisy_batch = torch.stack(x_boxes)
        curr_bboxes = self._raw_to_xyxy(x_noisy_batch, img_metas)

        t_input = t if self.diffusion_type == 'ddpm' else t * self.timesteps

        (
            all_cls_logits,
            all_pred_bboxes,
            all_pred_bboxes_raw,
            all_x0_raw,
            all_objectness,
            all_velocity,
            all_curr_proposals,
        ) = self(features, curr_bboxes, t_input, img_metas=img_metas,
                 x_noisy_raw=x_noisy_batch)

        norm_pred_bboxes = all_pred_bboxes

        # v-prediction + RF 模式: 用模型预测的 x0 归一化坐标作为 criterion 的 pred_boxes
        # 注意: 不能用 GT x_start 作为 pred_boxes, 否则所有 proposal 的 pred_boxes
        # 都等于 GT 坐标, Hungarian matching 无法区分正负样本, 导致 loss_cls 失效
        # criterion_pred_boxes 由 v_pred 经线性变换得来，GIoU 梯度可平滑回传
        if self.regression_mode == 'direct' and self.diffusion_type == 'rectified_flow':
            # 使用 forward 中计算的 x0 归一化坐标
            # all_x0_raw 是 raw 空间的 x0 预测, 转换为归一化坐标
            if all_x0_raw[-1] is not None:
                criterion_pred_boxes = self._raw_cxcywh_to_normed_xyxy(all_x0_raw[-1])
            else:
                criterion_pred_boxes = norm_pred_bboxes[-1]
        else:
            criterion_pred_boxes = norm_pred_bboxes[-1]

        outputs = ModelOutput(
            pred_logits=all_cls_logits[-1],
            pred_boxes=criterion_pred_boxes,
            pred_objectness=all_objectness[-1]
            if all_objectness[0] is not None
            else None,
        )
        if self.deep_supervision and self.num_heads > 1:
            outputs.aux_outputs = [
                ModelOutput(
                    pred_logits=all_cls_logits[i],
                    pred_boxes=self._raw_cxcywh_to_normed_xyxy(all_x0_raw[i])
                    if (self.regression_mode == 'direct' and self.diffusion_type == 'rectified_flow' and all_x0_raw[i] is not None)
                    else norm_pred_bboxes[i],
                    pred_objectness=all_objectness[i]
                    if all_objectness[0] is not None
                    else None,
                )
                for i in range(self.num_heads - 1)
            ]

        losses = self.criterion(outputs, targets)

        # v-prediction + RF: 保留 criterion 的 bbox/giou loss
        # GIoU 是尺度敏感的，对 MSE (尺度不敏感) 提供关键补充
        # criterion_pred_boxes 由 v_pred 经线性变换得来，梯度可平滑回传
        # 如需降低权重，在 config 中调整 loss_bbox/loss_giou 的 loss_weight

        self._add_velocity_loss(
            losses, all_velocity, x_starts, x_noises, device
        )
        self._add_raw_diffusion_loss(
            losses, all_pred_bboxes_raw, x_starts, x_noises, device
        )
        self._add_lsas_loss(losses, lsas_log_probs)

        # 诊断指标: 监控 raw 空间预测分布、归一化质量、匹配统计
        self._add_diagnostic_metrics(
            losses, all_pred_bboxes, all_pred_bboxes_raw,
            x_starts, x_noisy_batch, t, matched_gt_indices, device
        )

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

    def _add_raw_diffusion_loss(
        self, losses, all_pred_bboxes_raw, x_starts, x_noises, device
    ):
        """添加 velocity MSE 损失 (direct/v-prediction 模式专用)

        核心思想: direct 模式下 reg_head 输出 velocity v,
        主损失为 MSE(v_pred, v_target), v_target = x_noise - x_start。
        这确保 RF 向量场一致性, 且模型只需预测修正量 (比 x0-prediction 更容易)。
        归一化空间的 L1/GIoU loss 作为辅助检测损失 (由 criterion 计算)。
        """
        if self.regression_mode != 'direct':
            return
        if self.diffusion_type != 'rectified_flow':
            return

        x_starts_batch = torch.stack(x_starts)
        x_noises_batch = torch.stack(x_noises)
        v_target = x_noises_batch - x_starts_batch  # RF 理论速度
        vel_weight = 1.0  # 主损失权重 (降低以平衡 cls_head 梯度)

        # 最后一个 head 的 velocity 预测
        last_v_pred = all_pred_bboxes_raw[-1]
        if last_v_pred is not None:
            losses['loss_vel'] = F.mse_loss(last_v_pred, v_target) * vel_weight

        # Deep supervision: 中间 head 的 velocity 预测
        if self.deep_supervision and self.num_heads > 1:
            aux = torch.tensor(0.0, device=device)
            n = 0
            for hi in range(self.num_heads - 1):
                v_pred = all_pred_bboxes_raw[hi]
                if v_pred is not None:
                    aux = aux + F.mse_loss(v_pred, v_target)
                    n += 1
            if n > 0:
                losses['loss_vel_aux'] = aux / n * vel_weight * 0.5

    def _add_diagnostic_metrics(
        self, losses, all_pred_bboxes, all_pred_bboxes_raw,
        x_starts, x_noisy_batch, t, matched_gt_indices, device
    ):
        """添加诊断指标到 losses dict，由 MMEngine 自动记录到 SwanLab。

        关键监控维度:
        1. velocity 预测分布 — 确认 v-prediction 是否在合理范围
        2. 归一化空间预测质量 — 确认 x0 反推是否有效
        3. 时间步采样分布 — 确认 t 采样是否合理
        4. 匹配统计 — 确认 OT 匹配是否有效
        """
        with torch.no_grad():
            # 1. velocity 预测分布 (最后一个 head)
            if self.regression_mode == 'direct' and all_pred_bboxes_raw[-1] is not None:
                v_pred = all_pred_bboxes_raw[-1]
                v_target = x_noisy_batch - torch.stack(x_starts)
                losses['diag_v_pred_mean'] = v_pred.mean().detach()
                losses['diag_v_pred_std'] = v_pred.std().detach()
                losses['diag_v_target_mean'] = v_target.mean().detach()
                losses['diag_v_target_std'] = v_target.std().detach()
                # velocity 各维度 MAE
                v_mae = (v_pred - v_target).abs().mean(dim=(0, 1))  # [4]
                for i, name in enumerate(['cx', 'cy', 'w', 'h']):
                    losses[f'diag_v_mae_{name}'] = v_mae[i].detach()
                # 从 v 反推的 x0 误差
                t_view = t.view(-1, 1, 1)
                x0_from_v = x_noisy_batch - v_pred * t_view
                x0_target = torch.stack(x_starts)
                x0_mae = (x0_from_v - x0_target).abs().mean(dim=(0, 1))
                for i, name in enumerate(['cx', 'cy', 'w', 'h']):
                    losses[f'diag_x0_mae_{name}'] = x0_mae[i].detach()

            # 2. 归一化空间预测质量
            normed_pred = all_pred_bboxes[-1]  # [bs, N, 4] xyxy [0,1]
            losses['diag_normed_pred_mean'] = normed_pred.mean().detach()
            losses['diag_normed_pred_std'] = normed_pred.std().detach()
            losses['diag_normed_pred_min'] = normed_pred.min().detach()
            losses['diag_normed_pred_max'] = normed_pred.max().detach()
            # xyxy 有效性: x2>x1, y2>y1 的比例
            valid_w = (normed_pred[..., 2] > normed_pred[..., 0]).float().mean()
            valid_h = (normed_pred[..., 3] > normed_pred[..., 1]).float().mean()
            losses['diag_valid_w_ratio'] = valid_w.detach()
            losses['diag_valid_h_ratio'] = valid_h.detach()

            # 3. 时间步采样分布
            losses['diag_t_mean'] = t.mean().detach()
            losses['diag_t_std'] = t.std().detach()
            losses['diag_t_min'] = t.min().detach()
            losses['diag_t_max'] = t.max().detach()

            # 4. 匹配统计
            if matched_gt_indices is not None:
                # 每张图匹配到的 GT 数量
                for i, indices in enumerate(matched_gt_indices):
                    unique_gt = indices.unique()
                    losses[f'diag_matched_gt_count_{i}'] = torch.tensor(
                        len(unique_gt), dtype=torch.float, device=device
                    )

            # 5. noisy 输入分布 (确认噪声注入正确)
            losses['diag_noisy_mean'] = x_noisy_batch.mean().detach()
            losses['diag_noisy_std'] = x_noisy_batch.std().detach()

    def _normed_to_img(self, normed_bboxes, img_metas):
        img_bboxes = normed_bboxes.clone()
        for i, meta in enumerate(img_metas):
            h, w = self._get_img_shape(meta)[:2]
            scale = img_bboxes.new_tensor([w, h, w, h])
            img_bboxes[i] *= scale
        return img_bboxes

    @staticmethod
    def _add_lsas_loss(losses, lsas_log_probs):
        if lsas_log_probs is None:
            return
        with torch.no_grad():
            total = sum(losses.values())
        losses['loss_lsas'] = total.detach() * (-lsas_log_probs).mean() * 0.01

    # ---- 推理 ----

    def _forward_at_t(self, features, x_raw, t, img_metas):
        bs, device = x_raw.shape[0], x_raw.device
        curr_bboxes = self._raw_to_xyxy(x_raw, img_metas)
        t_input = torch.full((bs,), t * self.timesteps, device=device)
        # 推理时传入 x_raw 作为 x_noisy_raw, 使 forward 内部:
        # - 第一个 head: 用 x_raw (无 clamp) 计算 x0
        # - 后续 head: 用前一个 head 的 x0_raw (cascaded, 无 clamp)
        cls_logits_seq, pred_bboxes_seq, pred_bboxes_raw_seq, x0_raw_seq, _, _, _ = self(
            features, curr_bboxes, t_input, img_metas=img_metas,
            x_noisy_raw=x_raw,
        )
        last_cls_logits = cls_logits_seq[-1]
        last_pred_bboxes = pred_bboxes_seq[-1]
        last_pred_bboxes_img = self._normed_to_img(last_pred_bboxes, img_metas)

        if self.regression_mode == 'direct' and x0_raw_seq[-1] is not None:
            # v-prediction: 直接使用 forward 内部计算的 x0_raw (无 clamp 畸变)
            x0_raw = x0_raw_seq[-1]
            v_pred = pred_bboxes_raw_seq[-1]
            return last_cls_logits, last_pred_bboxes_img, x0_raw, v_pred
        else:
            x0_raw = self._xyxy_to_raw(last_pred_bboxes_img, img_metas)
            return last_cls_logits, last_pred_bboxes_img, x0_raw, None

    @torch.no_grad()
    def predict(
        self, features, img_metas, rescale=True, return_trajectory=False
    ):
        device = features[0].device
        bs = len(img_metas)

        if self.diffusion_type == 'ddpm':
            times = torch.linspace(
                -1,
                self.timesteps - 1,
                steps=self.sampling_timesteps + 1,
                device=device,
            )
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))
        else:
            times = torch.linspace(
                1.0, 0.0, steps=self.sampling_timesteps + 1, device=device
            )
            if self.rf_schedule == 'power':
                times = times.pow(self.rf_power)
            elif self.rf_schedule == 'shifted':
                s = self.rf_shift
                times = s * times / (1 + (s - 1) * times)
            time_pairs = []
            for i in range(len(times) - 1):
                time_pairs.append((times[i].item(), times[i + 1].item()))

        x_raw = self._init_inference_boxes(bs, device)
        x0_prev = None
        ensemble_results = []
        trajectory = []

        dpm_solver = None
        if (
            self.solver_type == 'dpm_solver_pp'
            or self.solver_type == 'dpm_solver_pp_3'
        ):
            from .rectified_flow import RFDPMSolverMultistep

            solver_order = 3 if self.solver_type == 'dpm_solver_pp_3' else 2
            dpm_solver = RFDPMSolverMultistep(
                num_steps=self.sampling_timesteps, solver_order=solver_order
            )

        for step_idx, (t_curr, t_next) in enumerate(time_pairs):
            cls_logits, pred_bboxes, x0_raw, v_pred = self._forward_at_t(
                features, x_raw, t_curr, img_metas
            )

            x0_prev = x0_raw.detach()

            if return_trajectory:
                trajectory.append((cls_logits.detach(), pred_bboxes.detach()))

            if self.diffusion_type == 'ddpm':
                if t_next < 0:
                    break
                x0 = self._xyxy_to_raw(pred_bboxes, img_metas)
                t_batch = torch.full(
                    (bs,), t_curr, device=device, dtype=torch.long
                )
                pred_noise = self.predict_noise_from_start(x_raw, t_batch, x0)
                alpha = self.alphas_cumprod[t_curr]
                alpha_next = self.alphas_cumprod[t_next]
                sigma = (
                    self.ddim_sampling_eta
                    * (
                        (1 - alpha / alpha_next)
                        * (1 - alpha_next)
                        / (1 - alpha)
                    ).sqrt()
                )
                c = (1 - alpha_next - sigma**2).sqrt()
                noise = torch.randn_like(x_raw)
                x_raw = x0 * alpha_next.sqrt() + c * pred_noise + sigma * noise
                if self.box_renewal:
                    x_raw = self._apply_box_renewal(x_raw, cls_logits)

                if self.use_ensemble:
                    ensemble_results.append((cls_logits, pred_bboxes))
            else:
                # RF 模式: 使用 x0-based step (而非 v-based step)
                # 因为 cascaded 设计中, v_pred 是最后一个 head 的 velocity,
                # 相对于其输入 (前一个 head 的 x0), 不是相对于 x_raw
                # x0-based step: v = (x_t - x0) / t, x_next = x_t + dt * v
                if dpm_solver is not None:
                    x_raw = dpm_solver.step(x_raw, x0_raw, t_curr, step_idx)
                elif self.solver_type == 'heun' and t_next > 0:

                    def model_fn(x_tmp, t_tmp):
                        _, _, x0_tmp, _ = self._forward_at_t(
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

                # RF 模式: 使用采样轨迹的 x_raw 转换为检测框
                if self.regression_mode == 'direct':
                    x_raw_normed = self._raw_cxcywh_to_normed_xyxy(x_raw)
                    x_raw_img = self._normed_to_img(x_raw_normed, img_metas)
                else:
                    x_raw_img = self._raw_to_xyxy(x_raw, img_metas)

                if self.use_ensemble:
                    ensemble_results.append((cls_logits, x_raw_img))

                if t_next <= 0:
                    break

        # 如果没有 ensemble 结果 (不应该发生), 使用最后一步的 x_raw
        if not ensemble_results:
            if self.regression_mode == 'direct':
                x_raw_normed = self._raw_cxcywh_to_normed_xyxy(x_raw)
                final_bboxes = self._normed_to_img(x_raw_normed, img_metas)
            else:
                final_bboxes = self._raw_to_xyxy(x_raw, img_metas)
            ensemble_results.append((cls_logits, final_bboxes))

        results = self._post_process(ensemble_results, img_metas, rescale)

        if return_trajectory:
            return results, trajectory
        return results

    def _apply_box_renewal(self, x_raw, cls_logits):
        bs, device = x_raw.shape[0], x_raw.device
        scores = torch.sigmoid(cls_logits).max(-1)[0]
        x_raw_new = x_raw.clone()
        for i in range(bs):
            keep = scores[i] > self.score_thr
            if keep.sum() < self.min_keep:
                _, topk_idx = scores[i].topk(
                    min(self.min_keep, scores.shape[1])
                )
                keep[topk_idx] = True
            num_renew = (~keep).sum()
            if num_renew > 0:
                if self.box_init_mode == 'spatial_prior' and hasattr(
                    self.box_tokenizer, 'anchor_boxes'
                ):
                    # spatial_prior 模式: 用 anchor 替换低分框，保持与训练一致的分布
                    anchors = self.box_tokenizer.anchor_boxes.to(device)
                    anchor_raw = (anchors * 2 - 1) * self.snr_scale
                    # 随机选取 anchor 作为替换
                    replace_idx = torch.randint(
                        0, anchor_raw.shape[0], (num_renew,), device=device
                    )
                    x_raw_new[i, ~keep] = anchor_raw[replace_idx]
                else:
                    x_raw_new[i, ~keep] = torch.randn(
                        num_renew, 4, device=device
                    )
        return x_raw_new

    def _post_process(self, ensemble_results, img_metas, rescale):
        results_list = []
        bs = len(img_metas)
        for i in range(bs):
            if self.use_ensemble and len(ensemble_results) > 1:
                # ensemble: 拼接所有 step 的结果，但按框位置去重
                # 同一位置不同 step 可能分配不同类别，NMS 无法抑制跨类别重复
                # 解决: 对每个 step 的结果独立做 NMS，再拼接做最终 NMS
                step_results = []
                for cls_logits, pred_bboxes in ensemble_results:
                    scores = torch.sigmoid(cls_logits[i])
                    conf, labels = scores.max(-1)
                    # 每个 step 先做一次 NMS 去除该 step 内的重复
                    if self.use_nms:
                        keep = batched_nms(
                            pred_bboxes[i], conf, labels, self.nms_thr
                        )
                        step_results.append((
                            conf[keep], pred_bboxes[i][keep], labels[keep]
                        ))
                    else:
                        step_results.append((conf, pred_bboxes[i], labels))
                all_scores = torch.cat([r[0] for r in step_results])
                all_bboxes = torch.cat([r[1] for r in step_results])
                all_labels = torch.cat([r[2] for r in step_results])
            else:
                # 单步: 直接使用最后一步结果
                cls_logits, pred_bboxes = ensemble_results[-1]
                scores = torch.sigmoid(cls_logits[i])
                all_scores, all_labels = scores.max(-1)
                all_bboxes = pred_bboxes[i]

            final_scores = all_scores
            final_bboxes = all_bboxes
            final_labels = all_labels
            if self.use_nms:
                keep = batched_nms(
                    final_bboxes, final_scores, final_labels, self.nms_thr
                )
                final_scores = final_scores[keep]
                final_bboxes = final_bboxes[keep]
                final_labels = final_labels[keep]
            if rescale:
                scale_factor = self._get_scale_factor(img_metas[i])
                if scale_factor is None:
                    scale_factor = [1.0, 1.0, 1.0, 1.0]
                if isinstance(scale_factor, (list, tuple, Tensor)):
                    if len(scale_factor) == 2:
                        if isinstance(scale_factor, Tensor):
                            scale_factor = scale_factor.repeat(2)
                        else:
                            scale_factor = [
                                scale_factor[0],
                                scale_factor[1],
                                scale_factor[0],
                                scale_factor[1],
                            ]
                if not isinstance(scale_factor, Tensor):
                    scale_factor = final_bboxes.new_tensor(scale_factor)
                if scale_factor.dim() == 1:
                    scale_factor = scale_factor.unsqueeze(0)
                final_bboxes /= scale_factor
            results_list.append(
                DetectionResult(
                    bboxes=final_bboxes,
                    scores=final_scores,
                    labels=final_labels,
                )
            )
        return results_list

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

    def predict_noise_from_start(self, x_t, t, x0):
        sqrt_recip_alphas_cumprod_t = load_buffer(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = load_buffer(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )
        return (
            sqrt_recip_alphas_cumprod_t * x_t - x0
        ) / sqrt_recipm1_alphas_cumprod_t
