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
)
from .rectified_flow import RectifiedFlow
from .structures import DetectionResult, InstanceData, ModelOutput
from .utils import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, sanitize_bboxes


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
        train_noise_source: str = 'gaussian',
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
        self.train_noise_source = train_noise_source
        self.use_ensemble = use_ensemble
        self.deep_supervision = deep_supervision
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
                    self._shared_head = DiTSingleHead(**single_head)
                    self.head_series = None
                else:
                    self._shared_head = None
                    self.head_series = nn.ModuleList(
                        [
                            DiTSingleHead(**single_head)
                            for _ in range(num_heads)
                        ]
                    )
            else:
                if share_heads:
                    self._shared_head = single_head
                    self.head_series = None
                else:
                    import copy

                    self._shared_head = None
                    self.head_series = nn.ModuleList(
                        [copy.deepcopy(single_head) for _ in range(num_heads)]
                    )
        else:
            self._shared_head = None
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

        for head in self._iter_heads():
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

    # ---- Head 迭代 ----

    def _iter_heads(self):
        """迭代所有 head 模块 (共享头重复 num_heads 次，独立头逐个返回)"""
        if self._shared_head is not None:
            for _ in range(self.num_heads):
                yield self._shared_head
        else:
            yield from self.head_series

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

    def _get_img_scale_tensors(self, img_metas):
        """从 img_metas 批量构建 [bs, 4] scale tensor [w, h, w, h]"""
        scales = []
        for meta in img_metas:
            h, w = self._get_img_shape(meta)[:2]
            scales.append([w, h, w, h])
        return (
            img_metas[0].new_tensor(scales)
            if isinstance(img_metas[0], torch.Tensor)
            else torch.tensor(scales, dtype=torch.float32)
        )

    def _xyxy_to_raw(self, bboxes, img_metas):
        scale = self._get_img_scale_tensors(img_metas).to(bboxes.device)
        x0 = bboxes / scale
        x0 = bbox_xyxy_to_cxcywh(x0)
        x0 = (x0 * 2 - 1) * self.snr_scale
        return x0

    def _init_inference_boxes(self, bs, device):
        """生成推理时的初始框

        核心问题: 纯 randn (raw空间) 经 tanh 变换后所有框聚集在图像中心，
        导致 Deformable Attention 参考点相同 → 所有提案获得相同图像条件 →
        ODE 积分从同质态出发永远无法分化 → mAP=0。

        训练时 proposal 分布在各 GT 的不同位置 (x_noisy = (1-t)*GT + t*noise)，
        推理时必须匹配此分布: 提案要均匀覆盖全图，使每个提案有独立参考点。
        """
        if self.box_init_mode == 'spatial_prior':
            # 空间均匀初始化: 归一化空间 cxcywh 网格 + 扰动 → atanh → raw
            N = self.num_proposals
            # 使用网格确保全图覆盖，扰动提供多样性
            grid_len = math.ceil(math.sqrt(N))
            cy = torch.linspace(0.05, 0.95, grid_len, device=device)
            cx = torch.linspace(0.05, 0.95, grid_len, device=device)
            gy, gx = torch.meshgrid(cy, cx, indexing='ij')  # [grid, grid]
            # 取前 N 个网格点的中心坐标
            centers = torch.stack(
                [gx.flatten()[:N], gy.flatten()[:N]], dim=-1
            )  # [N, 2]
            # 对每个网格点添加随机扰动 (0.03 标准差，相对 grid spacing ~0.1)
            # 构建 100 个中心点分布于全图，每个有微小随机偏移
            centers = (
                centers + torch.randn(N, 2, device=device) * 0.02
            ).clamp(0.01, 0.99)
            # 小框初始化: w,h ∈ [0.01, 0.15]，覆盖染色体大小范围 (1-15%)
            w = torch.rand(N, 1, device=device) * 0.14 + 0.01
            h = torch.rand(N, 1, device=device) * 0.14 + 0.01
            # 归一化 cxcywh → raw cxcywh via atanh
            cxcywh_normed = torch.cat(
                [centers, w, h], dim=-1
            )  # [N, 4], 在 [0,1]
            shifted = (2.0 * cxcywh_normed - 1.0).clamp(-0.999, 0.999)
            x_raw = torch.atanh(shifted) * self.snr_scale  # [N, 4], raw 空间
            return x_raw.unsqueeze(0).expand(bs, -1, -1)
        return torch.randn(bs, self.num_proposals, 4, device=device)

    def _raw_to_xyxy(self, raw_bboxes, img_metas):
        """raw 空间 cxcywh → 图像空间 xyxy (复用 _raw_cxcywh_to_normed_xyxy)"""
        normed = self._raw_cxcywh_to_normed_xyxy(raw_bboxes)
        normed[..., 0] = torch.min(normed[..., 0], normed[..., 2] - 1e-4)
        normed[..., 1] = torch.min(normed[..., 1], normed[..., 3] - 1e-4)
        scale = self._get_img_scale_tensors(img_metas).to(normed.device)
        return normed * scale.unsqueeze(1)

    def _normalize_bboxes_for_tokenizer(self, bboxes, img_metas):
        """将 xyxy 图像坐标转为 [0,1] 归一化坐标供 BoxTokenizer 使用

        归一化后 clamp 到 [0,1] 防止异常坐标传播到 reference_points。
        """
        scale = self._get_img_scale_tensors(img_metas).to(bboxes.device)
        return (bboxes / scale.unsqueeze(1)).clamp(0, 1)

    def _raw_cxcywh_to_normed_xyxy(self, raw_bboxes):
        """将 raw 空间 cxcywh 预测转换为归一化 xyxy [0,1]

        raw = (cxcywh_normed * 2 - 1) * snr_scale
        逆变换: cxcywh_normed = (raw / snr_scale + 1) / 2

        使用 tanh 替代 clamp 保留梯度流:
        - clamp 将 |raw| > snr_scale 的梯度归零，导致 GIoU loss 梯度被截断
        - tanh(raw/snr_scale) 输出仍在 [-1,1]，但梯度始终非零
        - 诊断发现 clamp 使 fc_feature 51% 梯度为零 (vs LDMDet sigmoid 的 0.17%)
        """
        cxcywh = (torch.tanh(raw_bboxes / self.snr_scale) + 1) / 2
        xyxy = bbox_cxcywh_to_xyxy(cxcywh)
        # 二次清洗: cxcywh→xyxy 转换可能产生 NaN (w/h 为 0 时除零)
        return sanitize_bboxes(xyxy)

    def _normed_xyxy_to_raw_cxcywh(self, normed_xyxy):
        """将归一化 xyxy [0,1] 转换为 raw 空间 cxcywh

        _raw_cxcywh_to_normed_xyxy 的逆变换。
        正向: cxcywh = (tanh(raw / snr_scale) + 1) / 2
        逆变换: raw = atanh(cxcywh * 2 - 1) * snr_scale

        注意: 必须使用 arctanh 而非线性反推, 否则推理时 ODE 积分
        因 x0_raw 不准确而偏离正确轨迹, 导致 mAP=0。
        线性反推 raw = (cxcywh*2-1)*snr_scale 在 raw=2 时误差达 54%。
        """
        cxcywh = bbox_xyxy_to_cxcywh(normed_xyxy)
        # Shift to [-1, 1]
        cxcywh_shifted = cxcywh * 2.0 - 1.0
        # 防御性 clamp: atanh(x) 在 x→±1 时趋向 ±∞
        # sigmoid 输出可能精确为 0 或 1, 导致 atanh 产生 Inf/NaN
        eps = 1e-5
        cxcywh_clamped = torch.clamp(
            cxcywh_shifted, min=-1.0 + eps, max=1.0 - eps
        )
        # tanh 的严格逆变换
        raw = torch.atanh(cxcywh_clamped) * self.snr_scale
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
        # 行归一化传输概率: 每个 proposal 分配到各 GT 的概率
        row_probs = transport / transport.sum(dim=1, keepdim=True).clamp_min(
            1e-10
        )
        if self.ot_sample:
            matched_idx = self._ot_multinomial(row_probs)
        else:
            matched_idx = transport.argmax(dim=1)
        # 每个 proposal 的最大传输概率，用于过滤低质量匹配
        max_prob = row_probs.max(dim=1).values  # [N]
        return matched_idx, max_prob

    # ---- 时间采样 ----

    def _sample_t(self, bs, device):
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

    # ---- 前向传播 ----

    def forward(
        self,
        features: Tuple[Tensor],
        bboxes: Tensor,
        t: Tensor,
        proposals: Optional[Tensor] = None,
        img_metas: Optional[List] = None,
        x_noisy_raw: Optional[Tensor] = None,
        normed_bboxes: Optional[Tensor] = None,
    ) -> Tuple:
        """DiT 前向传播，迭代去噪

        Args:
            features: FPN 特征元组
            bboxes: 当前边界框 [bs, num_proposals, 4] (xyxy, 图像坐标)
                    当 normed_bboxes 提供时可传 None
            t: 当前时间步 [bs]
            proposals: 未使用 (保留接口兼容)
            img_metas: 图像元信息列表，用于坐标归一化
            x_noisy_raw: 训练时传入的真实 x_t raw 空间坐标 (无 clamp 污染)，
                         用于 v-prediction 模式下精确反推 x0
            normed_bboxes: 预计算的归一化坐标 [bs, num_proposals, 4] (xyxy, [0,1])
                           提供时跳过内部归一化，避免冗余坐标转换

        Returns:
            all_cls_logits, all_pred_bboxes, all_pred_bboxes_raw, all_x0_raw, all_velocity, all_curr_proposals
        """
        if bboxes is not None:
            bs, num_boxes = bboxes.shape[:2]
            device = bboxes.device
        elif normed_bboxes is not None:
            bs, num_boxes = normed_bboxes.shape[:2]
            device = normed_bboxes.device
        else:
            raise ValueError('Either bboxes or normed_bboxes must be provided')

        time_emb = self.time_mlp(t)

        fpn_list = list(features)[: self.num_fpn_levels]
        while len(fpn_list) < self.num_fpn_levels:
            fpn_list.append(fpn_list[-1])
        fpn_flattened, spatial_shapes, level_start_index = (
            flatten_fpn_features(fpn_list)
        )

        if normed_bboxes is not None:
            # 调用方已提供归一化坐标，直接使用 (推理路径优化)
            pass
        elif img_metas is not None:
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
        inter_velocity = []
        inter_curr_proposals = []

        curr_tokens = box_tokens
        curr_normed = normed_bboxes
        # 跟踪 raw 空间坐标，避免从 clamped normed 反推时的信息丢失
        # curr_x_raw 始终保持为 x_noisy_raw (原始 x_t)，因为 v-prediction 的
        # 数学定义 v = (x_t - x_0) / t 要求起点始终是 x_t
        # curr_normed 则更新为 x0 预测 (cascade)，让后续 head 看到更准的位置
        curr_x_raw = x_noisy_raw

        for head in self._iter_heads():
            cls_logits, pred_bboxes, updated_tokens, _objectness, velocity = (
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
                # x0-prediction: pred_bboxes 是 sigmoid 输出的归一化 xyxy
                # 直接作为 criterion 的 pred_boxes, GIoU 梯度不受 t 缩放
                pred_bboxes_normed = pred_bboxes
                inter_pred_bboxes_raw.append(None)

                # 从 x0 反推 x0_raw (用于 velocity loss)
                x0_raw = self._normed_xyxy_to_raw_cxcywh(pred_bboxes_normed)
                inter_x0_raw.append(x0_raw)
            else:
                pred_bboxes_normed = pred_bboxes  # delta 模式已是归一化 xyxy
                inter_pred_bboxes_raw.append(None)
                inter_x0_raw.append(None)

            inter_cls_logits.append(cls_logits)
            inter_pred_bboxes.append(pred_bboxes_normed)
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
                inter_velocity,
                inter_curr_proposals,
            )
        else:
            return (
                torch.stack(inter_cls_logits[-1:]),
                torch.stack(inter_pred_bboxes[-1:]),
                inter_pred_bboxes_raw[-1:],
                inter_x0_raw[-1:],
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
        ot_match_probs = []

        for i in range(bs):
            num_gt = gt_bboxes[i].shape[0]
            if num_gt == 0:
                noise = self._make_train_noise(device)
                x_boxes.append(noise)
                x_starts.append(torch.zeros_like(noise))
                x_noises.append(noise)
                matched_gt_indices.append(
                    torch.zeros(
                        self.num_proposals, dtype=torch.long, device=device
                    )
                )
                ot_match_probs.append(
                    torch.zeros(self.num_proposals, device=device)
                )
                continue

            norm_gt_cxcywh = bbox_xyxy_to_cxcywh(targets[i].bboxes)
            gt_diffusion = (norm_gt_cxcywh * 2 - 1) * self.snr_scale
            noise = self._make_train_noise(device)

            if self.ot_coupling:
                if self.ot_matcher == 'sinkhorn':
                    matched_idx, max_prob = self._sinkhorn_match(
                        noise, gt_diffusion, device
                    )
                else:
                    dist = torch.cdist(noise, gt_diffusion, p=2)
                    matched_idx = dist.argmin(dim=1)
                    min_dist = dist.min(dim=1).values
                    max_prob = 1.0 / (min_dist + 1e-6)
                    max_prob = max_prob / max_prob.max()
            else:
                matched_idx = torch.randint(
                    0, num_gt, (self.num_proposals,), device=device
                )
                max_prob = torch.ones(self.num_proposals, device=device)

            x_start = gt_diffusion[matched_idx]
            matched_gt_indices.append(matched_idx)
            ot_match_probs.append(max_prob)

            x_noisy, _ = self.rf.q_sample(
                x_start, x_noise=noise, t=t[i : i + 1]
            )
            x_starts.append(x_start)
            x_noises.append(noise)
            x_boxes.append(x_noisy)

        return x_boxes, x_starts, x_noises, matched_gt_indices, ot_match_probs

    def _make_train_noise(self, device):
        """生成训练用的噪声提案，使其分布与推理一致。

        关键问题: 训练用 randn 生成噪声，推理用 grid 生成提案，
        两者分布不同导致训练-推理不匹配 → mAP=0。

        'gaussian': 原始行为，纯 randn
        'grid': 与 _init_inference_boxes 完全一致的网格初始化
        """
        if self.train_noise_source == 'grid':
            return self._init_inference_boxes(1, device).squeeze(0)
        else:
            return torch.randn(self.num_proposals, 4, device=device)

    def loss(self, features, img_metas, gt_bboxes, gt_labels):
        device = features[0].device
        bs = len(img_metas)

        targets = self._normalize_targets(gt_bboxes, gt_labels, img_metas, bs)
        t, _ = self._sample_t(bs, device)

        x_boxes, x_starts, x_noises, matched_gt_indices, ot_match_probs = (
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

        t_input = t * self.timesteps

        (
            all_cls_logits,
            all_pred_bboxes,
            all_pred_bboxes_raw,
            all_x0_raw,
            all_velocity,
            all_curr_proposals,
        ) = self(
            features,
            curr_bboxes,
            t_input,
            img_metas=img_metas,
            x_noisy_raw=x_noisy_batch,
        )

        norm_pred_bboxes = all_pred_bboxes

        # x0-prediction: reg_head 输出 sigmoid 归一化坐标，直接用于 criterion
        # GIoU 梯度不受时间步 t 缩放，与 LDMDet 完全一致
        if self.regression_mode == 'direct':
            criterion_pred_boxes = norm_pred_bboxes[-1]
        else:
            criterion_pred_boxes = norm_pred_bboxes[-1]

        outputs = ModelOutput(
            pred_logits=all_cls_logits[-1],
            pred_boxes=criterion_pred_boxes,
        )
        if self.deep_supervision and self.num_heads > 1:
            outputs.aux_outputs = [
                ModelOutput(
                    pred_logits=all_cls_logits[i],
                    pred_boxes=norm_pred_bboxes[i],
                )
                for i in range(self.num_heads - 1)
            ]

        losses = self.criterion(
            outputs,
            targets,
            ot_matched_gt_indices=matched_gt_indices,
            ot_match_probs=ot_match_probs,
        )

        # v-prediction + RF: 保留 criterion 的 bbox/giou loss
        # GIoU 是尺度敏感的，对 MSE (尺度不敏感) 提供关键补充
        # criterion_pred_boxes 由 v_pred 经线性变换得来，梯度可平滑回传
        # 如需降低权重，在 config 中调整 loss_bbox/loss_giou 的 loss_weight

        self._add_raw_diffusion_loss(
            losses,
            all_pred_bboxes_raw,
            x_starts,
            x_noises,
            device,
            all_x0_raw=all_x0_raw,
            x_noisy_batch=x_noisy_batch,
            t=t,
        )
        # 诊断指标: 监控所有关键组件状态
        self._add_diagnostic_metrics(
            losses,
            all_pred_bboxes,
            all_pred_bboxes_raw,
            x_starts,
            x_noisy_batch,
            t,
            matched_gt_indices,
            device,
            all_x0_raw=all_x0_raw,
            all_cls_logits=all_cls_logits,
            all_curr_proposals=all_curr_proposals,
        )

        return losses

    def _add_raw_diffusion_loss(
        self,
        losses,
        all_pred_bboxes_raw,
        x_starts,
        x_noises,
        device,
        all_x0_raw=None,
        x_noisy_batch=None,
        t=None,
    ):
        """添加 displacement MSE 损失 (direct/x0-prediction 模式专用)

        x0-prediction 模式: reg_head 输出 sigmoid 归一化坐标 x0,
        使用 displacement loss: MSE(x_noisy - x0_pred, x_noisy - x_start)
        等价于 MSE(v_pred * t, v_target * t)，避免 t→0 时 v→∞ 导致 loss 爆炸
        """
        if self.regression_mode != 'direct':
            return

        x_starts_batch = torch.stack(x_starts)
        # displacement target: x_noisy - x_start = t * (noise - x_start) = t * v
        # 正确 target 是时间缩放后的速度，而非完整速度 v = noise - x_start
        # 否则完美预测 x0_raw=x_start 时 loss = (1-t)²||v||² ≠ 0，
        # 梯度会错误地推 x0_raw 趋向 (1-t)*noise + t*x_start
        disp_target = x_noisy_batch - x_starts_batch
        vel_weight = 1.0

        # x0-prediction: displacement = x_noisy - x0_pred
        if all_x0_raw is not None and x_noisy_batch is not None:
            last_x0_raw = all_x0_raw[-1]
            if last_x0_raw is not None:
                disp_pred = x_noisy_batch - last_x0_raw
                losses['loss_vel'] = (
                    F.mse_loss(disp_pred, disp_target) * vel_weight
                )

            # Deep supervision
            if self.deep_supervision and self.num_heads > 1:
                aux = torch.tensor(0.0, device=device)
                n = 0
                for hi in range(self.num_heads - 1):
                    x0_raw_i = all_x0_raw[hi]
                    if x0_raw_i is not None:
                        disp_pred_i = x_noisy_batch - x0_raw_i
                        aux = aux + F.mse_loss(disp_pred_i, disp_target)
                        n += 1
                if n > 0:
                    losses['loss_vel_aux'] = aux / n * vel_weight * 0.5

    def _add_diagnostic_metrics(
        self,
        losses,
        all_pred_bboxes,
        all_pred_bboxes_raw,
        x_starts,
        x_noisy_batch,
        t,
        matched_gt_indices,
        device,
        all_x0_raw=None,
        all_cls_logits=None,
        all_curr_proposals=None,
        norm_pred_bboxes=None,
        curr_proposals_list=None,
    ):
        """添加诊断指标，覆盖训练流程每个环节的状态。

        关键监控维度:
        1. Box Token 多样性 — box_tokens 各 proposal 是否有差异
        2. 分类头输出分布 — sigmoid 后分数分布、正负例差异
        3. 回归头输出质量 — 框有效性、空间覆盖、多样性
        4. displacement 预测 — v-prediction 误差分解
        5. OT 匹配质量 — 每个 GT 分配了多少 proposal
        6. 损失分解 — 各 loss 项的数值与比例
        """
        from mmengine.logging import print_log

        with torch.no_grad():
            # ============ 0. 损失分解 (最关键: 确认各 loss 是否在下降) ============
            loss_items = []
            for k in sorted(losses.keys()):
                v = losses[k]
                if isinstance(v, torch.Tensor):
                    loss_items.append(f'{k}={v.item():.4f}')
            print_log(
                f'[LOSS] step_diag: {", ".join(loss_items)}',
                logger='current',
            )

            # ============ 1. 分类头输出诊断 ============
            if all_cls_logits is not None:
                cls_last = all_cls_logits[-1]  # [bs, N, num_classes]
                scores = torch.sigmoid(cls_last)  # [bs, N, num_classes]
                max_scores, pred_labels = scores.max(-1)  # [bs, N]
                # 分数分布
                top10_scores = max_scores.topk(
                    min(10, max_scores.shape[1]), dim=1
                )[0]
                print_log(
                    f'[CLS] max_score: mean={max_scores.mean():.4f} '
                    f'std={max_scores.std():.4f} '
                    f'min={max_scores.min():.4f} max={max_scores.max():.4f} '
                    f'top10_mean={top10_scores.mean():.4f}',
                    logger='current',
                )
                # 标签分布: 模型倾向于预测哪些类？
                label_unique = pred_labels.unique()
                print_log(
                    f'[CLS] pred_label_count={len(label_unique)}/{cls_last.shape[-1]} '
                    f'unique={sorted(label_unique.tolist())[:12]}...',
                    logger='current',
                )
                # 正负例分数分离度
                losses['diag_cls_max_mean'] = max_scores.mean().detach()
                losses['diag_cls_max_std'] = max_scores.std().detach()

            # ============ 2. 回归头输出诊断 ============
            normed_pred = all_pred_bboxes[-1]  # [bs, N, 4] xyxy [0,1]
            losses['diag_normed_pred_mean'] = normed_pred.mean().detach()
            losses['diag_normed_pred_std'] = normed_pred.std().detach()
            losses['diag_normed_pred_min'] = normed_pred.min().detach()
            losses['diag_normed_pred_max'] = normed_pred.max().detach()
            # 框有效性
            valid_w = (
                normed_pred[..., 2] > normed_pred[..., 0] + 1e-4
            ).float()
            valid_h = (
                normed_pred[..., 3] > normed_pred[..., 1] + 1e-4
            ).float()
            valid_both = (valid_w * valid_h).mean()
            losses['diag_valid_w_ratio'] = valid_w.mean().detach()
            losses['diag_valid_h_ratio'] = valid_h.mean().detach()
            losses['diag_valid_both'] = valid_both.detach()
            # 框空间覆盖: 是否所有框都预测在相同位置？
            cx = (normed_pred[..., 0] + normed_pred[..., 2]) / 2
            cy = (normed_pred[..., 1] + normed_pred[..., 3]) / 2
            losses['diag_cx_range'] = (cx.max() - cx.min()).detach()
            losses['diag_cy_range'] = (cy.max() - cy.min()).detach()
            # 框对多样性: 随机取 50 对的 pairwise IoU 均值
            bs_i, N_i = normed_pred.shape[:2]
            if N_i >= 2:
                idx_a = torch.randperm(N_i, device=device)[: min(50, N_i)]
                idx_b = torch.randperm(N_i, device=device)[: min(50, N_i)]
                from torchvision.ops import box_iou as _box_iou

                iou_pairs = _box_iou(
                    normed_pred[0, idx_a], normed_pred[0, idx_b]
                ).diag()
                losses['diag_pairwise_iou'] = iou_pairs.mean().detach()
            else:
                losses['diag_pairwise_iou'] = normed_pred.new_zeros(1).detach()
            print_log(
                f'[REG] cx_range={losses["diag_cx_range"]:.3f} '
                f'cy_range={losses["diag_cy_range"]:.3f} '
                f'valid_both={valid_both:.3f} '
                f'pairwise_iou={losses.get("diag_pairwise_iou", 0):.3f}',
                logger='current',
            )

            # ============ 3. displacement/velocity 预测分布 ============
            if self.regression_mode == 'direct' and all_x0_raw is not None:
                x0_raw_last = all_x0_raw[-1]
                if x0_raw_last is not None:
                    disp_pred = x_noisy_batch - x0_raw_last
                    disp_target = x_noisy_batch - torch.stack(x_starts)
                    losses['diag_v_pred_mean'] = disp_pred.mean().detach()
                    losses['diag_v_pred_std'] = disp_pred.std().detach()
                    losses['diag_v_target_mean'] = disp_target.mean().detach()
                    losses['diag_v_target_std'] = disp_target.std().detach()
                    disp_mae = (disp_pred - disp_target).abs().mean(dim=(0, 1))
                    for i, name in enumerate(['cx', 'cy', 'w', 'h']):
                        losses[f'diag_v_mae_{name}'] = disp_mae[i].detach()
                    x0_target = torch.stack(x_starts)
                    x0_mae = (x0_raw_last - x0_target).abs().mean(dim=(0, 1))
                    for i, name in enumerate(['cx', 'cy', 'w', 'h']):
                        losses[f'diag_x0_mae_{name}'] = x0_mae[i].detach()
                    print_log(
                        f'[VEL] mae: cx={disp_mae[0]:.3f} cy={disp_mae[1]:.3f} '
                        f'w={disp_mae[2]:.3f} h={disp_mae[3]:.3f}',
                        logger='current',
                    )

            # ============ 4. 时间步分布 ============
            losses['diag_t_mean'] = t.mean().detach()
            losses['diag_t_std'] = t.std().detach()
            losses['diag_t_min'] = t.min().detach()
            losses['diag_t_max'] = t.max().detach()

            # ============ 5. OT 匹配统计 ============
            if matched_gt_indices is not None:
                gt_counts = torch.tensor(
                    [
                        indices.unique().numel()
                        for indices in matched_gt_indices
                    ],
                    dtype=torch.float,
                    device=device,
                )
                losses['diag_matched_gt_mean'] = gt_counts.mean().detach()
                print_log(
                    f'[OT] matched_gt: mean={gt_counts.mean():.1f} '
                    f'range=[{gt_counts.min():.0f}, {gt_counts.max():.0f}]',
                    logger='current',
                )

            # ============ 6. noisy 输入分布 ============
            losses['diag_noisy_mean'] = x_noisy_batch.mean().detach()
            losses['diag_noisy_std'] = x_noisy_batch.std().detach()

            # ============ 7. 特征多样性 — 各 proposal 的 fc_feature 余弦相似度 ============
            if all_curr_proposals is not None and len(all_curr_proposals) > 0:
                tokens = all_curr_proposals[
                    -1
                ]  # can be [1, bs, N, C] or [bs, N, C]
                if tokens.dim() == 4:
                    tokens = tokens.squeeze(0)  # [bs, N, C]
                elif tokens.dim() == 3 and tokens.shape[1] == 1:
                    tokens = tokens.squeeze(1)
                if tokens.dim() == 3 and tokens.shape[1] >= 2:
                    t0 = tokens[0]  # [N, C]
                    t_norm = F.normalize(t0, dim=-1)
                    cos_sim = (t_norm @ t_norm.T).abs()  # [N, N]
                    mask = ~torch.eye(N_i, dtype=torch.bool, device=device)
                    losses['diag_token_cos_sim'] = (
                        cos_sim[mask].mean().detach()
                    )
                    print_log(
                        f'[FEAT] token_cos_sim={losses["diag_token_cos_sim"]:.4f} '
                        f'(1.0=all same, 0.0=orthogonal)',
                        logger='current',
                    )

    def _normed_to_img(self, normed_bboxes, img_metas):
        scale = self._get_img_scale_tensors(img_metas).to(normed_bboxes.device)
        return normed_bboxes * scale.unsqueeze(1)

    # ---- 推理 ----

    def _forward_at_t(self, features, x_raw, t, img_metas):
        bs, device = x_raw.shape[0], x_raw.device
        # 直接从 raw 转 normed，跳过图像坐标中转
        normed_bboxes = self._raw_cxcywh_to_normed_xyxy(x_raw)
        t_input = torch.full((bs,), t * self.timesteps, device=device)
        # 推理时传入 x_raw 作为 x_noisy_raw
        (
            cls_logits_seq,
            pred_bboxes_seq,
            pred_bboxes_raw_seq,
            x0_raw_seq,
            _,
            _,
        ) = self(
            features,
            None,
            t_input,
            img_metas=img_metas,
            x_noisy_raw=x_raw,
            normed_bboxes=normed_bboxes,
        )
        last_cls_logits = cls_logits_seq[-1]
        last_pred_bboxes = pred_bboxes_seq[-1]
        last_pred_bboxes_img = self._normed_to_img(last_pred_bboxes, img_metas)

        if self.regression_mode == 'direct' and x0_raw_seq[-1] is not None:
            # x0-prediction: 从 x0_raw 反推 velocity
            x0_raw = x0_raw_seq[-1]
            t_continuous = t  # 已经是连续时间 [0,1]
            v_pred = (x_raw - x0_raw) / max(t_continuous, 1e-4)
            return last_cls_logits, last_pred_bboxes_img, x0_raw, v_pred
        else:
            x0_raw = self._xyxy_to_raw(last_pred_bboxes_img, img_metas)
            return last_cls_logits, last_pred_bboxes_img, x0_raw, None

    @torch.inference_mode()
    def predict(
        self, features, img_metas, rescale=True, return_trajectory=False
    ):
        device = features[0].device
        bs = len(img_metas)
        # 重置 ODE 诊断计数器，每个 val epoch 只记录前 4 个 batch 的轨迹
        self._ode_diag_count = 0

        # 在 CPU 上计算 time_pairs, 避免 .item() 触发 GPU 同步
        # linspace/pow/算术运算在 CPU 与 GPU 上数值完全一致
        times_cpu = torch.linspace(1.0, 0.0, steps=self.sampling_timesteps + 1)
        if self.rf_schedule == 'power':
            times_cpu = times_cpu.pow(self.rf_power)
        elif self.rf_schedule == 'shifted':
            s = self.rf_shift
            times_cpu = s * times_cpu / (1 + (s - 1) * times_cpu)
        time_pairs = [
            (times_cpu[i].item(), times_cpu[i + 1].item())
            for i in range(len(times_cpu) - 1)
        ]

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

            # ensemble 使用模型预测的 pred_bboxes，而非 renewal 后的 x_raw
            # renewal 产生的随机/anchor 框不是模型预测结果，不应进入最终输出
            # 同时避免 cls_logits (renewal前) 与 x_raw_img (renewal后) 的分数-框错位
            if self.use_ensemble:
                ensemble_results.append((cls_logits, pred_bboxes))

            # ODE 轨迹诊断: 监控 x0_raw 反推精度和 velocity 方向
            # 验证 arctanh 逆变换修复后 ODE 积分是否正确
            # 仅在第一个 batch 的前 3 步和最后 1 步打印，避免日志爆炸
            if not hasattr(self, '_ode_diag_count'):
                self._ode_diag_count = 0
            if self._ode_diag_count < 4 or step_idx == len(time_pairs) - 1:
                _scores = torch.sigmoid(cls_logits).max(-1)[0]
                _x0_raw_norm = x0_raw.detach().norm(dim=-1).mean()
                _x_raw_norm = x_raw.detach().norm(dim=-1).mean()
                _v_norm = (x_raw - x0_raw).detach().norm(dim=-1).mean() / max(
                    t_curr, 1e-4
                )
                _x0_range = (
                    f'[{x0_raw.min().item():.2f}, {x0_raw.max().item():.2f}]'
                )
                from mmengine.logging import print_log

                print_log(
                    f'[ODE Step {step_idx}] t={t_curr:.3f}→{t_next:.3f} '
                    f'|x0_raw|={_x0_raw_norm:.3f} |x_raw|={_x_raw_norm:.3f} '
                    f'|v|={_v_norm:.1f} x0_range={_x0_range} '
                    f'max_score={_scores.max().item():.3f}',
                    logger='current',
                )
                if step_idx == len(time_pairs) - 1:
                    self._ode_diag_count += 1

            # renewal 仅影响下一步 x_raw，不污染当前模型预测
            if self.box_renewal:
                x_raw = self._apply_box_renewal(x_raw, cls_logits)

            if t_next <= 0:
                break

        # 如果没有 ensemble 结果 (不应该发生), 使用最后一步的模型预测
        if not ensemble_results:
            ensemble_results.append((cls_logits, pred_bboxes))

        results = self._post_process(ensemble_results, img_metas, rescale)

        if return_trajectory:
            return results, trajectory
        return results

    def _apply_box_renewal(self, x_raw, cls_logits):
        bs, device = x_raw.shape[0], x_raw.device
        scores = torch.sigmoid(cls_logits).max(-1)[0]
        x_raw_new = x_raw.clone()
        # 向量化: 一次性计算所有样本的 keep mask (避免 per-sample 比较)
        keep_mask = scores > self.score_thr
        # topk 补充和随机数生成保持 per-sample, 保证随机数序列与原实现一致
        for i in range(bs):
            keep = keep_mask[i]
            if keep.sum() < self.min_keep:
                _, topk_idx = scores[i].topk(
                    min(self.min_keep, scores.shape[1])
                )
                keep[topk_idx] = True
            num_renew = (~keep).sum()
            if num_renew > 0:
                # RF 模式下统一使用标准高斯噪声，与训练时的噪声分布一致
                # anchor prior 分布与训练时的高斯噪声分布不一致，
                # 会导致 RF 速度场在 renewal 后的积分路径偏离训练分布
                x_raw_new[i, ~keep] = torch.randn(num_renew, 4, device=device)
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
                        step_results.append(
                            (conf[keep], pred_bboxes[i][keep], labels[keep])
                        )
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
            # Score threshold 过滤: 清除低分框，防止噪声框污染评估
            # NMS 只能去掉高度重叠框，无法清除位置分散的低质量框
            score_keep = final_scores > self.score_thr
            final_scores = final_scores[score_keep]
            final_bboxes = final_bboxes[score_keep]
            final_labels = final_labels[score_keep]
            if self.use_nms and final_scores.numel() > 0:
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
