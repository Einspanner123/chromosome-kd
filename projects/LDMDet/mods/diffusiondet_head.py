import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import batched_nms

from .modules import (
    SinusoidalPositionEmbeddings,
    cosine_noise_schedule,
    load_buffer,
)
from .rectified_flow import RectifiedFlow
from .structures import DetectionResult, ImageMeta, InstanceData, ModelOutput
from .utils import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh


class DiffusionDetHead(nn.Module):
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
        solver_type: str = 'euler',  # "euler" or "heun"
        box_renewal: bool = True,
        use_ensemble: bool = True,
        deep_supervision: bool = True,
        ddim_sampling_eta: float = 1.0,
        diffusion_type: str = 'ddpm',  # "ddpm" or "rectified_flow"
        rf_schedule: str = 'linear',  # "linear" or "power" or "shifted"
        rf_power: float = 1.0,  # 用于 power schedule
        rf_shift: float = 1.0,  # 用于 shifted schedule (如 SD3 中的 3.0)
        single_head: nn.Module = None,
        roi_extractor: nn.Module = None,
        criterion: nn.Module = None,
        use_nms: bool = True,
        nms_thr: float = 0.5,
        score_thr: float = 0.05,
        min_keep: int = 60,
        # === FlowDet 新增参数 ===
        prediction_mode: str = 'x0',  # "x0" 或 "velocity"
        velocity_loss_weight: float = 1.0,  # velocity loss 权重
        ot_coupling: bool = False,  # 训练时用 OT 最优耦合 (noise, GT) 配对
        ot_matcher: str = 'nearest',  # "nearest" (原始 argmin) 或 "sinkhorn" (均衡配对)
        ot_epsilon: float = 1.0,  # Sinkhorn OT 的正则化参数
        ot_num_iters: int = 20,  # Sinkhorn OT 的迭代次数
        ot_sample: bool = False,  # Stochastic Coupling: 从传输矩阵采样替代 argmax
        ot_sample_seed: Optional[
            int
        ] = None,  # 固定 OT 采样 RNG；None 使用全局 RNG
        ot_group_hierarchical: bool = False,  # Group-Hierarchical Coupling: 按染色体组分组 OT
        ot_kcec: bool = False,  # Karyotype-Constrained Entropic Coupling
        kcec_morph_weight: float = 0.25,  # 形态 (w,h) 匹配代价权重
        kcec_group_weight: float = 0.1,  # 组别先验代价权重
        kcec_cls_weight: float = 0.0,  # 类别匹配代价权重
        kcec_quota_strength: float = 1.0,  # 软倍性/类别配额边际权重强度
        kcec_slack: float = 0.05,  # 边际质量平滑，保留异常核型弹性
        kcec_conf_threshold: float = 0.5,  # V3: 质量激活的代价阈值 (Cost Gating)
        kcec_scale_anneal: bool = False,  # V3: 是否启用尺度感知配额 (Scale-Aware Quota)
        kcec_log_interval: int = 100,  # KCEC 日志间隔 (每 N 次调用记录一次)
        daec_contrastive_weight: float = 0.1,  # DAEC 对比损失权重
        daec_temperature: float = 0.05,  # DAEC 对比对齐温度 (Phase 2 调低至 0.05)
        # === TRD 参数 ===
        use_trd: bool = False,  # Transport-Refinement Decomposition
        trd_self_cond_prob: float = 0.5,  # 训练时自条件化概率
        trd_delta_t: Optional[
            float
        ] = None,  # TRD 自条件步长；None 时沿用 cat_delta_t
        # === CAT 参数 ===
        use_cat: bool = False,  # Curvature-Aware Training
        cat_weight: float = 0.1,  # 曲率正则化权重
        cat_delta_t: float = 0.01,  # 曲率计算的时间步长
        cat_loss_type: str = 'x0_consistency',  # "x0_consistency" 或 "velocity_curvature"
        # === LSAS 参数 ===
        use_lsas: bool = False,  # Loss-Sensitive Adaptive Scheduling
        lsas_num_bins: int = 100,  # 时间分布离散化精度
        lsas_temp: float = 1.0,  # 采样温度
        # === 训练稳定化参数 ===
        t_sampling: str = 'uniform',  # 时间采样策略: "uniform" 或 "stratified"
        t_sampling_bins: int = 8,  # stratified 采样时的分箱数
        use_flash_attn: bool = False,  # 是否启用 Flash Attention (SDPA)
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
        self.velocity_loss_weight = velocity_loss_weight
        self.ot_coupling = ot_coupling
        self.ot_matcher = ot_matcher
        self.ot_epsilon = ot_epsilon
        self.ot_num_iters = ot_num_iters
        self.ot_sample = ot_sample
        self.ot_sample_seed = ot_sample_seed
        self.ot_group_hierarchical = ot_group_hierarchical
        self.ot_kcec = ot_kcec
        self.kcec_morph_weight = kcec_morph_weight
        self.kcec_group_weight = kcec_group_weight
        self.kcec_cls_weight = kcec_cls_weight
        self.kcec_quota_strength = kcec_quota_strength
        self.kcec_slack = kcec_slack
        self.kcec_conf_threshold = kcec_conf_threshold
        self.kcec_scale_anneal = kcec_scale_anneal
        self.kcec_log_interval = kcec_log_interval
        self.daec_contrastive_weight = daec_contrastive_weight
        self.daec_temperature = daec_temperature
        self._kcec_call_count = 0
        self.use_trd = use_trd
        self.trd_self_cond_prob = trd_self_cond_prob
        self.trd_delta_t = cat_delta_t if trd_delta_t is None else trd_delta_t
        self.use_cat = use_cat
        self.cat_weight = cat_weight
        self.cat_delta_t = cat_delta_t
        self.cat_loss_type = cat_loss_type
        self.use_lsas = use_lsas
        self.lsas_num_bins = lsas_num_bins
        self.lsas_temp = lsas_temp

        self.t_sampling = t_sampling
        self.t_sampling_bins = t_sampling_bins
        self.use_flash_attn = use_flash_attn

        # 测试配置
        self.use_nms = use_nms
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.min_keep = min_keep

        # ROI 特征提取器和损失函数
        self.roi_extractor = roi_extractor
        self.criterion = criterion

        # 构建扩散过程参数
        if self.diffusion_type == 'ddpm':
            self._build_diffusion_buffers()
        elif self.diffusion_type == 'rectified_flow':
            self.rf = RectifiedFlow(snr_scale=snr_scale)

        # 构建检测头序列 (迭代去噪)
        if (
            hasattr(single_head, 'prediction_mode')
            and single_head.prediction_mode != prediction_mode
        ):
            single_head.prediction_mode = prediction_mode
        self.head_series = nn.ModuleList(
            [copy.deepcopy(single_head) for _ in range(num_heads)]
        )

        if prediction_mode == 'velocity':
            for head in self.head_series:
                if (
                    not hasattr(head, 'velocity_head')
                    or head.velocity_head is None
                ):
                    head.velocity_head = nn.Sequential(
                        nn.Linear(feat_channels, feat_channels, bias=False),
                        nn.LayerNorm(feat_channels),
                        nn.ReLU(inplace=True),
                        nn.Linear(feat_channels, feat_channels, bias=False),
                        nn.LayerNorm(feat_channels),
                        nn.ReLU(inplace=True),
                        nn.Linear(feat_channels, 4),
                    )
                    head.prediction_mode = 'velocity'
        for head in self.head_series:
            head.use_flash_attn = use_flash_attn

        # 时间步嵌入 MLP
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

        # Chromosome group mapping: class_idx -> group_idx
        # Class order: A1,A2,A3, B4,B5, C10,C11,C12,C6,C7,C8,C9,
        #              D13,D14,D15, E16,E17,E18, F19,F20, G21,G22, X,Y
        # Groups: 0=A, 1=B, 2=C, 3=D, 4=E, 5=F, 6=G, 7=Sex
        self._chromo_group_of_class = [
            0,
            0,
            0,  # A1, A2, A3
            1,
            1,  # B4, B5
            2,
            2,
            2,
            2,
            2,
            2,
            2,  # C10, C11, C12, C6, C7, C8, C9
            3,
            3,
            3,  # D13, D14, D15
            4,
            4,
            4,  # E16, E17, E18
            5,
            5,  # F19, F20
            6,
            6,  # G21, G22
            7,
            7,  # X, Y
        ]
        # Expected karyotype slots per class. Autosomes are diploid; sex
        # chromosomes are smoothed at 1 here because XX/XY/aneuploidy should
        # remain soft and data-driven through the observed GT labels.
        self._chromo_quota_of_class = [
            2,
            2,
            2,  # A1, A2, A3
            2,
            2,  # B4, B5
            2,
            2,
            2,
            2,
            2,
            2,
            2,  # C10, C11, C12, C6, C7, C8, C9
            2,
            2,
            2,  # D13, D14, D15
            2,
            2,
            2,  # E16, E17, E18
            2,
            2,  # F19, F20
            2,
            2,  # G21, G22
            1,
            1,  # X, Y
        ]

    def _ot_multinomial(self, row_probs: Tensor) -> Tensor:
        """Sample one column per row, optionally with a reproducible generator."""
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

    def _sinkhorn_transport(
        self,
        cost: Tensor,
        row_mass: Optional[Tensor] = None,
        col_mass: Optional[Tensor] = None,
    ) -> Tensor:
        """Entropic OT transport matrix for a precomputed cost matrix."""
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

    def _run_kcec_ot(
        self,
        noise: Tensor,
        gt_diffusion: Tensor,
        gt_labels: Tensor,
        device: torch.device,
        cls_logits: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Karyotype-Constrained Entropic Coupling V3 (Scale-Aware & Gated).

        Evolved KCEC pipeline:
        1. Direct GT Quota (V2 baseline).
        2. Scale-Aware Quota (V3): Smaller GTs get softer quota strength to avoid noise injection.
        3. Confidence Gating (V3): Extra mass is only activated if nearby proposals exist.
        4. Semantic-Aware (Phase 2): Inject classification cost to guide coupling.
        """
        N, K = noise.shape[0], gt_diffusion.shape[0]
        if K == 0:
            return torch.zeros(N, dtype=torch.long, device=device), {}

        quotas = torch.tensor(
            self._chromo_quota_of_class,
            device=device,
            dtype=gt_diffusion.dtype,
        )
        group_ids_all = torch.tensor(
            self._chromo_group_of_class, device=device
        )
        labels = gt_labels.clamp(min=0, max=quotas.numel() - 1)

        # 0. 计算基础代价 (用于后续门控)
        box_cost = torch.cdist(noise, gt_diffusion, p=2)

        # 1. 计算每个 GT 的边际质量 (Mass)
        # 相同类别的 GT 共享该类别的总配额
        gt_mass = torch.zeros(K, device=device, dtype=gt_diffusion.dtype)
        unique_classes = torch.unique(labels)
        for cls_id in unique_classes:
            gt_mask = labels == cls_id
            n_observed = gt_mask.sum().float()
            q_c = quotas[cls_id.item()]
            gt_mass[gt_mask] = q_c / n_observed

        # V3: Confidence Gating (门控机制)
        # 如果某个 GT 附近没有任何 Proposal (min_cost > threshold)，则不激活额外配额
        if self.kcec_conf_threshold > 0:
            min_cost_per_gt = box_cost.min(dim=0).values
            # 只有 min_cost 小于阈值的 GT 才能获得 > 1.0 的质量注入
            gate = (min_cost_per_gt < self.kcec_conf_threshold).float()
            # 质量修正：质量 = 1.0 + (原始分配质量 - 1.0) * gate
            gt_mass = 1.0 + (gt_mass - 1.0) * gate

        # V3: Scale-Aware Quota (尺度感知)
        # 核心逻辑：大物体 (+3.7% gain) 保持强约束，小物体 (-3.1% loss) 减弱约束
        strength = self.kcec_quota_strength
        if self.kcec_scale_anneal:
            gt_areas = gt_diffusion[:, 2] * gt_diffusion[:, 3]  # w * h
            mean_area = gt_areas.mean().clamp_min(1e-6)
            # 相对面积因子：大物体 > 1, 小物体 < 1
            scale_factor = (gt_areas / mean_area).sqrt()
            strength = strength * scale_factor

        gt_mass = gt_mass.pow(strength)
        gt_mass = gt_mass + float(self.kcec_slack)
        col_mass = gt_mass / gt_mass.sum().clamp_min(1e-10)
        row_mass = torch.ones(N, device=device) / max(N, 1)

        # 2. 计算代价矩阵 (Direct noise-to-GT)
        noise_wh = noise[:, 2:4]
        gt_wh = gt_diffusion[:, 2:4]
        morph_cost = torch.cdist(noise_wh, gt_wh, p=2)

        group_cost = torch.zeros(N, K, device=device)
        if self.kcec_group_weight > 0:
            num_groups = int(group_ids_all.max().item()) + 1
            gt_group_ids = group_ids_all[labels]
            group_morph_reps = []
            for g in range(num_groups):
                g_mask = gt_group_ids == g
                if g_mask.any():
                    group_morph_reps.append(gt_diffusion[g_mask, 2:4].mean(0))
                else:
                    group_morph_reps.append(
                        torch.zeros(2, device=device, dtype=gt_diffusion.dtype)
                    )
            group_morph_tensor = torch.stack(group_morph_reps)
            morph_to_groups = torch.cdist(noise_wh, group_morph_tensor, p=2)
            group_compat = F.softmax(
                -morph_to_groups / max(self.ot_epsilon, 1e-6), dim=1
            )
            group_cost = 1.0 - group_compat[:, gt_group_ids]

        # DAEC Phase 2: Semantic-Aware Coupling
        cls_cost = torch.zeros(N, K, device=device)
        if self.kcec_cls_weight > 0 and cls_logits is not None:
            # cls_logits: [num_proposals, num_classes]
            # labels: [num_gt] (values in [0, num_classes-1])
            # 我们需要计算 [num_proposals, num_gt] 的分类代价
            # 使用负对数似然或交叉熵
            prob = torch.sigmoid(cls_logits)  # [N, C]
            # 对每个 GT，提取对应的概率
            cls_cost = 1.0 - prob[:, labels]  # [N, K]

        cost = (
            box_cost
            + self.kcec_morph_weight * morph_cost
            + self.kcec_group_weight * group_cost
            + self.kcec_cls_weight * cls_cost
        )

        # 3. Sinkhorn 迭代
        transport = self._sinkhorn_transport(
            cost, row_mass=row_mass, col_mass=col_mass
        )

        # 4. 采样/匹配
        if self.ot_sample:
            row_probs = transport / transport.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-10)
            matched_gt_idx = self._ot_multinomial(row_probs)
        else:
            matched_gt_idx = transport.argmax(dim=1)

        # 5. 日志监控
        self._kcec_call_count += 1
        log_stats: Dict[str, Tensor] = {}
        if self.kcec_log_interval > 0 and (
            self._kcec_call_count % self.kcec_log_interval == 0
        ):
            with torch.no_grad():
                row_entropy = (
                    -(transport * (transport + 1e-10).log()).sum(dim=1).mean()
                )
                col_entropy = (
                    -(transport * (transport + 1e-10).log()).sum(dim=0).mean()
                )
                gt_counts = torch.bincount(matched_gt_idx, minlength=K).float()
                max_transport_per_row = transport.max(dim=1).values
                max_transport_per_col = transport.max(dim=0).values
                row_probs = transport / transport.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-10)
                top1_prob = row_probs.max(dim=1).values.mean()
                transport_argmax = transport.argmax(dim=1)
                cost_argmin = cost.argmin(dim=1)
                argmax_agreement = (
                    (transport_argmax == cost_argmin).float().mean()
                )
                row_marginal_err = (
                    (transport.sum(dim=1) - row_mass).abs().mean()
                )
                col_marginal_err = (
                    (transport.sum(dim=0) - col_mass).abs().mean()
                )
                log_stats = {
                    'kcec_cost_mean': cost.mean().detach(),
                    'kcec_cost_std': cost.std().detach(),
                    'kcec_cost_min': cost.min().detach(),
                    'kcec_cost_max': cost.max().detach(),
                    'kcec_box_cost_mean': box_cost.mean().detach(),
                    'kcec_morph_cost_mean': morph_cost.mean().detach(),
                    'kcec_group_cost_mean': group_cost.mean().detach(),
                    'kcec_row_entropy': row_entropy.detach(),
                    'kcec_col_entropy': col_entropy.detach(),
                    'kcec_top1_prob': top1_prob.detach(),
                    'kcec_argmax_agreement': argmax_agreement.detach(),
                    'kcec_row_marginal_err': row_marginal_err.detach(),
                    'kcec_col_marginal_err': col_marginal_err.detach(),
                    'kcec_transport_max_row': max_transport_per_row.mean().detach(),
                    'kcec_transport_max_col': max_transport_per_col.mean().detach(),
                    'kcec_num_classes': torch.tensor(
                        float(len(unique_classes)), device=device
                    ),
                    'kcec_gt_max_count': gt_counts.max().detach(),
                    'kcec_gt_count_std': gt_counts.std().detach(),
                    'kcec_gt_zero_frac': (
                        (gt_counts == 0).float().mean().detach()
                    ),
                    'kcec_col_mass_entropy': -(
                        col_mass * (col_mass + 1e-10).log()
                    )
                    .sum()
                    .detach(),
                }

        return matched_gt_idx, log_stats

    def _run_group_hierarchical_ot(
        self, noise, gt_diffusion, gt_labels, device
    ):
        """按染色体组分层 OT 耦合, 每组独立运行 Sinkhorn"""
        N = noise.shape[0]  # total proposals
        num_gt = gt_labels.shape[0]
        if num_gt == 0:
            return torch.randint(0, max(num_gt, 1), (N,), device=device)

        group_ids = torch.tensor(self._chromo_group_of_class, device=device)
        gt_groups = group_ids[gt_labels]  # (K,) group index per GT box

        matched_gt_idx = torch.zeros(N, dtype=torch.long, device=device)
        offset = 0  # running offset into the N proposals

        unique_groups = torch.unique(gt_groups)
        for grp in unique_groups:
            grp_mask = gt_groups == grp
            grp_gt_idx = torch.where(grp_mask)[
                0
            ]  # indices into original GT list
            K_g = grp_gt_idx.shape[0]

            # Allocate proposals proportional to this group's share of GT boxes
            N_g = max(N * K_g // num_gt, 1)
            # Adjust last group to absorb rounding
            if grp == unique_groups[-1]:
                N_g = N - offset
            N_g = min(N_g, N - offset)
            if N_g <= 0:
                continue

            grp_noise = noise[offset : offset + N_g]
            grp_gt = gt_diffusion[grp_gt_idx]

            # Within-group Sinkhorn OT
            cost = torch.cdist(grp_noise, grp_gt, p=2)
            a = torch.ones(N_g, device=device) / N_g
            proposals_per_gt = max(N_g // K_g, 1)
            gt_mass = torch.full((K_g,), proposals_per_gt / N_g, device=device)
            b = gt_mass / gt_mass.sum()
            log_K = -cost / self.ot_epsilon
            log_u = torch.zeros(N_g, device=device)
            log_v = torch.zeros(K_g, device=device)
            for _ in range(self.ot_num_iters):
                log_u = torch.log(a + 1e-10) - torch.logsumexp(
                    log_K + log_v.unsqueeze(0), dim=1
                )
                log_v = torch.log(b + 1e-10) - torch.logsumexp(
                    log_K + log_u.unsqueeze(1), dim=0
                )
            transport = torch.exp(
                log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
            )

            if self.ot_sample:
                row_probs = transport / transport.sum(dim=1, keepdim=True)
                local_matched = self._ot_multinomial(row_probs)
            else:
                local_matched = transport.argmax(dim=1)

            # Map back to global GT indices
            matched_gt_idx[offset : offset + N_g] = grp_gt_idx[local_matched]
            offset += N_g

        return matched_gt_idx

    def _sample_time_lsas(
        self, bs: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        probs = F.softmax(self.lsas_logits / self.lsas_temp, dim=0)
        bin_indices = torch.multinomial(probs, bs, replacement=True)
        bin_width = 1.0 / self.lsas_num_bins
        t = (
            bin_indices.float() * bin_width
            + torch.rand(bs, device=device) * bin_width
        )
        t = t.clamp(1e-5, 1.0 - 1e-5)
        log_probs = torch.log(probs[bin_indices] + 1e-10)
        return t, log_probs

    def _build_training_targets(
        self,
        bs: int,
        device: torch.device,
        t: Tensor,
        targets: List[InstanceData],
        gt_bboxes: List[Tensor],
        img_metas: List[ImageMeta],
        cls_logits_for_coupling: Optional[Tensor] = None,
    ) -> Tuple[
        List[Tensor],
        List[Tensor],
        List[Tensor],
        Dict[str, Tensor],
        List[Tensor],
    ]:
        """构建训练配对: 对每张图生成 (x_noisy, x_start, x_noise) 三元组.

        Returns:
            x_boxes: 每个样本的 x_t 噪声框列表
            x_starts: 每个样本的扩散空间 GT (x_0) 列表
            x_noises: 每个样本的噪声源 (x_1) 列表
            kcec_log_stats: KCEC 监控统计 (合并 batch 内所有样本)
            matched_gt_indices: 每个样本对应的 GT 索引列表
        """
        x_boxes = []
        x_starts = []
        x_noises = []
        matched_gt_indices = []
        kcec_log_stats: Dict[str, Tensor] = {}

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

            # GT in diffusion space
            norm_gt_cxcywh = bbox_xyxy_to_cxcywh(targets[i].bboxes)
            gt_diffusion = (norm_gt_cxcywh * 2 - 1) * self.snr_scale
            noise = torch.randn(self.num_proposals, 4, device=device)

            # Coupling
            if self.ot_coupling and self.diffusion_type == 'rectified_flow':
                cls_logits_i = (
                    cls_logits_for_coupling[i]
                    if cls_logits_for_coupling is not None
                    else None
                )
                x_start, sample_log, matched_idx = self._couple_ot(
                    noise,
                    gt_diffusion,
                    targets[i].labels,
                    device,
                    cls_logits=cls_logits_i,
                )
                matched_gt_indices.append(matched_idx)
                for k, v in sample_log.items():
                    if k not in kcec_log_stats:
                        kcec_log_stats[k] = []
                    kcec_log_stats[k].append(v.detach())
            else:
                idx = torch.randint(
                    0, num_gt, (self.num_proposals,), device=device
                )
                sample_bboxes = bbox_xyxy_to_cxcywh(targets[i].bboxes[idx])
                x_start = (sample_bboxes * 2 - 1) * self.snr_scale
                matched_gt_indices.append(idx)

            # Forward diffusion
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

        merged_kcec_log: Dict[str, Tensor] = {}
        for k, v_list in kcec_log_stats.items():
            if v_list:
                merged_kcec_log[k] = torch.stack(v_list).mean()

        return x_boxes, x_starts, x_noises, merged_kcec_log, matched_gt_indices

    def _couple_ot(
        self,
        noise: Tensor,
        gt_diffusion: Tensor,
        gt_labels: Tensor,
        device: torch.device,
        cls_logits: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        """OT 耦合: 将噪声提案与 GT 框配对, 返回配对的 x_start, 日志统计, 以及匹配的 GT 索引."""
        if gt_diffusion.shape[0] == 0:
            return (
                noise,
                {},
                torch.zeros(noise.shape[0], dtype=torch.long, device=device),
            )

        kcec_log_stats: Dict[str, Tensor] = {}
        if self.ot_matcher == 'sinkhorn':
            if self.ot_kcec:
                matched_gt_idx, kcec_log_stats = self._run_kcec_ot(
                    noise,
                    gt_diffusion,
                    gt_labels,
                    device,
                    cls_logits=cls_logits,
                )
            elif self.ot_group_hierarchical:
                matched_gt_idx = self._run_group_hierarchical_ot(
                    noise, gt_diffusion, gt_labels, device
                )
            else:
                matched_gt_idx = self._sinkhorn_match(
                    noise, gt_diffusion, device
                )
        else:
            cost = torch.cdist(noise, gt_diffusion, p=2)
            matched_gt_idx = cost.argmin(dim=1)

        x_start = gt_diffusion[matched_gt_idx]
        return x_start, kcec_log_stats, matched_gt_idx

    def _sinkhorn_match(
        self, noise: Tensor, gt_diffusion: Tensor, device: torch.device
    ) -> Tensor:
        """Sinkhorn OT 匹配: 返回每个噪声提案对应的 GT 索引."""
        cost = torch.cdist(noise, gt_diffusion, p=2)
        transport = self._sinkhorn_transport(cost)

        if self.ot_sample:
            row_probs = transport / transport.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-10)
            return self._ot_multinomial(row_probs)
        return transport.argmax(dim=1)

    def loss(
        self,
        features: Tuple[Tensor],
        img_metas: List[ImageMeta],
        gt_bboxes: List[Tensor],
        gt_labels: List[Tensor],
    ) -> Dict[str, Tensor]:
        """训练损失计算接口"""
        device = features[0].device
        bs = len(img_metas)

        # 1. 准备真值数据并归一化
        targets = self._normalize_targets(gt_bboxes, gt_labels, img_metas, bs)

        # 2. 时间步采样
        t, lsas_log_probs = self._sample_t(bs, device)

        # DAEC Phase 2: 如果启用了类别匹配权重，需要先跑一次 no_grad forward 获取语义信息用于耦合
        cls_logits_for_coupling = None
        if (
            self.training
            and self.ot_coupling
            and self.ot_kcec
            and self.kcec_cls_weight > 0
        ):
            with torch.no_grad():
                # 使用当前 t 下的随机噪声框作为探测器
                noise_probe = torch.randn(
                    bs, self.num_proposals, 4, device=device
                )
                # 假设 x0 均值为 0，直接用噪声探测语义分布
                curr_bboxes_probe = self._raw_to_xyxy(noise_probe, img_metas)
                t_input_probe = (
                    t if self.diffusion_type == 'ddpm' else t * self.timesteps
                )
                all_cls_logits_probe, _, _, _, _ = self(
                    features, curr_bboxes_probe, t_input_probe
                )
                cls_logits_for_coupling = all_cls_logits_probe[-1]

        # 3. 构建训练配对
        x_boxes, x_starts, x_noises, kcec_log, matched_gt_indices = (
            self._build_training_targets(
                bs,
                device,
                t,
                targets,
                gt_bboxes,
                img_metas,
                cls_logits_for_coupling=cls_logits_for_coupling,
            )
        )
        x_noisy_batch = torch.stack(x_boxes)
        curr_bboxes = self._raw_to_xyxy(x_noisy_batch, img_metas)

        # 4. 前向传播 (含 TRD 自条件化)
        t_input = t if self.diffusion_type == 'ddpm' else t * self.timesteps
        (
            all_cls_logits,
            all_pred_bboxes,
            all_objectness,
            all_velocity,
            all_curr_proposals,
        ) = self._forward_trd(
            features, curr_bboxes, t, t_input, x_noisy_batch, img_metas, bs
        )

        # 5. 归一化并计算检测损失
        norm_pred_bboxes = self._normalize_pred_bboxes(
            all_pred_bboxes, img_metas
        )
        outputs = self._build_outputs(
            all_cls_logits, norm_pred_bboxes, all_objectness
        )
        losses = self.criterion(outputs, targets)

        # DAEC: 监督对比损失 (Supervised Contrastive Loss)
        # 对深层特征 all_curr_proposals[-1] 进行对比约束
        last_proposals = all_curr_proposals[
            -1
        ]  # [1, bs*num_proposals, feat_dim]
        if last_proposals is not None:
            # 调整 shape 从 [1, bs*num_proposals, feat_dim] -> [bs, num_proposals, feat_dim]
            last_proposals = last_proposals.squeeze(0).view(
                bs, self.num_proposals, -1
            )
            loss_contrastive = torch.tensor(0.0, device=device)
            valid_bs = 0
            for i in range(bs):
                gt_idx = matched_gt_indices[i]
                if len(targets[i].labels) == 0:
                    continue
                matched_labels = targets[i].labels[gt_idx]  # [num_proposals]

                features_i = F.normalize(last_proposals[i], dim=1)

                # Check dimensions to ensure they match
                if features_i.shape[0] != matched_labels.shape[0]:
                    continue

                temperature = self.daec_temperature
                sim_matrix = (
                    torch.matmul(features_i, features_i.T) / temperature
                )

                # 相同类别掩码 (过滤掉自己与自己的匹配)
                mask = torch.eq(
                    matched_labels.unsqueeze(1), matched_labels.unsqueeze(0)
                ).float()
                mask.fill_diagonal_(0)

                # 如果没有正样本对，跳过
                if mask.sum() == 0:
                    continue

                exp_sim = torch.exp(sim_matrix)
                # mask exp_sim to remove self similarity from denominator? No, InfoNCE includes self in denominator usually but removes it from numerator.
                # Actually, standard InfoNCE masks out the self similarity from the denominator as well to avoid trivial solutions, but for simplicity, we just compute the sum.
                # The issue was that mask shape and log_prob shape didn't match if features_i.shape[0] != matched_labels.shape[0], which we now check.

                log_prob = sim_matrix - torch.log(
                    exp_sim.sum(dim=1, keepdim=True)
                )

                mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (
                    mask.sum(dim=1) + 1e-8
                )
                loss_contrastive = loss_contrastive - mean_log_prob_pos.mean()
                valid_bs += 1

            if valid_bs > 0:
                # 权重可以配置化，暂时硬编码为 0.1 或加到配置中
                # 假设我们有一个 self.daec_contrastive_weight，如果没有则默认为 0.1
                weight = getattr(self, 'daec_contrastive_weight', 0.1)
                losses['loss_contrastive'] = (
                    loss_contrastive / valid_bs
                ) * weight

        # 6. 辅助损失
        self._add_velocity_loss(
            losses, all_velocity, x_starts, x_noises, device
        )
        self._add_cat_loss(
            losses,
            features,
            t,
            x_starts,
            x_noises,
            all_pred_bboxes,
            img_metas,
            bs,
            device,
        )
        self._add_lsas_loss(losses, lsas_log_probs)

        if kcec_log:
            for k, v in kcec_log.items():
                losses[k] = v.detach()

        return losses

    # ---- 损失计算辅助方法 ----

    def _normalize_targets(
        self,
        gt_bboxes: List[Tensor],
        gt_labels: List[Tensor],
        img_metas: List[ImageMeta],
        bs: int,
    ) -> List[InstanceData]:
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

    def _sample_t(
        self, bs: int, device: torch.device
    ) -> Tuple[Tensor, Optional[Tensor]]:
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

    def _forward_trd(
        self,
        features,
        curr_bboxes,
        t,
        t_input,
        x_noisy_batch,
        img_metas,
        bs,
    ) -> Tuple:
        """前向传播, 可选 TRD 自条件化."""
        if not (
            self.use_trd
            and self.diffusion_type == 'rectified_flow'
            and self.training
        ):
            return self(features, curr_bboxes, t_input)

        sc_mask = torch.rand(bs, device=t.device) < self.trd_self_cond_prob
        if not sc_mask.any():
            return self(features, curr_bboxes, t_input)

        with torch.no_grad():
            _, all_pred_sc, _, _, _ = self(features, curr_bboxes, t_input)
            x0_sc = self._xyxy_to_raw(all_pred_sc[-1], img_metas)
        x_noisy_sc = x_noisy_batch.clone()
        t_view = t.view(-1, 1, 1)
        v_ot_est = (x_noisy_sc - x0_sc) / torch.clamp(t_view, min=1e-5)
        x_noisy_sc[sc_mask] = (x_noisy_sc + self.trd_delta_t * v_ot_est)[
            sc_mask
        ]
        t_sc = t.clone()
        t_sc[sc_mask] = (t[sc_mask] + self.trd_delta_t).clamp(0, 1)
        curr_bboxes_sc = self._raw_to_xyxy(x_noisy_sc, img_metas)
        return self(features, curr_bboxes_sc, t_sc * self.timesteps)

    def _normalize_pred_bboxes(
        self, all_pred_bboxes: Tensor, img_metas
    ) -> Tensor:
        normed = all_pred_bboxes.clone()
        for i, meta in enumerate(img_metas):
            h, w = self._get_img_shape(meta)[:2]
            normed[:, i] /= normed.new_tensor([w, h, w, h])
        return normed

    def _build_outputs(
        self, all_cls_logits, norm_pred_bboxes, all_objectness
    ) -> ModelOutput:
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
                    pred_objectness=(
                        all_objectness[i]
                        if all_objectness[0] is not None
                        else None
                    ),
                )
                for i in range(self.num_heads - 1)
            ]
        return outputs

    def _add_velocity_loss(
        self,
        losses: dict,
        all_velocity: list,
        x_starts: list,
        x_noises: list,
        device: torch.device,
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

    def _add_cat_loss(
        self,
        losses: dict,
        features,
        t,
        x_starts,
        x_noises,
        all_pred_bboxes,
        img_metas,
        bs,
        device,
    ):
        if not (
            self.use_cat
            and self.diffusion_type == 'rectified_flow'
            and self.training
        ):
            return
        x_start_batch = torch.stack(x_starts)
        x_noise_batch = torch.stack(x_noises)
        dt = self.cat_delta_t
        with torch.no_grad():
            t2 = (t + dt).clamp(0, 1)
            t2_view = t2.view(-1, 1, 1)
            x_t2 = (1.0 - t2_view) * x_start_batch + t2_view * x_noise_batch
            curr_bboxes_t2 = self._raw_to_xyxy(x_t2, img_metas)
            _, all_pred_t2, _, _ = self(
                features, curr_bboxes_t2, t2 * self.timesteps
            )
            x0_t2 = self._xyxy_to_raw(all_pred_t2[-1], img_metas)
        x0_t1 = self._xyxy_to_raw(all_pred_bboxes[-1], img_metas)
        if self.cat_loss_type == 'velocity_curvature':
            t1_safe = t.view(-1, 1, 1).clamp_min(1e-3)
            t2_safe = t2_view.clamp_min(1e-3)
            v_t1 = (x_noise_batch - x0_t1) / t1_safe
            v_t2 = (x_noise_batch - x0_t2.detach()) / t2_safe
            losses['loss_curvature'] = F.mse_loss(v_t1, v_t2) * self.cat_weight
        else:
            losses['loss_curvature'] = (
                F.mse_loss(x0_t1, x0_t2.detach()) * self.cat_weight
            )

    @staticmethod
    def _add_lsas_loss(losses: dict, lsas_log_probs: Optional[Tensor]):
        if lsas_log_probs is None:
            return
        with torch.no_grad():
            total = sum(losses.values())
        losses['loss_lsas'] = total.detach() * (-lsas_log_probs).mean() * 0.01

    def _init_weights(self):
        """初始化权重"""
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

        # 重新零初始化 AdaLN-Zero 的输出层（_init_weights 会覆盖它）
        for head in self.head_series:
            if hasattr(head, 'adaln_mlp'):
                nn.init.zeros_(head.adaln_mlp[-1].weight)
                nn.init.zeros_(head.adaln_mlp[-1].bias)

    def _build_diffusion_buffers(self):
        """构建并注册扩散过程所需的常量 buffer"""
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

    def q_sample(
        self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        """前向扩散采样: x_t = sqrt(alpha_t_bar) * x_0 + sqrt(1 - alpha_t_bar) * noise"""
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

    def predict_noise_from_start(
        self, x_t: Tensor, t: Tensor, x0: Tensor
    ) -> Tensor:
        """从预测的 x0 还原噪声"""
        sqrt_recip_alphas_cumprod_t = load_buffer(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = load_buffer(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )
        return (
            sqrt_recip_alphas_cumprod_t * x_t - x0
        ) / sqrt_recipm1_alphas_cumprod_t

    def forward(
        self,
        features: Tuple[Tensor],
        bboxes: Tensor,
        t: Tensor,
        proposals: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List, List]:
        """前向传播，迭代去噪

        Args:
            features: FPN 特征元组
            bboxes: 当前边界框 [bs, num_proposals, 4] (xyxy)
            t: 当前时间步 [bs]
            proposals: 可选的提案特征

        Returns:
            all_cls_logits: [num_heads, bs, num_proposals, num_classes]
            all_pred_bboxes: [num_heads, bs, num_proposals, 4]
            all_objectness: list of [bs, num_proposals, 1] or [None, ...]
            all_velocity: list of [bs, num_proposals, 4] or [None, ...]
        """
        time_emb = self.time_mlp(t)

        inter_cls_logits = []
        inter_pred_bboxes = []
        inter_objectness = []
        inter_velocity = []
        inter_curr_proposals = []

        curr_bboxes = bboxes
        curr_proposals = proposals

        for head in self.head_series:
            result = head(
                features,
                curr_bboxes,
                curr_proposals,
                self.roi_extractor,
                time_emb,
            )

            # 兼容旧版 (3 返回值) 和新版 (5 返回值) 的 single_head
            if len(result) == 5:
                (
                    cls_logits,
                    pred_bboxes,
                    curr_proposals,
                    objectness,
                    velocity,
                ) = result
            else:
                cls_logits, pred_bboxes, curr_proposals = result
                objectness = None
                velocity = None

            inter_cls_logits.append(cls_logits)
            inter_pred_bboxes.append(pred_bboxes)
            inter_objectness.append(objectness)
            inter_velocity.append(velocity)
            inter_curr_proposals.append(curr_proposals)

            curr_bboxes = pred_bboxes.detach()

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

    def _forward_at_t(
        self,
        features: Tuple[Tensor],
        x_raw: Tensor,
        t: float,
        img_metas: List[ImageMeta],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """在指定时间步 t 进行前向预测"""
        bs, device = x_raw.shape[0], x_raw.device
        curr_bboxes = self._raw_to_xyxy(x_raw, img_metas)

        t_input = torch.full((bs,), t * self.timesteps, device=device)

        cls_logits_seq, pred_bboxes_seq, _, _, _ = self(
            features, curr_bboxes, t_input
        )

        last_cls_logits = cls_logits_seq[-1]
        last_pred_bboxes = pred_bboxes_seq[-1]

        x0_raw = self._xyxy_to_raw(last_pred_bboxes, img_metas)
        logits_0_raw = last_cls_logits

        return last_cls_logits, last_pred_bboxes, x0_raw, logits_0_raw

    @torch.no_grad()
    def predict(
        self,
        features: Tuple[Tensor],
        img_metas: List[ImageMeta],
        rescale: bool = True,
        return_trajectory: bool = False,
    ) -> List[DetectionResult]:
        """推理模式的前向传播"""
        device = features[0].device
        bs = len(img_metas)

        # 1. 准备采样时间序列
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

        # 初始噪声框
        x_raw = torch.randn(bs, self.num_proposals, 4, device=device)
        x0_prev = None

        ensemble_results = []
        trajectory = []

        dpm_solver = None
        if self.solver_type == 'dpm_solver_pp':
            from .rectified_flow import RFDPMSolverMultistep

            dpm_solver = RFDPMSolverMultistep(
                num_steps=self.sampling_timesteps, solver_order=2
            )

        # 2. 迭代采样
        for step_idx, (t_curr, t_next) in enumerate(time_pairs):
            if (
                self.use_trd
                and x0_prev is not None
                and self.diffusion_type == 'rectified_flow'
            ):
                t_curr_t = torch.full((bs,), t_curr, device=device)
                t_view = t_curr_t.view(-1, 1, 1)
                v_ot_est = (x_raw - x0_prev) / torch.clamp(t_view, min=1e-5)
                x_raw_refined = x_raw + self.trd_delta_t * v_ot_est
                cls_logits, pred_bboxes, x0_raw, logits_0_raw = (
                    self._forward_at_t(
                        features, x_raw_refined, t_curr, img_metas
                    )
                )
            else:
                cls_logits, pred_bboxes, x0_raw, logits_0_raw = (
                    self._forward_at_t(features, x_raw, t_curr, img_metas)
                )

            x0_prev = x0_raw.detach()

            if return_trajectory:
                trajectory.append((cls_logits.detach(), pred_bboxes.detach()))

            if self.use_ensemble:
                ensemble_results.append((cls_logits, pred_bboxes))

            if self.diffusion_type == 'ddpm':
                curr_bboxes_xyxy, x_raw = self._ddim_step(
                    t_curr, t_next, x_raw, cls_logits, pred_bboxes, img_metas
                )
                if t_next < 0:
                    break
            else:
                if dpm_solver is not None:
                    x_raw = dpm_solver.step(x_raw, x0_raw, t_curr, step_idx)
                elif self.solver_type == 'heun' and t_next > 0:

                    def model_fn(x_tmp, t_tmp):
                        _, _, x0_tmp, _ = self._forward_at_t(
                            features, x_tmp, t_tmp, img_metas
                        )
                        return x0_tmp, None

                    x_raw = self.rf.heun_step(
                        x_raw,
                        x0_raw,
                        t_curr,
                        t_next,
                        model_fn,
                    )
                else:
                    x_raw = self.rf.step(x_raw, x0_raw, t_curr, t_next)

                if self.box_renewal:
                    x_raw = self._apply_box_renewal(x_raw, cls_logits)

                if t_next <= 0:
                    break

        # 3. 后处理
        results = self._post_process(ensemble_results, img_metas, rescale)

        if return_trajectory:
            return results, trajectory

        return results

    def _apply_box_renewal(self, x_raw: Tensor, cls_logits: Tensor) -> Tensor:
        """通用的框更新策略"""
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
                x_raw_new[i, ~keep] = torch.randn(num_renew, 4, device=device)
        return x_raw_new

    @staticmethod
    def _get_img_shape(meta):
        """兼容 dict 和 ImageMeta 的 img_shape 获取"""
        if isinstance(meta, dict):
            return meta['img_shape']
        return meta.img_shape

    @staticmethod
    def _get_scale_factor(meta):
        """兼容 dict 和 ImageMeta 的 scale_factor 获取"""
        if isinstance(meta, dict):
            return meta.get('scale_factor')
        return meta.scale_factor

    def _xyxy_to_raw(self, bboxes: Tensor, img_metas) -> Tensor:
        """将图像空间的 xyxy 框转回扩散空间的 raw 框"""
        x0 = bboxes.clone()
        for i, meta in enumerate(img_metas):
            h, w = self._get_img_shape(meta)[:2]
            scale = x0.new_tensor([w, h, w, h])
            x0[i] /= scale
        x0 = bbox_xyxy_to_cxcywh(x0)
        x0 = (x0 * 2 - 1) * self.snr_scale
        return x0

    def _raw_to_xyxy(self, raw_bboxes: Tensor, img_metas) -> Tensor:
        """将扩散空间的 raw 框转为图像空间的 xyxy 框"""
        bboxes = (
            (
                raw_bboxes.clamp(-self.snr_scale, self.snr_scale)
                / self.snr_scale
            )
            + 1
        ) / 2
        bboxes = bbox_cxcywh_to_xyxy(bboxes)
        for i, meta in enumerate(img_metas):
            h, w = self._get_img_shape(meta)[:2]
            scale = bboxes.new_tensor([w, h, w, h])
            bboxes[i] *= scale
        return bboxes

    def _ddim_step(
        self,
        t_curr,
        t_next,
        x_raw,
        cls_logits,
        pred_bboxes,
        img_metas: List[ImageMeta],
    ):
        """执行一步 DDIM 采样"""
        bs, device = x_raw.shape[0], x_raw.device

        x0 = self._xyxy_to_raw(pred_bboxes, img_metas)
        if t_next < 0:
            return self._raw_to_xyxy(x0, img_metas), x0

        t_batch = torch.full((bs,), t_curr, device=device, dtype=torch.long)
        pred_noise = self.predict_noise_from_start(x_raw, t_batch, x0)

        alpha = self.alphas_cumprod[t_curr]
        alpha_next = self.alphas_cumprod[t_next]
        sigma = (
            self.ddim_sampling_eta
            * (
                (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
            ).sqrt()
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        noise = torch.randn_like(x_raw)
        x_raw_next = x0 * alpha_next.sqrt() + c * pred_noise + sigma * noise

        if self.box_renewal:
            x_raw_next = self._apply_box_renewal(x_raw_next, cls_logits)

        return self._raw_to_xyxy(x_raw_next, img_metas), x_raw_next

    def _post_process(
        self, ensemble_results, img_metas: List[ImageMeta], rescale: bool
    ) -> List[DetectionResult]:
        """后处理：集成、NMS、缩放"""
        results_list = []
        bs = len(img_metas)

        for i in range(bs):
            all_scores = []
            all_bboxes = []
            all_labels = []

            for cls_logits, pred_bboxes in ensemble_results:
                scores = torch.sigmoid(cls_logits[i])
                conf, labels = scores.max(-1)
                all_scores.append(conf)
                all_bboxes.append(pred_bboxes[i])
                all_labels.append(labels)

            final_scores = torch.cat(all_scores)
            final_bboxes = torch.cat(all_bboxes)
            final_labels = torch.cat(all_labels)

            if self.use_nms:
                keep = batched_nms(
                    final_bboxes,
                    final_scores,
                    final_labels,
                    self.nms_thr,
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
