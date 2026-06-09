"""最优传输 (OT) 耦合模块

包含 Sinkhorn OT、KCEC (Karyotype-Constrained Entropic Coupling)、
Group-Hierarchical OT 等耦合策略，用于训练时将噪声提案与 GT 框配对。
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .chromo_constants import CHROMO_GROUP_OF_CLASS, CHROMO_QUOTA_OF_CLASS


class OTCoupling(nn.Module):
    """最优传输耦合模块，封装所有 OT 匹配策略"""

    def __init__(
        self,
        ot_coupling: bool = False,
        ot_matcher: str = 'nearest',
        ot_epsilon: float = 1.0,
        ot_num_iters: int = 20,
        ot_sample: bool = False,
        ot_sample_seed: Optional[int] = None,
        ot_group_hierarchical: bool = False,
        ot_kcec: bool = False,
        kcec_morph_weight: float = 0.25,
        kcec_group_weight: float = 0.1,
        kcec_cls_weight: float = 0.0,
        kcec_quota_strength: float = 1.0,
        kcec_slack: float = 0.05,
        kcec_conf_threshold: float = 0.5,
        kcec_scale_anneal: bool = False,
        kcec_log_interval: int = 100,
    ):
        super().__init__()
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

        self._kcec_call_count = 0

        # 注册染色体常量为 buffer (不参与梯度，但随模型移动设备)
        self.register_buffer(
            '_chromo_group_of_class',
            torch.tensor(CHROMO_GROUP_OF_CLASS, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            '_chromo_quota_of_class',
            torch.tensor(CHROMO_QUOTA_OF_CLASS, dtype=torch.float32),
            persistent=False,
        )

    def couple(
        self,
        noise: Tensor,
        gt_diffusion: Tensor,
        gt_labels: Tensor,
        device: torch.device,
        cls_logits: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        """OT 耦合: 将噪声提案与 GT 框配对

        Returns:
            x_start: 配对的 GT 框 (扩散空间)
            log_stats: 监控统计
            matched_gt_idx: 匹配的 GT 索引
        """
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

    # ---- Sinkhorn OT ----

    def sinkhorn_transport(
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

    def _sinkhorn_match(
        self, noise: Tensor, gt_diffusion: Tensor, device: torch.device
    ) -> Tensor:
        """Sinkhorn OT 匹配: 返回每个噪声提案对应的 GT 索引"""
        cost = torch.cdist(noise, gt_diffusion, p=2)
        transport = self.sinkhorn_transport(cost)

        if self.ot_sample:
            row_probs = transport / transport.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-10)
            return self._ot_multinomial(row_probs)
        return transport.argmax(dim=1)

    # ---- KCEC OT ----

    def _run_kcec_ot(
        self,
        noise: Tensor,
        gt_diffusion: Tensor,
        gt_labels: Tensor,
        device: torch.device,
        cls_logits: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Karyotype-Constrained Entropic Coupling V3 (Scale-Aware & Gated)"""
        N, K = noise.shape[0], gt_diffusion.shape[0]
        if K == 0:
            return torch.zeros(N, dtype=torch.long, device=device), {}

        quotas = self._chromo_quota_of_class.to(
            device=device, dtype=gt_diffusion.dtype
        )
        group_ids_all = self._chromo_group_of_class.to(device=device)
        labels = gt_labels.clamp(min=0, max=quotas.numel() - 1)

        box_cost = torch.cdist(noise, gt_diffusion, p=2)

        # 1. 计算每个 GT 的边际质量
        gt_mass = torch.zeros(K, device=device, dtype=gt_diffusion.dtype)
        unique_classes = torch.unique(labels)
        for cls_id in unique_classes:
            gt_mask = labels == cls_id
            n_observed = gt_mask.sum().float()
            q_c = quotas[cls_id.item()]
            gt_mass[gt_mask] = q_c / n_observed

        # V3: Confidence Gating
        if self.kcec_conf_threshold > 0:
            min_cost_per_gt = box_cost.min(dim=0).values
            gate = (min_cost_per_gt < self.kcec_conf_threshold).float()
            gt_mass = 1.0 + (gt_mass - 1.0) * gate

        # V3: Scale-Aware Quota
        strength = self.kcec_quota_strength
        if self.kcec_scale_anneal:
            gt_areas = gt_diffusion[:, 2] * gt_diffusion[:, 3]
            mean_area = gt_areas.mean().clamp_min(1e-6)
            scale_factor = (gt_areas / mean_area).sqrt()
            strength = strength * scale_factor

        gt_mass = gt_mass.pow(strength)
        gt_mass = gt_mass + float(self.kcec_slack)
        col_mass = gt_mass / gt_mass.sum().clamp_min(1e-10)
        row_mass = torch.ones(N, device=device) / max(N, 1)

        # 2. 计算代价矩阵
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

        cls_cost = torch.zeros(N, K, device=device)
        if self.kcec_cls_weight > 0 and cls_logits is not None:
            prob = torch.sigmoid(cls_logits)
            cls_cost = 1.0 - prob[:, labels]

        cost = (
            box_cost
            + self.kcec_morph_weight * morph_cost
            + self.kcec_group_weight * group_cost
            + self.kcec_cls_weight * cls_cost
        )

        # 3. Sinkhorn 迭代
        transport = self.sinkhorn_transport(
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
            log_stats = self._compute_kcec_stats(
                transport,
                cost,
                box_cost,
                morph_cost,
                group_cost,
                matched_gt_idx,
                row_mass,
                col_mass,
                unique_classes,
                K,
                device,
            )

        return matched_gt_idx, log_stats

    def _compute_kcec_stats(
        self,
        transport,
        cost,
        box_cost,
        morph_cost,
        group_cost,
        matched_gt_idx,
        row_mass,
        col_mass,
        unique_classes,
        K,
        device,
    ) -> Dict[str, Tensor]:
        """计算 KCEC 监控统计"""
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
            argmax_agreement = (transport_argmax == cost_argmin).float().mean()
            row_marginal_err = (transport.sum(dim=1) - row_mass).abs().mean()
            col_marginal_err = (transport.sum(dim=0) - col_mass).abs().mean()
            return {
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
            }

    # ---- Group-Hierarchical OT ----

    def _run_group_hierarchical_ot(
        self, noise, gt_diffusion, gt_labels, device
    ):
        """按染色体组分层 OT 耦合, 每组独立运行 Sinkhorn"""
        N = noise.shape[0]
        num_gt = gt_labels.shape[0]
        if num_gt == 0:
            return torch.randint(0, max(num_gt, 1), (N,), device=device)

        group_ids = self._chromo_group_of_class.to(device=device)
        gt_groups = group_ids[gt_labels]

        matched_gt_idx = torch.zeros(N, dtype=torch.long, device=device)
        offset = 0

        unique_groups = torch.unique(gt_groups)
        for grp in unique_groups:
            grp_mask = gt_groups == grp
            grp_gt_idx = torch.where(grp_mask)[0]
            K_g = grp_gt_idx.shape[0]

            N_g = max(N * K_g // num_gt, 1)
            if grp == unique_groups[-1]:
                N_g = N - offset
            N_g = min(N_g, N - offset)
            if N_g <= 0:
                continue

            grp_noise = noise[offset : offset + N_g]
            grp_gt = gt_diffusion[grp_gt_idx]

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

            matched_gt_idx[offset : offset + N_g] = grp_gt_idx[local_matched]
            offset += N_g

        return matched_gt_idx

    # ---- 工具方法 ----

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
