"""最优传输 (OT) 耦合模块

包含 Sinkhorn Stochastic OT 和 Group-Hierarchical Stochastic OT 耦合策略，
用于训练时将噪声提案与 GT 框配对。

实验结论:
- 硬 OT (argmax) 在所有 ε 下都不如随机耦合，因为确定性分配消除了训练多样性
- Stochastic Coupling (从传输矩阵采样) 修复了多样性问题，eps=5 时 mAP=0.751
- Group-Hierarchical Stochastic OT (利用染色体分组先验) mAP=0.752
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .chromo_constants import CHROMO_GROUP_OF_CLASS


class OTCoupling(nn.Module):
    """最优传输耦合模块，封装 Stochastic OT 和 Group-Hierarchical OT"""

    def __init__(
        self,
        ot_coupling: bool = False,
        ot_epsilon: float = 1.0,
        ot_num_iters: int = 20,
        ot_sample_seed: Optional[int] = None,
        ot_group_hierarchical: bool = False,
        ot_sample: bool = True,
    ):
        super().__init__()
        self.ot_coupling = ot_coupling
        self.ot_epsilon = ot_epsilon
        self.ot_num_iters = ot_num_iters
        self.ot_sample_seed = ot_sample_seed
        self.ot_group_hierarchical = ot_group_hierarchical
        self.ot_sample = ot_sample  # True=multinomial采样, False=argmax

        # 诊断缓存: couple() 调用后可被 CouplingDiagnostics 读取
        self.last_transport = None  # [N, M] 传输矩阵
        self.last_cost = None  # [N, M] cost 矩阵
        self.last_coupling_mode = 'none'  # 'sinkhorn' / 'group_hierarchical' / 'none'

        # 注册染色体分组常量为 buffer (不参与梯度，但随模型移动设备)
        self.register_buffer(
            '_chromo_group_of_class',
            torch.tensor(CHROMO_GROUP_OF_CLASS, dtype=torch.long),
            persistent=False,
        )

    def couple(
        self,
        noise: Tensor,
        gt_diffusion: Tensor,
        gt_labels: Tensor,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """OT 耦合: 将噪声提案与 GT 框配对 (Stochastic 采样)

        Returns:
            x_start: 配对的 GT 框 (扩散空间)
            matched_gt_idx: 匹配的 GT 索引
        """
        if gt_diffusion.shape[0] == 0:
            return (
                noise,
                torch.zeros(noise.shape[0], dtype=torch.long, device=device),
            )

        if self.ot_group_hierarchical:
            matched_gt_idx = self._run_group_hierarchical_ot(
                noise, gt_diffusion, gt_labels, device
            )
        else:
            matched_gt_idx = self._sinkhorn_stochastic_match(
                noise, gt_diffusion, device
            )

        x_start = gt_diffusion[matched_gt_idx]
        return x_start, matched_gt_idx

    def _clear_diag_cache(self):
        """清除诊断缓存 (在每次 couple 调用前)."""
        self.last_transport = None
        self.last_cost = None
        self.last_coupling_mode = 'none'

    # ---- Sinkhorn Stochastic OT ----

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

    def _sinkhorn_stochastic_match(
        self, noise: Tensor, gt_diffusion: Tensor, device: torch.device
    ) -> Tensor:
        """Sinkhorn OT + Stochastic 采样/Argmax: 从传输矩阵采样或取 argmax"""
        cost = torch.cdist(noise, gt_diffusion, p=2)
        transport = self.sinkhorn_transport(cost)
        # 缓存供 CouplingDiagnostics 使用
        self.last_cost = cost.detach()
        self.last_transport = transport.detach()
        self.last_coupling_mode = 'sinkhorn' if self.ot_sample else 'sinkhorn_argmax'
        row_probs = transport / transport.sum(dim=1, keepdim=True).clamp_min(
            1e-10
        )
        if self.ot_sample:
            return self._ot_multinomial(row_probs)
        else:
            return row_probs.argmax(dim=1)

    # ---- Group-Hierarchical Stochastic OT ----

    def _run_group_hierarchical_ot(
        self, noise, gt_diffusion, gt_labels, device
    ):
        """按染色体组分层 OT 耦合, 每组独立运行 Sinkhorn + Stochastic 采样"""
        # 标记耦合模式 (分组 OT 有多个子传输矩阵, 不缓存单个)
        self.last_coupling_mode = 'group_hierarchical'
        self.last_transport = None
        self.last_cost = None
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

            row_probs = transport / transport.sum(dim=1, keepdim=True)
            local_matched = self._ot_multinomial(row_probs)

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
