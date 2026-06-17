"""群组层次 Sinkhorn 采样 (GHSS) — 论文 SOTA 耦合策略

在每个染色体组内独立运行 Sinkhorn stochastic OT，
消除无用的跨组匹配，保留组内多样性。
"""

from typing import Optional

import torch
from torch import Tensor

from ldmdet.coupling._sinkhorn_ops import ot_multinomial, sinkhorn_transport, sinkhorn_transport_batch
from ldmdet.coupling.base import CouplingStrategy, register_coupling
from ldmdet.utils.constants import CHROMO_GROUP_OF_CLASS


@register_coupling('ghss')
class GHSSCoupling(CouplingStrategy):
    """群组层次 Sinkhorn 采样。

    将 24 类染色体按 8 个生物学分组 (A-G + Sex)，
    每组独立运行 Sinkhorn stochastic OT。

    Args:
        epsilon: 熵正则化强度 (默认 5.0)
        num_iters: Sinkhorn 迭代次数 (默认 20)
        sample_seed: 可选随机种子
    """

    def __init__(
        self,
        epsilon: float = 5.0,
        num_iters: int = 20,
        sample_seed: Optional[int] = None,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.num_iters = num_iters
        self.sample_seed = sample_seed

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
    ) -> tuple[Tensor, Tensor]:
        N = noise.shape[0]
        num_gt = gt_labels.shape[0]
        if num_gt == 0:
            return noise, torch.zeros(N, dtype=torch.long, device=device)

        group_ids = self._chromo_group_of_class.to(device=device)
        gt_groups = group_ids[gt_labels]

        matched_gt_idx = torch.zeros(N, dtype=torch.long, device=device)
        offset = 0

        unique_groups, group_counts = torch.unique(gt_groups, return_counts=True)
        num_groups = unique_groups.shape[0]

        # 预计算每组的噪声分配和索引
        group_info = []
        for gi, grp in enumerate(unique_groups):
            grp_mask = gt_groups == grp
            grp_gt_idx = torch.where(grp_mask)[0]
            K_g = grp_gt_idx.shape[0]

            N_g = max(N * K_g // num_gt, 1)
            if gi == num_groups - 1:
                N_g = N - offset
            N_g = min(N_g, N - offset)
            if N_g <= 0:
                continue

            group_info.append((grp, grp_gt_idx, K_g, N_g, offset))
            offset += N_g

        if not group_info:
            x_start = gt_diffusion[matched_gt_idx]
            return x_start, matched_gt_idx

        # 批量构建代价矩阵和边缘分布
        costs = []
        row_masses = []
        col_masses = []
        for grp, grp_gt_idx, K_g, N_g, off in group_info:
            grp_noise = noise[off : off + N_g]
            grp_gt = gt_diffusion[grp_gt_idx]
            cost = torch.cdist(grp_noise, grp_gt, p=2)
            costs.append(cost)

            a = torch.ones(N_g, device=device) / N_g
            row_masses.append(a)

            proposals_per_gt = max(N_g // K_g, 1)
            gt_mass = torch.full((K_g,), proposals_per_gt / N_g, device=device)
            b = gt_mass / gt_mass.sum()
            col_masses.append(b)

        # 批量 Sinkhorn
        transports = sinkhorn_transport_batch(
            costs,
            epsilon=self.epsilon,
            num_iters=self.num_iters,
            row_masses=row_masses,
            col_masses=col_masses,
        )

        # 采样并写入结果
        for i, (grp, grp_gt_idx, K_g, N_g, off) in enumerate(group_info):
            row_probs = transports[i] / transports[i].sum(
                dim=1, keepdim=True
            ).clamp_min(1e-10)
            local_matched = ot_multinomial(
                row_probs, seed=self.sample_seed
            )
            matched_gt_idx[off : off + N_g] = grp_gt_idx[local_matched]

        x_start = gt_diffusion[matched_gt_idx]
        return x_start, matched_gt_idx
