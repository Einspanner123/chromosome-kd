"""非平衡 GHSS 耦合 — 方向一: 非平衡最优传输耦合

在 GHSS 的组内 Sinkhorn 基础上, 引入非平衡 OT (Chizat 2018),
允许边缘约束通过 KL 散度软化为可调松弛, 适应 GT 分布不均匀的场景.

与标准 GHSS 的区别:
1. 组内 OT 用非平衡 Sinkhorn, 允许边缘松弛
2. 不强制 proposals_per_gt 均分, 让 OT 自然决定
3. 空组 (K_g=0) 的 proposal 分配给最大组

开关: 通过 coupling type='unbalanced_ghss' 切换, 不影响 baseline 'ghss'.
失败时直接删除本文件 + _sinkhorn_ops.unbalanced_sinkhorn_transport 即可回滚.
"""

from typing import Optional

import torch
from torch import Tensor

from ldmdet.coupling._sinkhorn_ops import (
    ot_multinomial,
    unbalanced_sinkhorn_transport,
)
from ldmdet.coupling.base import CouplingStrategy, register_coupling
from ldmdet.utils.constants import CHROMO_GROUP_OF_CLASS


@register_coupling('unbalanced_ghss')
class UnbalancedGHSSCoupling(CouplingStrategy):
    """非平衡 GHSS 耦合.

    Args:
        epsilon: 熵正则化强度 (默认 5.0, 与 GHSS baseline 一致)
        num_iters: Sinkhorn 迭代次数 (默认 20)
        lambda_row: 行边缘松弛系数
            - ∞: 退化为标准 Sinkhorn (严格行约束, 等价于 GHSS)
            - 1.0: 软约束 (推荐起点)
            - 0.1: 强松弛 (更分散的匹配)
        lambda_col: 列边缘松弛系数 (同上)
        sample_seed: 可选随机种子
    """

    def __init__(
        self,
        epsilon: float = 5.0,
        num_iters: int = 20,
        lambda_row: float = 1.0,
        lambda_col: float = 1.0,
        sample_seed: Optional[int] = None,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.num_iters = num_iters
        self.lambda_row = lambda_row
        self.lambda_col = lambda_col
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

        # 按组分配 proposal 数 (与 GHSS 相同的比例分配)
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

        # 逐组非平衡 Sinkhorn (组数少, 通常 ≤8, 逐组开销可接受)
        # 不使用 batch 版本: 非平衡迭代公式与标准 Sinkhorn 不同, 需单独实现
        for grp, grp_gt_idx, K_g, N_g, off in group_info:
            grp_noise = noise[off : off + N_g]
            grp_gt = gt_diffusion[grp_gt_idx]
            cost = torch.cdist(grp_noise, grp_gt, p=2)

            # 行边缘: 均匀
            a = torch.ones(N_g, device=device) / N_g
            # 列边缘: 不再强制 proposals_per_gt, 用均匀目标让 OT 自由决定
            b = torch.ones(K_g, device=device) / K_g

            transport = unbalanced_sinkhorn_transport(
                cost,
                epsilon=self.epsilon,
                num_iters=self.num_iters,
                row_mass=a,
                col_mass=b,
                lambda_row=self.lambda_row,
                lambda_col=self.lambda_col,
            )

            row_probs = transport / transport.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-10)
            local_matched = ot_multinomial(row_probs, seed=self.sample_seed)
            matched_gt_idx[off : off + N_g] = grp_gt_idx[local_matched]

        x_start = gt_diffusion[matched_gt_idx]
        return x_start, matched_gt_idx
