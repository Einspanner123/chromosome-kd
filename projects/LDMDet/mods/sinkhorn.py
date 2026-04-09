"""
Sinkhorn OT 匹配器 — FlowDet 的核心创新组件

基于 Sinkhorn 算法的可微分最优传输 (OT) 匹配器，替代传统的 SimOTA 启发式匹配。
理论保证：OT 耦合天然提供噪声框→目标框的去重映射。

参考:
- Cuturi (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport.
- Peyrè & Cuturi (2019). Computational Optimal Transport.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import ops

from .structures import InstanceData, ModelOutput


class SinkhornOTMatcher(nn.Module):
    """基于 Sinkhorn 算法的可微分 OT 匹配器

    与 DiffusionDetMatcher 接口兼容，支持 drop-in 替换。

    核心设计:
    1. 代价矩阵 = 几何距离 + 分类代价 + IoU 代价
    2. Log 域 Sinkhorn 保证数值稳定性
    3. Dustbin 列处理背景 (未匹配提议框)
    4. O(N²·L) 复杂度，完全 GPU 并行
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        num_iters: int = 50,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        dustbin_cost: float = 10.0,
        match_costs: List = None,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.num_iters = num_iters
        self.dustbin_cost = dustbin_cost

        if match_costs is not None:
            self.costs = match_costs
        else:
            from .loss import BBoxL1Cost, FocalLossCost, IoUCost

            self.costs = [
                FocalLossCost(weight=cost_class),
                BBoxL1Cost(weight=cost_bbox),
                IoUCost(iou_mode="giou", weight=cost_giou),
            ]

    def sinkhorn_log_domain(
        self,
        cost_matrix: Tensor,
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        """Log 域 Sinkhorn 迭代，数值稳定

        Args:
            cost_matrix: (N, K+1) 代价矩阵，最后一列是 dustbin
            a: (N,) 源质量 (均匀: 1/N)
            b: (K+1,) 目标质量 (每个 GT 接收 ~N/K 个 proposal, dustbin 接收剩余)

        Returns:
            (N, K+1) 软分配矩阵
        """
        N, Kp1 = cost_matrix.shape

        # Log Kernel
        log_K = -cost_matrix / self.epsilon  # (N, K+1)
        log_a = torch.log(a + 1e-10)  # (N,)
        log_b = torch.log(b + 1e-10)  # (K+1,)

        # Sinkhorn 迭代 (log-domain for stability)
        log_u = torch.zeros_like(log_a)  # (N,)
        log_v = torch.zeros_like(log_b)  # (K+1,)

        for _ in range(self.num_iters):
            # Row update: u_i = a_i / sum_j K_ij * v_j
            log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            # Column update: v_j = b_j / sum_i K_ij * u_i
            log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)

        # 恢复传输矩阵
        log_T = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
        return torch.exp(log_T)

    @torch.no_grad()
    def forward(
        self, outputs: ModelOutput, targets: List[InstanceData]
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        与 DiffusionDetMatcher 相同接口

        Returns:
            List of (src_idx, gt_idx) tuples, one per batch element
        """
        pred_logits = outputs.pred_logits  # (B, N, C)
        pred_bboxes = outputs.pred_boxes  # (B, N, 4)
        batch_size = len(targets)

        batch_indices = []
        for i in range(batch_size):
            indices = self._single_assign(pred_logits[i], pred_bboxes[i], targets[i])
            batch_indices.append(indices)
        return batch_indices

    def _single_assign(
        self, pred_logits: Tensor, pred_bboxes: Tensor, target: InstanceData
    ) -> Tuple[Tensor, Tensor]:
        gt_bboxes = target.bboxes
        gt_labels = target.labels
        N = pred_logits.shape[0]
        K = gt_bboxes.size(0)
        device = pred_logits.device

        if K == 0:
            return (
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            )

        # 1. 计算代价矩阵 (N, K)
        cost_list = []
        for cost_fn in self.costs:
            cost = cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
            cost_list.append(cost)
        cost_matrix = torch.stack(cost_list).sum(0)  # (N, K)

        # 2. 添加 dustbin 列 (背景)
        dustbin_col = torch.full((N, 1), self.dustbin_cost, device=device)
        cost_with_dustbin = torch.cat([cost_matrix, dustbin_col], dim=1)  # (N, K+1)

        # 3. 设置质量约束
        # 源: 均匀分布
        a = torch.ones(N, device=device) / N

        # 目标: 每个 GT 接收 ~N/K 个 proposal 的质量
        # dustbin 接收剩余质量
        proposals_per_gt = max(N // max(K, 1), 1)
        gt_mass = torch.full((K,), proposals_per_gt / N, device=device)
        # 确保质量总和 = 1
        dustbin_mass = max(1.0 - gt_mass.sum().item(), 0.01)
        b = torch.cat([gt_mass, torch.tensor([dustbin_mass], device=device)])
        # 归一化
        b = b / b.sum()

        # 4. Sinkhorn 求解
        assignment = self.sinkhorn_log_domain(cost_with_dustbin, a, b)  # (N, K+1)

        # 5. 软→硬分配
        # 每个 proposal 分配到 cost 最小的 GT 或 dustbin
        gt_assignment = assignment[:, :K]  # (N, K)
        dustbin_assignment = assignment[:, K]  # (N,)

        # 前景: assignment 到 GT 的总概率 > dustbin 的概率
        fg_score, matched_gt_idx = gt_assignment.max(dim=1)  # (N,), (N,)
        is_fg = fg_score > dustbin_assignment

        # 返回前景 proposal 的索引和对应的 GT 索引
        src_idx = is_fg.nonzero().squeeze(1)
        gt_idx = matched_gt_idx[is_fg]

        return src_idx, gt_idx
