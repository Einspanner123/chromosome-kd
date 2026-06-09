"""动态 Top-K 匹配器 (SimOTA 风格)"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import ops

from .costs import BBoxL1Cost, FocalLossCost, IoUCost
from .structures import InstanceData, ModelOutput


class DiffusionDetMatcher(nn.Module):
    """DiffusionDet 动态 Top-K 匹配器 (SimOTA 风格)"""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        center_radius: float = 2.5,
        candidate_topk: int = 5,
        match_costs: List[dict] = None,
    ):
        super().__init__()
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        if match_costs is not None:
            self.costs = match_costs
        else:
            self.costs = [
                FocalLossCost(weight=cost_class),
                BBoxL1Cost(weight=cost_bbox),
                IoUCost(iou_mode='giou', weight=cost_giou),
            ]

    @torch.no_grad()
    def forward(
        self, outputs: ModelOutput, targets: List[InstanceData]
    ) -> List[Tuple[Tensor, Tensor]]:
        pred_logits = outputs.pred_logits
        pred_bboxes = outputs.pred_boxes
        batch_size = len(targets)

        batch_indices = []
        for i in range(batch_size):
            indices = self._single_assign(
                pred_logits[i], pred_bboxes[i], targets[i]
            )
            batch_indices.append(indices)

        return batch_indices

    def _single_assign(
        self, pred_logits: Tensor, pred_bboxes: Tensor, target: InstanceData
    ) -> Tuple[Tensor, Tensor]:
        gt_bboxes = target.bboxes
        gt_labels = target.labels
        num_gt = gt_bboxes.size(0)

        if num_gt == 0:
            device = pred_bboxes.device
            return torch.zeros(
                0, dtype=torch.long, device=device
            ), torch.zeros(0, dtype=torch.long, device=device)

        # 1. 计算各项 Cost
        cost_list = []
        for cost_fn in self.costs:
            cost = cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
            cost_list.append(cost)

        # 2. 计算是否在 GT 框内或中心范围内
        is_in_boxes_anchor, is_in_boxes_and_center = self._get_in_gt_info(
            pred_bboxes, gt_bboxes
        )

        cost_list.append((~is_in_boxes_and_center) * 100.0)
        cost_matrix = torch.stack(cost_list).sum(0)

        cost_matrix = torch.nan_to_num(
            cost_matrix, nan=1e6, posinf=1e6, neginf=1e6
        )

        cost_matrix[~is_in_boxes_anchor] += 10000.0

        # 3. 动态 K 匹配
        pairwise_ious = ops.box_iou(pred_bboxes, gt_bboxes)
        return self._dynamic_k_matching(cost_matrix, pairwise_ious, num_gt)

    def _get_in_gt_info(
        self, pred_bboxes: Tensor, gt_bboxes: Tensor
    ) -> Tuple[Tensor, Tensor]:
        pred_ctrs = (pred_bboxes[:, :2] + pred_bboxes[:, 2:]) / 2
        gt_ctrs = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        gt_wh = gt_bboxes[:, 2:] - gt_bboxes[:, :2]

        lt = pred_ctrs.unsqueeze(1) - gt_bboxes[:, :2].unsqueeze(0)
        rb = gt_bboxes[:, 2:].unsqueeze(0) - pred_ctrs.unsqueeze(1)
        is_in_boxes = torch.cat([lt, rb], dim=-1).min(-1)[0] > 0

        lt_c = pred_ctrs.unsqueeze(1) - (
            gt_ctrs - self.center_radius * gt_wh
        ).unsqueeze(0)
        rb_c = (gt_ctrs + self.center_radius * gt_wh).unsqueeze(
            0
        ) - pred_ctrs.unsqueeze(1)
        is_in_centers = torch.cat([lt_c, rb_c], dim=-1).min(-1)[0] > 0

        is_in_boxes_anchor = is_in_boxes.any(1) | is_in_centers.any(1)
        is_in_boxes_and_center = is_in_boxes & is_in_centers
        return is_in_boxes_anchor, is_in_boxes_and_center

    def _dynamic_k_matching(
        self, cost: Tensor, pairwise_ious: Tensor, num_gt: int
    ) -> Tuple[Tensor, Tensor]:
        matching_matrix = torch.zeros_like(cost)

        pairwise_ious = torch.nan_to_num(
            pairwise_ious, nan=0.0, posinf=1.0, neginf=0.0
        )

        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        dynamic_ks = torch.clamp(
            topk_ious.sum(0).int(), min=1, max=candidate_topk
        )

        for gt_idx in range(num_gt):
            k = dynamic_ks[gt_idx].item()
            _, pos_idx = torch.topk(cost[:, gt_idx], k=k, largest=False)
            matching_matrix[pos_idx, gt_idx] = 1.0

        duplicate_idx = matching_matrix.sum(1) > 1
        if duplicate_idx.any():
            _, cost_argmin = cost[duplicate_idx].min(1)
            matching_matrix[duplicate_idx] = 0.0
            matching_matrix[duplicate_idx, cost_argmin] = 1.0

        fg_mask = matching_matrix.sum(1) > 0
        matched_gt_inds = matching_matrix[fg_mask].argmax(1)
        return fg_mask.nonzero().squeeze(1), matched_gt_inds
