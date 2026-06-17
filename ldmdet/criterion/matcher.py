"""动态 Top-K 匹配器 (SimOTA 风格)"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import ops

from ldmdet.criterion.costs import BBoxL1Cost, FocalLossCost, IoUCost
from ldmdet.data.structures import InstanceData, ModelOutput


class DiffusionDetMatcher(nn.Module):
    """DiffusionDet 动态 Top-K 匹配器"""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        center_radius: float = 2.5,
        candidate_topk: int = 5,
        match_costs: List = None,
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
    def forward(self, outputs: ModelOutput, targets: List[InstanceData]) -> List[Tuple[Tensor, Tensor]]:
        pred_logits = outputs.pred_logits
        pred_bboxes = outputs.pred_boxes
        return [self._single_assign(pred_logits[i], pred_bboxes[i], targets[i]) for i in range(len(targets))]

    @torch.no_grad()
    def forward_with_gt_cache(self, outputs: ModelOutput, targets: List[InstanceData],
                              gt_cache: List[dict] = None) -> Tuple[List[Tuple[Tensor, Tensor]], List[dict]]:
        """带 GT 缓存的匹配，用于 deep_supervision 中共享 GT 相关计算。

        Returns:
            indices: 匹配结果
            gt_cache: GT 相关预计算结果，可传给下一次调用
        """
        pred_logits = outputs.pred_logits
        pred_bboxes = outputs.pred_boxes
        indices = []
        new_cache = []
        for i in range(len(targets)):
            idx, cache = self._single_assign_cached(
                pred_logits[i], pred_bboxes[i], targets[i],
                gt_cache[i] if gt_cache else None
            )
            indices.append(idx)
            new_cache.append(cache)
        return indices, new_cache

    def _single_assign(self, pred_logits: Tensor, pred_bboxes: Tensor, target: InstanceData) -> Tuple[Tensor, Tensor]:
        gt_bboxes = target.bboxes
        gt_labels = target.labels
        num_gt = gt_bboxes.size(0)
        if num_gt == 0:
            return torch.zeros(0, dtype=torch.long, device=pred_bboxes.device), torch.zeros(0, dtype=torch.long, device=pred_bboxes.device)

        # 预计算 GT 相关信息（只依赖 GT，不依赖 pred）
        pred_ctrs = (pred_bboxes[:, :2] + pred_bboxes[:, 2:]) / 2
        gt_ctrs = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        gt_wh = gt_bboxes[:, 2:] - gt_bboxes[:, :2]

        # is_in_boxes 和 is_in_centers: [N, M]
        lt = pred_ctrs.unsqueeze(1) - gt_bboxes[:, :2].unsqueeze(0)
        rb = gt_bboxes[:, 2:].unsqueeze(0) - pred_ctrs.unsqueeze(1)
        is_in_boxes = torch.cat([lt, rb], dim=-1).min(-1)[0] > 0

        lt_c = pred_ctrs.unsqueeze(1) - (gt_ctrs - self.center_radius * gt_wh).unsqueeze(0)
        rb_c = (gt_ctrs + self.center_radius * gt_wh).unsqueeze(0) - pred_ctrs.unsqueeze(1)
        is_in_centers = torch.cat([lt_c, rb_c], dim=-1).min(-1)[0] > 0

        is_in_boxes_anchor = is_in_boxes.any(1) | is_in_centers.any(1)
        is_in_boxes_and_center = is_in_boxes & is_in_centers

        # 代价计算 — 仅计算 FocalLossCost 和 BBoxL1Cost
        cost_list = []
        for cost_fn in self.costs:
            if isinstance(cost_fn, IoUCost):
                continue  # 稍后与 pairwise_ious 合并计算
            cost_list.append(cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes))

        # 合并计算: box_iou 用于 dynamic_k_matching，giou_cost 用于代价矩阵
        pairwise_ious = ops.box_iou(pred_bboxes, gt_bboxes)
        # 找到 IoUCost 并计算 giou_cost
        for cost_fn in self.costs:
            if isinstance(cost_fn, IoUCost):
                if cost_fn.iou_mode == 'giou':
                    giou = ops.generalized_box_iou(pred_bboxes, gt_bboxes)
                    iou_cost = (1 - giou) * cost_fn.weight
                else:
                    iou_cost = (1 - pairwise_ious) * cost_fn.weight
                cost_list.append(iou_cost)
                break

        cost_list.append((~is_in_boxes_and_center) * 100.0)
        cost_matrix = torch.stack(cost_list).sum(0)
        cost_matrix = torch.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=1e6)
        cost_matrix[~is_in_boxes_anchor] += 10000.0

        return self._dynamic_k_matching(cost_matrix, pairwise_ious, num_gt)

    def _single_assign_cached(self, pred_logits: Tensor, pred_bboxes: Tensor,
                               target: InstanceData, cache: dict = None) -> Tuple[Tuple[Tensor, Tensor], dict]:
        """带缓存的匹配，GT 相关信息只计算一次。"""
        gt_bboxes = target.bboxes
        gt_labels = target.labels
        num_gt = gt_bboxes.size(0)
        if num_gt == 0:
            empty = (torch.zeros(0, dtype=torch.long, device=pred_bboxes.device),
                     torch.zeros(0, dtype=torch.long, device=pred_bboxes.device))
            return empty, {}

        pred_ctrs = (pred_bboxes[:, :2] + pred_bboxes[:, 2:]) / 2

        # 使用或创建 GT 缓存
        if cache is None:
            gt_ctrs = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
            gt_wh = gt_bboxes[:, 2:] - gt_bboxes[:, :2]
            center_tl = gt_ctrs - self.center_radius * gt_wh
            center_br = gt_ctrs + self.center_radius * gt_wh
            cache = {
                'gt_ctrs': gt_ctrs,
                'gt_wh': gt_wh,
                'center_tl': center_tl,
                'center_br': center_br,
                'gt_labels': gt_labels,
                'gt_bboxes': gt_bboxes,
            }

        gt_ctrs = cache['gt_ctrs']
        gt_wh = cache['gt_wh']
        center_tl = cache['center_tl']
        center_br = cache['center_br']

        # is_in_boxes 和 is_in_centers: [N, M]
        lt = pred_ctrs.unsqueeze(1) - gt_bboxes[:, :2].unsqueeze(0)
        rb = gt_bboxes[:, 2:].unsqueeze(0) - pred_ctrs.unsqueeze(1)
        is_in_boxes = torch.cat([lt, rb], dim=-1).min(-1)[0] > 0

        lt_c = pred_ctrs.unsqueeze(1) - center_tl.unsqueeze(0)
        rb_c = center_br.unsqueeze(0) - pred_ctrs.unsqueeze(1)
        is_in_centers = torch.cat([lt_c, rb_c], dim=-1).min(-1)[0] > 0

        is_in_boxes_anchor = is_in_boxes.any(1) | is_in_centers.any(1)
        is_in_boxes_and_center = is_in_boxes & is_in_centers

        # 代价计算
        cost_list = []
        for cost_fn in self.costs:
            if isinstance(cost_fn, IoUCost):
                continue
            cost_list.append(cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes))

        pairwise_ious = ops.box_iou(pred_bboxes, gt_bboxes)
        for cost_fn in self.costs:
            if isinstance(cost_fn, IoUCost):
                if cost_fn.iou_mode == 'giou':
                    giou = ops.generalized_box_iou(pred_bboxes, gt_bboxes)
                    iou_cost = (1 - giou) * cost_fn.weight
                else:
                    iou_cost = (1 - pairwise_ious) * cost_fn.weight
                cost_list.append(iou_cost)
                break

        cost_list.append((~is_in_boxes_and_center) * 100.0)
        cost_matrix = torch.stack(cost_list).sum(0)
        cost_matrix = torch.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=1e6)
        cost_matrix[~is_in_boxes_anchor] += 10000.0

        result = self._dynamic_k_matching(cost_matrix, pairwise_ious, num_gt)
        return result, cache

    def _get_in_gt_info(self, pred_bboxes, gt_bboxes) -> Tuple[Tensor, Tensor]:
        # pred_ctrs: [N, 2], gt_ctrs: [M, 2]
        pred_ctrs = (pred_bboxes[:, :2] + pred_bboxes[:, 2:]) / 2
        gt_ctrs = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        gt_wh = gt_bboxes[:, 2:] - gt_bboxes[:, :2]

        # is_in_boxes: [N, M] — 预测中心是否在 GT 框内
        lt = pred_ctrs.unsqueeze(1) - gt_bboxes[:, :2].unsqueeze(0)   # [N, M, 2]
        rb = gt_bboxes[:, 2:].unsqueeze(0) - pred_ctrs.unsqueeze(1)   # [N, M, 2]
        is_in_boxes = torch.cat([lt, rb], dim=-1).min(-1)[0] > 0      # [N, M]

        # is_in_centers: [N, M] — 预测中心是否在 GT 中心区域
        lt_c = pred_ctrs.unsqueeze(1) - (gt_ctrs - self.center_radius * gt_wh).unsqueeze(0)
        rb_c = (gt_ctrs + self.center_radius * gt_wh).unsqueeze(0) - pred_ctrs.unsqueeze(1)
        is_in_centers = torch.cat([lt_c, rb_c], dim=-1).min(-1)[0] > 0

        is_in_boxes_anchor = is_in_boxes.any(1) | is_in_centers.any(1)
        is_in_boxes_and_center = is_in_boxes & is_in_centers
        return is_in_boxes_anchor, is_in_boxes_and_center

    def _dynamic_k_matching(self, cost, pairwise_ious, num_gt) -> Tuple[Tensor, Tensor]:
        matching_matrix = torch.zeros_like(cost)
        pairwise_ious = torch.nan_to_num(pairwise_ious, nan=0.0, posinf=1.0, neginf=0.0)
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1, max=candidate_topk)

        # 向量化: 使用 scatter 操作替代逐 GT 循环
        sorted_cost_indices = cost.argsort(dim=0)  # [N, num_gt]
        # 构建行/列索引对
        gt_cols = torch.arange(num_gt, device=cost.device).unsqueeze(0).expand(candidate_topk, -1)
        # 对每个 GT，取前 k 个候选的行索引
        max_k = dynamic_ks.max().item()
        top_rows = sorted_cost_indices[:max_k]  # [max_k, num_gt]
        # 创建 mask: row_idx < dynamic_ks[gt_idx]
        row_positions = torch.arange(max_k, device=cost.device).unsqueeze(1)  # [max_k, 1]
        valid_mask = row_positions < dynamic_ks.unsqueeze(0)  # [max_k, num_gt]
        # 使用 scatter 填充 matching_matrix
        valid_rows = top_rows[valid_mask]  # [total_valid]
        valid_cols = gt_cols[:max_k][valid_mask]  # [total_valid]
        matching_matrix[valid_rows, valid_cols] = 1.0

        duplicate_idx = matching_matrix.sum(1) > 1
        if duplicate_idx.any():
            _, cost_argmin = cost[duplicate_idx].min(1)
            matching_matrix[duplicate_idx] = 0.0
            matching_matrix[duplicate_idx, cost_argmin] = 1.0

        fg_mask = matching_matrix.sum(1) > 0
        matched_gt_inds = matching_matrix[fg_mask].argmax(1)
        return fg_mask.nonzero().squeeze(1), matched_gt_inds
