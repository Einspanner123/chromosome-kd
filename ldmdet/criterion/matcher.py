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
        pred_logits = outputs.pred_logits  # [bs, N, C]
        pred_bboxes = outputs.pred_boxes  # [bs, N, 4]
        bs = pred_logits.size(0)
        N = pred_logits.size(1)

        # Pad GT bboxes and labels to uniform shape
        max_gt = max(t.bboxes.size(0) for t in targets)
        if max_gt == 0:
            return [(torch.zeros(N, dtype=torch.bool, device=pred_bboxes.device),
                     torch.zeros(N, dtype=torch.long, device=pred_bboxes.device))] * bs

        gt_bboxes_padded = pred_bboxes.new_zeros(bs, max_gt, 4)
        gt_labels_padded = pred_logits.new_full((bs, max_gt), 0, dtype=torch.long)
        gt_num = []
        for i, t in enumerate(targets):
            n = t.bboxes.size(0)
            gt_num.append(n)
            if n > 0:
                gt_bboxes_padded[i, :n] = t.bboxes
                gt_labels_padded[i, :n] = t.labels

        # Batched cost computation
        cost_matrix, pairwise_ious = self._batched_cost(
            pred_logits, pred_bboxes, gt_labels_padded, gt_bboxes_padded, gt_num, max_gt
        )

        # Per-image dynamic_k_matching
        results = []
        for i in range(bs):
            n = gt_num[i]
            if n == 0:
                results.append((torch.zeros(N, dtype=torch.bool, device=pred_bboxes.device),
                                torch.zeros(N, dtype=torch.long, device=pred_bboxes.device)))
            else:
                results.append(self._dynamic_k_matching(cost_matrix[i, :, :n], pairwise_ious[i, :, :n], n))
        return results

    @torch.no_grad()
    def forward_with_gt_cache(self, outputs: ModelOutput, targets: List[InstanceData],
                              gt_cache: List[dict] = None) -> Tuple[List[Tuple[Tensor, Tensor]], List[dict]]:
        """带 GT 缓存的批量匹配，用于 deep_supervision 中共享 GT 相关计算。"""
        pred_logits = outputs.pred_logits
        pred_bboxes = outputs.pred_boxes
        bs = pred_logits.size(0)
        N = pred_logits.size(1)

        # Build or reuse GT cache
        if gt_cache is None:
            max_gt = max(t.bboxes.size(0) for t in targets)
            gt_bboxes_padded = pred_bboxes.new_zeros(bs, max_gt, 4)
            gt_labels_padded = pred_logits.new_full((bs, max_gt), 0, dtype=torch.long)
            gt_num = []
            new_cache = []
            for i, t in enumerate(targets):
                n = t.bboxes.size(0)
                gt_num.append(n)
                if n > 0:
                    gt_bboxes_padded[i, :n] = t.bboxes
                    gt_labels_padded[i, :n] = t.labels
                gt_ctrs = (t.bboxes[:, :2] + t.bboxes[:, 2:]) / 2 if n > 0 else t.bboxes.new_zeros(0, 2)
                gt_wh = (t.bboxes[:, 2:] - t.bboxes[:, :2]) if n > 0 else t.bboxes.new_zeros(0, 2)
                new_cache.append({
                    'gt_bboxes_padded': gt_bboxes_padded[i],
                    'gt_labels_padded': gt_labels_padded[i],
                    'gt_num': n,
                    'max_gt': max_gt,
                    'gt_ctrs': gt_ctrs,
                    'gt_wh': gt_wh,
                    'center_tl': gt_ctrs - self.center_radius * gt_wh if n > 0 else gt_ctrs,
                    'center_br': gt_ctrs + self.center_radius * gt_wh if n > 0 else gt_ctrs,
                })
        else:
            new_cache = gt_cache
            max_gt = new_cache[0]['max_gt']
            gt_bboxes_padded = torch.stack([c['gt_bboxes_padded'] for c in new_cache])
            gt_labels_padded = torch.stack([c['gt_labels_padded'] for c in new_cache])
            gt_num = [c['gt_num'] for c in new_cache]

        if max_gt == 0:
            empty = (torch.zeros(N, dtype=torch.bool, device=pred_bboxes.device),
                     torch.zeros(N, dtype=torch.long, device=pred_bboxes.device))
            return [empty] * bs, new_cache

        # Batched cost computation
        cost_matrix, pairwise_ious = self._batched_cost(
            pred_logits, pred_bboxes, gt_labels_padded, gt_bboxes_padded, gt_num, max_gt
        )

        # Per-image dynamic_k_matching
        indices = []
        for i in range(bs):
            n = gt_num[i]
            if n == 0:
                indices.append((torch.zeros(N, dtype=torch.bool, device=pred_bboxes.device),
                                torch.zeros(N, dtype=torch.long, device=pred_bboxes.device)))
            else:
                indices.append(self._dynamic_k_matching(cost_matrix[i, :, :n], pairwise_ious[i, :, :n], n))
        return indices, new_cache

    @torch.no_grad()
    def _batched_cost(self, pred_logits: Tensor, pred_bboxes: Tensor,
                      gt_labels_padded: Tensor, gt_bboxes_padded: Tensor,
                      gt_num: List[int], max_gt: int) -> Tuple[Tensor, Tensor]:
        """批量计算代价矩阵和 pairwise IoU。

        Args:
            pred_logits: [bs, N, C]
            pred_bboxes: [bs, N, 4]
            gt_labels_padded: [bs, max_gt]
            gt_bboxes_padded: [bs, max_gt, 4]
            gt_num: 每张图的 GT 数量
            max_gt: 最大 GT 数量

        Returns:
            cost_matrix: [bs, N, max_gt]
            pairwise_ious: [bs, N, max_gt]
        """
        bs, N = pred_bboxes.shape[:2]

        # Batched pairwise IoU: [bs, N, max_gt]
        pairwise_ious = self._batched_box_iou(pred_bboxes, gt_bboxes_padded)

        # Batched is_in_boxes and is_in_centers: [bs, N, max_gt]
        pred_ctrs = (pred_bboxes[:, :, :2] + pred_bboxes[:, :, 2:]) / 2  # [bs, N, 2]
        gt_ctrs = (gt_bboxes_padded[:, :, :2] + gt_bboxes_padded[:, :, 2:]) / 2  # [bs, max_gt, 2]
        gt_wh = gt_bboxes_padded[:, :, 2:] - gt_bboxes_padded[:, :, :2]  # [bs, max_gt, 2]

        # is_in_boxes: [bs, N, max_gt]
        lt = pred_ctrs.unsqueeze(2) - gt_bboxes_padded[:, :, :2].unsqueeze(1)  # [bs, N, max_gt, 2]
        rb = gt_bboxes_padded[:, :, 2:].unsqueeze(1) - pred_ctrs.unsqueeze(2)  # [bs, N, max_gt, 2]
        is_in_boxes = torch.cat([lt, rb], dim=-1).min(-1)[0] > 0  # [bs, N, max_gt]

        # is_in_centers: [bs, N, max_gt]
        center_tl = gt_ctrs - self.center_radius * gt_wh  # [bs, max_gt, 2]
        center_br = gt_ctrs + self.center_radius * gt_wh  # [bs, max_gt, 2]
        lt_c = pred_ctrs.unsqueeze(2) - center_tl.unsqueeze(1)  # [bs, N, max_gt, 2]
        rb_c = center_br.unsqueeze(1) - pred_ctrs.unsqueeze(2)  # [bs, N, max_gt, 2]
        is_in_centers = torch.cat([lt_c, rb_c], dim=-1).min(-1)[0] > 0  # [bs, N, max_gt]

        is_in_boxes_anchor = is_in_boxes.any(2) | is_in_centers.any(2)  # [bs, N]
        is_in_boxes_and_center = is_in_boxes & is_in_centers  # [bs, N, max_gt]

        # Build cost matrix
        cost_list = []

        # FocalLossCost (batched)
        for cost_fn in self.costs:
            if isinstance(cost_fn, IoUCost):
                continue
            if isinstance(cost_fn, FocalLossCost):
                cost_list.append(self._batched_focal_cost(cost_fn, pred_logits, gt_labels_padded, max_gt))
            elif isinstance(cost_fn, BBoxL1Cost):
                cost_list.append(self._batched_l1_cost(cost_fn, pred_bboxes, gt_bboxes_padded))
            else:
                # Fallback: per-image
                for i in range(bs):
                    n = gt_num[i]
                    if n > 0:
                        cost_i = cost_fn(pred_logits[i], pred_bboxes[i],
                                         gt_labels_padded[i, :n], gt_bboxes_padded[i, :n])
                    else:
                        cost_i = pred_bboxes.new_zeros(N, max_gt)
                    if i == 0:
                        batched_cost = cost_i.unsqueeze(0)
                    else:
                        batched_cost = torch.cat([batched_cost, cost_i.unsqueeze(0)], dim=0)
                cost_list.append(batched_cost)

        # IoUCost (batched GIoU)
        for cost_fn in self.costs:
            if isinstance(cost_fn, IoUCost):
                if cost_fn.iou_mode == 'giou':
                    giou = self._batched_giou(pred_bboxes, gt_bboxes_padded)
                    iou_cost = (1 - giou) * cost_fn.weight
                else:
                    iou_cost = (1 - pairwise_ious) * cost_fn.weight
                cost_list.append(iou_cost)
                break

        cost_list.append((~is_in_boxes_and_center) * 100.0)

        # 注: 测试表明 torch.stack().sum() 内部已高度优化,
        # 就地累加反而因多次 kernel launch 变慢 (0.92x), 保留原始实现。
        cost_matrix = torch.stack(cost_list).sum(0)  # [bs, N, max_gt]
        cost_matrix = torch.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=1e6)

        # Mask out padding GT columns
        gt_valid_mask = torch.zeros(bs, 1, max_gt, dtype=torch.bool, device=pred_bboxes.device)
        for i in range(bs):
            if gt_num[i] > 0:
                gt_valid_mask[i, 0, :gt_num[i]] = True
        cost_matrix[~gt_valid_mask.expand_as(cost_matrix)] += 10000.0
        cost_matrix[~is_in_boxes_anchor.unsqueeze(-1).expand_as(cost_matrix)] += 10000.0

        return cost_matrix, pairwise_ious

    @staticmethod
    def _batched_box_iou(pred: Tensor, gt: Tensor) -> Tensor:
        """批量 IoU: [bs, N, 4] x [bs, M, 4] -> [bs, N, M]"""
        pred_area = (pred[:, :, 2:] - pred[:, :, :2]).prod(-1)  # [bs, N]
        gt_area = (gt[:, :, 2:] - gt[:, :, :2]).prod(-1)  # [bs, M]
        lt = torch.max(pred[:, :, None, :2], gt[:, None, :, :2])  # [bs, N, M, 2]
        rb = torch.min(pred[:, :, None, 2:], gt[:, None, :, 2:])  # [bs, N, M, 2]
        wh = (rb - lt).clamp(min=0)
        inter = wh.prod(-1)  # [bs, N, M]
        union = pred_area[:, :, None] + gt_area[:, None, :] - inter
        return inter / union.clamp(min=1e-8)

    @staticmethod
    def _batched_giou(pred: Tensor, gt: Tensor) -> Tensor:
        """批量 GIoU: [bs, N, 4] x [bs, M, 4] -> [bs, N, M]"""
        pred_area = (pred[:, :, 2:] - pred[:, :, :2]).prod(-1)
        gt_area = (gt[:, :, 2:] - gt[:, :, :2]).prod(-1)
        lt = torch.max(pred[:, :, None, :2], gt[:, None, :, :2])
        rb = torch.min(pred[:, :, None, 2:], gt[:, None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh.prod(-1)
        union = pred_area[:, :, None] + gt_area[:, None, :] - inter
        iou = inter / union.clamp(min=1e-8)
        enc_lt = torch.min(pred[:, :, None, :2], gt[:, None, :, :2])
        enc_rb = torch.max(pred[:, :, None, 2:], gt[:, None, :, 2:])
        enc_wh = (enc_rb - enc_lt).clamp(min=0)
        enc_area = enc_wh.prod(-1)
        return iou - (enc_area - union) / enc_area.clamp(min=1e-8)

    @staticmethod
    def _batched_focal_cost(cost_fn, pred_logits: Tensor, gt_labels: Tensor, max_gt: int) -> Tensor:
        """批量 FocalLoss 代价: [bs, N, C] x [bs, M] -> [bs, N, M]"""
        num_classes = pred_logits.shape[-1]
        gt_labels_clamped = gt_labels.clamp(0, num_classes - 1)
        # Gather target logits: [bs, N, M]
        gt_labels_expanded = gt_labels_clamped.unsqueeze(1).expand(-1, pred_logits.shape[1], -1)
        target_logits = pred_logits.gather(2, gt_labels_expanded)  # [bs, N, M]
        p = torch.sigmoid(target_logits)
        neg_cost = -(1 - cost_fn.alpha) * (p ** cost_fn.gamma) * torch.log(1 - p + cost_fn.eps)
        pos_cost = -cost_fn.alpha * ((1 - p) ** cost_fn.gamma) * torch.log(p + cost_fn.eps)
        return (pos_cost - neg_cost) * cost_fn.weight

    @staticmethod
    def _batched_l1_cost(cost_fn, pred_bboxes: Tensor, gt_bboxes: Tensor) -> Tensor:
        """批量 L1 代价: [bs, N, 4] x [bs, M, 4] -> [bs, N, M]"""
        pred = torch.nan_to_num(pred_bboxes.detach(), nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)
        gt = torch.nan_to_num(gt_bboxes, nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)
        # [bs, N, 1, 4] - [bs, 1, M, 4] -> [bs, N, M, 4] -> sum -> [bs, N, M]
        return (pred.unsqueeze(2) - gt.unsqueeze(1)).abs().sum(-1) * cost_fn.weight

    def _dynamic_k_matching(self, cost, pairwise_ious, num_gt) -> Tuple[Tensor, Tensor]:
        matching_matrix = torch.zeros_like(cost)
        pairwise_ious = torch.nan_to_num(pairwise_ious, nan=0.0, posinf=1.0, neginf=0.0)
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1, max=candidate_topk)

        # 向量化: 使用 topk 替代完整排序 (只取 cost 最小的 candidate_topk 行)
        _, top_rows = torch.topk(cost, candidate_topk, dim=0, largest=False)  # [candidate_topk, num_gt]
        gt_cols = torch.arange(num_gt, device=cost.device).unsqueeze(0).expand(candidate_topk, -1)
        # 创建 mask: row_idx < dynamic_ks[gt_idx]
        row_positions = torch.arange(candidate_topk, device=cost.device).unsqueeze(1)  # [candidate_topk, 1]
        valid_mask = row_positions < dynamic_ks.unsqueeze(0)  # [candidate_topk, num_gt]
        # 使用 scatter 填充 matching_matrix
        valid_rows = top_rows[valid_mask]  # [total_valid]
        valid_cols = gt_cols[valid_mask]  # [total_valid]
        matching_matrix[valid_rows, valid_cols] = 1.0

        # 消除重复匹配: 始终计算 (避免 if any() 触发同步)
        duplicate_idx = matching_matrix.sum(1) > 1
        _, cost_argmin = cost.min(1)
        matching_matrix[duplicate_idx] = 0.0
        matching_matrix[duplicate_idx, cost_argmin[duplicate_idx]] = 1.0

        # 返回 bool mask 和全尺寸 GT 索引 (避免 nonzero 和变长张量)
        fg_mask = matching_matrix.sum(1) > 0  # [N] bool
        matched_gt_inds = matching_matrix.argmax(1)  # [N] long (非 fg 位置无效，后续 mask)
        return fg_mask, matched_gt_inds
