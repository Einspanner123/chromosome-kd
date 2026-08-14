"""检测损失计算核心类"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import ops

from ldmdet.data.structures import InstanceData, ModelOutput
from ldmdet.utils.box_ops import bbox_xyxy_to_cxcywh


class DiffusionDetCriterion(nn.Module):
    """DiffusionDet 损失计算核心"""

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        loss_cls: nn.Module,
        loss_bbox: nn.Module,
        loss_giou: nn.Module,
        deep_supervision: bool = True,
        quality_loss_weight: float = 0.25,
        quality_focal_alpha: float = 0.75,
        quality_focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.loss_giou = loss_giou
        self.quality_loss_weight = quality_loss_weight
        self.quality_focal_alpha = quality_focal_alpha
        self.quality_focal_gamma = quality_focal_gamma
        self.deep_supervision = deep_supervision

    def forward(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        t: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # 主输出: 使用 matcher.forward 并建立 GT 缓存
        indices, gt_cache = self.matcher.forward_with_gt_cache(
            outputs, targets
        )
        losses = self._get_loss(
            outputs, targets, indices, t, include_terminal_losses=True
        )

        if self.deep_supervision and outputs.aux_outputs is not None:
            for i, aux_out in enumerate(outputs.aux_outputs):
                # aux_outputs 复用 GT 缓存，避免重复计算 gt_ctrs/gt_wh/center 区域
                aux_indices, gt_cache = self.matcher.forward_with_gt_cache(
                    aux_out, targets, gt_cache
                )
                aux_losses = self._get_loss(
                    aux_out,
                    targets,
                    aux_indices,
                    t,
                    include_terminal_losses=False,
                )
                for name, val in aux_losses.items():
                    losses[f'aux_{i}_{name}'] = val
        self._last_indices = indices
        return losses

    def _get_loss(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        indices: List[Tuple[Tensor, Tensor]] = None,
        t: Optional[Tensor] = None,
        include_terminal_losses: bool = False,
    ) -> Dict[str, Tensor]:
        if indices is None:
            indices = self.matcher(outputs, targets)
        loss_cls = self._loss_classification(outputs, targets, indices)
        loss_bbox, loss_giou = self._loss_boxes(outputs, targets, indices)

        losses = {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
        }
        if outputs.pred_quality is not None:
            losses['loss_quality'] = self._loss_quality(
                outputs, targets, indices
            )
        return losses

    def _loss_quality(self, outputs, targets, indices) -> Tensor:
        """Varifocal-style continuous IoU quality supervision.

        Hungarian positives receive their detached aligned IoU as target;
        unmatched proposals receive zero. Geometry gradients therefore remain
        exclusively in the existing box loss, while the new branch learns the
        ranking statistic required by COCO AP at strict IoU thresholds.
        """
        logits = outputs.pred_quality
        if logits.ndim == 2:
            logits = logits.unsqueeze(-1)
        boxes = outputs.pred_boxes.detach()
        bs, num_queries, quality_dim = logits.shape
        if quality_dim != 1:
            raise ValueError('LQCR uses one continuous IoU quality target')
        aligned_iou = logits.new_zeros(bs, num_queries)
        fg_masks = torch.stack([idx[0] for idx in indices])
        matched_gt_inds = torch.stack([idx[1] for idx in indices]).clamp(min=0)

        max_gt = max(target.bboxes.shape[0] for target in targets)
        if max_gt > 0:
            gt_padded = boxes.new_zeros(bs, max_gt, 4)
            for batch_index, target in enumerate(targets):
                count = target.bboxes.shape[0]
                if count:
                    gt_padded[batch_index, :count] = target.bboxes
            matched = torch.gather(
                gt_padded, 1, matched_gt_inds.unsqueeze(-1).expand(-1, -1, 4)
            )
            lt = torch.maximum(boxes[..., :2], matched[..., :2])
            rb = torch.minimum(boxes[..., 2:], matched[..., 2:])
            wh = (rb - lt).clamp(min=0)
            intersection = wh[..., 0] * wh[..., 1]
            box_wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
            gt_wh = (matched[..., 2:] - matched[..., :2]).clamp(min=0)
            union = (
                box_wh[..., 0] * box_wh[..., 1]
                + gt_wh[..., 0] * gt_wh[..., 1]
                - intersection
            )
            matched_iou = intersection / union.clamp(min=1e-7)
            aligned_iou[fg_masks] = matched_iou[fg_masks].clamp(0, 1)

        quality_targets = aligned_iou.unsqueeze(-1)
        positive_weight = quality_targets

        probability = logits.sigmoid()
        negative_weight = self.quality_focal_alpha * probability.pow(
            self.quality_focal_gamma
        )
        focal_weight = torch.where(
            fg_masks.unsqueeze(-1), positive_weight, negative_weight
        ).detach()
        loss = (
            F.binary_cross_entropy_with_logits(
                logits, quality_targets, reduction='none'
            )
            * focal_weight
        )
        num_pos = fg_masks.sum().clamp(min=1) * quality_dim
        return self.quality_loss_weight * loss.sum() / num_pos

    def _loss_classification(self, outputs, targets, indices) -> Tensor:
        src_logits = outputs.pred_logits  # [bs, num_queries, num_classes+1]
        bs, num_queries = src_logits.shape[:2]

        # 构建 padded GT labels: [bs, max_gt]
        max_gt = max(t.labels.shape[0] for t in targets)
        if max_gt == 0:
            # 全部为背景
            target_classes = src_logits.new_full(
                (bs, num_queries), self.num_classes, dtype=torch.long
            )
            num_pos = src_logits.new_tensor(1, dtype=torch.long)
            loss_cls = self.loss_cls(
                src_logits.flatten(0, 1), target_classes.flatten(0, 1)
            )
            return loss_cls / num_pos

        gt_labels_padded = src_logits.new_full(
            (bs, max_gt), self.num_classes, dtype=torch.long
        )
        for i, t in enumerate(targets):
            n = t.labels.shape[0]
            if n > 0:
                gt_labels_padded[i, :n] = t.labels

        # 批量化: 将 indices 堆叠为 [bs, N] 张量
        fg_masks = torch.stack([idx[0] for idx in indices])  # [bs, N] bool
        matched_gt_inds = torch.stack(
            [idx[1] for idx in indices]
        )  # [bs, N] long
        matched_gt_inds_clamped = matched_gt_inds.clamp(min=0)

        # Gather GT labels: [bs, N]
        matched_gt_labels = torch.gather(
            gt_labels_padded, 1, matched_gt_inds_clamped
        )
        # 背景位置设为 num_classes
        matched_gt_labels[~fg_masks] = self.num_classes

        target_classes = matched_gt_labels
        # num_pos 保留为张量，避免 .item() 同步
        num_pos = fg_masks.sum().clamp(min=1)

        loss_cls = self.loss_cls(
            src_logits.flatten(0, 1), target_classes.flatten(0, 1)
        )
        return loss_cls / num_pos

    def _loss_boxes(
        self,
        outputs,
        targets,
        indices,
    ) -> Tuple[Tensor, Tensor]:
        """计算正样本框的 L1 与 GIoU 损失。"""
        src_boxes = (
            outputs.pred_boxes
        )  # [bs, num_queries, 4] 归一化 xyxy [0,1]
        bs = src_boxes.shape[0]

        # 构建 padded GT bboxes: [bs, max_gt, 4]
        max_gt = max(t_data.bboxes.shape[0] for t_data in targets)
        if max_gt == 0:
            # 无正样本，返回零损失
            return src_boxes.sum() * 0, src_boxes.sum() * 0

        # 批量化: 将 indices 堆叠为 [bs, N] 张量
        fg_masks = torch.stack([idx[0] for idx in indices])  # [bs, N] bool
        matched_gt_inds = torch.stack(
            [idx[1] for idx in indices]
        )  # [bs, N] long

        # 构建 padded GT bboxes: [bs, max_gt, 4]
        gt_bboxes_padded = src_boxes.new_zeros(bs, max_gt, 4)
        for i, t_data in enumerate(targets):
            n = t_data.bboxes.shape[0]
            if n > 0:
                gt_bboxes_padded[i, :n] = t_data.bboxes

        matched_gt_inds_clamped = matched_gt_inds.clamp(min=0)
        tgt_boxes = torch.gather(
            gt_bboxes_padded,
            1,
            matched_gt_inds_clamped.unsqueeze(-1).expand(-1, -1, 4),
        )

        # num_pos 保留为张量，避免 .item() 同步
        num_pos = fg_masks.sum().clamp(min=1)

        tgt_cxcywh = bbox_xyxy_to_cxcywh(tgt_boxes)  # [bs, N, 4]
        src_cxcywh = bbox_xyxy_to_cxcywh(src_boxes)  # [bs, N, 4]

        # L1 loss: 逐元素计算后 mask
        per_elem_l1 = F.l1_loss(
            src_cxcywh, tgt_cxcywh, reduction='none'
        )  # [bs, N, 4]
        masked_l1 = per_elem_l1 * fg_masks.unsqueeze(-1).float()

        loss_bbox = self.loss_bbox.loss_weight * masked_l1.sum() / num_pos
        # GIoU loss: 逐元素计算后 mask
        per_giou = ops.generalized_box_iou_loss(
            src_boxes.reshape(-1, 4),
            tgt_boxes.reshape(-1, 4),
            reduction='none',
        ).reshape(bs, -1)
        per_giou = torch.nan_to_num(per_giou, nan=0.0)

        loss_giou = (
            self.loss_giou.loss_weight
            * (per_giou * fg_masks.float()).sum()
            / num_pos
        )

        return loss_bbox, loss_giou
