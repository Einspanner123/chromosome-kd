"""SetCriterion — loss for joint diffusion detection.

方案 A (对齐 LDMDet/DiffusionDet): 标准 DETR 集合预测损失
1. Classification loss (sigmoid focal loss on class logits).
2. Box regression loss (L1 on predicted x_0, cxcywh 扩散空间).
3. GIoU loss (GIoU on predicted x_0, xyxy 空间).

删除 loss_diff (MSE): 与 L1+GIoU 计算同一对象 (pred_boxes vs matched_boxes),
属于冗余项; 检测领域 (DiffusionDet/LDMDet/DiffuDETR) 均无单独扩散 MSE,
L1+GIoU 已隐式承担 x_0 预测损失.

Matched slots (label >= 0) → predict their GT class.
Unmatched slots (label == -1) → predict "no object" (all class targets = 0).
All slots participate in classification loss; only matched slots contribute
to bbox/giou losses.

权重 2:5:2 对齐 DETR 家族 (DETR/Deformable DETR/DINO/LDMDet).
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ldmdet.utils.box_ops import bbox_cxcywh_to_xyxy


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """GIoU for cxcywh format boxes. Both [N, 4].

    Returns:
        giou: [N] GIoU values in [-1, 1].
    """
    # Convert cxcywh → xyxy
    b1 = bbox_cxcywh_to_xyxy(boxes1)
    b2 = bbox_cxcywh_to_xyxy(boxes2)

    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(
        min=0
    )
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(
        min=0
    )

    # Intersection
    lt = torch.max(b1[:, :2], b2[:, :2])
    rb = torch.min(b1[:, 2:], b2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-8)

    # Enclosing box
    enc_lt = torch.min(b1[:, :2], b2[:, :2])
    enc_rb = torch.max(b1[:, 2:], b2[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[:, 0] * enc_wh[:, 1]

    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-8)
    return giou


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Sigmoid focal loss (sum reduction).

    Args:
        inputs: [N, C] logits.
        targets: [N, C] one-hot targets.
        alpha: balancing factor.
        gamma: focusing parameter.

    Returns:
        loss: scalar (sum over elements).
    """
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    return loss.sum()


class SetCriterion(nn.Module):
    """Loss for joint diffusion detection (方案 A, 对齐 LDMDet/DiffusionDet).

    Combines:
    1. Classification loss (focal loss on class logits, w=2).
    2. Box regression loss (L1 on predicted x_0, w=5).
    3. GIoU loss (GIoU on predicted x_0, w=2).

    loss_diff (MSE) 已删除: 与 L1+GIoU 冗余, 检测领域无此实践.
    """

    def __init__(
        self,
        num_classes: int,
        weight_dict: Dict[str, float] | None = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        if weight_dict is None:
            # 方案 A: 2:5:2 对齐 DETR 家族 (cls : bbox : giou)
            weight_dict = {
                'loss_cls': 2.0,
                'loss_bbox': 5.0,
                'loss_giou': 2.0,
            }
        self.weight_dict = weight_dict

    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """Args:
            outputs: dict with 'pred_logits' [B, N, C], 'pred_boxes'
                [B, N, 4].
            targets: dict with 'matched_boxes' [B, N, 4], 'matched_labels'
                [B, N] (labels=-1 for padding/unmatched slots).

        Returns:
            loss_dict: dict of loss terms (loss_cls, loss_bbox, loss_giou).
            loss: scalar total loss (weighted sum).
        """
        pred_logits = outputs['pred_logits']  # [B, N, C]
        pred_boxes = outputs['pred_boxes']  # [B, N, 4]
        matched_boxes = targets['matched_boxes']  # [B, N, 4]
        matched_labels = targets['matched_labels']  # [B, N]

        loss_cls = self._loss_classification(pred_logits, matched_labels)
        loss_bbox = self._loss_bbox(pred_boxes, matched_boxes, matched_labels)
        loss_giou = self._loss_giou(pred_boxes, matched_boxes, matched_labels)

        loss_dict = {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
        }

        total = sum(
            loss_dict[k] * self.weight_dict.get(k, 1.0) for k in loss_dict
        )
        return loss_dict, total

    def _loss_classification(
        self,
        pred_logits: Tensor,
        matched_labels: Tensor,
    ) -> Tensor:
        """Sigmoid focal loss over ALL slots.

        - Matched slots (label >= 0): target is one-hot for their class.
        - Unmatched slots (label == -1): target is all zeros (no object).

        This is the standard DETR approach for sigmoid focal loss: every slot
        participates in classification, and unmatched slots learn to suppress
        all class scores.

        Args:
            pred_logits: [B, N, C]
            matched_labels: [B, N] (label=-1 for unmatched)
        """
        B, N, C = pred_logits.shape

        # One-hot targets: [B, N, C]
        targets = torch.zeros_like(pred_logits)  # all zeros initially
        valid_mask = matched_labels >= 0  # [B, N]
        if valid_mask.any():
            valid_labels = matched_labels[valid_mask]  # [num_valid]
            targets[valid_mask] = torch.zeros_like(
                targets[valid_mask]
            ).scatter_(1, valid_labels.clamp(0, C - 1).unsqueeze(1), 1.0)

        num_pos = valid_mask.sum().item()
        flatten_logits = pred_logits.reshape(-1, C)
        flatten_targets = targets.reshape(-1, C)

        loss = sigmoid_focal_loss(
            flatten_logits,
            flatten_targets,
            alpha=self.alpha,
            gamma=self.gamma,
        )
        # Normalize by number of positive slots (same as mmdet DETR).
        # With sigmoid focal loss on all slots, the effective normalization
        # balances the huge negative (unmatched) contribution.
        return loss / max(num_pos, 1)

    def _loss_bbox(
        self,
        pred_boxes: Tensor,
        matched_boxes: Tensor,
        matched_labels: Tensor,
    ) -> Tensor:
        """L1 loss on matched slots (x_0 prediction in cxcywh diffusion space).

        对齐 LDMDet/DiffusionDet: L1 损失承担 x_0 预测回归.
        """
        valid_mask = matched_labels >= 0

        if not valid_mask.any():
            return pred_boxes.sum() * 0.0

        pred = pred_boxes[valid_mask]  # [num_valid, 4]
        tgt = matched_boxes[valid_mask]  # [num_valid, 4]
        num_pos = pred.shape[0]

        return F.l1_loss(pred, tgt, reduction='sum') / max(num_pos, 1)

    def _loss_giou(
        self,
        pred_boxes: Tensor,
        matched_boxes: Tensor,
        matched_labels: Tensor,
    ) -> Tensor:
        """GIoU loss on matched slots (x_0 prediction).

        对齐 LDMDet/DiffusionDet: GIoU 提供几何重叠监督, 是检测 AP 的直接代理.
        Boxes 在 cxcywh 扩散空间, 内部转 xyxy 计算 GIoU.
        """
        valid_mask = matched_labels >= 0

        if not valid_mask.any():
            return pred_boxes.sum() * 0.0

        pred = pred_boxes[valid_mask]  # [num_valid, 4]
        tgt = matched_boxes[valid_mask]  # [num_valid, 4]
        num_pos = pred.shape[0]

        giou = generalized_box_iou(pred, tgt)  # [num_valid]
        return (1.0 - giou).sum() / max(num_pos, 1)
