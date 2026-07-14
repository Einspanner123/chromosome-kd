"""SetCriterion — loss for joint diffusion detection.

Combines:
1. Classification loss (sigmoid focal loss on class logits).
2. Box regression loss (L1 + GIoU on predicted x_0).
3. Diffusion loss (MSE on x_0 prediction — the velocity loss equivalent).

Only matched slots (label >= 0) contribute to losses; padding/unmatched
slots (label == -1) are ignored.
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

    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (
        b1[:, 3] - b1[:, 1]
    ).clamp(min=0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (
        b2[:, 3] - b2[:, 1]
    ).clamp(min=0)

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
    """Loss for joint diffusion detection.

    Combines:
    1. Classification loss (focal loss on class logits).
    2. Box regression loss (L1 + GIoU on predicted x_0).
    3. Diffusion loss (MSE on x_0 prediction, this is the velocity loss
       equivalent).
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
            weight_dict = {
                'loss_cls': 1.0,
                'loss_box': 1.0,
                'loss_diff': 1.0,
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
            loss_dict: dict of loss terms.
            loss: scalar total loss (weighted sum).
        """
        pred_logits = outputs['pred_logits']  # [B, N, C]
        pred_boxes = outputs['pred_boxes']  # [B, N, 4]
        matched_boxes = targets['matched_boxes']  # [B, N, 4]
        matched_labels = targets['matched_labels']  # [B, N]

        loss_cls = self._loss_classification(pred_logits, matched_labels)
        loss_box = self._loss_boxes(
            pred_boxes, matched_boxes, matched_labels
        )
        loss_diff = self._loss_diffusion(
            pred_boxes, matched_boxes, matched_labels
        )

        loss_dict = {
            'loss_cls': loss_cls,
            'loss_box': loss_box,
            'loss_diff': loss_diff,
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
        """Sigmoid focal loss on matched slots only.

        Args:
            pred_logits: [B, N, C]
            matched_labels: [B, N] (label=-1 for unmatched)
        """
        B, N, C = pred_logits.shape
        valid_mask = matched_labels >= 0  # [B, N]

        if not valid_mask.any():
            return pred_logits.sum() * 0.0

        flat_logits = pred_logits[valid_mask]  # [num_valid, C]
        flat_labels = matched_labels[valid_mask]  # [num_valid]

        # One-hot encode (labels are clamped to [0, C-1] for safety)
        one_hot = torch.zeros_like(flat_logits)
        one_hot.scatter_(1, flat_labels.clamp(0, C - 1).unsqueeze(1), 1.0)

        num_pos = flat_logits.shape[0]
        loss = sigmoid_focal_loss(
            flat_logits, one_hot, alpha=self.alpha, gamma=self.gamma
        )
        return loss / max(num_pos, 1)

    def _loss_boxes(
        self,
        pred_boxes: Tensor,
        matched_boxes: Tensor,
        matched_labels: Tensor,
    ) -> Tensor:
        """L1 + GIoU loss on matched slots.

        Boxes are in cxcywh (diffusion space).
        """
        valid_mask = matched_labels >= 0

        if not valid_mask.any():
            return pred_boxes.sum() * 0.0

        pred = pred_boxes[valid_mask]  # [num_valid, 4]
        tgt = matched_boxes[valid_mask]  # [num_valid, 4]
        num_pos = pred.shape[0]

        # L1 loss
        l1 = F.l1_loss(pred, tgt, reduction='sum') / max(num_pos, 1)

        # GIoU loss
        giou = generalized_box_iou(pred, tgt)  # [num_valid]
        giou_loss = (1.0 - giou).sum() / max(num_pos, 1)

        return l1 + giou_loss

    def _loss_diffusion(
        self,
        pred_boxes: Tensor,
        matched_boxes: Tensor,
        matched_labels: Tensor,
    ) -> Tensor:
        """MSE on x_0 prediction (diffusion loss / velocity equivalent).

        Only matched slots contribute — unmatched slots have velocity=0
        (matched_box = noise), so their diffusion target is trivial.
        """
        valid_mask = matched_labels >= 0

        if not valid_mask.any():
            return pred_boxes.sum() * 0.0

        pred = pred_boxes[valid_mask]  # [num_valid, 4]
        tgt = matched_boxes[valid_mask]  # [num_valid, 4]
        num_pos = pred.shape[0]

        return F.mse_loss(pred, tgt, reduction='sum') / max(num_pos, 1)
