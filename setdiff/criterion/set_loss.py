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

    GIoU 计算空间 (关键修复):
        GIoU 必须在 [0,1] 归一化空间计算, 不是扩散空间 [-s, s].
        根因: cxcywh 的 w,h 在扩散空间可能为负 (小目标), 导致框翻转.
        详见 _loss_giou 文档.
    """

    def __init__(
        self,
        num_classes: int,
        weight_dict: Dict[str, float] | None = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
        snr_scale: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        # snr_scale: GT 从 [0,1] 缩放到 [-s, +s] 匹配 N(0,1) 噪声.
        # _loss_giou 需要逆变换回 [0,1] 空间计算 GIoU (修复框翻转 bug).
        self.snr_scale = snr_scale
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
        """L1 loss on matched slots (x_0 prediction).

        对齐 LDMDet/DiffusionDet: L1 损失承担 x_0 预测回归.

        关键修复 — L1 必须在 [0,1] 归一化空间计算, 不是扩散空间 [-s, s]:
            根因: loss_bbox 在扩散空间计算时数值放大 2s 倍 (s=2 → 4x),
                  而 loss_giou 的逆变换引入 1/(2s) 梯度衰减 (d_pred_norm/d_pred_diff=1/(2s)),
                  两者不在同一空间导致有效权重失衡.
                  严格梯度比: L1 梯度 ~ O(1), GIoU 梯度 ~ O(1/(2s)), 比值下界 2s:1 (s=2 → 4:1);
                  但 GIoU 的几何依赖 (框重叠/包含关系) 使实际比值更大
                  (subagent1 实测 ~10:1, s=2), 有效权重从 2:5:2 失衡,
                  GIoU 几乎不学习 → mAP=0.
            修复: 与 _loss_giou 一致, 先把 pred/tgt 从 [-s, s] 逆变换到 [0, 1]
                  再计算 L1, 保证两项损失在同一空间, 梯度尺度一致.
        """
        valid_mask = matched_labels >= 0

        if not valid_mask.any():
            return pred_boxes.sum() * 0.0

        pred = pred_boxes[valid_mask]  # [num_valid, 4] in diffusion space [-s, s]
        tgt = matched_boxes[valid_mask]  # [num_valid, 4] in diffusion space [-s, s]
        num_pos = pred.shape[0]

        # 逆变换: [-s, s] → [0, 1] (与 _loss_giou 一致, 保证梯度尺度对齐)
        s = self.snr_scale
        pred_norm = (pred.clamp(-s, s) / s + 1.0) / 2.0
        tgt_norm = (tgt.clamp(-s, s) / s + 1.0) / 2.0

        return F.l1_loss(pred_norm, tgt_norm, reduction='sum') / max(num_pos, 1)

    def _loss_giou(
        self,
        pred_boxes: Tensor,
        matched_boxes: Tensor,
        matched_labels: Tensor,
    ) -> Tensor:
        """GIoU loss on matched slots (x_0 prediction).

        对齐 LDMDet/DiffusionDet: GIoU 提供几何重叠监督, 是检测 AP 的直接代理.

        关键修复 — GIoU 必须在 [0,1] 归一化空间计算, 不是扩散空间 [-s, s]:
            根因: cxcywh 的 w,h 在扩散空间可能为负 (小目标 GT 经
                  (x*2-1)*snr_scale 变换后). 例如 GT w=0.1 → 扩散空间 w=-1.6.
                  bbox_cxcywh_to_xyxy 后: x1=cx-w/2 > x2=cx+w/2 (框翻转),
                  area=0, giou=0, loss_giou=1.0 恒定.
                  染色体小目标 w,h 通常 0.05-0.3, 全部为负 → loss_giou 完全失效.
            修复: 调用 generalized_box_iou 前, 先把 pred/tgt 从 [-s, s]
                  逆变换到 [0, 1] (与 SetDiffDetector.diffusion_to_norm_space 一致).
        """
        valid_mask = matched_labels >= 0

        if not valid_mask.any():
            return pred_boxes.sum() * 0.0

        pred = pred_boxes[valid_mask]  # [num_valid, 4] in diffusion space [-s, s]
        tgt = matched_boxes[valid_mask]  # [num_valid, 4] in diffusion space [-s, s]
        num_pos = pred.shape[0]

        # 逆变换: [-s, s] → [0, 1] (GIoU 需要在有效框空间计算, 修复框翻转 bug)
        s = self.snr_scale
        pred_norm = (pred.clamp(-s, s) / s + 1.0) / 2.0
        tgt_norm = (tgt.clamp(-s, s) / s + 1.0) / 2.0

        giou = generalized_box_iou(pred_norm, tgt_norm)  # [num_valid]
        return (1.0 - giou).sum() / max(num_pos, 1)
