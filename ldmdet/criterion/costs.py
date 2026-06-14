"""匹配代价函数: FocalLossCost, BBoxL1Cost, IoUCost, RelativeL1Cost"""

import torch
from torch import Tensor
from torchvision import ops

from ldmdet.utils.box_ops import bbox_xyxy_to_cxcywh


class FocalLossCost:
    """分类匹配代价 (Focal Loss 形式)"""

    def __init__(self, alpha=0.25, gamma=2.0, weight=2.0, eps=1e-8):
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.eps = eps

    def __call__(self, pred_logits, pred_bboxes, gt_labels, gt_bboxes) -> Tensor:
        num_classes = pred_logits.shape[-1]
        out_prob = pred_logits.sigmoid()
        gt_labels = gt_labels.clamp(0, num_classes - 1)
        neg_cost_class = -(1 - self.alpha) * (out_prob ** self.gamma) * torch.log(1 - out_prob + self.eps)
        pos_cost_class = -self.alpha * ((1 - out_prob) ** self.gamma) * torch.log(out_prob + self.eps)
        return (pos_cost_class[:, gt_labels] - neg_cost_class[:, gt_labels]) * self.weight


class BBoxL1Cost:
    """边界框 L1 匹配代价"""

    def __init__(self, weight=5.0):
        self.weight = weight

    def __call__(self, pred_logits, pred_bboxes, gt_labels, gt_bboxes) -> Tensor:
        pred_bboxes = torch.nan_to_num(pred_bboxes.detach(), nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)
        gt_bboxes = torch.nan_to_num(gt_bboxes, nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)
        return torch.cdist(pred_bboxes, gt_bboxes, p=1) * self.weight


class RelativeL1Cost:
    """相对 L1 匹配代价 (按 GT 尺度归一化)"""

    def __init__(self, weight=5.0, eps=1e-2):
        self.weight = weight
        self.eps = eps

    def __call__(self, pred_logits, pred_bboxes, gt_labels, gt_bboxes) -> Tensor:
        pred_bboxes = torch.nan_to_num(pred_bboxes, nan=0.5, posinf=1.0, neginf=0.0).detach().clamp(0, 1)
        gt_bboxes = torch.nan_to_num(gt_bboxes, nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)
        pred_cxcywh = bbox_xyxy_to_cxcywh(pred_bboxes)
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        pred_cxcywh = torch.nan_to_num(pred_cxcywh, nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)
        gt_cxcywh = torch.nan_to_num(gt_cxcywh, nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)
        gt_w = gt_cxcywh[:, 2].clamp(min=self.eps)
        gt_h = gt_cxcywh[:, 3].clamp(min=self.eps)
        scale = torch.stack([gt_w, gt_h, gt_w, gt_h], dim=-1)
        diff = pred_cxcywh.unsqueeze(1) - gt_cxcywh.unsqueeze(0)
        cost = (diff.abs() / scale.unsqueeze(0)).sum(-1)
        cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)
        return cost * self.weight


class IoUCost:
    """IoU 匹配代价"""

    def __init__(self, iou_mode='giou', weight=2.0):
        self.iou_mode = iou_mode
        self.weight = weight

    def __call__(self, pred_logits, pred_bboxes, gt_labels, gt_bboxes) -> Tensor:
        pred_bboxes = torch.nan_to_num(pred_bboxes.detach(), nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)
        gt_bboxes = torch.nan_to_num(gt_bboxes, nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)
        if self.iou_mode == 'giou':
            iou = ops.generalized_box_iou(pred_bboxes, gt_bboxes)
        else:
            iou = ops.box_iou(pred_bboxes, gt_bboxes)
        return (1 - iou) * self.weight
