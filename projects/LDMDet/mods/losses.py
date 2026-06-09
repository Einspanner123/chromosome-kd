"""损失函数模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import ops


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
) -> Tensor:
    """Focal Loss 纯 PyTorch 实现 (Sigmoid 版本)

    参考: https://arxiv.org/abs/1708.02002
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction='none'
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


class FocalLoss(nn.Module):
    """Focal Loss 封装类"""

    def __init__(
        self,
        use_sigmoid=True,
        alpha=0.25,
        gamma=2.0,
        reduction='sum',
        loss_weight=2.0,
    ):
        super().__init__()
        assert use_sigmoid, 'Currently only supports sigmoid focal loss'
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        pred: [..., num_classes] (logits)
        target: [...] (class indices) or [..., num_classes] (one-hot)
        """
        if target.dim() == pred.dim() - 1:
            num_classes = pred.shape[-1]
            flat_pred = pred.reshape(-1, num_classes)
            flat_target = target.reshape(-1)
            flat_one_hot = torch.zeros_like(flat_pred)

            valid_mask = (flat_target >= 0) & (flat_target < num_classes)
            if valid_mask.any():
                flat_one_hot[valid_mask, flat_target[valid_mask]] = 1.0
            target = flat_one_hot.reshape(pred.shape)

        loss = sigmoid_focal_loss(
            pred, target, self.alpha, self.gamma, self.reduction
        )
        return loss * self.loss_weight


class GIoULoss(nn.Module):
    """GIoU Loss 封装类"""

    def __init__(self, reduction='sum', loss_weight=2.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """pred/target: [N, 4] (xyxy)"""
        loss = ops.generalized_box_iou_loss(
            pred, target, reduction=self.reduction
        )
        return loss * self.loss_weight


class L1Loss(nn.Module):
    """L1 Loss 封装类"""

    def __init__(self, reduction='sum', loss_weight=5.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = F.l1_loss(pred, target, reduction=self.reduction)
        return loss * self.loss_weight


class FlowMatchingVelocityLoss(nn.Module):
    """Flow Matching 速度场 MSE Loss

    L_flow = ||v_pred - (b_gt - z)||^2
    仅对前景 (matched) 提议框计算。
    """

    def __init__(self, loss_weight: float = 5.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        v_pred: Tensor,
        v_target: Tensor,
        fg_mask: Tensor = None,
    ) -> Tensor:
        if fg_mask is not None:
            v_pred = v_pred[fg_mask]
            v_target = v_target[fg_mask]
        if v_pred.numel() == 0:
            return v_pred.sum() * 0
        return F.mse_loss(v_pred, v_target) * self.loss_weight
