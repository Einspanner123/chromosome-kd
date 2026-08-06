"""损失函数: FocalLoss, GIoULoss, L1Loss, FeatureDistillLoss"""

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
    """Focal Loss (sigmoid 版本). 参考 https://arxiv.org/abs/1708.02002

    优化实现: 避免重复计算 sigmoid 和 log，
    使用数值稳定的 log_sigmoid 替代手动 sigmoid + log。
    """
    # 数值稳定的 log(sigmoid(x)) 和 log(1-sigmoid(x))
    log_p = F.logsigmoid(inputs)            # log(sigmoid(x))
    log_1_minus_p = log_p - inputs          # log(1-sigmoid(x)) = log(sigmoid(x)) - x

    p = torch.sigmoid(inputs)
    p_t = p * targets + (1 - p) * (1 - targets)

    # CE loss = -[y*log(p) + (1-y)*log(1-p)]
    ce_loss = -(targets * log_p + (1 - targets) * log_1_minus_p)

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
    """Focal Loss 封装

    优化实现: 使用 F.cross_entropy 的 label_smoothing 技巧，
    或直接用 scatter_ 高效创建 one-hot。
    """

    def __init__(self, use_sigmoid=True, alpha=0.25, gamma=2.0, reduction='sum', loss_weight=2.0):
        super().__init__()
        assert use_sigmoid, 'Only sigmoid focal loss supported'
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if target.dim() == pred.dim() - 1:
            num_classes = pred.shape[-1]
            flat_pred = pred.reshape(-1, num_classes)
            flat_target = target.reshape(-1)

            # 高效 one-hot: 使用 scatter_ 原地操作
            flat_one_hot = flat_pred.new_zeros(flat_pred.shape)
            valid_mask = (flat_target >= 0) & (flat_target < num_classes)
            if valid_mask.any():
                flat_one_hot[valid_mask] = flat_one_hot[valid_mask].scatter_(
                    1, flat_target[valid_mask].unsqueeze(1), 1.0
                )
            target = flat_one_hot.reshape(pred.shape)

        loss = sigmoid_focal_loss(pred, target, self.alpha, self.gamma, self.reduction)
        return loss * self.loss_weight


class GIoULoss(nn.Module):
    """GIoU Loss 封装"""

    def __init__(self, reduction='sum', loss_weight=2.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = ops.generalized_box_iou_loss(pred, target, reduction=self.reduction)
        return loss * self.loss_weight


class L1Loss(nn.Module):
    """L1 Loss 封装"""

    def __init__(self, reduction='sum', loss_weight=5.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = F.l1_loss(pred, target, reduction=self.reduction)
        return loss * self.loss_weight


class FeatureDistillLoss(nn.Module):
    """Feature-level 蒸馏损失 (MSE on fc_feature)

    用于 Head Distillation v2: Student head 的 fc_feature (box head 前的中间特征)
    与 Teacher head 的 fc_feature 对齐。

    详见 docs/research/proposals/REFLOW_HEAD_DISTILL_IMPL_PLAN.md §2.4

    Args:
        loss_weight: 损失权重 (对应 distill_lambda, 默认 0.05)
        normalize: 是否对特征做 L2 归一化后再计算 MSE (消除幅度差异, 关注方向对齐)
    """

    def __init__(self, loss_weight: float = 0.05, normalize: bool = False):
        super().__init__()
        self.loss_weight = loss_weight
        self.normalize = normalize

    def forward(self, student_feat: Tensor, teacher_feat: Tensor) -> Tensor:
        if self.normalize:
            student_feat = F.normalize(student_feat, dim=-1)
            teacher_feat = F.normalize(teacher_feat, dim=-1)
        loss = F.mse_loss(student_feat, teacher_feat)
        return loss * self.loss_weight
