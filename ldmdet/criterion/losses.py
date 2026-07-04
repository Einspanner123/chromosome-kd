"""损失函数: FocalLoss, GIoULoss, L1Loss, SeesawLoss"""

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


class SeesawLoss(nn.Module):
    """Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021).

    Sigmoid 版本适配: 对负类损失乘以 Seesaw 因子 S_{y,j} = (N_j / N_y)^p,
    衰减头类样本对尾类负类的负梯度。当 N_j << N_y (尾类负类 j, 头类样本 y) 时 S << 1。

    误分类自校准: 当样本被误分类 (最大负类 logit > 正类 logit) 时,
    对误分类类恢复 S=1 (完整惩罚), 以学习区分。

    与 mmdet 官方实现的差异 (已知, 不影响 Y 类坍塌核心场景):
    1. 补偿因子简化: 当前只恢复单个最大误分类类为 S=1;
       mmdet 对所有 p_j > p_y 的类施加 (p_j/p_y)^q 放大 (理论精度更高).
    2. cum_samples 初始化: 当前 torch.ones(num_classes) (初始1, 不含背景位);
       mmdet torch.zeros(num_classes+1) (初始0, 含背景位). 初始1避免除零.

    Args:
        num_classes: 类别数 (不含背景)
        p: Seesaw 指数, 默认 0.8
        alpha: 正负样本平衡因子, <0 时不使用
        gamma: Focal 聚焦参数
        reduction: 'sum' 或 'mean'
        loss_weight: 损失权重
    """

    def __init__(self, num_classes: int, p: float = 0.8, alpha: float = 0.25,
                 gamma: float = 2.0, reduction: str = 'sum', loss_weight: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.register_buffer('cum_samples', torch.ones(num_classes))

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if target.dim() == pred.dim() - 1:
            num_classes = pred.shape[-1]
            flat_pred = pred.reshape(-1, num_classes)
            flat_target = target.reshape(-1)
        else:
            # target 与 pred 同形状 (one-hot), 转换为类别索引
            num_classes = pred.shape[-1]
            flat_pred = pred.reshape(-1, num_classes)
            flat_target = target.reshape(-1, num_classes).argmax(dim=1)

        if flat_pred.shape[0] == 0:
            return flat_pred.sum() * 0.0

        loss = self._compute_loss(flat_pred, flat_target)

        with torch.no_grad():
            valid_mask = (flat_target >= 0) & (flat_target < self.num_classes)
            if valid_mask.any():
                batch_counts = torch.bincount(
                    flat_target[valid_mask], minlength=self.num_classes
                ).to(self.cum_samples.dtype)
                self.cum_samples += batch_counts

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss * self.loss_weight

    def _compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        N, C = pred.shape

        p = torch.sigmoid(pred)
        log_p = F.logsigmoid(pred)
        log_1_minus_p = log_p - pred

        one_hot = torch.zeros_like(pred)
        valid = (target >= 0) & (target < C)
        if valid.any():
            one_hot[valid] = one_hot[valid].scatter_(
                1, target[valid].unsqueeze(1), 1.0
            )

        p_t = p * one_hot + (1 - p) * (1 - one_hot)
        ce_loss = -(one_hot * log_p + (1 - one_hot) * log_1_minus_p)
        focal_weight = (1 - p_t) ** self.gamma

        seesaw_weight = torch.ones_like(pred)
        positive_mask = valid
        if positive_mask.any():
            pos_classes = target[positive_mask]
            n_y = self.cum_samples[pos_classes]
            n_j = self.cum_samples.unsqueeze(0)
            # mmdet 官方方向: ratio = N_j / N_y, clamp(max=1.0) 确保只衰减不放大
            # 语义: 头类样本(N_y大)对尾类负类(N_j小)时 S<1 衰减, 保护尾类
            ratio = n_j / n_y.unsqueeze(1)
            s = ratio.clamp(max=1.0) ** self.p
            s[range(len(pos_classes)), pos_classes] = 1.0

            # 误分类自校准: 当样本被误分类(最大负类 logit > 正类 logit)时,
            # 对误分类类恢复 S=1 (完整惩罚), 以学习区分
            pos_logits = pred[positive_mask]
            pos_class_logits = pos_logits[range(len(pos_classes)), pos_classes]
            masked_logits = pos_logits.clone()
            masked_logits[range(len(pos_classes)), pos_classes] = float('-inf')
            max_neg_logits, max_neg_classes = masked_logits.max(dim=1)
            misclassified = max_neg_logits > pos_class_logits
            if misclassified.any():
                mis_idx = misclassified.nonzero(as_tuple=True)[0]
                mis_neg_classes = max_neg_classes[mis_idx]
                s[mis_idx, mis_neg_classes] = 1.0

            seesaw_weight[positive_mask] = s

        weighted_loss = ce_loss * focal_weight * seesaw_weight

        if self.alpha >= 0:
            alpha_t = self.alpha * one_hot + (1 - self.alpha) * (1 - one_hot)
            weighted_loss = alpha_t * weighted_loss

        return weighted_loss
