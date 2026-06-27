"""ContrastiveLoss — 方向 C2: 对比学习辅助损失

拉大形态相似类 (G-F, Y-E18 等) 的特征距离, 提升判别力.
基于 InfoNCE: 对每个 anchor, 同类为正, 不同类为负, 用余弦相似度.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """InfoNCE 风格的监督对比损失.

    Args:
        temp: 温度参数 (默认 0.07)
    """

    def __init__(self, temp: float = 0.07):
        super().__init__()
        self.temp = temp

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算对比损失.

        Args:
            feats: (N, D) 归一化特征 (建议 L2 归一化)
            labels: (N,) 类别标签

        Returns:
            scalar loss
        """
        if feats.shape[0] < 2:
            return feats.new_zeros(())

        # L2 归一化 (若未归一化)
        feats = F.normalize(feats, dim=-1)
        # 相似度矩阵: (N, N)
        sim = feats @ feats.t() / self.temp
        # 数值稳定: 减去最大值
        sim = sim - sim.max(dim=-1, keepdim=True).values.detach()
        # 对角线置 -inf (自身不参与对比)
        mask_self = torch.eye(feats.shape[0], device=feats.device, dtype=torch.bool)
        sim = sim.masked_fill(mask_self, float('-inf'))

        # log-sum-exp 分母: 对所有非自身样本
        # exp(sim) 求和 (非对角)
        exp_sim = torch.exp(sim)
        exp_sum = exp_sim.sum(dim=-1)  # (N,)
        # 避免分母为 0 (若每个样本都无负样本)
        exp_sum = exp_sum.clamp(min=1e-8)

        # 正对 mask: 同类且非自身
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
        pos_mask = label_eq & ~mask_self  # (N, N)

        # 对每个 anchor, 正对的 log-sum-exp
        # 若某 anchor 无正对, 跳过 (贡献 0)
        pos_count = pos_mask.sum(dim=-1)  # (N,)
        valid = pos_count > 0  # (N,)

        if not valid.any():
            return feats.new_zeros(())

        # 正对的 exp 之和
        pos_exp = (exp_sim * pos_mask.float()).sum(dim=-1)  # (N,)
        pos_exp = pos_exp.clamp(min=1e-8)

        # -log(pos_exp / exp_sum) 对每个有效 anchor
        log_prob = torch.log(pos_exp / exp_sum)
        loss_per_anchor = -log_prob  # (N,)

        # 仅对有正对的 anchor 求平均
        loss = loss_per_anchor[valid].mean()
        return loss
