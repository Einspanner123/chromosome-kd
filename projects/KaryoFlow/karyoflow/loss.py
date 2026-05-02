"""
KaryoFlow 训练损失

MDLM 风格: 仅对被 mask 的位置计算 Cross-Entropy loss。
每个位置 i 的合法值为 {0, ..., N-1-i}，超出范围的 logits 被 mask 为 -inf。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class KaryoFlowLoss(nn.Module):
    """MDLM 训练 loss: 对 masked 位置计算 CE

    Args:
        label_smoothing: 标签平滑系数
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Args:
            logits: (B, N, vocab) — 模型预测 logits (无效位置已被 flow_module 设为 -inf)
            target: (B, N) — 真值 Lehmer code
            mask: (B, N) bool — True = 该位置被 masked = 需要预测

        Returns:
            loss: scalar
        """
        if not mask.any():
            return logits.sum() * 0.0

        # 提取被 mask 的位置
        masked_logits = logits[mask]   # (num_masked, vocab)
        masked_target = target[mask]   # (num_masked,)

        loss = F.cross_entropy(
            masked_logits,
            masked_target,
            label_smoothing=self.label_smoothing,
        )

        return loss
