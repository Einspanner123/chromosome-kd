"""BoxRefineNet — 方向 D1: 框细化网络

在扩散采样最后一步后, 加一个独立的细粒度回归头, 专门优化高 IoU 精度.
类似 Cascade R-CNN 的级联精化, 但不增加采样步数.

设计:
- 输入: fc_feature (bs*num_boxes, C)
- 输出: 4 维残差偏移 (dx, dy, dw, dh)
- 零初始化最后一层, 保证初始时残差为 0, 不破坏 baseline
"""

import torch
import torch.nn as nn


class BoxRefineNet(nn.Module):
    """框细化网络 (残差结构).

    Args:
        feat_channels: 特征通道数 (默认 256)
    """

    def __init__(self, feat_channels: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, 4),
        )
        # D1: 零初始化最后一层, 保证初始时残差为 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向.

        Args:
            x: (N, C) 特征

        Returns:
            (N, 4) 残差偏移
        """
        return self.mlp(x)
