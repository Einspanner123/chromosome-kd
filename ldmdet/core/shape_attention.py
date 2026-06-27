"""ShapeAttention — 方向 C1: 局部形状注意力

在 RoI 特征 (7x7) 上加方向性卷积, 显式编码染色体臂长比、着丝粒位置等形态信息.
- 水平分支 (7,1): 捕获水平方向形态 (臂长)
- 垂直分支 (1,7): 捕获垂直方向形态 (着丝粒)
- fuse: 融合后 + 残差

设计依据: RoIAlign 已将每条染色体提取到独立 7x7 特征, RoI 内部无重叠.
"""

import torch
import torch.nn as nn


class ShapeAttention(nn.Module):
    """局部形状注意力模块.

    Args:
        channels: RoI 特征通道数 (默认 256)
        reduction: 方向分支的通道缩减比 (默认 4, 即 channels//4)
    """

    def __init__(self, channels: int = 256, reduction: int = 4):
        super().__init__()
        hidden = channels // reduction
        # C1: 水平方向 (7,1) — 捕获臂长
        self.h_branch = nn.Conv2d(channels, hidden, kernel_size=(7, 1))
        # C1: 垂直方向 (1,7) — 捕获着丝粒
        self.v_branch = nn.Conv2d(channels, hidden, kernel_size=(1, 7))
        # 融合: hidden*2 → channels
        self.fuse = nn.Conv2d(hidden * 2, channels, kernel_size=1)
        # 激活
        self.act = nn.ReLU(inplace=True)

        # 初始化 fuse 为小值, 保证初始时残差占主导
        nn.init.zeros_(self.fuse.weight)
        if self.fuse.bias is not None:
            nn.init.zeros_(self.fuse.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向.

        Args:
            x: (N, C, 7, 7) RoI 特征

        Returns:
            (N, C, 7, 7) 增强后的 RoI 特征 (残差)
        """
        # h_branch: (N, hidden, 1, 7) — 水平方向压缩到 1
        h = self.act(self.h_branch(x))
        # v_branch: (N, hidden, 7, 1) — 垂直方向压缩到 1
        v = self.act(self.v_branch(x))
        # 将 h 扩展到 (N, hidden, 7, 7) 以与 v 对齐
        h = h.expand(-1, -1, 7, -1)
        v = v.expand(-1, -1, -1, 7)
        # 拼接并融合
        fused = self.fuse(torch.cat([h, v], dim=1))
        # 残差连接
        return x + fused
