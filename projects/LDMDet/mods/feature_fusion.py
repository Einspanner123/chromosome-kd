"""特征融合模块"""

from typing import List

import torch.nn as nn
from torch import Tensor


class PurePyTorchSimpleFeatureFusion(nn.Module):
    """简单的特征融合模块，用于 ViT 输出的多级特征对齐

    将 ViT 的单一分辨率输出还原为多尺度特征金字塔:
    P2: 1/4, P3: 1/8, P4: 1/16, P5: 1/32
    """

    def __init__(
        self, in_channels: List[int], out_channels: int, num_outs: int = 4
    ):
        super().__init__()
        self.upsample_p2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels[0], out_channels, kernel_size=2, stride=2
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=2, stride=2
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.upsample_p3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels[1], out_channels, kernel_size=2, stride=2
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.p4_conv = nn.Sequential(
            nn.Conv2d(in_channels[2], out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.downsample_p5 = nn.Sequential(
            nn.Conv2d(in_channels[3], out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.num_outs = num_outs

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        p2 = self.upsample_p2(x[0])
        p3 = self.upsample_p3(x[1])
        p4 = self.p4_conv(x[2])
        p5 = self.downsample_p5(x[3])
        return [p2, p3, p4, p5]
