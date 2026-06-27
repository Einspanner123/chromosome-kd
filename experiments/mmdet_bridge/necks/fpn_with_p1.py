"""FPNWithP1 — 在标准 FPN 基础上增加 stride 2 的 P1 层

染色体数据集有大量小目标 (Y, 14, 15 号染色体等面积 <4000),
原始 FPN 最细为 P2 (stride 4), 对小目标特征表达不足。
P1 (stride 2) 通过上采样 C2 (backbone stage 0) 得到更细粒度的特征图,
保留更多小目标空间细节。

设计:
    输入: backbone 输出 (C2, C3, C4, C5) at strides (4, 8, 16, 32)
    输出: (P1, P2, P3, P4, P5) at strides (2, 4, 8, 16, 32)
    P1 = P1_Conv3x3( P1_LateralConv1x1( Upsample2x(C2) ) )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.necks import FPN
from mmdet.registry import MODELS


@MODELS.register_module()
class FPNWithP1(FPN):
    """FPN + P1 (stride 2) 层。

    在标准 FPN 输出 (P2-P5) 基础上, 预置一个 P1 层 (stride 2):
      P1 = P1_Conv(Upsample2x(P1_Lateral(C2)))

    与 mmdet FPN 的 add_extra_convs 不同 (它只能加更粗的 P6/P7),
    本模块前置更细的 P1 层用于小目标检测。

    Args:
        in_channels: backbone 各 stage 输出通道 (如 ResNet [256, 512, 1024, 2048])
        out_channels: FPN 输出通道
        num_outs: FPN 原有输出数 (不含 P1), 默认 = len(in_channels) = 4 (P2-P5)
        **kwargs: 透传给 FPN (start_level, relu_before_extra_convs 等)
    """

    def __init__(self, in_channels, out_channels, num_outs=None, **kwargs):
        if num_outs is None:
            num_outs = len(in_channels)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs
        )
        # P1 来自 C2 (in_channels[start_level], 即 backbone stage 0)
        start_level = kwargs.get('start_level', 0)
        c2_channels = in_channels[start_level]
        self.p1_lateral_conv = nn.Conv2d(c2_channels, out_channels, kernel_size=1)
        self.p1_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, inputs):
        # 标准 FPN 输出: (P2, P3, P4, P5) at strides (4, 8, 16, 32)
        outs = super().forward(inputs)
        # 构建 P1: C2 (stride 4) → 1x1 投影 → 2x 上采样 → 3x3 平滑
        c2 = inputs[0]  # backbone stage 0, stride 4
        p1 = self.p1_lateral_conv(c2)
        p1 = F.interpolate(p1, scale_factor=2.0, mode='nearest')
        p1 = self.p1_conv(p1)
        # P1 置于最前, 使 strides = [2, 4, 8, 16, 32]
        return (p1,) + tuple(outs)
