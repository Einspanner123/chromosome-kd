"""DeformableRoIExtractor — RoIAlign + DeformConv2d (可学习空间偏移)

在 SingleRoIExtractor 基础上增加可变形卷积, 让 RoI 特征采样位置可学习,
提升对染色体形态变化的适应能力 (相比固定 RoIAlign 网格采样)。

设计:
    RoIAlign (7x7) → offset_conv (预测 2*kh*kw*offset_groups 通道偏移)
                   → DeformConv2d (按偏移采样) + residual
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import DeformConv2d

from ldmdet.core.roi_extractor import SingleRoIExtractor


class DeformableRoIExtractor(SingleRoIExtractor):
    """RoIAlign + DeformConv2d 残差增强。

    相比 SingleRoIExtractor:
      - 额外学习 RoI 内部的空间偏移 (offset_conv)
      - 用 DeformConv2d 按偏移采样, 自适应染色体形态
      - 残差连接保留 RoIAlign 原始信息

    Args:
        roi_layer: 同 SingleRoIExtractor
        out_channels: RoI 特征通道数
        featmap_strides: 各 FPN 层级步幅
        deform_groups: 偏移分组数 (类似 deform_groups in DeformConv2d)
        finest_scale: 同 SingleRoIExtractor
        kernel_size: DeformConv 核大小 (默认 3)
    """

    def __init__(
        self,
        roi_layer: Dict,
        out_channels: int,
        featmap_strides: List[int],
        deform_groups: int = 1,
        finest_scale: int = 56,
        kernel_size: int = 3,
    ) -> None:
        super().__init__(roi_layer, out_channels, featmap_strides, finest_scale)
        self.deform_groups = deform_groups
        padding = kernel_size // 2

        # 偏移预测器: 从 RoI 特征预测每个采样点的 (dx, dy) 偏移
        # 输出通道 = 2 * kh * kw * deform_groups (torchvision 用 groups 同时控制通道和偏移分组)
        self.offset_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=2 * kernel_size * kernel_size * deform_groups,
            kernel_size=kernel_size,
            padding=padding,
        )
        # 初始化为 0 → 初始偏移为 0 → 等价于普通卷积 (稳定训练初期)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        # 可变形卷积 (groups 同时控制通道分组和偏移分组, 需 out_channels % deform_groups == 0)
        self.deform_conv = DeformConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=deform_groups,
        )
        # 初始化 deform_conv 权重 → 初始输出接近 0, 残差后 = RoIAlign 输出
        nn.init.zeros_(self.deform_conv.weight)

    def forward(
        self,
        feats,
        rois: Tensor,
        roi_scale_factor: Optional[float] = None,
    ) -> Tensor:
        # 1. 标准 RoIAlign 提取
        roi_feats = super().forward(feats, rois, roi_scale_factor)

        # 2. 预测空间偏移 (从 RoI 特征自身)
        offset = self.offset_conv(roi_feats)

        # 3. 可变形卷积采样 + 残差连接
        roi_feats = roi_feats + self.deform_conv(roi_feats, offset)
        return roi_feats
