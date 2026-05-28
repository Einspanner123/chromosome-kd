import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS


class WeightedFusion(nn.Module):
    def __init__(self, num_inputs: int, eps: float = 1e-4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = eps

    def forward(self, inputs: List[Tensor]) -> Tensor:
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)
        return sum(w_i * x_i for w_i, x_i in zip(w, inputs))


@MODELS.register_module()
class BiFPN(BaseModule):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        num_repeats: int = 1,
        start_level: int = 0,
        end_level: int = -1,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        upsample_cfg=dict(mode='nearest'),
        init_cfg=dict(
            type='Xavier', layer='Conv2d', distribution='uniform'
        ),
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level + 1
        self.end_level = end_level

        self.num_levels = self.backbone_end_level - self.start_level
        assert self.num_levels >= 2, 'BiFPN requires at least 2 feature levels'

        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_levels):
            self.lateral_convs.append(
                ConvModule(
                    in_channels[i + self.start_level],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
            )

        extra_levels = num_outs - self.num_levels
        if extra_levels > 0:
            self.extra_convs = nn.ModuleList()
            for i in range(extra_levels):
                in_ch = in_channels[-1] if i == 0 else out_channels
                self.extra_convs.append(
                    ConvModule(
                        in_ch,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False,
                    )
                )
        else:
            self.extra_convs = None

        self.bifpn_convs = nn.ModuleList()
        self.bifpn_fusions_td = nn.ModuleList()
        self.bifpn_fusions_bu = nn.ModuleList()

        for _ in range(num_repeats):
            td_convs = nn.ModuleList()
            bu_convs = nn.ModuleList()
            td_fusions = nn.ModuleList()
            bu_fusions = nn.ModuleList()

            for i in range(self.num_levels - 1):
                td_fusions.append(WeightedFusion(2))
                td_convs.append(
                    ConvModule(
                        out_channels,
                        out_channels,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False,
                    )
                )

            for i in range(self.num_levels - 1):
                bu_fusions.append(WeightedFusion(3 if i > 0 else 2))
                bu_convs.append(
                    ConvModule(
                        out_channels,
                        out_channels,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False,
                    )
                )

            self.bifpn_fusions_td.append(td_fusions)
            self.bifpn_fusions_bu.append(bu_fusions)
            self.bifpn_convs.append(
                nn.ModuleDict({'td': td_convs, 'bu': bu_convs})
            )

        self.num_repeats = num_repeats

    def _upsample_add(self, x: Tensor, y: Tensor) -> Tensor:
        if 'scale_factor' in self.upsample_cfg:
            return y + F.interpolate(x, **self.upsample_cfg)
        else:
            return y + F.interpolate(x, size=y.shape[2:], **self.upsample_cfg)

    def _downsample_add(self, x: Tensor, y: Tensor) -> Tensor:
        return y + F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        assert len(inputs) == len(self.in_channels)

        laterals = [
            self.lateral_convs[i](inputs[i + self.start_level])
            for i in range(self.num_levels)
        ]

        if self.extra_convs is not None:
            last_backbone = inputs[self.backbone_end_level - 1]
            p_extra = self.extra_convs[0](last_backbone)
            laterals.append(p_extra)
            for i in range(1, len(self.extra_convs)):
                p_extra = self.extra_convs[i](p_extra)
                laterals.append(p_extra)
            num_levels = len(laterals)
        else:
            num_levels = self.num_levels

        features = list(laterals)

        for r in range(self.num_repeats):
            td_fusions = self.bifpn_fusions_td[r]
            bu_fusions = self.bifpn_fusions_bu[r]
            td_convs = self.bifpn_convs[r]['td']
            bu_convs = self.bifpn_convs[r]['bu']

            td_outs = [None] * num_levels
            td_outs[-1] = features[-1]

            for i in range(num_levels - 2, -1, -1):
                up = F.interpolate(
                    td_outs[i + 1],
                    size=features[i].shape[2:],
                    **self.upsample_cfg,
                )
                fused = td_fusions[i]([features[i], up])
                td_outs[i] = td_convs[i](fused)

            bu_outs = [None] * num_levels
            bu_outs[0] = td_outs[0]

            for i in range(1, num_levels):
                down = F.max_pool2d(td_outs[i - 1], kernel_size=3, stride=2, padding=1)
                if i < num_levels - 1:
                    fused = bu_fusions[i - 1]([features[i], td_outs[i], down])
                else:
                    fused = bu_fusions[i - 1]([td_outs[i], down])
                bu_outs[i] = bu_convs[i - 1](fused)

            features = bu_outs

        if self.num_outs > num_levels:
            for i in range(self.num_outs - num_levels):
                features.append(F.max_pool2d(features[-1], 1, stride=2))

        return tuple(features)
