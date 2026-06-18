"""SingleRoIExtractor — 纯 PyTorch RoI 特征提取

从 FPN 特征金字塔中提取 RoI 特征，支持多尺度分配合和 RoIAlign。
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import RoIAlign


class SingleRoIExtractor(nn.Module):
    """从多级特征图中提取 RoI 特征"""

    def __init__(
        self,
        roi_layer: Dict,
        out_channels: int,
        featmap_strides: List[int],
        finest_scale: int = 56,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.roi_layers = self._build_roi_layers(roi_layer, featmap_strides)

    def _build_roi_layers(
        self, layer_cfg: Dict, featmap_strides: List[int]
    ) -> nn.ModuleList:
        cfg = layer_cfg.copy()
        cfg.pop('type', None)
        output_size = cfg.pop('output_size')
        sampling_ratio = cfg.pop('sampling_ratio', 0)
        aligned = cfg.pop('aligned', True)

        return nn.ModuleList([
            RoIAlign(
                output_size=output_size,
                spatial_scale=1 / s,
                sampling_ratio=sampling_ratio,
                aligned=aligned,
            )
            for s in featmap_strides
        ])

    def map_roi_levels(self, rois: Tensor, num_levels: int) -> Tensor:
        """根据尺度将 ROI 映射到对应 FPN 层级"""
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2])
        )
        target_lvls = torch.floor(
            torch.log2(torch.clamp(scale / self.finest_scale, min=1e-6))
        )
        return target_lvls.clamp(min=0, max=num_levels - 1).long()

    def roi_rescale(self, rois: Tensor, scale_factor: float) -> Tensor:
        """缩放 ROI 坐标"""
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1]
        h = rois[:, 4] - rois[:, 2]
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5
        x2 = cx + new_w * 0.5
        y1 = cy - new_h * 0.5
        y2 = cy + new_h * 0.5
        return torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)

    def forward(
        self,
        feats: Tuple[Tensor, ...],
        rois: Tensor,
        roi_scale_factor: Optional[float] = None,
    ) -> Tensor:
        """Args:
            feats: 多尺度特征图 Tuple[Tensor]
            rois: (N, 5) [batch_idx, x1, y1, x2, y2]
        Returns: (N, out_channels, H, W)
        """
        rois = rois.type_as(feats[0])
        num_levels = len(feats)
        output_size = self.roi_layers[0].output_size
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        if num_levels == 1:
            if len(rois) == 0:
                return feats[0].new_zeros(
                    rois.shape[0], self.out_channels, *output_size
                )
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        # 按层级分组 ROI，减少循环内条件判断
        roi_feats = feats[0].new_zeros(
            rois.shape[0], self.out_channels, *output_size
        )

        # 对每个 FPN 层级提取 ROI 特征，使用 scatter 合并结果
        # 避免逐层 nonzero + 索引赋值导致的 GPU→CPU 同步
        all_roi_feats = []
        all_roi_indices = []
        for i in range(num_levels):
            mask = target_lvls == i
            idxs = mask.nonzero(as_tuple=False).squeeze(1)
            if idxs.shape[0] > 0:
                feat_i = self.roi_layers[i](feats[i], rois[idxs])
                all_roi_feats.append(feat_i)
                all_roi_indices.append(idxs)

        if len(all_roi_feats) > 0:
            all_roi_feats = torch.cat(all_roi_feats, dim=0)
            all_roi_indices = torch.cat(all_roi_indices, dim=0)
            roi_feats[all_roi_indices] = all_roi_feats

        return roi_feats
