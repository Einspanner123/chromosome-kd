from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import RoIAlign


class SingleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.
    (Pure PyTorch implementation)

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale.
    """

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

        # 构建 RoI 层
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)

    def build_roi_layers(
        self, layer_cfg: Dict, featmap_strides: List[int]
    ) -> nn.ModuleList:
        """构建 torchvision 的 RoIAlign 层"""
        cfg = layer_cfg.copy()
        layer_type = cfg.pop("type", "RoIAlign")

        # 仅支持 RoIAlign，因为 torchvision 只有这个常用实现
        assert layer_type == "RoIAlign", (
            "Pure torch version only supports 'RoIAlign' currently."
        )

        output_size = cfg.pop("output_size")
        sampling_ratio = cfg.pop("sampling_ratio", 0)
        # mmdet/mmcv 默认 aligned=True，torchvision 默认 aligned=False
        # 这里为了保持一致性，默认设为 True，如果 config 里没写
        aligned = cfg.pop("aligned", True)

        roi_layers = nn.ModuleList(
            [
                RoIAlign(
                    output_size=output_size,
                    spatial_scale=1 / s,
                    sampling_ratio=sampling_ratio,
                    aligned=aligned,
                )
                for s in featmap_strides
            ]
        )
        return roi_layers

    def map_roi_levels(self, rois: Tensor, num_levels: int) -> Tensor:
        """根据尺度将 ROI 映射到对应的特征层级 (FPN 策略)"""
        # rois shape (num_boxes, 5)
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2])
        )  # (num_boxes,)
        target_lvls = torch.floor(
            torch.log2(torch.clamp(scale / self.finest_scale, min=1e-6))
        )
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls  # (num_boxes,)

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
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    def forward(
        self,
        feats: Tuple[Tensor],
        rois: Tensor,
        roi_scale_factor: Optional[float] = None,
    ) -> Tensor:
        """
        Args:
            feats (Tuple[Tensor]): 多尺度特征图
            rois (Tensor): shape(num_boxes, 5)，第一列是 batch index
            roi_scale_factor (Optional[float]): ROI 缩放因子
        """
        # 确保数据类型一致 (例如在混合精度训练时)
        rois = rois.type_as(feats[0])

        # 获取输出尺寸 (例如 7x7)
        num_levels = len(feats)  # 多尺度特征个数
        output_size = self.roi_layers[0].output_size
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        # 初始化输出特征
        roi_feats = feats[0].new_zeros(
            rois.shape[0], self.out_channels, *output_size
        )  # (num_boxes, out_dim, 7, 7)

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        # 计算每个 ROI 应该去哪个层
        target_lvls = self.map_roi_levels(rois, num_levels)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        has_params = len(list(self.parameters())) > 0
        # 逐层提取特征
        for i in range(num_levels):
            mask = target_lvls == i
            idxs = mask.nonzero(as_tuple=False).squeeze(1)
            if idxs.numel() > 0:
                rois_ = rois[idxs]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[idxs] = roi_feats_t
            else:
                # 保持计算图完整性，避免 DDP 挂死
                # 如果某一层没有用到，加上它的 0*sum 确保梯度能回传（虽然是0）
                # roi_feats += (
                #     sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                #     + feats[i].sum() * 0.0
                # )
                # 优化: 仅使用特征图的一个元素来构建梯度依赖，而不是全图求和
                # feats[i].sum() 会触发巨大的 kernel，而 feats[i][0:1].sum() 几乎无开销
                fake_loss = feats[i][0:1].sum() * 0.0
                if has_params:
                    fake_loss += sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                roi_feats += fake_loss

        return roi_feats


if __name__ == "__main__":
    # 假设你有 4 层特征金字塔 (stride 4, 8, 16, 32)
    featmap_strides = [4, 8, 16, 32]
    out_channels = 256
    roi_layer_cfg = dict(type="RoIAlign", output_size=(7, 7), sampling_ratio=0)

    # 初始化
    roi_extractor = SingleRoIExtractor(
        roi_layer=roi_layer_cfg,
        out_channels=out_channels,
        featmap_strides=featmap_strides,
    )

    # 模拟输入
    feats = [torch.randn(2, 256, 200 // s, 200 // s) for s in featmap_strides]
    rois = torch.tensor(
        [
            [0, 10.0, 10.0, 50.0, 50.0],
            [1, 30.0, 30.0, 100.0, 100.0],
            [2, 20.0, 20.0, 90.0, 80.0],
        ]
    )  # batch 1

    # 前向传播
    roi_feats = roi_extractor(feats, rois)
    print(roi_feats.shape)  # Should be (2, 256, 7, 7)
