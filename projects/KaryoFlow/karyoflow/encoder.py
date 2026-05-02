"""
染色体编码器

将每个染色体 crop 图像编码为 d 维特征向量。
支持额外的几何信息 (bbox 坐标、面积、长宽比) 融合。
"""

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models


class ChromosomeEncoder(nn.Module):
    """染色体 crop 编码器

    使用 ResNet18 前几层提取视觉特征，融合几何信息。

    Args:
        d_model: 输出特征维度
        crop_size: 输入 crop 尺寸 (默认 64)
        use_geometry: 是否融合 bbox 几何信息
        pretrained: 是否使用 ImageNet 预训练权重
    """

    def __init__(
        self,
        d_model: int = 256,
        crop_size: int = 64,
        use_geometry: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_geometry = use_geometry

        # 视觉编码器: ResNet18 去掉最后的 FC 层
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # 保留到 layer3 (输出 256 通道), 去掉 layer4 和 fc (减少参数)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 64 channels
            resnet.layer2,  # 128 channels
            resnet.layer3,  # 256 channels
        )
        backbone_out_dim = 256

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.visual_proj = nn.Linear(backbone_out_dim, d_model)

        # 几何信息编码
        if use_geometry:
            # bbox: 4d (归一化 xyxy) + 面积 + 长宽比 = 6d
            self.geometry_proj = nn.Sequential(
                nn.Linear(6, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model),
            )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        crops: Tensor,
        bboxes: Tensor = None,
        image_size: tuple = None,
    ) -> Tensor:
        """
        Args:
            crops: (N, 3, crop_H, crop_W) — 染色体裁剪图像
            bboxes: (N, 4) xyxy 格式 (可选，用于几何信息融合)
            image_size: (H, W) 原图尺寸 (用于归一化 bbox)

        Returns:
            features: (N, d_model)
        """
        N = crops.shape[0]

        # 视觉特征
        feat = self.backbone(crops)      # (N, 256, h, w)
        feat = self.pool(feat).flatten(1)  # (N, 256)
        feat = self.visual_proj(feat)      # (N, d_model)

        # 融合几何信息
        if self.use_geometry and bboxes is not None:
            if image_size is not None:
                H, W = image_size
                norm_bboxes = bboxes.clone()
                norm_bboxes[:, 0::2] /= W
                norm_bboxes[:, 1::2] /= H
            else:
                norm_bboxes = bboxes

            # 计算面积和长宽比
            w = norm_bboxes[:, 2] - norm_bboxes[:, 0]
            h = norm_bboxes[:, 3] - norm_bboxes[:, 1]
            area = w * h
            aspect_ratio = w / (h + 1e-6)

            geometry = torch.stack([
                norm_bboxes[:, 0], norm_bboxes[:, 1],
                norm_bboxes[:, 2], norm_bboxes[:, 3],
                area, aspect_ratio,
            ], dim=1)  # (N, 6)

            geo_feat = self.geometry_proj(geometry)  # (N, d_model)
            feat = feat + geo_feat

        feat = self.norm(feat)
        return feat
