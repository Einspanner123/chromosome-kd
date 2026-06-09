"""动态卷积模块"""

import torch
import torch.nn as nn
from torch import Tensor


class DynamicConv(nn.Module):
    """动态卷积模块

    通过 ROI 特征与提案特征的交互，生成动态卷积参数并应用。
    """

    def __init__(
        self,
        feat_channels: int,
        dynamic_dim: int = 64,
        dynamic_num: int = 2,
        pooler_resolution: int = 7,
    ):
        super().__init__()

        self.feat_channels = feat_channels
        self.dynamic_dim = dynamic_dim
        self.dynamic_num = dynamic_num
        self.num_params = self.feat_channels * self.dynamic_dim

        # 动态层: 生成动态卷积参数
        self.dynamic_layer = nn.Linear(
            self.feat_channels, self.dynamic_num * self.num_params
        )

        # LayerNorm 层
        self.norm1 = nn.LayerNorm(self.dynamic_dim)
        self.norm2 = nn.LayerNorm(self.feat_channels)

        # 激活函数
        self.act = nn.ReLU(inplace=True)

        # 输出层
        num_output = self.feat_channels * pooler_resolution**2
        self.out_layer = nn.Linear(num_output, self.feat_channels)
        self.norm3 = nn.LayerNorm(self.feat_channels)

    def forward(self, proposals: Tensor, roi_feats: Tensor) -> Tensor:
        """
        Args:
            proposals: 提案特征, shape: (1, Bs * num_boxes, feat_channels)
            roi_feats: ROI特征, shape: (pooler_res**2, Bs * num_boxes, feat_channels)

        Returns:
            features: shape: (1, Bs * num_boxes, feat_channels)
        """
        # (pooler_res**2, N, C) -> (N, pooler_res**2, C)
        features = roi_feats.transpose(0, 1)
        # (1, N, C) -> (N, C) -> (N, dynamic_num * num_params)
        parameters = self.dynamic_layer(proposals.squeeze(0))

        # 分割并应用动态卷积层
        param_list = parameters.chunk(self.dynamic_num, dim=1)

        # 第一层动态卷积
        param1 = param_list[0].view(-1, self.feat_channels, self.dynamic_dim)
        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.act(features)

        # 第二层动态卷积
        param2 = param_list[1].view(-1, self.dynamic_dim, self.feat_channels)
        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.act(features)

        # 展平并通过输出层
        features = features.reshape(features.size(0), -1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.act(features)

        return features.unsqueeze(0)
