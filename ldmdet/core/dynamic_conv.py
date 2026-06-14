"""动态卷积模块

通过 ROI 特征与提案特征的交互，生成动态卷积参数并应用。
"""

import torch
import torch.nn as nn
from torch import Tensor


class DynamicConv(nn.Module):
    """动态卷积: 提案特征 → 卷积参数 → 作用于 ROI 特征"""

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

        self.dynamic_layer = nn.Linear(
            self.feat_channels, self.dynamic_num * self.num_params
        )
        self.norm1 = nn.LayerNorm(self.dynamic_dim)
        self.norm2 = nn.LayerNorm(self.feat_channels)
        self.act = nn.ReLU(inplace=True)
        num_output = self.feat_channels * pooler_resolution**2
        self.out_layer = nn.Linear(num_output, self.feat_channels)
        self.norm3 = nn.LayerNorm(self.feat_channels)

    def forward(self, proposals: Tensor, roi_feats: Tensor) -> Tensor:
        """Args:
            proposals: (1, N, C) 提案特征
            roi_feats: (P*P, N, C) ROI 特征
        Returns: (1, N, C)
        """
        features = roi_feats.transpose(0, 1)  # (N, P*P, C)
        parameters = self.dynamic_layer(proposals.squeeze(0))  # (N, num_params)
        param_list = parameters.chunk(self.dynamic_num, dim=1)

        param1 = param_list[0].view(-1, self.feat_channels, self.dynamic_dim)
        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.act(features)

        param2 = param_list[1].view(-1, self.dynamic_dim, self.feat_channels)
        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.act(features)

        features = features.reshape(features.size(0), -1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.act(features)
        return features.unsqueeze(0)
