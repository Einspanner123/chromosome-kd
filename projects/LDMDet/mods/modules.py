import math
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


def cosine_noise_schedule(T, s=0.008, device='cpu'):
    t = torch.linspace(0, T, T + 1, dtype=torch.float64, device=device)
    alphas_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2)**2
    alphas_t = alphas_t / alphas_t[0]
    betas_t = 1 - (alphas_t[1:] / alphas_t[:-1])
    return torch.clamp(betas_t, 0, 0.999)


def load_buffer(arr: Tensor, steps: Tensor, x_shape: List[int]):
    """
    从序列a中提取时间步t对应的值,并调整形状以匹配x_shape
    a: 序列参数,如alphas_cumprod等,shape: [timesteps]
    t: 时间步索引,shape: [batch_size]
    x_shape: 目标形状,用于reshape
    """
    step_arr = arr.gather(-1, steps)
    return step_arr.reshape(
        -1,
        *[1] * (len(x_shape) - 1))  # shape: [bs, 1, 1, 1, ..., 1(输入的维度数-1)]


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码模块,用于时间步编码"""

    def __init__(self, dim: int):
        """
        初始化正弦位置编码
        dim: 编码维度
        """
        super().__init__()
        self.dim = dim  # 位置编码的维度

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        timesteps: [batch_size] (float or int)
        return: [batch_size, dim]
        """
        device = timesteps.device
        half_dim = self.dim // 2

        # exp(-log(10000) * i / (half_dim - 1))
        exponent = -math.log(10000) / (half_dim - 1)
        i = torch.arange(half_dim, device=device, dtype=timesteps.dtype)
        freqs = torch.exp(i * exponent)

        # args = timesteps[:, None] * freqs[None, :]  # 元素乘法, 自动广播
        args = torch.outer(timesteps, freqs)  # 效率更高
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return emb


class DynamicConv(nn.Module):
    """动态卷积模块"""

    def __init__(
        self,
        feat_channels: int,  # 特征通道数
        dynamic_dim: int = 64,  # 动态维度
        dynamic_num: int = 2,  # 动态层数
        pooler_resolution: int = 7,
    ) -> None:  # 池化分辨率
        super().__init__()

        self.feat_channels = feat_channels  # 特征通道数
        self.dynamic_dim = dynamic_dim  # 动态维度
        self.dynamic_num = dynamic_num  # 动态层数
        self.num_params = self.feat_channels * self.dynamic_dim  # 参数数量
        # 动态层: 生成动态卷积参数
        self.dynamic_layer = nn.Linear(self.feat_channels,
                                       self.dynamic_num * self.num_params)

        # LayerNorm层
        self.norm1 = nn.LayerNorm(self.dynamic_dim)
        self.norm2 = nn.LayerNorm(self.feat_channels)

        # 激活函数
        self.act = nn.ReLU(inplace=True)

        # 输出层
        num_output = self.feat_channels * pooler_resolution**2  # 输出维度
        self.out_layer = nn.Linear(num_output, self.feat_channels)  # 输出线性层
        self.norm3 = nn.LayerNorm(self.feat_channels)  # 输出归一化层

    def forward(self, proposals: Tensor, roi_feats: Tensor) -> Tensor:
        """前向传播

        Args:
            proposals: 提案特征,shape: (1, Bs * num_boxes, self.feat_channels)
            roi_feats: ROI特征,shape: (pooler_res**2, Bs * num_boxes, self.feat_channels)

        Returns:
            features: 处理后的特征,shape: (1, Bs * num_boxes, self.feat_channels)
        """
        # 1. 准备特征和动态参数
        # (pooler_res**2, N, C) -> (N, pooler_res**2, C)
        features = roi_feats.transpose(0, 1)
        # (1, N, C) -> (N, C) -> (N, dynamic_num * num_params)
        parameters = self.dynamic_layer(proposals.squeeze(0))

        # 2. 分割并应用动态卷积层
        # 使用 chunk 减少切片操作
        param_list = parameters.chunk(self.dynamic_num, dim=1)

        # 第一层动态卷积
        param1 = param_list[0].view(-1, self.feat_channels, self.dynamic_dim)
        features = torch.bmm(features, param1)  # (N, 49, dynamic_dim)
        features = self.norm1(features)
        features = self.act(features)

        # 第二层动态卷积
        param2 = param_list[1].view(-1, self.dynamic_dim, self.feat_channels)
        features = torch.bmm(features, param2)  # (N, 49, feat_channels)
        features = self.norm2(features)
        features = self.act(features)

        # 3. 展平并通过输出层
        # 使用 reshape 而非 flatten 以保持兼容性，squeeze(0) 的逆操作
        features = features.reshape(features.size(0),
                                    -1)  # (N, 49*feat_channels)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.act(features)

        return features.unsqueeze(0)  # (1, N, feat_channels)


if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256))
    x = x.permute(0, 2, 3, 1).reshape(1, -1, 3)
    attn = nn.MultiheadAttention(3, 3)
    out = attn(x, x, x)

    print(out[0].shape)
    print(out[1].shape)
