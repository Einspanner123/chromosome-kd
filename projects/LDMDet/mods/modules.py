import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# 将当前文件所在目录添加到 sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))


DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


def cosine_noise_schedule(T, s=0.008, device="cpu"):
    t = torch.linspace(0, T, T + 1, dtype=torch.float64, device=device)
    alphas_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
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
        -1, *[1] * (len(x_shape) - 1)
    )  # shape: [bs, 1, 1, 1, ..., 1(输入的维度数-1)]


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

    def __init__(
        self,
        feat_channels: int,
        dynamic_dim: int = 64,
        dynamic_num: int = 2,
        pooler_resolution: int = 7,
    ) -> None:
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
        features = roi_feats.transpose(0, 1)
        parameters = self.dynamic_layer(proposals.squeeze(0))

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


class LinearCrossAttention(nn.Module):

    def __init__(
        self,
        feat_channels: int,
        pooler_resolution: int = 7,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feat_channels = feat_channels
        self.num_heads = num_heads
        self.head_dim = feat_channels // num_heads
        self.pooler_res = pooler_resolution

        self.q_proj = nn.Linear(feat_channels, feat_channels)
        self.k_proj = nn.Linear(feat_channels, feat_channels)
        self.v_proj = nn.Linear(feat_channels, feat_channels)
        self.out_proj = nn.Linear(feat_channels, feat_channels)

        self.norm1 = nn.LayerNorm(feat_channels)
        self.norm2 = nn.LayerNorm(feat_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, proposals: Tensor, roi_feats: Tensor) -> Tensor:
        proposals_squeezed = proposals.squeeze(0)
        N = proposals_squeezed.shape[0]

        roi_flat = roi_feats.permute(1, 0, 2).reshape(
            N, self.pooler_res**2, self.feat_channels
        )

        q = self.q_proj(proposals_squeezed)
        k = self.k_proj(roi_flat)
        v = self.v_proj(roi_flat)

        q = q.reshape(N, self.num_heads, self.head_dim)
        k = k.reshape(N, self.pooler_res**2, self.num_heads, self.head_dim)
        v = v.reshape(N, self.pooler_res**2, self.num_heads, self.head_dim)

        q = nn.functional.elu(q) + 1
        k = nn.functional.elu(k) + 1

        kv = torch.einsum("nshd,nshd->nhd", k, v)
        k_sum = k.sum(dim=1)

        z = 1.0 / (q * k_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        out = q * kv * z

        out = out.reshape(N, self.feat_channels)
        out = self.out_proj(out)
        out = self.norm1(out)
        out = self.act(out)

        return out.unsqueeze(0)


class LinearSelfAttention(nn.Module):

    def __init__(
        self,
        feat_channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feat_channels = feat_channels
        self.num_heads = num_heads
        self.head_dim = feat_channels // num_heads

        self.q_proj = nn.Linear(feat_channels, feat_channels)
        self.k_proj = nn.Linear(feat_channels, feat_channels)
        self.v_proj = nn.Linear(feat_channels, feat_channels)
        self.out_proj = nn.Linear(feat_channels, feat_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> Tuple[Tensor, None]:
        L, N, C = query.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.reshape(L, N, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        k = k.reshape(L, N, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        v = v.reshape(L, N, self.num_heads, self.head_dim).permute(1, 0, 2, 3)

        q = nn.functional.elu(q) + 1
        k = nn.functional.elu(k) + 1

        kv = torch.einsum("nshd,nshd->nhd", k, v)
        k_sum = k.sum(dim=1)

        z = 1.0 / (q * k_sum.unsqueeze(1)).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        out = q * kv.unsqueeze(1) * z

        out = out.permute(1, 0, 2, 3).reshape(L, N, C)
        out = self.out_proj(out)

        return self.dropout(out), None


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    x = x.permute(0, 2, 3, 1).reshape(1, -1, 3)
    attn = nn.MultiheadAttention(3, 3)
    out = attn(x, x, x)

    print(out[0].shape)
    print(out[1].shape)
