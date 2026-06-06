import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class RoPE1D(nn.Module):
    """1D 旋转位置编码 (Rotary Positional Embedding)

    用于对 Box 坐标 (x, y, w, h) 进行编码。
    相比于简单的线性映射，RoPE 能够保持相对位置的旋转不变性。
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (bs, N, 4) 归一化坐标
        Returns:
            pe: (bs, N, 4, dim) 坐标嵌入
        """
        # 将坐标映射到不同频率
        # x: (bs, N, 4, 1), inv_freq: (dim/2)
        # out: (bs, N, 4, dim/2)
        sinusoid_inp = torch.einsum('bni,j->bnij', x, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb


def rotate_half(x: Tensor) -> Tensor:
    """旋转一半的维度，用于 RoPE 计算"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> Tuple[Tensor, Tensor]:
    """将旋转编码应用到 Query 和 Key 上"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def cosine_noise_schedule(T, s=0.008, device='cpu'):
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


class PurePyTorchSimpleFeatureFusion(nn.Module):
    """简单的特征融合模块，用于 ViT 输出的多级特征对齐"""

    def __init__(
        self, in_channels: List[int], out_channels: int, num_outs: int = 4
    ):
        super().__init__()
        # 对应 ViT 默认输出 (H/16, W/16)，我们需要通过反卷积或上采样还原多尺度
        # P2: 1/4, P3: 1/8, P4: 1/16, P5: 1/32
        self.upsample_p2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels[0], out_channels, kernel_size=2, stride=2
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=2, stride=2
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.upsample_p3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels[1], out_channels, kernel_size=2, stride=2
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.p4_conv = nn.Sequential(
            nn.Conv2d(in_channels[2], out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.downsample_p5 = nn.Sequential(
            nn.Conv2d(in_channels[3], out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.num_outs = num_outs

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        # x[i] shape: (B, 384, H/16, W/16)
        p2 = self.upsample_p2(x[0])
        p3 = self.upsample_p3(x[1])
        p4 = self.p4_conv(x[2])
        p5 = self.downsample_p5(x[3])
        return [p2, p3, p4, p5]


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


if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256))
    x = x.permute(0, 2, 3, 1).reshape(1, -1, 3)
    attn = nn.MultiheadAttention(3, 3)
    out = attn(x, x, x)

    print(out[0].shape)
    print(out[1].shape)
