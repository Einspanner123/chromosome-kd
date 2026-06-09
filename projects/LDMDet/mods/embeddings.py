"""位置编码模块"""

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码模块，用于时间步编码"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: Tensor) -> Tensor:
        """
        Args:
            timesteps: [batch_size] (float or int)
        Returns:
            [batch_size, dim]
        """
        device = timesteps.device
        half_dim = self.dim // 2

        exponent = -math.log(10000) / (half_dim - 1)
        i = torch.arange(half_dim, device=device, dtype=timesteps.dtype)
        freqs = torch.exp(i * exponent)

        args = torch.outer(timesteps, freqs)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


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
