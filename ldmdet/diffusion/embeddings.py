"""正弦位置编码 (时间步编码)"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码，用于时间步 → 高维特征。

    PE(t, 2i)   = sin(t * 10000^{-2i/d})
    PE(t, 2i+1) = cos(t * 10000^{-2i/d})
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: Tensor) -> Tensor:
        """Args: timesteps [B] → Returns: [B, dim]"""
        device = timesteps.device
        half_dim = self.dim // 2
        exponent = -math.log(10000) / (half_dim - 1)
        i = torch.arange(half_dim, device=device, dtype=timesteps.dtype)
        freqs = torch.exp(i * exponent)
        args = torch.outer(timesteps, freqs)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
