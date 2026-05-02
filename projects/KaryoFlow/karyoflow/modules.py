"""
KaryoFlow 基础模块 (自包含，不依赖外部项目)

- SinusoidalPositionEmbeddings: 正弦时间步编码
- AdaLNZeroLayer: AdaLN-Zero 调制的 Transformer 解码层
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码，用于编码连续时间步 t ∈ [0, 1]

    输出维度 = dim (偶数)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (B,) 时间步

        Returns:
            emb: (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device, dtype=t.dtype) * exponent)
        args = torch.outer(t, freqs)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AdaLNZeroLayer(nn.Module):
    """AdaLN-Zero 调制的 Transformer 解码层

    结构: Self-Attn → Cross-Attn → FFN
    每个子块前用 AdaLN 调制，gate 零初始化 → 初始时为恒等映射

    Args:
        d_model: 特征维度
        nhead: 注意力头数
        dim_feedforward: FFN 中间层维度
        dropout: dropout 率
        time_dim: 时间嵌入维度 (通常 = 4 * d_model)
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        time_dim: int = 1024,
    ):
        super().__init__()

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # LayerNorm (用于 AdaLN 调制前的归一化)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # AdaLN-Zero: 9 组参数
        # Self-Attn: γ₁, β₁, α₁
        # Cross-Attn: γ₂, β₂, α₂
        # FFN: γ₃, β₃, α₃
        self.adaln_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, d_model * 9),
        )

        # 初始化: gate (α) 设为小正数而非零，确保信息从一开始就能流过
        # AdaLN-Zero 原版用零初始化使初始层为恒等映射，
        # 但这会导致 cross-attention 完全不贡献梯度，在小数据场景下训练极慢
        nn.init.zeros_(self.adaln_mlp[-1].weight)
        nn.init.zeros_(self.adaln_mlp[-1].bias)
        # 对 gate 位 (α₁, α₂, α₃) 的 bias 初始化为 0.1
        d = d_model
        with torch.no_grad():
            bias = self.adaln_mlp[-1].bias
            # 9 组 * d_model: [γ₁, β₁, α₁, γ₂, β₂, α₂, γ₃, β₃, α₃]
            # α₁ at [2d:3d], α₂ at [5d:6d], α₃ at [8d:9d]
            bias[2*d:3*d] = 0.1   # α₁ (self-attn gate)
            bias[5*d:6*d] = 0.1   # α₂ (cross-attn gate)
            bias[8*d:9*d] = 0.1   # α₃ (FFN gate)

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        time_emb: Tensor,
        self_attn_mask: Tensor = None,
        cross_attn_mask: Tensor = None,
    ) -> Tensor:
        """
        Args:
            query: (B, N, d) — 排列 tokens
            memory: (B, M, d) — 染色体特征
            time_emb: (B, time_dim) — 时间嵌入

        Returns:
            query: (B, N, d) — 更新后的排列 tokens
        """
        # 计算 AdaLN 参数
        params = self.adaln_mlp(time_emb)  # (B, 9*d)
        params = params.unsqueeze(1)  # (B, 1, 9*d)
        g1, b1, a1, g2, b2, a2, g3, b3, a3 = params.chunk(9, dim=-1)

        # === Self-Attention + AdaLN-Zero ===
        q_norm = self.norm1(query) * (1 + g1) + b1
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm, attn_mask=self_attn_mask)
        query = query + a1 * sa_out

        # === Cross-Attention + AdaLN-Zero ===
        q_norm = self.norm2(query) * (1 + g2) + b2
        ca_out, _ = self.cross_attn(q_norm, memory, memory, attn_mask=cross_attn_mask)
        query = query + a2 * ca_out

        # === FFN + AdaLN-Zero ===
        q_norm = self.norm3(query) * (1 + g3) + b3
        ffn_out = self.ffn(q_norm)
        query = query + a3 * ffn_out

        return query
