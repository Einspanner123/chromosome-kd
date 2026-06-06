import torch
import torch.nn as nn
from torch import Tensor

from .box_tokenizer import (
    bbox_to_reference_points,
    reference_points_with_levels,
)
from .deformable_attn import MultiScaleDeformableAttention
from .modules import RoPE1D, apply_rope


class DiTBlock(nn.Module):
    """DiT Block: Self-Attention + Deformable Cross-Attention + FFN

    全部使用 AdaLN-Zero 时间条件化 (9 参数: 3 子块 × 3 参数)
    备选: 6 参数 (2×3), 共享 Self-Attn 和 Cross-Attn 的调制

    AdaLN-Zero 参数:
      (γ1, β1, α1) for Self-Attention
      (γ2, β2, α2) for Deformable Cross-Attention
      (γ3, β3, α3) for FFN
    """

    def __init__(
        self,
        feat_channels: int = 256,
        num_heads: int = 8,
        num_fpn_levels: int = 4,
        num_ref_points: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        adaln_params: int = 9,
        use_adaln_zero: bool = True,
    ):
        """
        Args:
            feat_channels: 特征通道数
            num_heads: 注意力头数
            num_fpn_levels: FPN 层级数
            num_ref_points: 每个查询的采样点数 K
            dim_feedforward: FFN 隐藏层维度
            dropout: dropout 率
            adaln_params: AdaLN-Zero 参数组数, 9 (3×3) 或 6 (2×3)
            use_adaln_zero: 是否使用零初始化。如果为 False，则初始化为小随机数或 1.0，
                           让梯度在初期就能流过残差路径。
        """
        super().__init__()
        self.feat_channels = feat_channels
        self.num_heads = num_heads
        self.num_fpn_levels = num_fpn_levels
        self.adaln_params = adaln_params
        self.use_adaln_zero = use_adaln_zero

        self.dropout = dropout

        self.qkv = nn.Linear(feat_channels, feat_channels * 3)
        self.proj = nn.Linear(feat_channels, feat_channels)
        self.attn_drop = nn.Dropout(dropout)

        self.cross_attn = MultiScaleDeformableAttention(
            embed_dim=feat_channels,
            num_heads=num_heads,
            num_levels=num_fpn_levels,
            num_points=num_ref_points,
            dropout=dropout,
        )

        self.ffn = nn.Sequential(
            nn.Linear(feat_channels, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, feat_channels),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(feat_channels)
        self.norm2 = nn.LayerNorm(feat_channels)
        self.norm3 = nn.LayerNorm(feat_channels)

        # RoPE 旋转位置编码 (Sora/DiT 风格，对齐 DINOv3)
        # DINOv3 使用的是标准 RoPE，基数为 10000
        self.rope = RoPE1D(feat_channels // num_heads, base=10000)

        # 调制参数映射 (AdaLN)
        time_dim = feat_channels * 4
        self.adaln_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, feat_channels * adaln_params),
        )

        # 初始化逻辑
        if use_adaln_zero:
            nn.init.zeros_(self.adaln_mlp[-1].weight)
            nn.init.zeros_(self.adaln_mlp[-1].bias)
        else:
            # 即使不使用零初始化，也使用较小的初始化值以保证训练初期稳定
            nn.init.xavier_uniform_(self.adaln_mlp[-1].weight, gain=0.02)
            nn.init.constant_(self.adaln_mlp[-1].bias, 0.0)

    def _modulate(self, x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        """AdaLN 调制: x * (1 + gamma) + beta"""
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def forward(
        self,
        box_tokens: Tensor,
        fpn_flattened: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        time_emb: Tensor,
        bbox_coords: Tensor,
    ) -> Tensor:
        """前向传播

        Args:
            box_tokens: (bs, N, C) Box Tokens
            fpn_flattened: (bs, sum(H_l*W_l), C) 展平的多尺度特征
            spatial_shapes: (num_levels, 2) 各层分辨率
            level_start_index: (num_levels,) 各层起始索引
            time_emb: (bs, time_dim) 时间嵌入
            bbox_coords: (bs, N, 4) 归一化 xyxy 坐标

        Returns:
            box_tokens: (bs, N, C) 更新后的 Box Tokens
        """
        bs, num_query, _ = box_tokens.shape

        if self.adaln_params == 9:
            params = self.adaln_mlp(time_emb)
            g1, b1, a1, g2, b2, a2, g3, b3, a3 = params.chunk(9, dim=-1)
        else:
            params = self.adaln_mlp(time_emb)
            g1, b1, a1, g2, b2, a2 = params.chunk(6, dim=-1)
            g3, b3, a3 = g2, b2, a2

        # 1. Self-Attention + AdaLN-Zero + RoPE
        x = self.norm1(box_tokens)
        x = self._modulate(x, g1, b1)

        # 应用 RoPE 到 Self-Attention
        # bbox_coords: (bs, N, 4) -> RoPE: (bs, N, 4, C/H) cat(sin, cos)
        rope_emb = self.rope(bbox_coords)  # (bs, N, 4, C/H)
        dim_half = rope_emb.shape[-1] // 2
        rope_sin = rope_emb[..., :dim_half]  # (bs, N, 4, dim_half)
        rope_cos = rope_emb[..., dim_half:]  # (bs, N, 4, dim_half)

        # 按坐标维度平均, 然后重复以匹配 head_dim
        cos = rope_cos.mean(dim=2)  # (bs, N, dim_half)
        sin = rope_sin.mean(dim=2)  # (bs, N, dim_half)
        cos = torch.cat([cos, cos], dim=-1).unsqueeze(1)  # (bs, 1, N, C/H)
        sin = torch.cat([sin, sin], dim=-1).unsqueeze(1)  # (bs, 1, N, C/H)

        # 手动实现带 RoPE 的 Self-Attention
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, C/H)

        q, k = apply_rope(q, k, cos, sin)

        # Attention
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.proj(attn_out)
        attn_out = self.attn_drop(attn_out)

        box_tokens = box_tokens + a1.unsqueeze(1) * attn_out
        box_tokens = torch.nan_to_num(box_tokens, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Deformable Cross-Attention
        x = self.norm2(box_tokens)
        x = self._modulate(x, g2, b2)
        ref_points = bbox_to_reference_points(bbox_coords)
        ref_points = reference_points_with_levels(
            ref_points, self.num_fpn_levels
        )
        cross_out = self.cross_attn(
            query=x,
            reference_points=ref_points,
            value=fpn_flattened,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        box_tokens = box_tokens + a2.unsqueeze(1) * cross_out
        box_tokens = torch.nan_to_num(box_tokens, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. FFN + AdaLN-Zero
        x = self.norm3(box_tokens)
        x = self._modulate(x, g3, b3)
        ffn_out = self.ffn(x)
        box_tokens = box_tokens + a3.unsqueeze(1) * ffn_out
        box_tokens = torch.nan_to_num(box_tokens, nan=0.0, posinf=0.0, neginf=0.0)

        return box_tokens
