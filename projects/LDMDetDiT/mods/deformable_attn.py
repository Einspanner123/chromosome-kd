import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def bilinear_sample_by_ref_points(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    reference_points: Tensor,
    sampling_locations: Tensor,
    num_heads: int,
) -> Tensor:
    """从多尺度特征图中按参考点+偏移做双线性插值采样

    Args:
        value: (bs, num_value, num_heads * embed_dim_per_head) 展平的多尺度特征
        spatial_shapes: (num_levels, 2) 各层分辨率 [H, W]
        level_start_index: (num_levels,) 各层在 value 中的起始索引
        reference_points: (bs, num_query, num_levels, 2) 归一化参考点 [0,1]
        sampling_locations: (bs, num_query, num_heads, num_levels, num_points, 2)
            归一化采样位置 [0,1]
        num_heads: 注意力头数

    Returns:
        output: (bs, num_query, num_heads * num_levels * num_points * embed_dim_per_head)
    """
    bs = value.shape[0]
    num_query = sampling_locations.shape[1]
    num_levels = spatial_shapes.shape[0]
    num_points = sampling_locations.shape[-2]

    # value: (bs, num_value, num_heads, embed_dim_per_head)
    embed_dim_per_head = value.shape[-1] // num_heads
    value = value.view(bs, -1, num_heads, embed_dim_per_head)

    # 逐层采样
    sampled_features = []
    for level_idx in range(num_levels):
        h, w = spatial_shapes[level_idx]
        start = level_start_index[level_idx]
        end = start + h * w

        # 当前层特征: (bs, h*w, num_heads, embed_dim_per_head)
        value_level = value[:, start:end]
        # reshape 为 2D 空间: (bs, num_heads, embed_dim_per_head, h, w)
        value_level = value_level.transpose(1, 2).reshape(
            bs, num_heads, embed_dim_per_head, h, w
        )

        # 当前层采样位置: (bs, num_query, num_heads, num_points, 2)
        # reference_points: (bs, num_query, num_levels, 2)
        sampling_loc_level = sampling_locations[:, :, :, level_idx, :, :]
        # (bs, num_query, num_heads, num_points, 2)

        # 对每个 head 做 grid_sample
        # 需要将 shape 转为 (bs*num_heads, embed_dim_per_head, h, w)
        value_2d = value_level.reshape(
            bs * num_heads, embed_dim_per_head, h, w
        )

        # grid_sample 需要 (N, H_out, W_out, 2) 的 grid
        # sampling_loc_level: (bs, num_query, num_heads, num_points, 2)
        # → (bs*num_heads, num_query*num_points, 2)
        grid = sampling_loc_level.permute(0, 2, 1, 3, 4).reshape(
            bs * num_heads, num_query * num_points, 2
        )
        # grid_sample 需要 (N, H_out, W_out, 2)
        grid = grid.reshape(bs * num_heads, num_query, num_points, 2)

        # IMPORTANT: Convert normalized [0, 1] sampling locations to grid space [-1, 1]
        grid = grid * 2.0 - 1.0

        # F.grid_sample: input (N,C,H_in,W_in), grid (N,H_out,W_out,2)
        sampled = F.grid_sample(
            value_2d,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        # sampled: (bs*num_heads, embed_dim_per_head, num_query, num_points)
        sampled = sampled.reshape(
            bs, num_heads, embed_dim_per_head, num_query, num_points
        )
        # → (bs, num_query, num_heads, num_points, embed_dim_per_head)
        sampled = sampled.permute(0, 3, 1, 4, 2)
        sampled_features.append(sampled)

    # (bs, num_query, num_heads, num_levels*num_points, embed_dim_per_head)
    output = torch.stack(sampled_features, dim=3).reshape(
        bs, num_query, num_heads, num_levels * num_points, embed_dim_per_head
    )
    return output


class MultiScaleDeformableAttention(nn.Module):
    """纯 PyTorch 实现的多尺度可变形注意力

    每个 Query 从多尺度特征图中采样 K 个点，
    通过可学习的偏移和注意力权重聚合特征。

    复杂度: O(N_query × K × num_levels) 而非 O(N_query × N_img)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 8,
        dropout: float = 0.0,
        offset_scale: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        self.offset_scale = offset_scale

        assert embed_dim % num_heads == 0, (
            f'embed_dim {embed_dim} must be divisible by num_heads {num_heads}'
        )

        self.sampling_offsets = nn.Linear(
            embed_dim, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dim, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

        # attention_weights 初始化为均匀分布
        nn.init.xavier_uniform_(self.attention_weights.weight)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 1, 2).repeat(
            1, self.num_levels, self.num_points, 1, 1
        )
        grid_init = grid_init.reshape(-1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init)

        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        bbox_coords: Tensor = None,
    ) -> Tensor:
        """前向传播

        Args:
            query: (bs, num_query, embed_dim) - Box Tokens
            reference_points: (bs, num_query, num_levels, 2) - 归一化参考点 [0,1]
            value: (bs, num_value, embed_dim) - 展平的多尺度特征
            spatial_shapes: (num_levels, 2) - 各层分辨率 [H, W]
            level_start_index: (num_levels,) - 各层起始索引
            bbox_coords: (bs, num_query, 4) - 归一化 xyxy 坐标 [0,1]，用于尺度感知采样

        Returns:
            output: (bs, num_query, embed_dim)
        """
        bs, num_query, _ = query.shape

        # Value 投影
        value_proj = self.value_proj(value)
        # (bs, num_value, num_heads * head_dim)

        # 预测采样偏移
        offsets = self.sampling_offsets(query)
        # (bs, num_query, num_heads * num_levels * num_points * 2)
        offsets = offsets.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        # 预测注意力权重
        attn_weights = self.attention_weights(query)
        # (bs, num_query, num_heads * num_levels * num_points)
        attn_weights = attn_weights.view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        # 计算采样位置 = 参考点 + 偏移
        # reference_points: (bs, num_query, num_levels, 2) → 扩展到 (bs, num_query, 1, num_levels, 1, 2)
        ref_points_expanded = reference_points.unsqueeze(2).unsqueeze(4)
        # 尺度感知采样: 偏移量与 bbox 宽高成正比
        # 大框采样范围大，小框采样范围小，避免小目标采样点大量落到目标外部
        if bbox_coords is not None:
            # bbox_coords: (bs, num_query, 4) xyxy [0,1] → box_wh: (bs, num_query, 2)
            box_wh = (bbox_coords[..., 2:] - bbox_coords[..., :2]).clamp(min=1e-4)
            # box_wh: (bs, num_query, 1, 1, 1, 2) 用于广播到 offsets shape
            box_wh = box_wh.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            offsets = offsets.tanh() * box_wh * self.offset_scale
        else:
            # 回退: 固定半径采样
            offsets = offsets.tanh() * 0.1
        sampling_locations = ref_points_expanded + offsets
        # (bs, num_query, num_heads, num_levels, num_points, 2)

        # 双线性插值采样
        sampled = bilinear_sample_by_ref_points(
            value_proj,
            spatial_shapes,
            level_start_index,
            reference_points,
            sampling_locations,
            self.num_heads,
        )
        # (bs, num_query, num_heads, num_levels*num_points, head_dim)

        # 加权求和
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        # (bs, num_query, num_heads, num_levels, num_points, 1)
        attn_weights_flat = attn_weights_expanded.reshape(
            bs, num_query, self.num_heads, self.num_levels * self.num_points, 1
        )

        output = (sampled * attn_weights_flat).sum(dim=3)
        # (bs, num_query, num_heads, head_dim)
        output = output.reshape(bs, num_query, self.embed_dim)

        # 输出投影
        output = self.output_proj(output)
        output = self.dropout(output)

        return output


def build_fpn_level_info(
    spatial_shapes: list,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """构建 FPN 层级信息

    Args:
        spatial_shapes: [(H_0, W_0), (H_1, W_1), ...] 各层分辨率
        device: 设备

    Returns:
        spatial_shapes_tensor: (num_levels, 2)
        level_start_index: (num_levels,)
    """
    shapes_tensor = torch.tensor(
        spatial_shapes, device=device, dtype=torch.long
    )
    level_start_index = torch.zeros(
        len(spatial_shapes), device=device, dtype=torch.long
    )
    for i in range(1, len(spatial_shapes)):
        level_start_index[i] = (
            level_start_index[i - 1]
            + spatial_shapes[i - 1][0] * spatial_shapes[i - 1][1]
        )
    return shapes_tensor, level_start_index


def flatten_fpn_features(
    fpn_features: list,
) -> Tuple[Tensor, Tensor, Tensor]:
    """将 FPN 多尺度特征展平为统一序列

    Args:
        fpn_features: List[(bs, C, H_l, W_l)] P2-P5 特征

    Returns:
        flattened: (bs, sum(H_l*W_l), C)
        spatial_shapes: (num_levels, 2)
        level_start_index: (num_levels,)
    """
    spatial_shapes = []
    level_start_index = []
    flattened_parts = []
    start = 0

    for feat in fpn_features:
        bs, c, h, w = feat.shape
        spatial_shapes.append((h, w))
        level_start_index.append(start)
        start += h * w
        flattened_parts.append(feat.flatten(2).transpose(1, 2))

    flattened = torch.cat(flattened_parts, dim=1)
    spatial_shapes_tensor = torch.tensor(
        spatial_shapes, device=feat.device, dtype=torch.long
    )
    level_start_index_tensor = torch.tensor(
        level_start_index, device=feat.device, dtype=torch.long
    )

    return flattened, spatial_shapes_tensor, level_start_index_tensor
