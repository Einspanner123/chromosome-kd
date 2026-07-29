"""简化版 DINO Transformer — DiffuDETR 移植到 mmdet

从原仓库 (MBadran2000/DiffuDETR) 的 dino_transformer.py + layers_diffu_detr/transformer.py
移植, 做以下简化以适配当前环境 (PyTorch 2.1 + cu118, 无 detrex/detectron2):

  1. 用标准 nn.MultiheadAttention 替代 MultiScaleDeformableAttention
     (当前环境无 MMCV 的 MSDeformableAttn 实现)
  2. 去除 two-stage proposal 生成, 改用 learned query embeddings
  3. 去除 fairscale checkpoint_wrapper (可选依赖)
  4. 保留核心: TimeStepBlock 每层注入 + 中间层输出 (deep supervision)

关键设计决策:
  - num_layers=6, embed_dim=256, num_heads=8, dim_feedforward=2048
  - 时间步注入: time_embed (SiLU MLP, embed_dim → 4*embed_dim) +
    每层 TimeStepBlock (FiLM 调制) 在 self_attn/cross_attn/ffn 前注入
  - 查询初始化: tgt_embed (learned) + ref_point_head(sine_embed(noisy_boxes))
  - 返回中间层特征供 head 做 deep supervision + x0 预测
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_scheduler import timestep_embedding
from .timestep_block import TimeStepBlock


class MLP(nn.Module):
    """多层感知机 (对齐 detrex.layers.MLP).

    用于边界框预测 (box regression head).

    Args:
        input_dim: 输入维度.
        hidden_dim: 隐藏层维度.
        output_dim: 输出维度.
        num_layers: 层数 (含输出层).
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim, *h], [*h, output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FFN(nn.Module):
    """前馈网络 (对齐 detrex.layers.FFN).

    Args:
        embed_dim: 输入/输出维度.
        feedforward_dim: 中间层维度.
        output_dim: 输出维度 (默认 = embed_dim).
        num_fcs: 全连接层数 (默认 2).
        ffn_drop: dropout 概率.
    """

    def __init__(
        self,
        embed_dim: int,
        feedforward_dim: int,
        output_dim: Optional[int] = None,
        num_fcs: int = 2,
        ffn_drop: float = 0.1,
    ):
        super().__init__()
        output_dim = output_dim or embed_dim
        layers = []
        in_dim = embed_dim
        for _ in range(num_fcs - 1):
            layers.append(nn.Linear(in_dim, feedforward_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(ffn_drop))
            in_dim = feedforward_dim
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, identity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """FFN 前向 + 残差.

        Args:
            x: [B, N, C] 输入.
            identity: 残差连接的 identity, None 则用 x.
        """
        out = self.layers(x)
        if identity is None:
            identity = x
        return identity + out


class MultiheadAttention(nn.Module):
    """标准多头注意力包装 (对齐 detrex MultiheadAttention).

    用 nn.MultiheadAttention 替代, 支持 identity 残差 + 位置嵌入.

    Args:
        embed_dim: 嵌入维度.
        num_heads: 注意力头数.
        attn_drop: attention dropout.
        proj_drop: 输出 dropout.
        batch_first: 输入格式 (bs, n, c) 还是 (n, bs, c).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        batch_first: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,
            **kwargs,
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """前向: 加位置嵌入 → nn.MultiheadAttention → 残差.

        对齐原仓库 _full_attention.py MultiheadAttention.forward.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None and query_pos is not None:
            if query_pos.shape == key.shape:
                key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        return identity + self.proj_drop(out)


def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
) -> torch.Tensor:
    """生成正弦位置编码 (对齐 detrex get_sine_pos_embed).

    Args:
        pos_tensor: [..., 2 or 4] 归一化坐标 (reference points).
        num_pos_feats: 每个维度的编码特征数 (默认 128, 2*128=256=embed_dim).
        temperature: 温度参数.

    Returns:
        [..., num_pos_feats*2 or *4] 正弦位置编码.
    """
    scale = 2 * math.pi
    dim_t = torch.arange(
        num_pos_feats, dtype=torch.float32, device=pos_tensor.device
    )
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos_tensor[..., 0] * scale
    pos_y = pos_tensor[..., 1] * scale
    pos_x = pos_x[..., None] / dim_t
    pos_y = pos_y[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    if pos_tensor.size(-1) == 2:
        pos_embed = torch.cat((pos_y, pos_x), dim=-1)
    elif pos_tensor.size(-1) == 4:
        pos_w = pos_tensor[..., 2] * scale
        pos_h = pos_tensor[..., 3] * scale
        pos_w = pos_w[..., None] / dim_t
        pos_h = pos_h[..., None] / dim_t
        pos_w = torch.stack(
            (pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1
        ).flatten(-2)
        pos_h = torch.stack(
            (pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1
        ).flatten(-2)
        pos_embed = torch.cat((pos_y, pos_x, pos_h, pos_w), dim=-1)
    else:
        raise ValueError(f'Unknown pos_tensor size: {pos_tensor.size(-1)}')
    return pos_embed


class DiffuDETRDecoderLayer(nn.Module):
    """简化版 DINO Decoder 层 (含 TimeStepBlock 注入).

    对齐原仓库 BaseTransformerLayer 的 operation_order:
        ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")

    每个子操作前注入 TimeStepBlock (FiLM 调制时间步信息), 对齐原仓库
    BaseTransformerLayer.forward 中 t_index 递增逻辑.

    Args:
        embed_dim: 嵌入维度 (256).
        num_heads: 注意力头数 (8).
        dim_feedforward: FFN 中间维度 (2048).
        dropout: dropout 概率.
        time_embed_dim: 时间步嵌入维度 (4*embed_dim = 1024).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        time_embed_dim: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # 自注意力 + 交叉注意力 (标准 nn.MultiheadAttention)
        self.self_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=dropout,
            batch_first=True,
        )
        self.cross_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=dropout,
            batch_first=True,
        )
        self.ffn = FFN(
            embed_dim=embed_dim,
            feedforward_dim=dim_feedforward,
            output_dim=embed_dim,
            ffn_drop=dropout,
        )

        # 三个 TimeStepBlock: 分别在 self_attn / cross_attn / ffn 前注入
        # 对齐原仓库 time_step_embeds (num_attn + num_ffns = 3 个)
        self.time_step_embeds = nn.ModuleList(
            [
                TimeStepBlock(
                    channels=embed_dim,
                    emb_channels=time_embed_dim,
                    out_channels=embed_dim,
                )
                for _ in range(3)
            ]
        )

        # LayerNorm (post-norm, 对齐 DINO operation_order)
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(3)])

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        time_embed: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decoder 层前向.

        流程 (对齐原仓库 BaseTransformerLayer):
          1. TimeStepBlock(target, time) → self_attn → norm
          2. TimeStepBlock(target, time) → cross_attn → norm
          3. TimeStepBlock(target, time) → ffn → norm

        Args:
            target: [B, N, C] 查询特征.
            memory: [B, HW, C] 编码器/图像特征.
            query_pos: [B, N, C] 查询位置嵌入.
            key_pos: [B, HW, C] 键位置嵌入 (图像位置编码).
            time_embed: [B, time_ch] 时间步嵌入.
            attn_mask: 注意力掩码 (DN-DETR 用).
            key_padding_mask: [B, HW] 键填充掩码.

        Returns:
            [B, N, C] 更新后的查询特征.
        """
        # 1. self_attn (注入 time_step → self_attn → norm)
        if time_embed is not None:
            target = self.time_step_embeds[0](target, time_embed)
        target = self.self_attn(
            query=target,
            key=target,
            value=target,
            identity=target,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=attn_mask,
        )
        target = self.norms[0](target)

        # 2. cross_attn (注入 time_step → cross_attn → norm)
        if time_embed is not None:
            target = self.time_step_embeds[1](target, time_embed)
        target = self.cross_attn(
            query=target,
            key=memory,
            value=memory,
            identity=target,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
        )
        target = self.norms[1](target)

        # 3. ffn (注入 time_step → ffn → norm)
        if time_embed is not None:
            target = self.time_step_embeds[2](target, time_embed)
        target = self.ffn(target, identity=target)
        target = self.norms[2](target)

        return target


class DiffuDETRTransformer(nn.Module):
    """简化版 DINO Transformer (仅 Decoder, 无 Encoder/two-stage).

    对齐原仓库 DINOTransformer + DINOTransformerDecoder, 简化:
      - 无 encoder (图像特征直接 flatten 作为 memory)
      - 无 two-stage proposal (用 learned tgt_embed)
      - 无 deformable attention (标准 attention)
      - 保留: time_embed + 每层 TimeStepBlock + 中间层输出

    Args:
        embed_dim: 嵌入维度 (256).
        num_heads: 注意力头数 (8).
        dim_feedforward: FFN 中间维度 (2048).
        num_layers: decoder 层数 (6).
        num_queries: 查询数 (300).
        num_feature_levels: 特征层数 (用于 level_embed).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 6,
        num_queries: int = 300,
        num_feature_levels: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels

        # 学习的查询内容嵌入 (对齐原仓库 tgt_embed)
        self.tgt_embed = nn.Embedding(num_queries, embed_dim)

        # reference point → query_pos (对齐原仓库 ref_point_head)
        # sine_embed 维度 = 2*embed_dim (4 坐标 × embed_dim/2)
        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)

        # 时间步嵌入: timestep_embedding → time_embed MLP
        # 对齐原仓库: linear(embed_dim, 4*embed_dim) → SiLU → linear(4*embed_dim, 4*embed_dim)
        time_embed_dim = embed_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Decoder 层
        self.layers = nn.ModuleList(
            [
                DiffuDETRDecoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    time_embed_dim=time_embed_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # 层级嵌入 (对齐原仓库 level_embeds)
        self.level_embeds = nn.Parameter(
            torch.Tensor(num_feature_levels, embed_dim)
        )

        # 最终 norm (对齐原仓库 decoder.norm)
        self.norm = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self) -> None:
        """初始化权重 (对齐原仓库 init_weights)."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.level_embeds)

    def _flatten_features(
        self, multi_level_feats: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """展平多尺度特征: List[[B,C,H,W]] → [B, sum(HW), C] + spatial_shapes.

        对齐原仓库 DINOTransformer.forward 中的 flatten 逻辑.
        """
        feat_list = []
        spatial_shapes = []
        for lvl, feat in enumerate(multi_level_feats):
            bs, c, h, w = feat.shape
            spatial_shapes.append((h, w))
            # [B,C,H,W] → [B,HW,C] + level_embed
            feat = feat.flatten(2).transpose(1, 2)
            feat = feat + self.level_embeds[lvl].view(1, 1, -1)
            feat_list.append(feat)
        memory = torch.cat(feat_list, dim=1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=memory.device
        )
        return memory, spatial_shapes

    def _get_key_pos(
        self,
        spatial_shapes: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """生成图像特征的正弦位置编码 (key_pos).

        对齐原仓库 multi_level_pos_embeds 的作用.
        """
        pos_list = []
        for h, w in spatial_shapes:
            # 生成归一化网格坐标 [h*w, 2]
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, device=device),
                torch.linspace(0.5, w - 0.5, w, device=device),
                indexing='ij',
            )
            ref = torch.stack(
                (ref_x.reshape(-1) / w, ref_y.reshape(-1) / h), dim=-1
            )
            pos = get_sine_pos_embed(ref, num_pos_feats=self.embed_dim // 2)
            pos_list.append(pos)
        key_pos = torch.cat(pos_list, dim=0)  # [sum(HW), embed_dim]
        key_pos = key_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        return key_pos

    def forward(
        self,
        multi_level_feats: List[torch.Tensor],
        noisy_boxes: torch.Tensor,
        time_steps: torch.Tensor,
        scale: float = 2.0,
    ) -> torch.Tensor:
        """Transformer 前向.

        对齐原仓库 DINOTransformer.forward (简化):
          1. flatten 多尺度特征 → memory
          2. noisy_boxes (扩散空间 [-scale,scale]) → 归一化 [0,1] → reference_points
          3. query_pos = ref_point_head(sine_embed(reference_points))
          4. target = tgt_embed (learned)
          5. time_emb = time_embed(timestep_embedding(time_steps))
          6. decoder 逐层处理, 收集中间输出

        Args:
            multi_level_feats: List[[B,C,H,W]] 多尺度图像特征.
            noisy_boxes: [B, N, 4] 噪声框 (扩散空间 [-scale, scale]).
            time_steps: [B] 时间步.
            scale: 扩散空间缩放因子.

        Returns:
            inter_states: [num_layers, B, N, C] 中间层特征 (deep supervision).
        """
        bs = noisy_boxes.shape[0]
        device = noisy_boxes.device

        # 1. 展平多尺度特征 → memory [B, sum(HW), C]
        memory, spatial_shapes = self._flatten_features(multi_level_feats)

        # 2. key_pos: 图像特征位置编码
        key_pos = self._get_key_pos(spatial_shapes, bs, device)

        # 3. noisy_boxes (扩散空间) → 归一化 [0,1] → reference_points
        # 对齐 setdiff_detector.diffusion_to_norm_space:
        #   (clamp(x, -s, s) / s + 1) / 2
        s = scale
        ref = (noisy_boxes.clamp(-s, s) / s + 1.0) / 2.0  # [B, N, 4] in [0,1]

        # 4. query_pos = ref_point_head(sine_embed(reference_points))
        # sine_embed 维度 = 2*embed_dim (4 坐标 × embed_dim/2)
        query_sine = get_sine_pos_embed(ref, num_pos_feats=self.embed_dim // 2)
        query_pos = self.ref_point_head(query_sine)  # [B, N, C]

        # 5. target = learned query embeddings
        target = self.tgt_embed.weight.unsqueeze(0).repeat(
            bs, 1, 1
        )  # [B, N, C]

        # 6. 时间步嵌入: timestep_embedding → time_embed MLP
        time_emb = timestep_embedding(time_steps.float(), self.embed_dim)
        time_emb = self.time_embed(time_emb)  # [B, 4*embed_dim]

        # 7. decoder 逐层处理, 收集中间输出 (deep supervision)
        intermediate = []
        for layer in self.layers:
            target = layer(
                target=target,
                memory=memory,
                query_pos=query_pos,
                key_pos=key_pos,
                time_embed=time_emb,
            )
            intermediate.append(self.norm(target))

        # [num_layers, B, N, C]
        return torch.stack(intermediate)
