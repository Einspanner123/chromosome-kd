"""SetEncoder — Transformer-based joint set encoder.

Self-Attention (box-box) + Cross-Attention (box-image). This is a Joint
Function Approximator — NOT the contribution of SetDiff. Any architecture
that models x_i's dependence on x_j would work (GNN, MLP-Mixer, etc.).
Transformer is chosen for compatibility with the detection community
(DETR family).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class SetEncoder(nn.Module):
    """Joint Set Encoder: Self-Attn (box-box) + Cross-Attn (box-image).

    Input: noisy boxes ``x_t`` [B, N, 4] (cxcywh, diffusion space), time
    embedding ``t_emb`` [B, dim], image features [B, HW, C].

    Output: ``cls_logits`` [B, N, num_classes], ``pred_boxes`` [B, N, 4]
    (predicted x_0 in diffusion space, NOT velocity).

    方向 C (2026-07-19, per-proposal 退化诊断性消融):
        enable_self_attn=False 时, 用对角 mask 阻断 slot 间交互, 每个 slot
        只 attend to 自己. self-attention 退化为 per-slot 线性变换 (保留
        value/out 投影), 但满足 "slot 间无信息流动" 的核心需求, 等价于
        per-proposal 范式 (对齐 DiffusionDet).

        ⚠️ 对角 mask ≠ 严格跳过 self-attn 层 (跳过层方案需重写
        nn.TransformerDecoderLayer, 改动量大). mask 方案更简洁, 且
        满足方向 C 的诊断目的: 验证 self-attention 是否是 mAP=0 根因.

        互斥处理: enable_self_attn=False 优先于 matched_mask. 对角 mask
        是 matched_mask 的超集 (阻断所有 inter-slot, 包括 unmatched→matched
        路径), 所以 matched_mask 在 enable_self_attn=False 时完全冗余.
    """

    def __init__(
        self,
        num_queries: int,
        feat_channels: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        num_classes: int,
        enable_self_attn: bool = True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        # 方向 C: False 时用对角 mask 阻断 slot 间 self-attention 交互
        self.enable_self_attn = enable_self_attn

        # Box position encoding: MLP(4 → C → C)
        self.box_pos_embed = MLP(
            input_dim=4,
            hidden_dim=feat_channels,
            output_dim=feat_channels,
            num_layers=2,
        )

        # Content query: learnable embedding [N, C]
        self.query_embed = nn.Embedding(num_queries, feat_channels)

        # Transformer Decoder: self-attn (among N boxes) + cross-attn
        # (boxes ↔ image_features).
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_channels,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Output heads
        self.cls_head = nn.Linear(feat_channels, num_classes)
        self.box_head = MLP(
            input_dim=feat_channels,
            hidden_dim=feat_channels,
            output_dim=4,
            num_layers=3,
        )

        # DETR prior_prob init: bias = -log((1-p)/p) so that initial sigmoid ≈ p.
        # With p=0.01, bias ≈ -4.6, which keeps unmatched queries' scores low
        # at start and stabilizes focal loss training.
        prior_prob = 0.01
        bias_init = -math.log((1 - prior_prob) / prior_prob)
        self.cls_head.bias.data = torch.ones(num_classes) * bias_init

    def forward(
        self,
        x_t: Tensor,
        t_emb: Tensor,
        image_features: Tensor,
        matched_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Args:
            x_t: [B, N, 4] noisy boxes (joint state, cxcywh).
            t_emb: [B, dim] time embedding.
            image_features: [B, HW, C] flattened multi-scale features.
            matched_mask: [B, N] or None. True = matched slot (has GT),
                False = unmatched/padding. When provided during training,
                an attention bias is added to prevent unmatched→matched
                noise contamination in self-attention.

        Returns:
            cls_logits: [B, N, num_classes]
            pred_boxes: [B, N, 4] (predicted x_0 in diffusion space)
        """
        B, N, _ = x_t.shape

        # Box position encoding: [B, N, 4] → [B, N, C]
        box_pos = self.box_pos_embed(x_t)

        # Time conditioning via simple addition: box_pos + t_emb
        t = t_emb.unsqueeze(1)  # [B, 1, C]

        # Content query: [N, C] → [B, N, C]
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Combined query for the decoder's tgt.
        tgt = query + box_pos + t  # [B, N, C]

        # Attention mask 构造 (互斥处理, 优先级: enable_self_attn > matched_mask):
        #
        # 方向 C (enable_self_attn=False): 对角 mask 阻断所有 inter-slot 交互,
        #   每个 slot 只 attend to 自己. 是 matched_mask 的超集, 所以
        #   matched_mask 在此模式下完全冗余, 被忽略.
        #
        # 原行为 (enable_self_attn=True + matched_mask): 仅阻断 unmatched→matched
        #   的噪声污染路径, 保留 matched→matched 和 unmatched→unmatched 的交互.
        tgt_mask = None
        if not self.enable_self_attn:
            # 方向 C: 对角 mask, 非对角位置 -inf (每个 slot 只 attend to 自己)
            diag_mask = torch.full(
                (N, N), float('-inf'), device=x_t.device, dtype=x_t.dtype
            )
            diag_mask.fill_diagonal_(0.0)
            # 数值稳定性防御 (C-2): 确保对角线为 0, 避免 softmax NaN
            assert diag_mask.diagonal().abs().max() < 1e-6, (
                "对角 mask 的对角线必须为 0, 否则 softmax 会产生 NaN"
            )
            # PyTorch 2.x 期望 [B * H, N, N] (C-1: 用 repeat_interleave 而非 expand)
            H = self.decoder.layers[0].self_attn.num_heads
            tgt_mask = diag_mask.unsqueeze(0).repeat_interleave(B * H, dim=0)
        elif matched_mask is not None and self.training:
            # 原行为: 阻断 unmatched→matched 的噪声污染
            mm = matched_mask.bool()  # [B, N]
            # block_mask[b, i, j] = True if i is unmatched & j is matched
            block_mask = (~mm.unsqueeze(-1)) & mm.unsqueeze(-2)  # [B, N, N]
            attn_bias = torch.zeros(
                B, N, N, device=x_t.device, dtype=x_t.dtype
            )
            tgt_mask = attn_bias.masked_fill(block_mask, float('-inf'))
            # PyTorch 2.x expects 3D attn_mask shape [B * H, N, N].
            H = self.decoder.layers[0].self_attn.num_heads
            tgt_mask = tgt_mask.repeat_interleave(H, dim=0)

        # Transformer Decoder: self-attn (tgt ↔ tgt) + cross-attn
        # (tgt ↔ image_features).
        # tgt_mask of shape [B*H, N, N] (expanded from [B, N, N]).
        hs = self.decoder(
            tgt, image_features, tgt_mask=tgt_mask
        )  # [B, N, C]

        cls_logits = self.cls_head(hs)  # [B, N, num_classes]
        pred_boxes = self.box_head(hs)  # [B, N, 4]
        return cls_logits, pred_boxes
