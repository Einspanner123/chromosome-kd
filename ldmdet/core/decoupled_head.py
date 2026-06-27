"""DecoupledSingleHead — 方向 B: 解耦分类/定位双分支头

B1: cls 和 reg 各有独立的 self_attn + FFN (共享 inst_interact 用于 RoI 特征提取)
B2: 每分支独立的 LayerNorm
B3: 阶段化 cls loss 权重 (在 criterion 层实现, 见集成测试)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldmdet.core.single_head import SingleDiffusionDetHead


class DecoupledSingleHead(SingleDiffusionDetHead):
    """解耦双分支扩散检测头。

    与 SingleDiffusionDetHead 的区别:
    - cls 分支: cls_self_attn + cls FFN + cls LayerNorms
    - reg 分支: reg_self_attn + reg FFN + reg LayerNorms
    - 共享: inst_interact (DynamicConv), time_mlp/adaln_mlp, cls_head/reg_head
    """

    def __init__(
        self,
        num_classes=80,
        feat_channels=256,
        dim_feedforward=2048,
        num_cls_convs=1,
        num_reg_convs=3,
        num_heads=8,
        dropout=0.0,
        pooler_resolution=7,
        scale_clamp=math.log(100000.0 / 16),
        bbox_weights=(2.0, 2.0, 1.0, 1.0),
        use_focal_loss=True,
        use_fed_loss=False,
        dynamic_dim=64,
        dynamic_num=2,
        time_conditioning='scale_shift',
        use_sdpa=True,
        attn_half=False,
        shape_attention=None,
        box_refine=None,
    ):
        super().__init__(
            num_classes=num_classes,
            feat_channels=feat_channels,
            dim_feedforward=dim_feedforward,
            num_cls_convs=num_cls_convs,
            num_reg_convs=num_reg_convs,
            num_heads=num_heads,
            dropout=dropout,
            pooler_resolution=pooler_resolution,
            scale_clamp=scale_clamp,
            bbox_weights=bbox_weights,
            use_focal_loss=use_focal_loss,
            use_fed_loss=use_fed_loss,
            dynamic_dim=dynamic_dim,
            dynamic_num=dynamic_num,
            time_conditioning=time_conditioning,
            use_sdpa=use_sdpa,
            attn_half=attn_half,
            shape_attention=shape_attention,
            box_refine=box_refine,
        )

        # B1: cls 分支独立的 self_attn
        self.cls_self_attn = nn.MultiheadAttention(
            feat_channels, num_heads, dropout=dropout
        )
        # B1: reg 分支独立的 self_attn
        self.reg_self_attn = nn.MultiheadAttention(
            feat_channels, num_heads, dropout=dropout
        )

        # B1: cls 分支独立的 FFN
        self.cls_linear1 = nn.Linear(feat_channels, dim_feedforward)
        self.cls_linear2 = nn.Linear(dim_feedforward, feat_channels)
        # B1: reg 分支独立的 FFN
        self.reg_linear1 = nn.Linear(feat_channels, dim_feedforward)
        self.reg_linear2 = nn.Linear(dim_feedforward, feat_channels)

        # B2: cls 分支独立的 LayerNorm
        self.cls_norm1 = nn.LayerNorm(feat_channels)
        self.cls_norm2 = nn.LayerNorm(feat_channels)
        self.cls_norm3 = nn.LayerNorm(feat_channels)
        # B2: reg 分支独立的 LayerNorm
        self.reg_norm1 = nn.LayerNorm(feat_channels)
        self.reg_norm2 = nn.LayerNorm(feat_channels)
        self.reg_norm3 = nn.LayerNorm(feat_channels)

        # cls/reg 各自的 dropout
        self.cls_dropout1 = nn.Dropout(dropout)
        self.cls_dropout2 = nn.Dropout(dropout)
        self.cls_dropout3 = nn.Dropout(dropout)
        self.reg_dropout1 = nn.Dropout(dropout)
        self.reg_dropout2 = nn.Dropout(dropout)
        self.reg_dropout3 = nn.Dropout(dropout)

    def _cls_self_attn(self, q):
        """cls 分支的 self-attention (复用 SDPA 加速)"""
        if self.use_sdpa:
            return self._sdpa_self_attn_cls(q), None
        return self.cls_self_attn(q, q, q)

    def _reg_self_attn(self, q):
        """reg 分支的 self-attention (复用 SDPA 加速)"""
        if self.use_sdpa:
            return self._sdpa_self_attn_reg(q), None
        return self.reg_self_attn(q, q, q)

    def _sdpa_self_attn_cls(self, x_seq_bs_dim):
        """cls 分支 SDPA self-attention"""
        return self._sdpa_self_attn_with(x_seq_bs_dim, self.cls_self_attn)

    def _sdpa_self_attn_reg(self, x_seq_bs_dim):
        """reg 分支 SDPA self-attention"""
        return self._sdpa_self_attn_with(x_seq_bs_dim, self.reg_self_attn)

    def _sdpa_self_attn_with(self, x_seq_bs_dim, attn):
        """通用 SDPA self-attention (复用指定 attn 层的投影权重)"""
        seq_len, bs, dim = x_seq_bs_dim.shape
        x_flat = x_seq_bs_dim.reshape(seq_len * bs, dim)
        qkv = F.linear(x_flat, attn.in_proj_weight, attn.in_proj_bias)
        qkv = qkv.reshape(seq_len, bs, 3, dim)
        q, k, v = qkv.unbind(2)

        q = q.reshape(seq_len, bs, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.reshape(seq_len, bs, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.reshape(seq_len, bs, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        input_dtype = q.dtype
        if self.attn_half and input_dtype == torch.float32:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        if out.dtype != input_dtype:
            out = out.to(input_dtype)

        out = out.permute(2, 0, 1, 3).reshape(seq_len, bs, dim)
        out_flat = out.reshape(seq_len * bs, dim)
        out_flat = F.linear(out_flat, attn.out_proj.weight, attn.out_proj.bias)
        return out_flat.reshape(seq_len, bs, dim)

    def _forward_scale_shift(self, proposals, roi_features, time_emb, bs, num_boxes):
        """B1+B2: scale_shift 模式下双分支前向"""
        proposals_seq = proposals.view(bs, num_boxes, self.feat_channels).permute(1, 0, 2)

        # --- cls 分支 ---
        cls_attn, _ = self._cls_self_attn(proposals_seq)
        cls_feat = proposals_seq + self.cls_dropout1(cls_attn)
        cls_feat = self.cls_norm1(cls_feat)

        cls_feat = cls_feat.permute(1, 0, 2).reshape(1, bs * num_boxes, self.feat_channels)
        cls_inst = self.inst_interact(cls_feat, roi_features)
        cls_feat = cls_feat + self.cls_dropout2(cls_inst)
        cls_feat = self.cls_norm2(cls_feat)

        cls_ffn = self.cls_linear2(self.dropout(self.act(self.cls_linear1(cls_feat))))
        cls_feat = cls_feat + self.cls_dropout3(cls_ffn)
        cls_feat = self.cls_norm3(cls_feat)

        # --- reg 分支 ---
        reg_attn, _ = self._reg_self_attn(proposals_seq)
        reg_feat = proposals_seq + self.reg_dropout1(reg_attn)
        reg_feat = self.reg_norm1(reg_feat)

        reg_feat = reg_feat.permute(1, 0, 2).reshape(1, bs * num_boxes, self.feat_channels)
        reg_inst = self.inst_interact(reg_feat, roi_features)
        reg_feat = reg_feat + self.reg_dropout2(reg_inst)
        reg_feat = self.reg_norm2(reg_feat)

        reg_ffn = self.reg_linear2(self.dropout(self.act(self.reg_linear1(reg_feat))))
        reg_feat = reg_feat + self.reg_dropout3(reg_ffn)
        reg_feat = self.reg_norm3(reg_feat)

        # --- 时间条件化 (共享 time_mlp) ---
        scale_shift = self.time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, num_boxes, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)

        cls_fc = cls_feat.squeeze(0)
        cls_fc = cls_fc * (scale + 1) + shift
        reg_fc = reg_feat.squeeze(0)
        reg_fc = reg_fc * (scale + 1) + shift

        # 返回 cls 和 reg 各自的 fc_feature
        return cls_fc, reg_fc

    def _forward_adaln_zero(self, proposals, roi_features, time_emb, bs, num_boxes):
        """B1+B2: adaln_zero 模式下双分支前向"""
        proposals_seq = proposals.view(bs, num_boxes, self.feat_channels).permute(1, 0, 2)
        adaln_params = self.adaln_mlp(time_emb)
        adaln_params = torch.repeat_interleave(adaln_params, num_boxes, dim=0)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = adaln_params.chunk(6, dim=-1)

        # --- cls 分支 ---
        proposals_flat = proposals_seq.reshape(num_boxes * bs, self.feat_channels)
        q_cls = F.layer_norm(proposals_flat, [self.feat_channels]) * (1 + gamma1) + beta1
        q_cls = q_cls.view(num_boxes, bs, self.feat_channels)
        cls_attn, _ = self._cls_self_attn(q_cls)
        cls_attn_flat = cls_attn.reshape(num_boxes * bs, self.feat_channels)
        cls_feat = proposals_flat + alpha1 * cls_attn_flat

        cls_feat = cls_feat.view(num_boxes, bs, self.feat_channels).permute(1, 0, 2)
        cls_feat = cls_feat.reshape(1, bs * num_boxes, self.feat_channels)
        cls_inst = self.inst_interact(cls_feat, roi_features)
        cls_feat = cls_feat + self.cls_dropout2(cls_inst)
        cls_feat = self.cls_norm2(cls_feat)

        cls_obj = cls_feat.squeeze(0)
        cls_ffn_in = F.layer_norm(cls_obj, [self.feat_channels]) * (1 + gamma2) + beta2
        cls_ffn = self.cls_linear2(self.dropout(self.act(self.cls_linear1(cls_ffn_in))))  # noqa
        cls_fc = cls_obj + alpha2 * cls_ffn

        # --- reg 分支 ---
        proposals_flat = proposals_seq.reshape(num_boxes * bs, self.feat_channels)
        q_reg = F.layer_norm(proposals_flat, [self.feat_channels]) * (1 + gamma1) + beta1
        q_reg = q_reg.view(num_boxes, bs, self.feat_channels)
        reg_attn, _ = self._reg_self_attn(q_reg)
        reg_attn_flat = reg_attn.reshape(num_boxes * bs, self.feat_channels)
        reg_feat = proposals_flat + alpha1 * reg_attn_flat

        reg_feat = reg_feat.view(num_boxes, bs, self.feat_channels).permute(1, 0, 2)
        reg_feat = reg_feat.reshape(1, bs * num_boxes, self.feat_channels)
        reg_inst = self.inst_interact(reg_feat, roi_features)
        reg_feat = reg_feat + self.reg_dropout2(reg_inst)
        reg_feat = self.reg_norm2(reg_feat)

        reg_obj = reg_feat.squeeze(0)
        reg_ffn_in = F.layer_norm(reg_obj, [self.feat_channels]) * (1 + gamma2) + beta2
        reg_ffn = self.reg_linear2(self.dropout(self.act(self.reg_linear1(reg_ffn_in))))  # noqa
        reg_fc = reg_obj + alpha2 * reg_ffn

        return cls_fc, reg_fc

    def _conditioned_forward(self, proposals, roi_features, time_emb, bs, num_boxes):
        """重写: 返回 (cls_fc, reg_fc) 元组而非单一 fc_feature"""
        if self.time_conditioning == 'adaln_zero':
            return self._forward_adaln_zero(proposals, roi_features, time_emb, bs, num_boxes)
        return self._forward_scale_shift(proposals, roi_features, time_emb, bs, num_boxes)

    def forward(self, features, bboxes, proposals, pooler, time_emb):
        """重写 forward: 解耦双分支 → cls_head/reg_head 各自预测"""
        bs, num_boxes = bboxes.shape[:2]
        from ldmdet.utils.box_ops import bbox2roi
        rois = bbox2roi([bboxes[i] for i in range(bs)])
        roi_features = pooler(features, rois)

        if proposals is None:
            proposals = roi_features.flatten(2).mean(-1).view(bs, num_boxes, self.feat_channels)

        roi_features = roi_features.view(
            bs * num_boxes, self.feat_channels, -1
        ).permute(2, 0, 1)

        cls_fc, reg_fc = self._conditioned_forward(
            proposals, roi_features, time_emb, bs, num_boxes
        )

        class_logits = self.cls_head(cls_fc)
        pred_bboxes = self._predict_bboxes(reg_fc, bboxes)

        # 返回的 fc_feature 取 cls/reg 均值, 供下一扩散步使用
        fc_feature = (cls_fc + reg_fc) * 0.5

        return (
            class_logits.view(bs, num_boxes, -1),
            pred_bboxes.view(bs, num_boxes, -1),
            fc_feature.view(1, bs * num_boxes, self.feat_channels),
        )
