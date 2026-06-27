"""SingleDiffusionDetHead — 单个扩散检测头

支持 AdaLN-Zero 和传统 scale-shift 两种时间条件化模式。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldmdet.core.dynamic_conv import DynamicConv
from ldmdet.utils.box_ops import bbox2roi

# 检测 SDPA (PyTorch 2.0+) 是否可用，用于高效 Flash Attention 后端
_SDPA_AVAILABLE = hasattr(F, 'scaled_dot_product_attention')


class SingleDiffusionDetHead(nn.Module):
    """单步扩散检测头。

    Self-Attention + Instance Interaction + FFN 三段式结构，
    末端接分类和回归预测头。
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
    ):
        super().__init__()
        self.feat_channels = feat_channels
        self.time_conditioning = time_conditioning
        # use_sdpa: True = 使用 SDPA, False = 使用 nn.MultiheadAttention
        # attn_half: True = attention 核心用 FP16 加速, False = 保持原始精度
        self.use_sdpa = use_sdpa and _SDPA_AVAILABLE
        self.attn_half = attn_half

        # 方向 C1: 可选的局部形状注意力 (在 RoI 特征上应用)
        self.shape_attention = shape_attention

        self.self_attn = nn.MultiheadAttention(
            feat_channels, num_heads, dropout=dropout
        )
        self.inst_interact = DynamicConv(
            feat_channels=feat_channels,
            pooler_resolution=pooler_resolution,
            dynamic_dim=dynamic_dim,
            dynamic_num=dynamic_num,
        )

        self.linear1 = nn.Linear(feat_channels, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, feat_channels)

        self.norm1 = nn.LayerNorm(feat_channels)
        self.norm2 = nn.LayerNorm(feat_channels)
        self.norm3 = nn.LayerNorm(feat_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.act = nn.ReLU(inplace=True)

        if self.time_conditioning == 'adaln_zero':
            self.adaln_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(feat_channels * 4, feat_channels * 6),
            )
            nn.init.zeros_(self.adaln_mlp[-1].weight)
            nn.init.zeros_(self.adaln_mlp[-1].bias)
        else:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(feat_channels * 4, feat_channels * 2),
            )

        self.cls_head = self._build_cls_head(
            feat_channels, num_cls_convs, num_classes, use_focal_loss, use_fed_loss
        )
        self.reg_head = self._build_reg_head(feat_channels, num_reg_convs)

        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights
        self.num_heads = num_heads
        self.head_dim = feat_channels // num_heads

    def _sdpa_self_attn(self, x_seq_bs_dim):
        """使用 SDPA 的高效 self-attention。

        输入/输出格式与 nn.MultiheadAttention 一致: [seq_len, bs, dim]。
        复用 self.self_attn 的投影权重，仅替换核心 attention 计算。

        精度由 self.attn_half 控制:
          - False (默认): 全精度，输出与 nn.MHA 几乎一致
          - True: attention 核心转 FP16 加速，有微小精度差异
        """
        seq_len, bs, dim = x_seq_bs_dim.shape
        attn = self.self_attn
        # QKV 投影: [seq*bs, dim]
        x_flat = x_seq_bs_dim.reshape(seq_len * bs, dim)
        qkv = F.linear(x_flat, attn.in_proj_weight, attn.in_proj_bias)
        qkv = qkv.reshape(seq_len, bs, 3, dim)
        q, k, v = qkv.unbind(2)  # 各 [seq, bs, dim]

        # 转为 [bs, num_heads, seq, head_dim]
        q = q.reshape(seq_len, bs, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.reshape(seq_len, bs, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.reshape(seq_len, bs, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # 可选: attention 核心转 FP16 以启用高效后端
        input_dtype = q.dtype
        if self.attn_half and input_dtype == torch.float32:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        if out.dtype != input_dtype:
            out = out.to(input_dtype)

        # 转回 [seq, bs, dim]
        out = out.permute(2, 0, 1, 3).reshape(seq_len, bs, dim)

        # 输出投影
        out_flat = out.reshape(seq_len * bs, dim)
        out_flat = F.linear(out_flat, attn.out_proj.weight, attn.out_proj.bias)
        return out_flat.reshape(seq_len, bs, dim)

    def _self_attn(self, q, k=None, v=None):
        """统一的 self-attention 接口。"""
        if k is None:
            k = q
        if v is None:
            v = q
        if self.use_sdpa and q is k is v:
            return self._sdpa_self_attn(q), None
        return self.self_attn(q, k, v)

    @staticmethod
    def _build_cls_head(feat_channels, num_convs, num_classes, use_focal_loss, use_fed_loss):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Sequential(
                nn.Linear(feat_channels, feat_channels, bias=False),
                nn.LayerNorm(feat_channels),
                nn.ReLU(inplace=True),
            ))
        out_dim = num_classes if (use_focal_loss or use_fed_loss) else num_classes + 1
        layers.append(nn.Linear(feat_channels, out_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_reg_head(feat_channels, num_convs):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Sequential(
                nn.Linear(feat_channels, feat_channels, bias=False),
                nn.LayerNorm(feat_channels),
                nn.ReLU(inplace=True),
            ))
        layers.append(nn.Linear(feat_channels, 4))
        return nn.Sequential(*layers)

    def forward(self, features, bboxes, proposals, pooler, time_emb):
        bs, num_boxes = bboxes.shape[:2]
        rois = bbox2roi([bboxes[i] for i in range(bs)])
        roi_features = pooler(features, rois)

        # 方向 C1: 在 RoI 特征上应用局部形状注意力 (可选)
        if self.shape_attention is not None:
            roi_features = self.shape_attention(roi_features)

        if proposals is None:
            proposals = roi_features.flatten(2).mean(-1).view(bs, num_boxes, self.feat_channels)

        roi_features = roi_features.view(
            bs * num_boxes, self.feat_channels, -1
        ).permute(2, 0, 1)

        fc_feature = self._conditioned_forward(proposals, roi_features, time_emb, bs, num_boxes)
        return self._predict(fc_feature, bboxes, bs, num_boxes)

    def _conditioned_forward(self, proposals, roi_features, time_emb, bs, num_boxes):
        if self.time_conditioning == 'adaln_zero':
            return self._forward_adaln_zero(proposals, roi_features, time_emb, bs, num_boxes)
        return self._forward_scale_shift(proposals, roi_features, time_emb, bs, num_boxes)

    def _predict(self, fc_feature, bboxes, bs, num_boxes):
        class_logits = self.cls_head(fc_feature)
        pred_bboxes = self._predict_bboxes(fc_feature, bboxes)
        return (
            class_logits.view(bs, num_boxes, -1),
            pred_bboxes.view(bs, num_boxes, -1),
            fc_feature.view(1, bs * num_boxes, self.feat_channels),
        )

    def _forward_adaln_zero(self, proposals, roi_features, time_emb, bs, num_boxes):
        proposals = proposals.view(bs, num_boxes, self.feat_channels).permute(1, 0, 2)
        adaln_params = self.adaln_mlp(time_emb)
        adaln_params = torch.repeat_interleave(adaln_params, num_boxes, dim=0)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = adaln_params.chunk(6, dim=-1)

        proposals_flat = proposals.reshape(num_boxes * bs, self.feat_channels)
        q_modulated = F.layer_norm(proposals_flat, [self.feat_channels]) * (1 + gamma1) + beta1
        q_modulated = q_modulated.view(num_boxes, bs, self.feat_channels)
        attn_out, _ = self._self_attn(q_modulated)
        attn_out_flat = attn_out.reshape(num_boxes * bs, self.feat_channels)
        proposals_flat = proposals_flat + alpha1 * attn_out_flat

        proposals = proposals_flat.view(num_boxes, bs, self.feat_channels)
        proposals = proposals.permute(1, 0, 2).reshape(1, bs * num_boxes, self.feat_channels)
        inst_out = self.inst_interact(proposals, roi_features)
        proposals = proposals + self.dropout2(inst_out)
        obj_features = self.norm2(proposals)

        obj_flat = obj_features.squeeze(0)
        ffn_input = F.layer_norm(obj_flat, [self.feat_channels]) * (1 + gamma2) + beta2
        ffn_out = self.linear2(self.dropout(self.act(self.linear1(ffn_input))))
        obj_flat = obj_flat + alpha2 * ffn_out
        return obj_flat

    def _forward_scale_shift(self, proposals, roi_features, time_emb, bs, num_boxes):
        proposals = proposals.view(bs, num_boxes, self.feat_channels).permute(1, 0, 2)
        attn_shortcut, _ = self._self_attn(proposals)
        proposals = proposals + self.dropout1(attn_shortcut)
        proposals = self.norm1(proposals)

        proposals = proposals.permute(1, 0, 2).reshape(1, bs * num_boxes, self.feat_channels)
        attn_shortcut = self.inst_interact(proposals, roi_features)
        proposals = proposals + self.dropout2(attn_shortcut)
        obj_features = self.norm2(proposals)

        obj_shortcut = self.linear2(self.dropout(self.act(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_shortcut)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(bs * num_boxes, -1)
        scale_shift = self.time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, num_boxes, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift
        return fc_feature

    def _predict_bboxes(self, fc_feature, bboxes):
        bboxes_deltas = self.reg_head(fc_feature)
        return self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

    def apply_deltas(self, deltas, boxes):
        boxes = boxes.to(deltas.dtype)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
        return pred_boxes
