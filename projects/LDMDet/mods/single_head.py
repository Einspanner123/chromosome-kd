import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .modules import DynamicConv, LinearCrossAttention, LinearSelfAttention
from .utils import bbox2roi


class SingleDiffusionDetHead(nn.Module):

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
        interact_type="dynamic_conv",
        self_attn_type="standard",
        time_conditioning="scale_shift",
    ):
        super().__init__()
        self.feat_channels = feat_channels
        self.interact_type = interact_type
        self.self_attn_type = self_attn_type
        self.time_conditioning = time_conditioning

        if self_attn_type == "linear":
            self.self_attn = LinearSelfAttention(
                feat_channels, num_heads=num_heads, dropout=dropout
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                feat_channels, num_heads, dropout=dropout
            )

        if interact_type == "linear_cross_attn":
            self.inst_interact = LinearCrossAttention(
                feat_channels=feat_channels,
                pooler_resolution=pooler_resolution,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
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

        if self.time_conditioning == "adaln_zero":
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

        cls_layers = []
        for _ in range(num_cls_convs):
            cls_layers.append(
                nn.Sequential(
                    nn.Linear(feat_channels, feat_channels, bias=False),
                    nn.LayerNorm(feat_channels),
                    nn.ReLU(inplace=True),
                )
            )
        cls_layers.append(
            nn.Linear(
                feat_channels,
                num_classes if use_focal_loss or use_fed_loss else num_classes + 1,
            )
        )
        self.cls_head = nn.Sequential(*cls_layers)

        reg_layers = []
        for _ in range(num_reg_convs):
            reg_layers.append(
                nn.Sequential(
                    nn.Linear(feat_channels, feat_channels, bias=False),
                    nn.LayerNorm(feat_channels),
                    nn.ReLU(inplace=True),
                )
            )
        reg_layers.append(nn.Linear(feat_channels, 4))
        self.reg_head = nn.Sequential(*reg_layers)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, proposals, pooler, time_emb):
        bs, num_boxes = bboxes.shape[:2]

        rois = bbox2roi([bboxes[i] for i in range(bs)])
        roi_features = pooler(features, rois)

        cls_logits, pred_bboxes, obj_features = self._dispatch(
            roi_features, bboxes, proposals, bs, num_boxes, time_emb
        )

        return cls_logits, pred_bboxes, obj_features, roi_features

    def forward_with_cached_roi(self, features, bboxes, proposals, cached_roi, time_emb):
        bs, num_boxes = bboxes.shape[:2]

        cls_logits, pred_bboxes, obj_features = self._dispatch(
            cached_roi, bboxes, proposals, bs, num_boxes, time_emb
        )

        return cls_logits, pred_bboxes, obj_features

    def _dispatch(self, roi_features, bboxes, proposals, bs, num_boxes, time_emb):
        if self.time_conditioning == "adaln_zero":
            return self._forward_adaln_zero(
                roi_features, bboxes, proposals, bs, num_boxes, time_emb
            )
        else:
            return self._forward_scale_shift(
                roi_features, bboxes, proposals, bs, num_boxes, time_emb
            )

    def _forward_adaln_zero(self, roi_features, bboxes, proposals, bs, num_boxes, time_emb):
        if proposals is None:
            proposals = (
                roi_features.flatten(2).mean(-1).view(bs, num_boxes, self.feat_channels)
            )

        roi_features = roi_features.view(
            bs * num_boxes, self.feat_channels, -1
        ).permute(2, 0, 1)

        N = bs * num_boxes
        adaln_params = self.adaln_mlp(time_emb)
        adaln_params = torch.repeat_interleave(adaln_params, num_boxes, dim=0)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = adaln_params.chunk(6, dim=-1)

        # Block 1: Self-Attention + AdaLN-Zero (in flat space, matching SOTA checkpoint)
        proposals = proposals.view(bs, num_boxes, self.feat_channels).permute(1, 0, 2)
        proposals_flat = proposals.reshape(N, self.feat_channels)

        q_modulated = F.layer_norm(proposals_flat, [self.feat_channels]) * (1 + gamma1) + beta1
        q_modulated = q_modulated.view(num_boxes, bs, self.feat_channels)

        attn_out, _ = self.self_attn(q_modulated, q_modulated, value=q_modulated)
        attn_out_flat = attn_out.reshape(N, self.feat_channels)
        proposals_flat = proposals_flat + alpha1 * attn_out_flat
        proposals = proposals_flat.view(num_boxes, bs, self.feat_channels)

        # Block 2: Instance Interaction
        proposals = (
            proposals.view(num_boxes, bs, self.feat_channels)
            .permute(1, 0, 2)
            .reshape(1, N, self.feat_channels)
        )
        attn_shortcut = self.inst_interact(proposals, roi_features)
        proposals = proposals + self.dropout2(attn_shortcut)
        obj_features = self.norm2(proposals)

        # Block 3: FFN + AdaLN-Zero (in flat space)
        obj_flat = obj_features.squeeze(0)
        ffn_input = F.layer_norm(obj_flat, [self.feat_channels]) * (1 + gamma2) + beta2
        ffn_out = self.linear2(self.dropout(self.act(self.linear1(ffn_input))))
        obj_flat = obj_flat + alpha2 * ffn_out

        class_logits = self.cls_head(obj_flat)
        bboxes_deltas = self.reg_head(obj_flat)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        return (
            class_logits.view(bs, num_boxes, -1),
            pred_bboxes.view(bs, num_boxes, -1),
            obj_flat.unsqueeze(0),
        )

    def _forward_scale_shift(self, roi_features, bboxes, proposals, bs, num_boxes, time_emb):
        if proposals is None:
            proposals = (
                roi_features.flatten(2).mean(-1).view(bs, num_boxes, self.feat_channels)
            )

        roi_features = roi_features.view(
            bs * num_boxes,
            self.feat_channels,
            -1,
        ).permute(2, 0, 1)

        proposals = proposals.view(
            bs,
            num_boxes,
            self.feat_channels,
        ).permute(1, 0, 2)

        attn_shortcut, _ = self.self_attn(
            proposals,
            proposals,
            value=proposals,
        )
        proposals = proposals + self.dropout1(attn_shortcut)
        proposals = self.norm1(proposals)

        proposals = (
            proposals.view(num_boxes, bs, self.feat_channels)
            .permute(1, 0, 2)
            .reshape(1, bs * num_boxes, self.feat_channels)
        )
        attn_shortcut = self.inst_interact(proposals, roi_features)

        proposals = proposals + self.dropout2(attn_shortcut)
        obj_features = self.norm2(proposals)

        obj_shortcut = self.linear2(
            self.dropout(self.act(self.linear1(obj_features)))
        )
        obj_features = obj_features + self.dropout3(obj_shortcut)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(
            bs * num_boxes, -1
        )

        scale_shift = self.time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(
            scale_shift, num_boxes, dim=0
        )
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift

        class_logits = self.cls_head(fc_feature)
        bboxes_deltas = self.reg_head(fc_feature)

        pred_bboxes = self.apply_deltas(
            bboxes_deltas, bboxes.view(-1, 4)
        )

        return (
            class_logits.view(bs, num_boxes, -1),
            pred_bboxes.view(bs, num_boxes, -1),
            obj_features,
        )

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
