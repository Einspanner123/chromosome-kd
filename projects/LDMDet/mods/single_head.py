import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import DynamicConv
from .utils import bbox2roi


class SingleDiffusionDetHead(nn.Module):
    """单个DiffusionDet检测头，支持 AdaLN-Zero 和传统 scale-shift 条件化

    两种条件化模式共享:
    - Self-Attention + Instance Interaction + FFN 三段式结构
    - 分类/回归/objectness/velocity 预测头
    差异仅在于时间嵌入如何注入各子块。
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
        use_objectness=False,
        prediction_mode='x0',
        use_flash_attn=False,
    ):
        super().__init__()
        self.feat_channels = feat_channels
        self.time_conditioning = time_conditioning
        self.use_objectness = use_objectness
        self.prediction_mode = prediction_mode
        self.use_flash_attn = use_flash_attn

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            feat_channels, num_heads, dropout=dropout
        )
        # Instance Interaction (Dynamic Conv)
        self.inst_interact = DynamicConv(
            feat_channels=feat_channels,
            pooler_resolution=pooler_resolution,
            dynamic_dim=dynamic_dim,
            dynamic_num=dynamic_num,
        )

        # FFN
        self.linear1 = nn.Linear(feat_channels, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, feat_channels)

        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(feat_channels)
        self.norm2 = nn.LayerNorm(feat_channels)
        self.norm3 = nn.LayerNorm(feat_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.act = nn.ReLU(inplace=True)

        # 时间嵌入条件化
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

        # 分类头
        self.cls_head = self._build_cls_head(
            feat_channels,
            num_cls_convs,
            num_classes,
            use_focal_loss,
            use_fed_loss,
        )
        # 回归头
        self.reg_head = self._build_reg_head(feat_channels, num_reg_convs)
        # Objectness 头
        self.objectness_head = (
            self._build_objectness_head(feat_channels)
            if use_objectness
            else None
        )
        # Velocity 头
        self.velocity_head = (
            self._build_velocity_head(feat_channels)
            if prediction_mode == 'velocity'
            else None
        )

        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    # ------------------------------------------------------------------
    # 预测头构建 (静态工厂，便于子类复用)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cls_head(
        feat_channels, num_convs, num_classes, use_focal_loss, use_fed_loss
    ):
        layers = []
        for _ in range(num_convs):
            layers.append(
                nn.Sequential(
                    nn.Linear(feat_channels, feat_channels, bias=False),
                    nn.LayerNorm(feat_channels),
                    nn.ReLU(inplace=True),
                )
            )
        out_dim = (
            num_classes
            if (use_focal_loss or use_fed_loss)
            else num_classes + 1
        )
        layers.append(nn.Linear(feat_channels, out_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_reg_head(feat_channels, num_convs):
        layers = []
        for _ in range(num_convs):
            layers.append(
                nn.Sequential(
                    nn.Linear(feat_channels, feat_channels, bias=False),
                    nn.LayerNorm(feat_channels),
                    nn.ReLU(inplace=True),
                )
            )
        layers.append(nn.Linear(feat_channels, 4))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_objectness_head(feat_channels):
        return nn.Sequential(
            nn.Linear(feat_channels, feat_channels, bias=False),
            nn.LayerNorm(feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, 1),
        )

    @staticmethod
    def _build_velocity_head(feat_channels):
        return nn.Sequential(
            nn.Linear(feat_channels, feat_channels, bias=False),
            nn.LayerNorm(feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels, bias=False),
            nn.LayerNorm(feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, 4),
        )

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------

    def forward(self, features, bboxes, proposals, pooler, time_emb):
        bs, num_boxes = bboxes.shape[:2]

        # ROI 特征提取
        rois = bbox2roi([bboxes[i] for i in range(bs)])
        roi_features = pooler(features, rois)

        if proposals is None:
            proposals = (
                roi_features.flatten(2)
                .mean(-1)
                .view(bs, num_boxes, self.feat_channels)
            )

        roi_features = roi_features.view(
            bs * num_boxes,
            self.feat_channels,
            -1,
        ).permute(2, 0, 1)  # (49, bs*num_boxes, feat_channels)

        # 条件化前向 → 得到 (bs*num_boxes, feat_channels) 的特征
        fc_feature = self._conditioned_forward(
            proposals,
            roi_features,
            time_emb,
            bs,
            num_boxes,
        )

        # 统一预测头
        return self._predict(fc_feature, bboxes, bs, num_boxes)

    def _conditioned_forward(
        self, proposals, roi_features, time_emb, bs, num_boxes
    ):
        """根据条件化模式执行前向传播，返回展平的特征 (bs*num_boxes, feat_channels)"""
        if self.time_conditioning == 'adaln_zero':
            return self._forward_adaln_zero(
                proposals,
                roi_features,
                time_emb,
                bs,
                num_boxes,
            )
        return self._forward_scale_shift(
            proposals,
            roi_features,
            time_emb,
            bs,
            num_boxes,
        )

    def _predict(self, fc_feature, bboxes, bs, num_boxes):
        """统一预测头: 分类 + 回归 + objectness + velocity"""
        class_logits = self.cls_head(fc_feature)
        pred_bboxes = self._predict_bboxes(fc_feature, bboxes)

        objectness = None
        if self.objectness_head is not None:
            objectness = self.objectness_head(fc_feature).view(
                bs, num_boxes, 1
            )

        pred_velocity = None
        if self.velocity_head is not None:
            pred_velocity = self.velocity_head(fc_feature).view(
                bs, num_boxes, 4
            )

        return (
            class_logits.view(bs, num_boxes, -1),
            pred_bboxes.view(bs, num_boxes, -1),
            fc_feature.view(1, bs * num_boxes, self.feat_channels),
            objectness,
            pred_velocity,
        )

    # ------------------------------------------------------------------
    # 条件化前向: AdaLN-Zero
    # ------------------------------------------------------------------

    def _forward_adaln_zero(
        self, proposals, roi_features, time_emb, bs, num_boxes
    ):
        proposals = proposals.view(bs, num_boxes, self.feat_channels).permute(
            1, 0, 2
        )  # (num_boxes, bs, feat_channels)

        # 计算 AdaLN 参数: 6 组 (γ₁, β₁, α₁, γ₂, β₂, α₂)
        adaln_params = self.adaln_mlp(time_emb)  # (bs, 6*feat_channels)
        adaln_params = torch.repeat_interleave(adaln_params, num_boxes, dim=0)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = adaln_params.chunk(
            6, dim=-1
        )

        # Block 1: Self-Attention + AdaLN-Zero
        proposals_flat = proposals.reshape(num_boxes * bs, self.feat_channels)
        q_modulated = (
            F.layer_norm(proposals_flat, [self.feat_channels]) * (1 + gamma1)
            + beta1
        )
        q_modulated = q_modulated.view(num_boxes, bs, self.feat_channels)

        attn_out, _ = self.self_attn(q_modulated, q_modulated, q_modulated)
        attn_out_flat = attn_out.reshape(num_boxes * bs, self.feat_channels)
        proposals_flat = proposals_flat + alpha1 * attn_out_flat

        # Block 2: Instance Interaction (Dynamic Conv)
        proposals = proposals_flat.view(num_boxes, bs, self.feat_channels)
        proposals = proposals.permute(1, 0, 2).reshape(
            1, bs * num_boxes, self.feat_channels
        )
        inst_out = self.inst_interact(proposals, roi_features)
        proposals = proposals + self.dropout2(inst_out)
        obj_features = self.norm2(
            proposals
        )  # (1, bs*num_boxes, feat_channels)

        # Block 3: FFN + AdaLN-Zero
        obj_flat = obj_features.squeeze(0)  # (bs*num_boxes, feat_channels)
        ffn_input = (
            F.layer_norm(obj_flat, [self.feat_channels]) * (1 + gamma2) + beta2
        )
        ffn_out = self.linear2(self.dropout(self.act(self.linear1(ffn_input))))
        obj_flat = obj_flat + alpha2 * ffn_out

        return obj_flat  # (bs*num_boxes, feat_channels)

    # ------------------------------------------------------------------
    # 条件化前向: Scale-Shift
    # ------------------------------------------------------------------

    def _forward_scale_shift(
        self, proposals, roi_features, time_emb, bs, num_boxes
    ):
        # Block 1: Self-Attention + Post-Norm
        proposals = proposals.view(bs, num_boxes, self.feat_channels).permute(
            1, 0, 2
        )
        attn_shortcut, _ = self.self_attn(proposals, proposals, proposals)
        proposals = proposals + self.dropout1(attn_shortcut)
        proposals = self.norm1(proposals)

        # Block 2: Instance Interaction + Post-Norm
        proposals = proposals.permute(1, 0, 2).reshape(
            1, bs * num_boxes, self.feat_channels
        )
        attn_shortcut = self.inst_interact(proposals, roi_features)
        proposals = proposals + self.dropout2(attn_shortcut)
        obj_features = self.norm2(proposals)

        # Block 3: FFN + Post-Norm
        obj_shortcut = self.linear2(
            self.dropout(self.act(self.linear1(obj_features)))
        )
        obj_features = obj_features + self.dropout3(obj_shortcut)
        obj_features = self.norm3(obj_features)

        # 时间嵌入条件化 (scale-shift)
        fc_feature = obj_features.transpose(0, 1).reshape(bs * num_boxes, -1)
        scale_shift = self.time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, num_boxes, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift

        return fc_feature  # (bs*num_boxes, feat_channels)

    # ------------------------------------------------------------------
    # 边界框预测
    # ------------------------------------------------------------------

    def _predict_bboxes(self, fc_feature, bboxes):
        """预测边界框 — 始终使用 delta regression

        注意: velocity 模式不改变 bbox 预测方式。velocity_head 仅作为辅助
        loss 分支，预测扩散空间的速度 v = x_0 - noise，不直接参与 bbox 更新。
        """
        bboxes_deltas = self.reg_head(fc_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))
        return pred_bboxes

    def apply_deltas(self, deltas, boxes):
        """将变换deltas (dx, dy, dw, dh) 应用到boxes"""
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
