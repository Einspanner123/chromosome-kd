import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .modules import DynamicConv
from .utils import bbox2roi

_HAS_SDPA = hasattr(torch.nn.functional, 'scaled_dot_product_attention')


class SingleDiffusionDetHead(nn.Module):
    """单个DiffusionDet检测头，支持 AdaLN-Zero 和传统 scale-shift 条件化"""

    def __init__(
        self,
        num_classes=80,  # 类别数
        feat_channels=256,  # 特征通道数
        dim_feedforward=2048,  # 前馈网络维度
        num_cls_convs=1,  # 分类卷积层数
        num_reg_convs=3,  # 回归卷积层数
        num_heads=8,  # 注意力头数
        dropout=0.0,  # dropout率
        pooler_resolution=7,  # 池化分辨率
        scale_clamp=math.log(100000.0 / 16),  # 缩放截断值
        bbox_weights=(2.0, 2.0, 1.0, 1.0),  # 边界框权重
        use_focal_loss=True,  # 是否使用focal loss
        use_fed_loss=False,  # 是否使用fed loss
        dynamic_dim=64,
        dynamic_num=2,
        # === FlowDet 新增参数 ===
        time_conditioning='scale_shift',  # "scale_shift" 或 "adaln_zero"
        use_objectness=False,  # 是否使用 objectness 预测头
        prediction_mode='x0',  # "x0" (delta regression) 或 "velocity" (直接速度预测)
        use_flash_attn=False,  # 是否启用 Flash Attention (SDPA)
    ):
        super().__init__()
        self.feat_channels = feat_channels
        self.time_conditioning = time_conditioning
        self.use_objectness = use_objectness
        self.prediction_mode = prediction_mode
        self.use_flash_attn = use_flash_attn

        # 动态模块
        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(
            feat_channels, num_heads, dropout=dropout
        )
        # 实例交互模块（动态卷积）
        self.inst_interact = DynamicConv(
            feat_channels=feat_channels,
            pooler_resolution=pooler_resolution,
            dynamic_dim=dynamic_dim,
            dynamic_num=dynamic_num,
        )

        # 前馈网络
        self.linear1 = nn.Linear(feat_channels, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, feat_channels)

        # LayerNorm层
        self.norm1 = nn.LayerNorm(feat_channels)  # 自注意力后归一化
        self.norm2 = nn.LayerNorm(feat_channels)  # 实例交互后归一化
        self.norm3 = nn.LayerNorm(feat_channels)  # 前馈网络后归一化
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # 激活函数
        self.act = nn.ReLU(inplace=True)

        # 时间嵌入条件化
        if self.time_conditioning == 'adaln_zero':
            # AdaLN-Zero: 产生 6 组参数 (γ₁, β₁, α₁ for SA+InstInteract, γ₂, β₂, α₂ for FFN)
            self.adaln_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(feat_channels * 4, feat_channels * 6),
            )
            # 关键: 零初始化最后一层 → 初始时各层为恒等映射
            nn.init.zeros_(self.adaln_mlp[-1].weight)
            nn.init.zeros_(self.adaln_mlp[-1].bias)
        else:
            # 传统 scale-shift 条件化
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(feat_channels * 4, feat_channels * 2),
            )

        # 分类模块
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
                num_classes
                if use_focal_loss or use_fed_loss
                else num_classes + 1,
            )
        )
        self.cls_head = nn.Sequential(*cls_layers)

        # 回归模块
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

        # Objectness 预测头 (Phase 3B)
        self.objectness_head = None
        if use_objectness:
            self.objectness_head = nn.Sequential(
                nn.Linear(feat_channels, feat_channels, bias=False),
                nn.LayerNorm(feat_channels),
                nn.ReLU(inplace=True),
                nn.Linear(feat_channels, 1),
            )

        # Velocity 预测头 (Phase 4) - 直接预测 4D 速度
        self.velocity_head = None
        if prediction_mode == 'velocity':
            self.velocity_head = nn.Sequential(
                nn.Linear(feat_channels, feat_channels, bias=False),
                nn.LayerNorm(feat_channels),
                nn.ReLU(inplace=True),
                nn.Linear(feat_channels, feat_channels, bias=False),
                nn.LayerNorm(feat_channels),
                nn.ReLU(inplace=True),
                nn.Linear(feat_channels, 4),
            )

        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features: Tensor, bboxes, proposals, pooler, time_emb):
        """
        前向传播

        Args:
            features: 特征金字塔,列表形式
            bboxes (N, num_boxes, 4): 边界框
            proposals (N, num_boxes, feat_channels): 提案特征
            pooler: ROI池化器
            time_emb (N, time_dim): 时间嵌入

        Returns:
            class_logits (N, num_boxes, num_classes): 分类logits
            pred_bboxes (N, num_boxes, 4): 预测边界框
            obj_features (1, N*num_boxes, feat_channels): 对象特征
            objectness (N, num_boxes, 1) or None: objectness 预测
            pred_velocity (N, num_boxes, 4) or None: 速度预测
        """
        bs, num_boxes = bboxes.shape[:2]

        # ROI特征提取
        rois = bbox2roi([bboxes[i] for i in range(bs)])
        roi_features = pooler(features, rois)  # (bs*num_boxes, c, 7, 7)

        # 处理提案特征
        if proposals is None:
            proposals = (
                roi_features.flatten(2)
                .mean(-1)
                .view(bs, num_boxes, self.feat_channels)
            )

        # 调整ROI特征形状
        roi_features = roi_features.view(
            bs * num_boxes,
            self.feat_channels,
            -1,
        ).permute(2, 0, 1)  # (49, bs*num_boxes, feat_channels)

        if self.time_conditioning == 'adaln_zero':
            return self._forward_adaln_zero(
                features,
                bboxes,
                proposals,
                roi_features,
                time_emb,
                bs,
                num_boxes,
            )
        else:
            return self._forward_scale_shift(
                features,
                bboxes,
                proposals,
                roi_features,
                time_emb,
                bs,
                num_boxes,
            )

    def _attn(self, q, k, v):
        if self.use_flash_attn and _HAS_SDPA:
            orig_dp = self.self_attn.dropout
            self.self_attn.dropout = 0.0
            with torch.cuda.amp.autocast(enabled=True):
                if not getattr(self, '_flash_attn_logged', False):
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CUDA],
                    ) as prof:
                        _ = self.self_attn(q, k, value=v)
                    kernels = ' '.join(
                        evt.key for evt in prof.key_averages()
                    ).lower()
                    if 'flash' in kernels:
                        backend = 'flash_attention'
                    elif 'cutlass' in kernels or 'softmax_warp' in kernels:
                        backend = 'efficient_attention'
                    else:
                        backend = 'math_fallback'
                    print(
                        f'[SDPA] use_flash_attn=True, dispatched backend: {backend}'
                    )
                    self._flash_attn_logged = True
                    self.self_attn.dropout = orig_dp
                    return _
                out = self.self_attn(q, k, value=v)
            self.self_attn.dropout = orig_dp
            return out
        return self.self_attn(q, k, value=v)

    def _forward_adaln_zero(
        self,
        features,
        bboxes,
        proposals,
        roi_features,
        time_emb,
        bs,
        num_boxes,
    ):
        """AdaLN-Zero 条件化的前向传播"""
        # 计算 AdaLN 参数: 6 组
        adaln_params = self.adaln_mlp(time_emb)  # (bs, 6 * feat_channels)
        adaln_params = torch.repeat_interleave(adaln_params, num_boxes, dim=0)
        # (bs*num_boxes, 6*feat_channels)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = adaln_params.chunk(
            6, dim=-1
        )

        # === Block 1: Self-Attention + AdaLN-Zero ===
        proposals = proposals.view(bs, num_boxes, self.feat_channels).permute(
            1, 0, 2
        )  # (num_boxes, bs, feat_channels)

        # Pre-norm + modulate
        proposals_flat = proposals.reshape(
            num_boxes * bs, self.feat_channels
        )  # for AdaLN
        q_modulated = (
            F.layer_norm(proposals_flat, [self.feat_channels]) * (1 + gamma1)
            + beta1
        )
        q_modulated = q_modulated.view(num_boxes, bs, self.feat_channels)

        attn_out, _ = self._attn(q_modulated, q_modulated, q_modulated)
        # Gated residual
        attn_out_flat = attn_out.reshape(num_boxes * bs, self.feat_channels)
        proposals_flat = proposals_flat + alpha1 * attn_out_flat
        proposals = proposals_flat.view(num_boxes, bs, self.feat_channels)

        # === Block 2: Instance Interaction (Dynamic Conv) ===
        proposals = (
            proposals.view(num_boxes, bs, self.feat_channels)
            .permute(1, 0, 2)
            .reshape(1, bs * num_boxes, self.feat_channels)
        )

        inst_out = self.inst_interact(proposals, roi_features)
        proposals = proposals + self.dropout2(inst_out)
        obj_features = self.norm2(
            proposals
        )  # (1, bs*num_boxes, feat_channels)

        # === Block 3: FFN + AdaLN-Zero ===
        obj_flat = obj_features.squeeze(0)  # (bs*num_boxes, feat_channels)
        # Pre-norm + modulate for FFN
        ffn_input = (
            F.layer_norm(obj_flat, [self.feat_channels]) * (1 + gamma2) + beta2
        )
        ffn_out = self.linear2(self.dropout(self.act(self.linear1(ffn_input))))
        # Gated residual
        obj_flat = obj_flat + alpha2 * ffn_out
        obj_features = obj_flat.unsqueeze(
            0
        )  # (1, bs*num_boxes, feat_channels)

        # === 预测头 ===
        fc_feature = obj_flat  # (bs*num_boxes, feat_channels)

        class_logits = self.cls_head(fc_feature)
        pred_bboxes = self._predict_bboxes(fc_feature, bboxes)

        # Objectness
        objectness = None
        if self.objectness_head is not None:
            objectness = self.objectness_head(fc_feature).view(
                bs, num_boxes, 1
            )

        # Velocity
        pred_velocity = None
        if self.velocity_head is not None:
            pred_velocity = self.velocity_head(fc_feature).view(
                bs, num_boxes, 4
            )

        return (
            class_logits.view(bs, num_boxes, -1),
            pred_bboxes.view(bs, num_boxes, -1),
            obj_features,
            objectness,
            pred_velocity,
        )

    def _forward_scale_shift(
        self,
        features,
        bboxes,
        proposals,
        roi_features,
        time_emb,
        bs,
        num_boxes,
    ):
        """传统 scale-shift 条件化的前向传播（向后兼容）"""
        # 自注意力
        proposals = proposals.view(bs, num_boxes, self.feat_channels).permute(
            1, 0, 2
        )  # (num_boxes, bs, feat_channels)

        attn_shortcut, _ = self._attn(proposals, proposals, proposals)
        proposals = proposals + self.dropout1(attn_shortcut)
        proposals = self.norm1(proposals)

        # 实例交互
        proposals = (
            proposals.view(num_boxes, bs, self.feat_channels)
            .permute(1, 0, 2)
            .reshape(1, bs * num_boxes, self.feat_channels)
        )
        attn_shortcut = self.inst_interact(proposals, roi_features)
        proposals = proposals + self.dropout2(attn_shortcut)
        obj_features = self.norm2(proposals)

        # FFN
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

        # 预测头
        class_logits = self.cls_head(fc_feature)
        pred_bboxes = self._predict_bboxes(fc_feature, bboxes)

        # Objectness
        objectness = None
        if self.objectness_head is not None:
            objectness = self.objectness_head(fc_feature).view(
                bs, num_boxes, 1
            )

        # Velocity
        pred_velocity = None
        if self.velocity_head is not None:
            pred_velocity = self.velocity_head(fc_feature).view(
                bs, num_boxes, 4
            )

        return (
            class_logits.view(bs, num_boxes, -1),
            pred_bboxes.view(bs, num_boxes, -1),
            obj_features,
            objectness,
            pred_velocity,
        )

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
