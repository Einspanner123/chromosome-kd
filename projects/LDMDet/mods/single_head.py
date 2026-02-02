import math

import torch
import torch.nn as nn
from torch import Tensor

from .modules import DynamicConv


class SingleDiffusionDetHead(nn.Module):
    """单个DiffusionDet检测头"""

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
    ):
        super().__init__()
        self.feat_channels = feat_channels  # 特征通道数

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
        self.linear1 = nn.Linear(feat_channels, dim_feedforward)  # 第一线性层
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.linear2 = nn.Linear(dim_feedforward, feat_channels)  # 第二线性层

        # LayerNorm层
        self.norm1 = nn.LayerNorm(feat_channels)  # 自注意力后归一化
        self.norm2 = nn.LayerNorm(feat_channels)  # 实例交互后归一化
        self.norm3 = nn.LayerNorm(feat_channels)  # 前馈网络后归一化
        self.dropout1 = nn.Dropout(dropout)  # 自注意力dropout
        self.dropout2 = nn.Dropout(dropout)  # 实例交互dropout
        self.dropout3 = nn.Dropout(dropout)  # 前馈网络dropout

        # 激活函数
        self.act = nn.ReLU(inplace=True)

        # 时间嵌入块
        self.time_mlp = nn.Sequential(
            nn.SiLU(),  # SiLU激活
            nn.Linear(feat_channels * 4, feat_channels * 2),
        )  # 线性变换

        # 分类和回归头
        self.num_cls_convs = num_cls_convs
        self.num_reg_convs = num_reg_convs
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.use_fed_loss = use_fed_loss

        # 共享卷积/线性层 (如果层数相同可以共享更多，这里保持独立逻辑但优化结构)
        self.cls_convs = self._make_fc_layers(num_cls_convs, feat_channels)
        self.reg_convs = self._make_fc_layers(num_reg_convs, feat_channels)

        # 最终输出层：合并分类和回归以减少算子调用
        out_dim = num_classes if use_focal_loss or use_fed_loss else num_classes + 1
        self.fc_logits = nn.Linear(feat_channels, out_dim)
        self.fc_reg = nn.Linear(feat_channels, 4)

        self.scale_clamp = scale_clamp  # 缩放截断值
        self.bbox_weights = bbox_weights  # 边界框权重

    def _make_fc_layers(self, num_convs, feat_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(
                nn.Sequential(
                    nn.Linear(feat_channels, feat_channels, bias=False),
                    nn.LayerNorm(feat_channels),
                    nn.ReLU(inplace=True),
                )
            )
        return nn.Sequential(*layers)

    def forward(
        self,
        features: Tensor,
        bboxes: Tensor,
        proposals: Tensor,
        pooler,
        time_emb: Tensor,
    ):
        """
        前向传播

        Args:
            features: 特征金字塔, 列表形式
            bboxes (N, num_boxes, 4): 边界框
            proposals (N, num_boxes, feat_channels): 提案特征
            pooler: ROI池化器
            time_emb (N, time_dim): 时间嵌入

        Returns:
            class_logits (N, num_boxes, num_classes): 分类logits
            pred_bboxes (N, num_boxes, 4): 预测边界框
            obj_features (1, N*num_boxes, feat_channels): 对象特征
        """
        bs, num_boxes = bboxes.shape[:2]

        # 1. ROI 特征提取 (向量化处理)
        # 构造 batch 索引: [0,0...1,1...bs-1,bs-1...]
        device = bboxes.device
        ids = (
            torch.arange(bs, device=device)
            .view(-1, 1)
            .expand(bs, num_boxes)
            .reshape(-1, 1)
        )
        rois = torch.cat([ids.to(bboxes.dtype), bboxes.reshape(-1, 4)], dim=1)
        roi_features = pooler(features, rois)  # (bs*num_boxes, c, 7, 7)

        # 2. 提案特征初始化
        if proposals is None:
            # (bs*num_boxes, c, 7, 7) -> (bs, num_boxes, c)
            proposals = roi_features.view(bs, num_boxes, self.feat_channels, -1).mean(
                -1
            )

        # 3. 实例交互与注意力机制
        # 统一 Layout: (L, N, E) 适合 MultiheadAttention
        # proposals: (num_boxes, bs, feat_channels)
        proposals = proposals.permute(1, 0, 2)

        # 自注意力
        attn_shortcut, _ = self.self_attn(proposals, proposals, value=proposals)
        proposals = self.norm1(proposals + self.dropout1(attn_shortcut))

        # 实例交互 (DynamicConv)
        # roi_features: (49, bs*num_boxes, feat_channels)
        roi_features = roi_features.view(
            bs * num_boxes, self.feat_channels, -1
        ).permute(2, 0, 1)

        # proposals_flat: (1, bs*num_boxes, feat_channels)
        proposals_flat = proposals.permute(1, 0, 2).reshape(1, -1, self.feat_channels)
        attn_shortcut = self.inst_interact(proposals_flat, roi_features)

        obj_features = self.norm2(proposals_flat + self.dropout2(attn_shortcut))

        # FFN
        obj_shortcut = self.linear2(self.dropout(self.act(self.linear1(obj_features))))
        obj_features = self.norm3(obj_features + self.dropout3(obj_shortcut))

        # 4. 时间嵌入条件化 (使用广播优化)
        # obj_features: (1, bs*num_boxes, feat_channels) -> (bs, num_boxes, feat_channels)
        fc_feature = obj_features.view(bs, num_boxes, -1)

        scale_shift = self.time_mlp(time_emb)  # (bs, feat_channels * 2)
        scale, shift = scale_shift.chunk(2, dim=1)
        # 广播: (bs, 1, c) 与 (bs, num_boxes, c) 相乘
        fc_feature = fc_feature * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
        fc_feature = fc_feature.view(bs * num_boxes, -1)

        # 5. 分类和回归分支
        class_logits = self.fc_logits(self.cls_convs(fc_feature))
        bboxes_deltas = self.fc_reg(self.reg_convs(fc_feature))

        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        return (
            class_logits.view(bs, num_boxes, -1),
            pred_bboxes.view(bs, num_boxes, -1),
            obj_features,
        )

    def apply_deltas(self, deltas: Tensor, boxes: Tensor) -> Tensor:
        """将变换`deltas` (dx, dy, dw, dh) 应用到`boxes`

        Args:
            deltas (Tensor): 变换偏移, shape: (N, 4)
            boxes (Tensor): 要变换的框, shape: (N, 4)

        Returns:
            pred_boxes: 变换后的框, shape: (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        # 1. 计算框的中心和宽高
        # boxes: (x1, y1, x2, y2)
        wh = boxes[:, 2:] - boxes[:, :2]
        ctr = boxes[:, :2] + 0.5 * wh

        # 2. 解包偏移量并应用权重
        # deltas: (dx, dy, dw, dh)
        wx, wy, ww, wh_w = self.bbox_weights
        dx, dy, dw, dh = deltas.chunk(4, dim=-1)

        dx = dx / wx
        dy = dy / wy
        dw = dw / ww
        dh = dh / wh_w

        # 3. 防止数值溢出
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        # 4. 计算预测的中心和宽高
        pred_ctr = torch.cat([dx, dy], dim=-1) * wh + ctr
        pred_wh = torch.exp(torch.cat([dw, dh], dim=-1)) * wh

        # 5. 构造预测框: (x1, y1, x2, y2)
        x1y1 = pred_ctr - 0.5 * pred_wh
        x2y2 = pred_ctr + 0.5 * pred_wh

        return torch.cat([x1y1, x2y2], dim=-1)
