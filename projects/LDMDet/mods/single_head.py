import math

import torch
import torch.nn as nn
from torch import Tensor

from .modules import DynamicConv
from .utils import bbox2roi


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
                num_classes if use_focal_loss or use_fed_loss else num_classes + 1,
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
        reg_layers.append(nn.Linear(feat_channels, 4))  # 边界框回归层
        self.reg_head = nn.Sequential(*reg_layers)
        self.scale_clamp = scale_clamp  # 缩放截断值
        self.bbox_weights = bbox_weights  # 边界框权重

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
        """
        bs, num_boxes = bboxes.shape[:2]  # 获取批次大小和框数量

        # ROI特征提取
        rois = bbox2roi([bboxes[i] for i in range(bs)])  # 转换为ROI格式
        roi_features = pooler(features, rois)  # (bs*num_boxess, c, 7, 7)

        # 处理提案特征
        if proposals is None:
            # 如果没有提供提案特征,则从ROI特征计算均值
            # 优化: 使用 flatten + mean 更简洁高效
            proposals = (
                roi_features.flatten(2).mean(-1).view(bs, num_boxes, self.feat_channels)
            )
            # shape: (bs, num_boxes, feat_channels)

        # 调整ROI特征形状以适应注意力机制
        roi_features = roi_features.view(
            bs * num_boxes,
            self.feat_channels,
            -1,
        ).permute(2, 0, 1)  # shape: (49, bs*num_boxes, feat_channels)

        # 自注意力
        proposals = proposals.view(
            bs,
            num_boxes,
            self.feat_channels,
        ).permute(1, 0, 2)  # shape: (num_boxes, N, feat_channels)

        attn_shortcut, _ = self.self_attn(
            proposals,
            proposals,
            value=proposals,
        )  # shape: (num_boxes, N, feat_channels)
        # 残差连接和归一化
        proposals = proposals + self.dropout1(attn_shortcut)
        proposals = self.norm1(proposals)

        # 实例交互
        # 调整形状以进行实例交互
        proposals = (
            proposals.view(num_boxes, bs, self.feat_channels)
            .permute(1, 0, 2)
            .reshape(1, bs * num_boxes, self.feat_channels)
        )  # (1, N*num_boxes, feat_channels)
        # 动态卷积交互
        attn_shortcut = self.inst_interact(proposals, roi_features)

        # 残差连接和归一化
        proposals = proposals + self.dropout2(attn_shortcut)
        obj_features = self.norm2(proposals)  # (1, N*num_boxes, feat_channels)

        # 对象特征处理（前馈网络）
        obj_shortcut = self.linear2(
            self.dropout(self.act(self.linear1(obj_features)))
        )  # 前馈网络
        obj_features = obj_features + self.dropout3(obj_shortcut)  # 残差连接
        obj_features = self.norm3(obj_features)  # 归一化

        # 时间嵌入条件化
        fc_feature = obj_features.transpose(0, 1).reshape(
            bs * num_boxes, -1
        )  # (N*num_boxes, feat_channels)

        scale_shift = self.time_mlp(time_emb)  # 时间嵌入MLP
        scale_shift = torch.repeat_interleave(
            scale_shift, num_boxes, dim=0
        )  # 扩展到每个框
        scale, shift = scale_shift.chunk(2, dim=1)  # 分割为缩放和偏移
        # 时间条件化: feature = feature * (scale + 1) + shift
        fc_feature = fc_feature * (scale + 1) + shift

        # 分类和回归分支
        # 通过分类模块
        class_logits = self.cls_head(fc_feature)
        # out shape: (N*num_boxes, num_classes)

        # 通过回归模块
        bboxes_deltas = self.reg_head(fc_feature)
        # out shape: (N*num_boxes, 4)

        pred_bboxes = self.apply_deltas(
            bboxes_deltas, bboxes.view(-1, 4)
        )  # 应用偏移到边界框

        # 调整输出形状
        return (
            class_logits.view(bs, num_boxes, -1),
            pred_bboxes.view(bs, num_boxes, -1),
            obj_features,
        )

    def apply_deltas(self, deltas, boxes):
        """将变换`deltas` (dx, dy, dw, dh) 应用到`boxes`

        Args:
            deltas (Tensor): 变换偏移,shape: (N, k*4), k >= 1
            boxes (Tensor): 要变换的框,shape: (N, 4)

        Returns:
            pred_boxes: 变换后的框,shape: (N, k*4)
        """
        boxes = boxes.to(deltas.dtype)  # 确保数据类型一致

        # 计算框的宽度和高度
        widths = boxes[:, 2] - boxes[:, 0]  # x2 - x1
        heights = boxes[:, 3] - boxes[:, 1]  # y2 - y1
        ctr_x = boxes[:, 0] + 0.5 * widths  # 中心x坐标
        ctr_y = boxes[:, 1] + 0.5 * heights  # 中心y坐标

        wx, wy, ww, wh = self.bbox_weights  # 边界框权重
        dx = deltas[:, 0::4] / wx  # x方向偏移
        dy = deltas[:, 1::4] / wy  # y方向偏移
        dw = deltas[:, 2::4] / ww  # 宽度偏移
        dh = deltas[:, 3::4] / wh  # 高度偏移

        # 防止将过大值送入torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)  # 截断宽度偏移
        dh = torch.clamp(dh, max=self.scale_clamp)  # 截断高度偏移

        # 计算预测的中心坐标和宽高
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]  # 预测中心x
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]  # 预测中心y
        pred_w = torch.exp(dw) * widths[:, None]  # 预测宽度
        pred_h = torch.exp(dh) * heights[:, None]  # 预测高度

        # 构造预测框: (x1, y1, x2, y2)
        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes
