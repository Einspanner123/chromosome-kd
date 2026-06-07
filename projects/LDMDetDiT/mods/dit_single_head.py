import math

import torch
import torch.nn as nn
from torch import Tensor

from .dit_block import DiTBlock


class DiTSingleHead(nn.Module):
    """单个 DiT Block + 预测头

    替代 SingleDiffusionDetHead，去掉 RoIAlign 和 DynamicConv，
    使用 Deformable Cross-Attention 实现框-图像交互。

    regression_mode:
      - 'delta': 传统 delta regression (apply_deltas)，适合图像坐标空间
      - 'direct': reg_head 直接输出 velocity v (v-prediction 模式)
        v = x_noise - x_start, 推理时 x_next = x_t + v * dt
        无 sigmoid/激活函数, 确保 RF 向量场一致性
    """

    def __init__(
        self,
        num_classes: int = 80,
        feat_channels: int = 256,
        dim_feedforward: int = 2048,
        num_cls_convs: int = 1,
        num_reg_convs: int = 3,
        num_heads: int = 8,
        num_fpn_levels: int = 4,
        num_ref_points: int = 8,
        dropout: float = 0.0,
        use_focal_loss: bool = True,
        use_fed_loss: bool = False,
        scale_clamp: float = math.log(8.0),
        bbox_weights: tuple = (1.0, 1.0, 1.0, 1.0),
        use_objectness: bool = False,
        prediction_mode: str = 'x0',
        adaln_params: int = 9,
        regression_mode: str = 'direct',
        use_adaln_zero: bool = True,
        num_blocks: int = 1,
    ):
        """
        Args:
            num_classes: 类别数
            feat_channels: 特征通道数
            dim_feedforward: FFN 隐藏层维度
            num_cls_convs: 分类头卷积层数
            num_reg_convs: 回归头卷积层数
            num_heads: 注意力头数
            num_fpn_levels: FPN 层级数
            num_ref_points: Deformable Attention 采样点数 K
            dropout: dropout 率
            use_focal_loss: 是否使用 focal loss
            use_fed_loss: 是否使用 fed loss
            scale_clamp: bbox 缩放截断值 (仅 delta 模式)
            bbox_weights: bbox 回归权重 (仅 delta 模式)
            use_objectness: 是否使用 objectness 预测头
            prediction_mode: "x0" 或 "velocity"
            adaln_params: AdaLN-Zero 参数组数, 9 或 6
            regression_mode: "delta" (传统 delta regression) 或 "direct" (raw 空间直接预测)
            use_adaln_zero: 是否开启 AdaLN-Zero 零初始化
            num_blocks: DiTBlock 堆叠数量
        """
        super().__init__()
        self.feat_channels = feat_channels
        self.use_objectness = use_objectness
        self.prediction_mode = prediction_mode
        self.regression_mode = regression_mode
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

        self.dit_blocks = nn.ModuleList([
            DiTBlock(
                feat_channels=feat_channels,
                num_heads=num_heads,
                num_fpn_levels=num_fpn_levels,
                num_ref_points=num_ref_points,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                adaln_params=adaln_params,
                use_adaln_zero=use_adaln_zero,
            )
            for _ in range(num_blocks)
        ])

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

        self.objectness_head = None
        if use_objectness:
            self.objectness_head = nn.Sequential(
                nn.Linear(feat_channels, feat_channels, bias=False),
                nn.LayerNorm(feat_channels),
                nn.ReLU(inplace=True),
                nn.Linear(feat_channels, 1),
            )

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

    def forward(
        self,
        box_tokens: Tensor,
        fpn_flattened: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        time_emb: Tensor,
        bbox_coords: Tensor,
    ) -> tuple:
        """前向传播

        Args:
            box_tokens: (bs, N, C)
            fpn_flattened: (bs, sum(H_l*W_l), C)
            spatial_shapes: (num_levels, 2)
            level_start_index: (num_levels,)
            time_emb: (bs, time_dim)
            bbox_coords: (bs, N, 4) 归一化 xyxy [0,1]

        Returns:
            class_logits: (bs, N, num_classes)
            pred_bboxes: (bs, N, 4) 归一化 xyxy [0,1]
            updated_tokens: (bs, N, C)
            objectness: (bs, N, 1) or None
            pred_velocity: (bs, N, 4) or None
        """
        updated_tokens = box_tokens
        for dit_block in self.dit_blocks:
            updated_tokens = dit_block(
                updated_tokens,
                fpn_flattened,
                spatial_shapes,
                level_start_index,
                time_emb,
                bbox_coords,
            )

        fc_feature = updated_tokens

        class_logits = self.cls_head(fc_feature)
        pred_bboxes = self._predict_bboxes(fc_feature, bbox_coords)

        objectness = None
        if self.objectness_head is not None:
            objectness = self.objectness_head(fc_feature)

        pred_velocity = None
        if self.velocity_head is not None:
            pred_velocity = self.velocity_head(fc_feature)

        return (
            class_logits,
            pred_bboxes,
            updated_tokens,
            objectness,
            pred_velocity,
        )

    def _predict_bboxes(self, fc_feature: Tensor, bboxes: Tensor) -> Tensor:
        if self.regression_mode == 'direct':
            # v-prediction 模式: reg_head 直接输出 velocity v
            # v = x_noise - x_start (RF 理论速度)
            # 推理时: x_next = x_t + v * dt
            # 训练时: loss = MSE(v_pred, v_target)
            # 无 sigmoid/激活函数, 确保 RF 向量场一致性
            return self.reg_head(fc_feature)
        bboxes_deltas = self.reg_head(fc_feature)
        bs, n, _ = bboxes.shape
        pred_bboxes = self.apply_deltas(
            bboxes_deltas.reshape(-1, 4), bboxes.reshape(-1, 4)
        )
        return pred_bboxes.reshape(bs, n, 4)

    def apply_deltas(self, deltas: Tensor, boxes: Tensor) -> Tensor:
        """将变换 deltas 应用到 boxes (归一化坐标 [0,1])

        对 widths/heights 取 abs + clamp(min=eps) 防止负值/零值导致 NaN。
        结果 clamp 到 [0,1] 防止越界。
        """
        boxes = boxes.to(deltas.dtype)

        widths = (boxes[:, 2] - boxes[:, 0]).abs().clamp(min=1e-4)
        heights = (boxes[:, 3] - boxes[:, 1]).abs().clamp(min=1e-4)
        ctr_x = boxes[:, 0] + 0.5 * (boxes[:, 2] - boxes[:, 0])
        ctr_y = boxes[:, 1] + 0.5 * (boxes[:, 3] - boxes[:, 1])

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

        pred_boxes = pred_boxes.clamp(0, 1)

        return pred_boxes
