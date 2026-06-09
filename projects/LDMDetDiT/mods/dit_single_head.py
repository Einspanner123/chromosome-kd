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
            prediction_mode: "x0" 或 "velocity"
            adaln_params: AdaLN-Zero 参数组数, 9 或 6
            regression_mode: "delta" (传统 delta regression) 或 "direct" (raw 空间直接预测)
            use_adaln_zero: 是否开启 AdaLN-Zero 零初始化
            num_blocks: DiTBlock 堆叠数量
        """
        super().__init__()
        self.feat_channels = feat_channels
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

        # 初始化回归头最后一层的 bias：w 和 h 偏向小框
        # sigmoid(-3) ≈ 0.047, sigmoid(0) = 0.5
        # 对于染色体检测 (目标占图 1-5%)，初始小框避免 100 个 proposal 全部
        # 预测相同的 [0.25, 0.25, 0.75, 0.75] 导致 GIoU 梯度同质化
        self._init_reg_bias()

    def _init_reg_bias(self):
        """初始化 reg_head 最后层 bias，使初始预测多样化

        默认 Xavier init 下 sigmoid(0)=0.5，导致 cx=cy=0.5, w=h=0.5，
        所有 proposal 预测同一个 [0.25, 0.25, 0.75, 0.75] 框。
        这会引发 GIoU 梯度同质化 (所有 proposal 接收相同梯度方向)
        并因 NMS 把 100 个框压成 1 个，导致 mAP=0。

        修复: w/h bias 设为 -3 → sigmoid(-3)≈0.047，初始框足够小以避免
        完全重叠。不缩小权重 (保留 Xavier init 方差)，使不同 proposal 的
        fc_feature 差异通过 reg_head 产生不同的预测位置。
        """
        last_layer = list(self.reg_head.children())[-1]
        with torch.no_grad():
            # cx, cy → 0 (中心点在 0.5，由 fc_feature 方差提供多样性)
            last_layer.bias[0] = 0.0
            last_layer.bias[1] = 0.0
            # w, h → -3 (小框, 约 5% 图像面积)
            last_layer.bias[2] = -3.0
            last_layer.bias[3] = -3.0

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
            None: placeholder (原 objectness, 已移除)
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

        return (
            class_logits,
            pred_bboxes,
            updated_tokens,
            None,  # placeholder (原 objectness, 已移除)
            None,  # pred_velocity (unused, kept for interface compatibility)
        )

    def _predict_bboxes(self, fc_feature: Tensor, bboxes: Tensor) -> Tensor:
        if self.regression_mode == 'direct':
            # x0-prediction 模式: reg_head 输出归一化坐标 (与 LDMDet 一致)
            # 使用 sigmoid 保证输出在 [0,1]，GIoU 梯度不受时间步 t 缩放
            # velocity 由 DitHead 从 x0 反推: v = (x_t - x0) / t
            raw = self.reg_head(fc_feature)
            cx = torch.sigmoid(raw[..., 0])
            cy = torch.sigmoid(raw[..., 1])
            w = torch.sigmoid(raw[..., 2])
            h = torch.sigmoid(raw[..., 3])
            x1 = cx - w * 0.5
            y1 = cy - h * 0.5
            x2 = cx + w * 0.5
            y2 = cy + h * 0.5
            return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1)
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
