"""DiffuDETR 检测器 — mmdet BaseDetector 桥接

参考 experiments/mmdet_bridge/setdiff_detector.py 的桥接模式:
  - backbone/neck 通过 mmdet MODELS.build 构建
  - bbox_head (DiffuDETRHead) 通过纯 PyTorch 构建
  - GT boxes: xyxy 像素 → cxcywh 归一化 → [-scale, scale] 扩散空间
  - 预测框: [-scale, scale] → [0,1] → cxcywh 像素 → xyxy 像素

关键设计决策:
  - GT 转换: (norm_cxcywh * 2 - 1) * scale  (对齐 setdiff gt_to_diffusion_space)
  - 预测转换: (clamp(x, -s, s) / s + 1) / 2  (对齐 setdiff diffusion_to_norm_space)
  - 多尺度特征展平: head 内部处理 (transformer._flatten_features)
  - 推理结果: DetDataSample.pred_instances (bboxes xyxy像素, scores, labels)
"""

import copy
from typing import List, Tuple

import torch
from mmengine.structures import InstanceData as MMInstanceData
from torch import Tensor

from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .criterion import box_xyxy_to_cxcywh
from .diffudetr_head import DiffuDETRHead


@MODELS.register_module(name='DiffuDETR', force=True)
class DiffuDETRDetector(BaseDetector):
    """DiffuDETR 检测器 — mmdet BaseDetector 兼容包装.

    backbone/neck 通过 mmdet 构建, bbox_head 通过 DiffuDETRHead (纯 PyTorch) 构建.

    关键点:
        - GT boxes 格式转换: mmdet xyxy 像素 → cxcywh 归一化 → 扩散 [-scale, scale]
          (对齐 setdiff_detector.gt_to_diffusion_space)
        - 预测框格式转换: 扩散 [-scale, scale] → [0,1] → cxcywh 像素 → xyxy 像素
          (对齐 setdiff_detector.diffusion_to_norm_space)
    """

    def __init__(
        self,
        backbone: ConfigType,
        neck: ConfigType,
        bbox_head: ConfigType,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg
        )

        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.bbox_head = self._build_head(bbox_head)

    def _build_head(self, cfg: ConfigType) -> DiffuDETRHead:
        """从配置构建 DiffuDETRHead (纯 PyTorch).

        Args:
            cfg: head 配置 dict (含 type='DiffuDETRHead' 等参数).
        """
        cfg = copy.deepcopy(cfg)
        cfg.pop('type', None)
        return DiffuDETRHead(**cfg)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor, ...]:
        """backbone → neck → multi-scale features."""
        x = self.backbone(batch_inputs)
        if self.neck is not None:
            x = self.neck(x)
        return x

    @staticmethod
    def gt_to_diffusion_space(gt_cxcywh_norm: Tensor, scale: float) -> Tensor:
        """GT 从 [0,1] cxcywh 缩放到 [-scale, +scale] 匹配 N(0,1) 噪声.

        对齐 setdiff_detector.gt_to_diffusion_space:
            gt_diffusion = (norm_gt_cxcywh * 2 - 1) * scale

        Args:
            gt_cxcywh_norm: [N, 4] GT boxes in [0, 1] cxcywh.
            scale: 缩放因子 (典型 2.0).

        Returns:
            gt_diffusion: [N, 4] in [-scale, +scale].
        """
        return (gt_cxcywh_norm * 2.0 - 1.0) * scale

    @staticmethod
    def diffusion_to_norm_space(
        pred_diffusion: Tensor, scale: float
    ) -> Tensor:
        """预测框从 [-scale, +scale] 逆缩放到 [0, 1] cxcywh.

        对齐 setdiff_detector.diffusion_to_norm_space:
            bboxes = (raw.clamp(-s, s) / s + 1) / 2

        Args:
            pred_diffusion: [N, 4] predicted boxes in diffusion space.
            scale: 缩放因子.

        Returns:
            pred_norm: [N, 4] in [0, 1] cxcywh.
        """
        s = scale
        clamped = pred_diffusion.clamp(-s, s)
        return (clamped / s + 1.0) / 2.0

    def loss(
        self,
        batch_inputs: Tensor,
        batch_data_samples: List[DetDataSample],
    ) -> dict:
        """训练前向: 计算 loss.

        步骤:
            1. extract_feat → multi-scale features
            2. 从 batch_data_samples 提取 gt_bboxes (xyxy 像素) 和 gt_labels
            3. GT boxes 格式转换: xyxy 像素 → cxcywh 归一化 → diffusion [-scale, scale]
            4. 调用 self.bbox_head.forward_train(features, gt_boxes_list, gt_labels_list)
            5. 返回 loss dict
        """
        x = self.extract_feat(batch_inputs)
        multi_level_feats = list(x) if isinstance(x, (list, tuple)) else [x]

        scale = self.bbox_head.scale
        gt_boxes_list: List[Tensor] = []
        gt_labels_list: List[Tensor] = []
        for ds in batch_data_samples:
            img_shape = ds.metainfo['img_shape']
            img_h, img_w = img_shape[0], img_shape[1]

            gt_bboxes = ds.gt_instances.bboxes  # [M, 4] xyxy 像素坐标
            gt_labels = ds.gt_instances.labels  # [M]

            # xyxy 像素 → cxcywh 像素 → cxcywh 归一化 (÷ img_w, ÷ img_h)
            gt_cxcywh = box_xyxy_to_cxcywh(gt_bboxes)
            norm_scale = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h])
            gt_cxcywh_norm = gt_cxcywh / norm_scale

            # cxcywh 归一化 [0,1] → diffusion [-scale, +scale]
            gt_diffusion = self.gt_to_diffusion_space(gt_cxcywh_norm, scale)

            gt_boxes_list.append(gt_diffusion)
            gt_labels_list.append(gt_labels)

        return self.bbox_head.forward_train(
            multi_level_feats, gt_boxes_list, gt_labels_list
        )

    def predict(
        self,
        batch_inputs: Tensor,
        batch_data_samples: List[DetDataSample],
        rescale: bool = True,
    ) -> List[DetDataSample]:
        """推理前向: 生成检测结果.

        步骤:
            1. extract_feat → multi-scale features
            2. 调用 self.bbox_head.forward_inference(features, img_shapes)
               → 每张图 dict('boxes' xyxy像素, 'scores', 'labels')
            3. 如果 rescale=True, 除以 scale_factor 恢复到原图尺寸
            4. 格式化为 DetDataSample (pred_instances)
        """
        x = self.extract_feat(batch_inputs)
        multi_level_feats = list(x) if isinstance(x, (list, tuple)) else [x]

        # 提取图像尺寸
        img_shapes = []
        for ds in batch_data_samples:
            img_shape = ds.metainfo['img_shape']
            img_shapes.append((img_shape[0], img_shape[1]))

        # 推理 (head 内部处理扩散空间 → xyxy 像素)
        results = self.bbox_head.forward_inference(
            multi_level_feats, img_shapes
        )

        # 格式化 + rescale
        for i, ds in enumerate(batch_data_samples):
            boxes = results[i]['boxes']  # xyxy 像素 (当前尺度)
            scores = results[i]['scores']
            labels = results[i]['labels']

            # rescale: 除以 scale_factor 恢复到原图尺寸
            if rescale:
                scale_factor = ds.metainfo.get('scale_factor', None)
                if scale_factor is not None:
                    if not isinstance(scale_factor, Tensor):
                        scale_factor = boxes.new_tensor(scale_factor)
                    else:
                        scale_factor = scale_factor.to(
                            device=boxes.device, dtype=boxes.dtype
                        )
                    # 2-element [w, h] → 4-element [w, h, w, h]
                    if scale_factor.numel() == 2:
                        scale_factor = scale_factor.repeat(2)
                    boxes = boxes / scale_factor

            pred = MMInstanceData()
            pred.bboxes = boxes
            pred.scores = scores
            pred.labels = labels
            ds.pred_instances = pred

        return batch_data_samples

    def _forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples: List[DetDataSample],
    ) -> Tuple[Tensor, Tensor]:
        """raw forward (mode='tensor'), 返回原始张量不做后处理.

        主要用于 ONNX/TensorRT 导出.
        """
        x = self.extract_feat(batch_inputs)
        multi_level_feats = list(x) if isinstance(x, (list, tuple)) else [x]

        # 简化: 仅返回最后一步预测 (pred_x0_norm 在 [0,1] cxcywh)
        device = multi_level_feats[0].device
        B = multi_level_feats[0].shape[0]
        N = self.bbox_head.num_queries
        noise = torch.randn(B, N, 4, device=device)
        time_pairs = self.bbox_head.scheduler.get_time_pairs()
        time, time_next = time_pairs[0]
        pred_x0_norm, pred_logits = self.bbox_head._single_step_predict(
            multi_level_feats, noise, time, device
        )
        return pred_logits, pred_x0_norm
