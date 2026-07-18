"""SetDiffDetector — mmdet BaseDetector 桥接

将 setdiff 纯 PyTorch 核心包装为 mmdet 兼容的检测器。
"""

import copy
from typing import List, Tuple

import torch
from mmengine.structures import InstanceData as MMInstanceData
from torch import Tensor

from ldmdet.utils.box_ops import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from setdiff.models.set_head import JointDiffusionHead


@MODELS.register_module(name='SetDiff', force=True)
class SetDiffDetector(BaseDetector):
    """SetDiff 检测器 — mmdet BaseDetector 兼容包装。

    backbone/neck 通过 mmdet 构建，bbox_head 通过 setdiff 纯 PyTorch 构建
    (JointDiffusionHead)。

    关键点:
        - GT boxes 格式转换: mmdet xyxy 像素坐标 → setdiff cxcywh 归一化
        - snr_scale 缩放: GT [0,1] → diffusion [-snr_scale, +snr_scale]
          匹配 N(0,1) 噪声尺度 (对齐 LDMDet head.py:789-790)
        - 预测框格式转换: diffusion [-s,s] → [0,1] → cxcywh 像素 → xyxy 像素
    """

    @staticmethod
    def gt_to_diffusion_space(
        gt_cxcywh_norm: Tensor, snr_scale: float
    ) -> Tensor:
        """GT 从 [0,1] cxcywh 缩放到 [-snr_scale, +snr_scale] 匹配 N(0,1) 噪声.

        对齐 LDMDet head.py:789-790:
            gt_diffusion = (norm_gt_cxcywh * 2 - 1) * snr_scale

        Args:
            gt_cxcywh_norm: [N, 4] GT boxes in [0, 1] cxcywh.
            snr_scale: 缩放因子 (典型 2.0).

        Returns:
            gt_diffusion: [N, 4] in [-snr_scale, +snr_scale].
        """
        return (gt_cxcywh_norm * 2.0 - 1.0) * snr_scale

    @staticmethod
    def diffusion_to_norm_space(
        pred_diffusion: Tensor, snr_scale: float
    ) -> Tensor:
        """预测框从 [-snr_scale, +snr_scale] 逆缩放到 [0, 1] cxcywh.

        对齐 LDMDet sampling.py:278-284:
            bboxes = (raw.clamp(-s, s) / s + 1) / 2

        clamp 防止模型输出超出范围 (训练初期可能发生).

        Args:
            pred_diffusion: [N, 4] predicted boxes in diffusion space.
            snr_scale: 缩放因子.

        Returns:
            pred_norm: [N, 4] in [0, 1] cxcywh.
        """
        s = snr_scale
        clamped = pred_diffusion.clamp(-s, s)
        return (clamped / s + 1.0) / 2.0

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

    def _build_head(self, cfg: ConfigType) -> JointDiffusionHead:
        """从配置构建 setdiff JointDiffusionHead。

        JointDiffusionHead.__init__ 参数:
            num_queries, feat_channels, num_heads, num_layers,
            dim_feedforward, num_classes, snr_scale=2.0,
            num_sample_steps=4, sampler='euler'
        """
        cfg = copy.deepcopy(cfg)
        cfg.pop('type', None)
        return JointDiffusionHead(**cfg)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor, ...]:
        """backbone → neck → multi-scale features."""
        x = self.backbone(batch_inputs)
        if self.neck is not None:
            x = self.neck(x)
        return x

    @staticmethod
    def _flatten_features(features) -> Tensor:
        """Flatten multi-scale features: tuple of [B, C, H_i, W_i] → [B, sum(H_i*W_i), C].

        参考 setdiff/models/detector.py 中的 _flatten_features 实现。
        """
        if isinstance(features, Tensor):
            features = [features]
        flat_list = []
        for feat in features:
            # [B, C, H, W] → [B, H*W, C]
            flat_list.append(feat.flatten(2).transpose(1, 2))
        return torch.cat(flat_list, dim=1)

    def loss(
        self,
        batch_inputs: Tensor,
        batch_data_samples: List[DetDataSample],
    ) -> dict:
        """训练前向: 计算 loss。

        步骤:
            1. extract_feat → multi-scale features
            2. _flatten_features → [B, HW, C]
            3. 从 batch_data_samples 提取 gt_bboxes (xyxy 像素) 和 gt_labels
            4. GT boxes 格式转换: xyxy 像素 → cxcywh 归一化 → diffusion [-s, s]
               (snr_scale 缩放, 对齐 LDMDet head.py:789-790)
            5. 调用 self.bbox_head(flat_features, gt_boxes_list, gt_labels_list)
            6. 返回 loss dict
        """
        x = self.extract_feat(batch_inputs)
        flat_features = self._flatten_features(x)

        snr_scale = self.bbox_head.snr_scale
        gt_boxes_list: List[Tensor] = []
        gt_labels_list: List[Tensor] = []
        for ds in batch_data_samples:
            img_shape = ds.metainfo['img_shape']
            img_h, img_w = img_shape[0], img_shape[1]

            gt_bboxes = ds.gt_instances.bboxes  # [M, 4] xyxy 像素坐标
            gt_labels = ds.gt_instances.labels  # [M]

            # xyxy 像素 → cxcywh 像素 → cxcywh 归一化 (÷ img_w, ÷ img_h)
            gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
            scale = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h])
            gt_cxcywh_norm = gt_cxcywh / scale

            # cxcywh 归一化 [0,1] → diffusion [-snr_scale, +snr_scale]
            # 匹配 N(0,1) 噪声尺度 (修复 SetDiff mAP=0 根因)
            gt_diffusion = self.gt_to_diffusion_space(
                gt_cxcywh_norm, snr_scale
            )

            gt_boxes_list.append(gt_diffusion)
            gt_labels_list.append(gt_labels)

        return self.bbox_head(flat_features, gt_boxes_list, gt_labels_list)

    def predict(
        self,
        batch_inputs: Tensor,
        batch_data_samples: List[DetDataSample],
        rescale: bool = True,
    ) -> List[DetDataSample]:
        """推理前向: 生成检测结果。

        步骤:
            1. extract_feat → flatten → [B, HW, C]
            2. 调用 self.bbox_head.predict(flat_features)
               → pred_logits [B, N, C], pred_boxes [B, N, 4] (diffusion [-s, s])
            3. 预测框格式转换: diffusion [-s, s] → [0, 1] → cxcywh 像素 → xyxy 像素
               (snr_scale 逆缩放, 对齐 LDMDet sampling.py:278-284)
            4. 如果 rescale=True, 除以 scale_factor 恢复到原图尺寸
            5. 格式化为 DetDataSample (pred_instances)
        """
        x = self.extract_feat(batch_inputs)
        flat_features = self._flatten_features(x)

        # SetDiff head.predict 返回 dict: 'pred_logits' [B, N, C],
        # 'pred_boxes' [B, N, 4] (diffusion [-snr_scale, +snr_scale])
        outputs = self.bbox_head.predict(flat_features)
        pred_logits = outputs['pred_logits']  # [B, N, C]
        pred_boxes = outputs['pred_boxes']  # [B, N, 4]

        snr_scale = self.bbox_head.snr_scale
        B = pred_logits.shape[0]
        for i in range(B):
            ds = batch_data_samples[i]
            img_shape = ds.metainfo['img_shape']
            img_h, img_w = img_shape[0], img_shape[1]

            # diffusion [-s, s] → [0, 1] cxcywh → cxcywh 像素 → xyxy 像素
            boxes = pred_boxes[i]  # [N, 4] in diffusion space
            boxes = self.diffusion_to_norm_space(boxes, snr_scale)
            scale = boxes.new_tensor([img_w, img_h, img_w, img_h])
            boxes = boxes * scale
            boxes = bbox_cxcywh_to_xyxy(boxes)

            # 分类: sigmoid → max 得到 (score, label)
            scores = torch.sigmoid(pred_logits[i])  # [N, C]
            conf, labels = scores.max(dim=-1)  # [N], [N]

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
            pred.scores = conf
            pred.labels = labels
            ds.pred_instances = pred

        return batch_data_samples

    def _forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples: List[DetDataSample],
    ) -> Tuple[Tensor, Tensor]:
        """raw forward (mode='tensor'), 返回原始张量不做后处理。

        主要用于 ONNX/TensorRT 导出。
        """
        x = self.extract_feat(batch_inputs)
        flat_features = self._flatten_features(x)
        outputs = self.bbox_head(flat_features)
        return outputs['pred_logits'], outputs['pred_boxes']
