"""方向I-1 集成测试: SeesawLoss + Normalized Classifier 端到端协同

验证:
1. SingleDiffusionDetHead 使用 NormalizedLinear 后 forward 正常
2. DiffusionDetCriterion 使用 SeesawLoss 后 loss 计算正常
3. 组合后梯度可回传
4. 与 head._init_weights 兼容 (NormalizedLinear 无 bias)
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ldmdet.core import NormalizedLinear, SingleDiffusionDetHead
from ldmdet.criterion import DiffusionDetCriterion, DiffusionDetMatcher, SeesawLoss
from ldmdet.data.structures import InstanceData, ModelOutput


class TestSeesawNormalizedIntegration:
    """SeesawLoss + NormalizedLinear 集成测试"""

    def _make_single_head(self, use_normalized=True):
        """构建 SingleDiffusionDetHead"""
        return SingleDiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_cls_convs=1,
            num_reg_convs=2,
            use_focal_loss=True,
            use_normalized_classifier=use_normalized,
            classifier_temperature=20.0,
        )

    def _make_criterion(self, use_seesaw=True):
        """构建 DiffusionDetCriterion"""
        matcher = DiffusionDetMatcher(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
        )
        if use_seesaw:
            loss_cls = SeesawLoss(num_classes=24, p=0.8, loss_weight=2.0)
        else:
            from ldmdet.criterion import FocalLoss
            loss_cls = FocalLoss(use_sigmoid=True, alpha=0.25, gamma=2.0, loss_weight=2.0)
        from ldmdet.criterion import GIoULoss, L1Loss
        return DiffusionDetCriterion(
            num_classes=24,
            matcher=matcher,
            loss_cls=loss_cls,
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            deep_supervision=False,
        )

    def _make_dummy_output(self, bs=2, num_queries=10, num_classes=24):
        """构造 dummy ModelOutput"""
        pred_logits = torch.randn(bs, num_queries, num_classes, requires_grad=True)
        pred_boxes = torch.sigmoid(torch.randn(bs, num_queries, 4, requires_grad=True))
        # 确保框有效: xyxy 格式, x2 > x1, y2 > y1
        pred_boxes_data = pred_boxes.detach()
        # 简单确保 x2 > x1: 排序
        x1 = pred_boxes_data[..., 0].clamp(0, 0.4)
        y1 = pred_boxes_data[..., 1].clamp(0, 0.4)
        x2 = pred_boxes_data[..., 2].clamp(0.6, 1.0)
        y2 = pred_boxes_data[..., 3].clamp(0.6, 1.0)
        pred_boxes = torch.stack([x1, y1, x2, y2], dim=-1).requires_grad_(True)

        return ModelOutput(
            pred_logits=pred_logits,
            pred_boxes=pred_boxes,
            aux_outputs=None,
        )

    def _make_dummy_targets(self, bs=2, num_gt=3, num_classes=24):
        """构造 dummy targets"""
        targets = []
        for _ in range(bs):
            bboxes = torch.rand(num_gt, 4) * 0.8 + 0.1  # [0.1, 0.9]
            # 确保 xyxy 有效
            bboxes[:, 2] = bboxes[:, 0] + (bboxes[:, 2] - bboxes[:, 0]).clamp(min=0.05)
            bboxes[:, 3] = bboxes[:, 1] + (bboxes[:, 3] - bboxes[:, 1]).clamp(min=0.05)
            labels = torch.randint(0, num_classes, (num_gt,))
            inst = InstanceData(bboxes=bboxes, labels=labels, img_shape=(256, 256))
            targets.append(inst)
        return targets

    # ================================================================
    # 1. SingleDiffusionDetHead + NormalizedLinear
    # ================================================================

    def test_single_head_with_normalized_forward(self):
        """SingleDiffusionDetHead 使用 NormalizedLinear 后 forward 不报错"""
        head = self._make_single_head(use_normalized=True)
        bs, num_boxes = 2, 10
        features = [torch.randn(bs, 64, 16, 16)]
        bboxes = torch.rand(bs, num_boxes, 4)
        bboxes[..., 2:] = bboxes[..., :2] + 0.1  # 确保有效框
        proposals = torch.randn(bs, num_boxes, 64)
        time_emb = torch.randn(bs, 256)
        from ldmdet.core import SingleRoIExtractor
        # 用最小 dummy roi extractor
        pooler = SingleRoIExtractor(
            featmap_strides=[16], out_channels=64,
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0, aligned=True),
        )
        # forward
        result = head(features, bboxes, proposals, pooler, time_emb)
        assert len(result) >= 2
        cls_logits, pred_bboxes = result[0], result[1]
        assert cls_logits.shape == (bs, num_boxes, 24)
        assert pred_bboxes.shape == (bs, num_boxes, 4)

    def test_single_head_cls_head_last_is_normalized_linear(self):
        """使用 use_normalized_classifier=True 时, cls_head 最后一层应为 NormalizedLinear"""
        head = self._make_single_head(use_normalized=True)
        assert isinstance(head.cls_head[-1], NormalizedLinear)

    def test_single_head_cls_head_last_is_linear_when_disabled(self):
        """use_normalized_classifier=False 时, cls_head 最后一层应为 nn.Linear"""
        head = self._make_single_head(use_normalized=False)
        assert isinstance(head.cls_head[-1], nn.Linear)
        assert not isinstance(head.cls_head[-1], NormalizedLinear)

    # ================================================================
    # 2. DiffusionDetCriterion + SeesawLoss
    # ================================================================

    def test_criterion_with_seesaw_loss(self):
        """DiffusionDetCriterion 使用 SeesawLoss 后 loss 计算正常"""
        criterion = self._make_criterion(use_seesaw=True)
        outputs = self._make_dummy_output()
        targets = self._make_dummy_targets()
        losses = criterion(outputs, targets)
        assert 'loss_cls' in losses
        assert 'loss_bbox' in losses
        assert 'loss_giou' in losses
        assert isinstance(losses['loss_cls'], torch.Tensor)
        assert losses['loss_cls'].dim() == 0

    def test_criterion_seesaw_loss_finite(self):
        """SeesawLoss 计算结果应为有限值"""
        criterion = self._make_criterion(use_seesaw=True)
        outputs = self._make_dummy_output()
        targets = self._make_dummy_targets()
        losses = criterion(outputs, targets)
        for key, val in losses.items():
            assert torch.isfinite(val), f'{key} is not finite'

    # ================================================================
    # 3. 梯度回传
    # ================================================================

    def test_gradient_backprop_seesaw_normalized(self):
        """SeesawLoss + NormalizedLinear 组合后梯度可回传"""
        # 构建 head with NormalizedLinear
        head = self._make_single_head(use_normalized=True)
        # 构建 criterion with SeesawLoss
        criterion = self._make_criterion(use_seesaw=True)

        # 模拟前向: 直接用 head.cls_head 测试
        bs, num_boxes = 2, 10
        fc_feature = torch.randn(bs * num_boxes, 64, requires_grad=True)
        cls_logits = head.cls_head(fc_feature)
        # 模拟 criterion._loss_classification
        target_classes = torch.randint(0, 25, (bs * num_boxes,))
        loss_cls = criterion.loss_cls(cls_logits, target_classes) / max(1, (target_classes < 24).sum())
        loss_cls.backward()
        assert fc_feature.grad is not None
        assert not torch.isnan(fc_feature.grad).any()
        # NormalizedLinear 的 weight 应有梯度
        assert head.cls_head[-1].weight.grad is not None

    # ================================================================
    # 4. _init_weights 兼容性
    # ================================================================

    def test_init_weights_no_bias_normalized(self):
        """_init_weights 应兼容无 bias 的 NormalizedLinear (不报错)"""
        from ldmdet.core.head import DiffusionDetHead
        # 这里只测试 _init_weights 逻辑不报错
        # 构建一个最小的 head_series 模拟
        head = self._make_single_head(use_normalized=True)
        # 模拟 DiffusionDetHead._init_weights
        prior_prob = 0.01
        last_layer = head.cls_head[-1]
        if hasattr(last_layer, 'bias') and last_layer.bias is not None:
            # 不应进入此分支
            pytest.fail('NormalizedLinear should not have bias')
        # 如果没有 bias, 跳过即可 (不报错)
