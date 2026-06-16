"""测试 ldmdet.data.structures — 数据结构"""

import torch
import pytest
from ldmdet.data.structures import ImageMeta, InstanceData, ModelOutput, DetectionResult


class TestImageMeta:
    def test_basic_creation(self):
        meta = ImageMeta(img_shape=(512, 512))
        assert meta.img_shape == (512, 512)

    def test_all_fields(self):
        meta = ImageMeta(
            img_shape=(512, 512),
            pad_shape=(512, 512),
            ori_shape=(600, 600),
            scale_factor=(0.85, 0.85),
        )
        assert meta.img_shape == (512, 512)
        assert meta.ori_shape == (600, 600)
        assert meta.scale_factor == (0.85, 0.85)

    def test_defaults(self):
        meta = ImageMeta(img_shape=(256, 256))
        assert meta.pad_shape is None
        assert meta.ori_shape is None
        assert meta.scale_factor is None


class TestInstanceData:
    def test_basic_creation(self):
        inst = InstanceData(
            bboxes=torch.rand(10, 4),
            labels=torch.randint(0, 24, (10,)),
            img_shape=(512, 512),
        )
        assert inst.bboxes.shape == (10, 4)
        assert inst.labels.shape == (10,)
        assert inst.img_shape == (512, 512)

    def test_empty(self):
        inst = InstanceData(
            bboxes=torch.zeros(0, 4),
            labels=torch.zeros(0, dtype=torch.long),
            img_shape=(512, 512),
        )
        assert inst.bboxes.shape == (0, 4)

    def test_attribute_access(self):
        inst = InstanceData(bboxes=torch.rand(5, 4), labels=torch.zeros(5, dtype=torch.long), img_shape=(512, 512))
        assert hasattr(inst, 'bboxes')
        assert hasattr(inst, 'labels')
        assert hasattr(inst, 'img_shape')


class TestModelOutput:
    def test_basic_creation(self):
        out = ModelOutput(
            pred_logits=torch.randn(2, 100, 24),
            pred_boxes=torch.rand(2, 100, 4),
        )
        assert out.pred_logits.shape == (2, 100, 24)
        assert out.pred_boxes.shape == (2, 100, 4)

    def test_with_aux_outputs(self):
        aux = [
            ModelOutput(pred_logits=torch.randn(2, 100, 24), pred_boxes=torch.rand(2, 100, 4))
            for _ in range(5)
        ]
        out = ModelOutput(
            pred_logits=torch.randn(2, 100, 24),
            pred_boxes=torch.rand(2, 100, 4),
            aux_outputs=aux,
        )
        assert len(out.aux_outputs) == 5

    def test_with_pred_count(self):
        out = ModelOutput(
            pred_logits=torch.randn(2, 100, 24),
            pred_boxes=torch.rand(2, 100, 4),
            pred_count=torch.randn(2, 1),
        )
        assert out.pred_count.shape == (2, 1)

    def test_defaults(self):
        out = ModelOutput(
            pred_logits=torch.randn(2, 100, 24),
            pred_boxes=torch.rand(2, 100, 4),
        )
        assert out.pred_count is None
        assert out.aux_outputs is None


class TestDetectionResult:
    def test_basic_creation(self):
        result = DetectionResult(
            bboxes=torch.rand(10, 4),
            scores=torch.rand(10),
            labels=torch.randint(0, 24, (10,)),
        )
        assert result.bboxes.shape == (10, 4)
        assert result.scores.shape == (10,)
        assert result.labels.shape == (10,)

    def test_empty(self):
        result = DetectionResult(
            bboxes=torch.zeros(0, 4),
            scores=torch.zeros(0),
            labels=torch.zeros(0, dtype=torch.long),
        )
        assert result.bboxes.shape == (0, 4)
