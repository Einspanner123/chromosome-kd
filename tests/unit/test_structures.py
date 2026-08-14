"""测试纯 PyTorch 数据结构"""

from ldmdet.data.structures import (
    DetectionResult,
    ImageMeta,
    InstanceData,
    ModelOutput,
)


class TestImageMeta:
    def test_create(self):
        meta = ImageMeta(img_shape=(800, 1216))
        assert meta.img_shape == (800, 1216)
        assert meta.pad_shape is None

    def test_full(self):
        meta = ImageMeta(
            img_shape=(800, 1000),
            pad_shape=(800, 1024),
            ori_shape=(1600, 2000),
            scale_factor=[0.5, 0.5],
        )
        assert meta.scale_factor == [0.5, 0.5]


class TestDetectionResult:
    def test_create(self, device):
        import torch

        bboxes = torch.randn(5, 4, device=device)
        scores = torch.rand(5, device=device)
        labels = torch.randint(0, 24, (5,), device=device)
        result = DetectionResult(bboxes=bboxes, scores=scores, labels=labels)
        assert result.bboxes.shape == (5, 4)
        assert result.scores.shape == (5,)
        assert result.labels.shape == (5,)


class TestInstanceData:
    def test_create(self, device):
        import torch

        data = InstanceData(
            bboxes=torch.randn(3, 4, device=device),
            labels=torch.tensor([0, 1, 2], device=device),
            img_shape=(800, 1216),
        )
        assert data.bboxes.shape == (3, 4)


class TestModelOutput:
    def test_create(self, device):
        import torch

        out = ModelOutput(
            pred_logits=torch.randn(2, 500, 24, device=device),
            pred_boxes=torch.randn(2, 500, 4, device=device),
        )
        assert out.pred_logits.shape == (2, 500, 24)
        assert out.pred_boxes.shape == (2, 500, 4)
        assert out.aux_outputs is None
