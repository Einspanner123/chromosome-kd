import torch

from ldmdet.core.head import calibrate_class_logits
from ldmdet.core.head import DiffusionDetHead
from ldmdet.core.single_head import SingleDiffusionDetHead
from ldmdet.criterion.criterion import DiffusionDetCriterion
from ldmdet.data.structures import InstanceData, ModelOutput


def make_single_head(enabled):
    return SingleDiffusionDetHead(
        num_classes=3,
        feat_channels=16,
        dim_feedforward=32,
        num_heads=4,
        dynamic_dim=8,
        predict_iou_quality=enabled,
        quality_hidden=8,
    )


def test_quality_head_shape_initialization_and_gradient():
    head = make_single_head(True)
    features = torch.randn(6, 16, requires_grad=True)
    bboxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]] * 3] * 2)
    result = head._predict(features, bboxes, 2, 3)
    assert len(result) == 4
    quality = result[3]
    assert quality.shape == (2, 3, 1)
    assert torch.allclose(quality.sigmoid().mean(), torch.tensor(0.9), atol=0.01)
    quality.sum().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert head.quality_head[0].weight.grad.abs().sum() > 0


def test_disabled_head_preserves_three_tensor_contract():
    head = make_single_head(False)
    features = torch.randn(3, 16)
    bboxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]] * 3])
    assert len(head._predict(features, bboxes, 1, 3)) == 3


def test_calibration_matches_probability_product():
    cls = torch.tensor([[[0.0, 1.0]]])
    quality = torch.tensor([[[0.0]]])
    fused = calibrate_class_logits(cls, quality, beta=2.0).sigmoid()
    assert torch.allclose(fused, cls.sigmoid() * 0.25, atol=1e-6)


def test_varifocal_quality_loss_is_finite_and_backpropagates():
    criterion = DiffusionDetCriterion(
        num_classes=2,
        matcher=None,
        loss_cls=None,
        loss_bbox=None,
        loss_giou=None,
        quality_loss_weight=1.0,
    )
    quality = torch.zeros(1, 2, 1, requires_grad=True)
    output = ModelOutput(
        pred_logits=torch.zeros(1, 2, 2),
        pred_boxes=torch.tensor([[[0.0, 0.0, 1.0, 1.0],
                                  [0.0, 0.0, 0.5, 0.5]]]),
        pred_quality=quality,
    )
    targets = [InstanceData(
        bboxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        labels=torch.tensor([0]), img_shape=(1, 1))]
    indices = [(torch.tensor([True, False]), torch.tensor([0, -1]))]
    loss = criterion._loss_quality(output, targets, indices)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert quality.grad is not None and quality.grad.abs().sum() > 0


def test_quality_only_mode_requires_enabled_quality_head():
    try:
        DiffusionDetHead(
            num_classes=3, feat_channels=16, num_proposals=3, num_heads=2,
            single_head=make_single_head(False), roi_extractor=None,
            criterion=None, quality_only_training=True)
    except ValueError as error:
        assert 'predict_iou_quality=True' in str(error)
    else:
        raise AssertionError('quality-only mode accepted a missing quality head')


def test_quality_calibration_mode_is_validated():
    try:
        DiffusionDetHead(
            num_classes=3, feat_channels=16, num_proposals=3, num_heads=2,
            single_head=make_single_head(True), roi_extractor=None,
            criterion=None, quality_calibration_mode='unknown')
    except ValueError as error:
        assert 'final-stage quality ranking only' in str(error)
    else:
        raise AssertionError('invalid calibration mode was accepted')


def test_final_only_ranking_does_not_overwrite_internal_logits():
    head = DiffusionDetHead(
        num_classes=3, feat_channels=16, num_proposals=3, num_heads=2,
        single_head=make_single_head(True), roi_extractor=None,
        criterion=None, quality_calibration_mode='final_only')
    internal = torch.tensor([[[0.0, 1.0, -1.0]]])
    head._last_quality_logits = torch.zeros(1, 1, 1)
    ranked = head._quality_ranking_logits(internal)
    assert torch.allclose(ranked.sigmoid(), internal.sigmoid() * 0.25, atol=1e-6)
    assert torch.equal(internal, torch.tensor([[[0.0, 1.0, -1.0]]]))
