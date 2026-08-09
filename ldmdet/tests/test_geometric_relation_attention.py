"""Tests for overlap-conditional geometric relation attention."""

import torch

from ldmdet.core.geometric_relation_attention import (
    OverlapConditionalGeometryBias,
)
from ldmdet.core.single_head import SingleDiffusionDetHead


def valid_boxes(batch=2, count=8):
    top_left = torch.rand(batch, count, 2) * 50
    size = torch.rand(batch, count, 2) * 30 + 5
    return torch.cat((top_left, top_left + size), -1)


def test_zero_initialization_is_exact_identity_bias():
    module = OverlapConditionalGeometryBias(8, hidden_channels=16)
    output = module(valid_boxes())
    assert output.shape == (2, 8, 8, 8)
    assert torch.equal(output, torch.zeros_like(output))


def test_joint_translation_and_scale_invariance():
    module = OverlapConditionalGeometryBias(4, hidden_channels=16)
    torch.nn.init.normal_(module.mlp[-1].weight)
    boxes = valid_boxes(batch=1)
    first = module(boxes)
    transformed = boxes * 3.5 + torch.tensor([20.0, -7.0, 20.0, -7.0])
    second = module(transformed)
    assert torch.allclose(first, second, atol=2e-5, rtol=2e-5)


def test_bias_is_finite_and_receives_gradients():
    module = OverlapConditionalGeometryBias(4, hidden_channels=16)
    output = module(valid_boxes(batch=1))
    output.square().sum().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert all(torch.isfinite(parameter.grad).all()
               for parameter in module.parameters())


def test_single_head_accepts_geometry_attention_configuration():
    head = SingleDiffusionDetHead(
        num_classes=24, feat_channels=32, dim_feedforward=64,
        num_heads=4, dynamic_dim=8,
        geometric_relation_attention=True,
        geometric_relation_hidden=16,
    )
    assert isinstance(
        head.geometric_relation_attn, OverlapConditionalGeometryBias)
