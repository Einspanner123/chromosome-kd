"""Regression tests for detector-side head-distillation checkpoint binding."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from mmdet.models.detectors.base import BaseDetector

from experiments.mmdet_bridge.detector import LDMDetDetector


def make_bare_detector():
    detector = LDMDetDetector.__new__(LDMDetDetector)
    torch.nn.Module.__init__(detector)
    detector.init_cfg = None
    detector._is_init = False
    detector.backbone = torch.nn.Linear(2, 3)
    detector.neck = torch.nn.Linear(3, 4)
    detector.bbox_head = SimpleNamespace(
        use_distillation=True,
        freeze_backbone=False,
        _teacher=None,
    )
    detector._teacher_checkpoint = 'parent.pth'
    return detector


def test_component_checkpoint_loads_only_requested_prefix():
    module = torch.nn.Linear(2, 3)
    expected_weight = torch.randn_like(module.weight)
    expected_bias = torch.randn_like(module.bias)
    checkpoint = {
        'state_dict': {
            'backbone.weight': expected_weight,
            'backbone.bias': expected_bias,
            'bbox_head.unrelated': torch.ones(1),
        }
    }
    with patch('torch.load', return_value=checkpoint):
        LDMDetDetector._load_component_checkpoint(
            make_bare_detector(),
            module,
            'parent.pth',
            prefix='backbone.',
            component_name='student backbone',
        )
    assert torch.equal(module.weight, expected_weight)
    assert torch.equal(module.bias, expected_bias)


def test_component_checkpoint_rejects_missing_prefix():
    with patch(
        'torch.load',
        return_value={'state_dict': {'neck.x': torch.ones(1)}},
    ), pytest.raises(RuntimeError, match='contains no'):
        LDMDetDetector._load_component_checkpoint(
            make_bare_detector(),
            torch.nn.Linear(2, 3),
            'parent.pth',
            prefix='backbone.',
            component_name='student backbone',
        )


def test_distillation_init_loads_parent_backbone_and_neck():
    detector = make_bare_detector()
    with patch.object(BaseDetector, 'init_weights'), patch.object(
        detector, '_load_component_checkpoint'
    ) as load:
        detector.init_weights()
    assert load.call_count == 2
    assert load.call_args_list[0].kwargs['prefix'] == 'backbone.'
    assert load.call_args_list[1].kwargs['prefix'] == 'neck.'


def test_freeze_backbone_and_neck_is_explicit():
    detector = make_bare_detector()
    detector._freeze_backbone_and_neck()
    assert not any(p.requires_grad for p in detector.backbone.parameters())
    assert not any(p.requires_grad for p in detector.neck.parameters())
