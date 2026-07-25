"""测试 Head Distillation 方案A: 解冻 backbone + 从 A4 checkpoint 加载 backbone/neck

方案A 核心逻辑 (detector.py):
  1. _load_backbone_from_checkpoint: 从完整 detector checkpoint 提取 backbone.*/neck.*
     前缀并加载到 Student 自身 (不触及 bbox_head.*, 保护 head 映射权重)
  2. init_weights: super().init_weights() 加载 ImageNet 后, 调用 _load_backbone_from_checkpoint
     覆盖为 A4 backbone/neck (修复特征不匹配 root cause)
  3. 条件守卫: 仅 use_distillation + not freeze_backbone + teacher_checkpoint 时加载

关键不变量:
  - head 映射权重 (init_student_from_teacher 设置) 不被 init_weights 覆盖
  - 不使用 load_from (会破坏 head 映射), backbone 由自定义 init_weights 加载

测试策略: 绕过 LDMDetDetector.__init__ (避免重 mmdet 构建), 用 nn.Linear 模拟
backbone/neck, mock torch.load 返回 fake checkpoint, 直接测试方法逻辑。
"""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# import 触发 mmdet registry 注册 (chromo 环境有 mmdet)
from mmdet.models.detectors.base import BaseDetector
from experiments.mmdet_bridge.detector import LDMDetDetector


def _make_bare_detector():
    """绕过 __init__ 创建 LDMDetDetector 实例 (避免重 mmdet 构建)

    设置最小属性: backbone/neck 用 nn.Linear 模拟, bbox_head 用 MagicMock。
    需先调用 nn.Module.__init__ 初始化 _modules 等内部数据结构,
    否则赋值子模块 (nn.Linear) 时报 "cannot assign module before Module.__init__"。
    """
    detector = LDMDetDetector.__new__(LDMDetDetector)
    nn.Module.__init__(detector)  # 初始化 nn.Module 内部数据结构 (不调用完整 __init__)
    detector.backbone = nn.Linear(10, 5)
    detector.neck = nn.Linear(5, 3)
    detector.bbox_head = MagicMock()
    detector.bbox_head.use_distillation = True
    detector.bbox_head.freeze_backbone = False
    detector._teacher_checkpoint = 'fake.pth'
    return detector


def _make_fake_checkpoint():
    """构造 fake A4 完整 detector checkpoint state_dict

    含 backbone.*/neck.*/bbox_head.* 三类前缀, 验证前缀提取逻辑。
    """
    return {
        'state_dict': {
            # backbone 权重 (应加载到 detector.backbone)
            'backbone.weight': torch.randn(5, 10),
            'backbone.bias': torch.randn(5),
            # neck 权重 (应加载到 detector.neck)
            'neck.weight': torch.randn(3, 5),
            'neck.bias': torch.randn(3),
            # bbox_head 权重 (不应被 _load_backbone_from_checkpoint 触及)
            'bbox_head.head_series.0.fc.weight': torch.randn(5, 5),
            'bbox_head.head_series.0.fc.bias': torch.randn(5),
            'bbox_head.head_series.5.fc.weight': torch.randn(5, 5),
        }
    }


# ============================================================
# 1. _load_backbone_from_checkpoint: 前缀提取与加载
# ============================================================

class TestLoadBackboneFromCheckpoint:
    """_load_backbone_from_checkpoint 方法测试"""

    def test_loads_backbone_weights(self):
        """backbone.* 前缀权重正确加载到 detector.backbone"""
        detector = _make_bare_detector()
        fake_ckpt = _make_fake_checkpoint()
        expected_w = fake_ckpt['state_dict']['backbone.weight'].clone()
        expected_b = fake_ckpt['state_dict']['backbone.bias'].clone()

        with patch('torch.load', return_value=fake_ckpt):
            detector._load_backbone_from_checkpoint('fake.pth')

        assert torch.equal(detector.backbone.weight.data, expected_w)
        assert torch.equal(detector.backbone.bias.data, expected_b)

    def test_loads_neck_weights(self):
        """neck.* 前缀权重正确加载到 detector.neck"""
        detector = _make_bare_detector()
        fake_ckpt = _make_fake_checkpoint()
        expected_w = fake_ckpt['state_dict']['neck.weight'].clone()
        expected_b = fake_ckpt['state_dict']['neck.bias'].clone()

        with patch('torch.load', return_value=fake_ckpt):
            detector._load_backbone_from_checkpoint('fake.pth')

        assert torch.equal(detector.neck.weight.data, expected_w)
        assert torch.equal(detector.neck.bias.data, expected_b)

    def test_does_not_touch_bbox_head(self):
        """bbox_head.* 权重不被加载到 backbone/neck (保护 head 映射)

        方案A 关键不变量: init_weights 仅加载 backbone/neck, head 映射权重
        (由 init_student_from_teacher 设置) 必须保留, 不被覆盖。
        """
        detector = _make_bare_detector()
        fake_ckpt = _make_fake_checkpoint()

        with patch('torch.load', return_value=fake_ckpt):
            detector._load_backbone_from_checkpoint('fake.pth')

        # bbox_head 是 MagicMock, 不应有 load_state_dict 被调用
        detector.bbox_head.load_state_dict.assert_not_called()

    def test_handles_missing_backbone_prefix(self):
        """checkpoint 无 backbone.* 前缀时发出 warning, 不崩溃"""
        detector = _make_bare_detector()
        original_w = detector.backbone.weight.data.clone()
        fake_ckpt = {'state_dict': {'neck.weight': torch.randn(3, 5)}}

        with patch('torch.load', return_value=fake_ckpt):
            detector._load_backbone_from_checkpoint('fake.pth')

        # backbone 权重保持不变 (未被覆盖)
        assert torch.equal(detector.backbone.weight.data, original_w)

    def test_handles_checkpoint_without_state_dict_key(self):
        """checkpoint 直接是 state_dict (无 'state_dict' 键) 也能处理"""
        detector = _make_bare_detector()
        expected_w = torch.randn(5, 10)
        fake_ckpt = {'backbone.weight': expected_w.clone()}

        with patch('torch.load', return_value=fake_ckpt):
            detector._load_backbone_from_checkpoint('fake.pth')

        assert torch.equal(detector.backbone.weight.data, expected_w)

    def test_handles_none_neck(self):
        """模型无 neck 时, checkpoint 含 neck 权重则 warning, 不崩溃"""
        detector = _make_bare_detector()
        detector.neck = None
        fake_ckpt = _make_fake_checkpoint()

        with patch('torch.load', return_value=fake_ckpt):
            # 不应抛异常
            detector._load_backbone_from_checkpoint('fake.pth')

        # backbone 仍正确加载
        assert torch.equal(
            detector.backbone.weight.data,
            fake_ckpt['state_dict']['backbone.weight'],
        )


# ============================================================
# 2. init_weights: 条件守卫
# ============================================================

class TestInitWeightsConditional:
    """init_weights 重写的条件逻辑测试

    守卫: 仅 use_distillation + not freeze_backbone + teacher_checkpoint 时
    调用 _load_backbone_from_checkpoint。
    """

    def test_loads_backbone_when_unfrozen_with_checkpoint(self):
        """方案A 标准路径: freeze_backbone=False + use_distillation + checkpoint → 加载"""
        detector = _make_bare_detector()
        # 默认配置: use_distillation=True, freeze_backbone=False, _teacher_checkpoint='fake.pth'

        with patch.object(BaseDetector, 'init_weights'), \
             patch.object(detector, '_load_backbone_from_checkpoint') as mock_load:
            detector.init_weights()
            mock_load.assert_called_once_with('fake.pth')

    def test_skips_when_frozen(self):
        """v2 路径: freeze_backbone=True → 不加载 backbone (保持 v2 行为)"""
        detector = _make_bare_detector()
        detector.bbox_head.freeze_backbone = True

        with patch.object(BaseDetector, 'init_weights'), \
             patch.object(detector, '_load_backbone_from_checkpoint') as mock_load:
            detector.init_weights()
            mock_load.assert_not_called()

    def test_skips_when_no_distillation(self):
        """非蒸馏模式: use_distillation=False → 不加载 backbone"""
        detector = _make_bare_detector()
        detector.bbox_head.use_distillation = False

        with patch.object(BaseDetector, 'init_weights'), \
             patch.object(detector, '_load_backbone_from_checkpoint') as mock_load:
            detector.init_weights()
            mock_load.assert_not_called()

    def test_skips_when_no_checkpoint(self):
        """无 teacher_checkpoint → 不加载 backbone (Teacher 随机权重场景)"""
        detector = _make_bare_detector()
        detector._teacher_checkpoint = None

        with patch.object(BaseDetector, 'init_weights'), \
             patch.object(detector, '_load_backbone_from_checkpoint') as mock_load:
            detector.init_weights()
            mock_load.assert_not_called()

    def test_calls_super_init_weights(self):
        """init_weights 调用 super().init_weights() (加载 ImageNet backbone)"""
        detector = _make_bare_detector()

        with patch.object(BaseDetector, 'init_weights') as mock_super, \
             patch.object(detector, '_load_backbone_from_checkpoint'):
            detector.init_weights()
            mock_super.assert_called_once()


# ============================================================
# 3. _freeze_backbone 行为 (方案A 不触发)
# ============================================================

class TestFreezeBackboneBehavior:
    """_freeze_backbone 行为测试 (方案A freeze_backbone=False 时不触发)"""

    def test_freeze_backbone_sets_requires_grad_false(self):
        """_freeze_backbone 设置 backbone/neck 参数 requires_grad=False"""
        detector = _make_bare_detector()
        # 确保初始可训练
        assert detector.backbone.weight.requires_grad is True
        assert detector.neck.weight.requires_grad is True

        detector._freeze_backbone()

        assert detector.backbone.weight.requires_grad is False
        assert detector.neck.weight.requires_grad is False

    def test_plan_a_backbone_stays_trainable(self):
        """方案A: freeze_backbone=False 时 backbone 参数保持可训练

        (init_weights 加载 A4 backbone 后, 仍参与梯度更新)
        """
        detector = _make_bare_detector()
        # 方案A 不调用 _freeze_backbone, backbone 保持 requires_grad=True
        assert detector.bbox_head.freeze_backbone is False
        assert detector.backbone.weight.requires_grad is True
        assert detector.neck.weight.requires_grad is True

    def test_load_backbone_keeps_requires_grad_true(self):
        """方案A: 加载 A4 backbone 权重后, 参数仍 requires_grad=True (可微调)"""
        detector = _make_bare_detector()
        fake_ckpt = _make_fake_checkpoint()

        with patch('torch.load', return_value=fake_ckpt):
            detector._load_backbone_from_checkpoint('fake.pth')

        # 加载权重后仍可训练 (load_state_dict 不改变 requires_grad)
        assert detector.backbone.weight.requires_grad is True
        assert detector.neck.weight.requires_grad is True
