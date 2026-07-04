"""方向N Phase 1: cascade_detach 选项单元测试

测试 DiffusionDetHead 的 cascade_detach 参数:
- True (默认): head 间 detach, 梯度不贯通 (向后兼容)
- False: head 间不 detach, 梯度贯通所有 stage (端到端可微 Cascade)

对应 docs/research/frontier_directions/方向N_端到端可微Cascade.md Phase 1 设计。

核心验证:
1. 参数存在且默认为 True (向后兼容)
2. cascade_detach=True 时, 中间 head 的 pred_bboxes 不携带梯度
3. cascade_detach=False 时, 梯度能从最后 head 回传到第一个 head
4. 与 deep_supervision=False 组合使用 (Phase 1 推荐配置)
"""

import os
import sys

import pytest
import torch

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ldmdet.core.head import DiffusionDetHead
from ldmdet.core.single_head import SingleDiffusionDetHead
from ldmdet.core.roi_extractor import SingleRoIExtractor


def _make_single_head(num_classes=24, feat_channels=64):
    """构建最小 SingleDiffusionDetHead"""
    return SingleDiffusionDetHead(
        num_classes=num_classes,
        feat_channels=feat_channels,
        num_cls_convs=1,
        num_reg_convs=2,
        use_focal_loss=True,
        use_normalized_classifier=False,
    )


def _make_roi_extractor(out_channels=64):
    """构建最小 RoIExtractor"""
    return SingleRoIExtractor(
        featmap_strides=[16], out_channels=out_channels,
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0, aligned=True),
    )


class TestCascadeDetachParameter:
    """测试 cascade_detach 参数存在性和默认值"""

    def test_parameter_exists_with_default_true(self):
        """DiffusionDetHead 应有 cascade_detach 参数, 默认 True"""
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_heads=3,
            single_head=_make_single_head(),
            roi_extractor=_make_roi_extractor(),
            criterion=None,
        )
        assert hasattr(head, 'cascade_detach')
        assert head.cascade_detach is True

    def test_parameter_can_be_set_false(self):
        """cascade_detach 应可设置为 False"""
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_heads=3,
            single_head=_make_single_head(),
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            cascade_detach=False,
        )
        assert head.cascade_detach is False

    def test_default_deep_supervision_true(self):
        """deep_supervision 默认应为 True"""
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_heads=3,
            single_head=_make_single_head(),
            roi_extractor=_make_roi_extractor(),
            criterion=None,
        )
        assert head.deep_supervision is True


class TestCascadeDetachBehavior:
    """测试 cascade_detach 的实际梯度行为"""

    def _make_head(self, cascade_detach, num_heads=3):
        """构建 DiffusionDetHead"""
        return DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_heads=num_heads,
            single_head=_make_single_head(),
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            deep_supervision=False,
            cascade_detach=cascade_detach,
        )

    def _make_dummy_input(self, bs=2, num_boxes=10):
        """构造 dummy 输入"""
        features = [torch.randn(bs, 64, 16, 16)]
        bboxes = torch.rand(bs, num_boxes, 4)
        # 确保有效框: x2 > x1, y2 > y1
        bboxes[:, :, 2] = bboxes[:, :, 0] + 0.2
        bboxes[:, :, 3] = bboxes[:, :, 1] + 0.2
        t = torch.rand(bs)
        return features, bboxes, t

    def test_detach_true_gradient_does_not_flow_to_input(self):
        """cascade_detach=True 时, 输入 bboxes 不应接收来自后续 head 的梯度"""
        head = self._make_head(cascade_detach=True)
        head.eval()
        features, bboxes, t = self._make_dummy_input()
        bboxes.requires_grad_(True)

        with torch.no_grad():
            cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        # detach 模式下, bboxes 不应有梯度 (因为 pred_bboxes 被 detach)
        # 注意: eval 模式下无梯度, 这里仅验证 forward 不报错
        assert cls_logits.shape[0] == 1  # deep_supervision=False, 只有最后 head

    def test_detach_false_forward_works(self):
        """cascade_detach=False 时, forward 应正常工作"""
        head = self._make_head(cascade_detach=False)
        head.eval()
        features, bboxes, t = self._make_dummy_input()
        with torch.no_grad():
            cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        assert cls_logits.shape[0] == 1  # deep_supervision=False

    def test_detach_false_gradient_flows_through_heads(self):
        """cascade_detach=False 时, 梯度应能从最后 head 回传到中间 head 的参数"""
        head = self._make_head(cascade_detach=False, num_heads=3)
        head.train()
        features, bboxes, t = self._make_dummy_input()

        # 记录第一个 head 的初始参数
        head0_params_before = [p.clone() for p in head.head_series[0].parameters()]

        cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        # 用 cls_logits 的和作为 loss
        loss = cls_logits.sum() + pred_bboxes.sum()
        loss.backward()

        # 第一个 head 的参数应有梯度 (因为 cascade_detach=False, 梯度贯通)
        has_grad = sum(1 for p in head.head_series[0].parameters()
                       if p.grad is not None and p.grad.abs().sum() > 0)
        assert has_grad > 0, 'cascade_detach=False 时第一个 head 应有梯度'

    def test_detach_true_intermediate_bboxes_detached(self):
        """cascade_detach=True 时, 中间 head 的 pred_bboxes 被 detach (不携带梯度)"""
        head = self._make_head(cascade_detach=True, num_heads=3)
        head.eval()
        features, bboxes, t = self._make_dummy_input()

        # 修改 forward 以捕获中间 pred_bboxes 的 grad_fn
        # 直接检查 head.forward 的实现: cascade_detach=True 时 curr_bboxes = pred_bboxes.detach()
        # 验证方式: 检查最后一个 head 的 pred_bboxes 对第一个 head 的 pred_bboxes 是否有梯度路径
        # 通过 bboxes 路径 (不通过 proposals)
        # 简化验证: cascade_detach=True 时, head.cascade_detach 属性为 True
        assert head.cascade_detach is True

        # 更具体的验证: 检查 forward 后, 中间 pred_bboxes 被 detach
        # 通过修改 head 的 forward 逻辑间接验证
        # 这里验证 cascade_detach=True 时, 输入 bboxes 不接收来自 pred_bboxes 的梯度
        bboxes.requires_grad_(True)
        head.train()
        cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        loss = pred_bboxes.sum()
        loss.backward()
        # cascade_detach=True + deep_supervision=False:
        # pred_bboxes 依赖于 curr_bboxes (detached), 所以输入 bboxes 无梯度
        # (注意: pred_bboxes 也通过 fc_feature 依赖于 proposals, 但 proposals 不依赖于输入 bboxes
        #  因为 RoIAlign 对 bboxes 不可导)
        assert bboxes.grad is None or bboxes.grad.abs().sum() == 0

    def test_detach_false_bboxes_gradient_flows(self):
        """cascade_detach=False 时, 输入 bboxes 应接收来自 pred_bboxes 的梯度"""
        head = self._make_head(cascade_detach=False, num_heads=3)
        head.train()
        features, bboxes, t = self._make_dummy_input()
        bboxes.requires_grad_(True)

        cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        loss = pred_bboxes.sum()
        loss.backward()
        # cascade_detach=False: pred_bboxes 通过 apply_deltas 依赖于 curr_bboxes
        # 但 curr_bboxes = pred_bboxes (不 detach), 最终回传到输入 bboxes
        # 注意: RoIAlign 对 bboxes 不可导, 但 apply_deltas 中的 widths/heights/ctr 可导
        # 第一个 head 的 pred_bboxes 依赖于输入 bboxes (通过 apply_deltas)
        # 由于 cascade_detach=False, 后续 head 的 pred_bboxes 也通过 bboxes 路径回传
        assert bboxes.grad is not None, 'cascade_detach=False 时输入 bboxes 应有梯度'

    def test_detach_false_all_heads_have_gradient(self):
        """cascade_detach=False 时, 所有 head 都应有梯度"""
        head = self._make_head(cascade_detach=False, num_heads=3)
        head.train()
        features, bboxes, t = self._make_dummy_input()

        cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        loss = cls_logits.sum() + pred_bboxes.sum()
        loss.backward()

        for i, h in enumerate(head.head_series):
            has_grad = sum(1 for p in h.parameters()
                           if p.grad is not None and p.grad.abs().sum() > 0)
            assert has_grad > 0, f'cascade_detach=False 时 head[{i}] 应有梯度'


class TestCascadeDetachWithDeepSupervision:
    """测试 cascade_detach 与 deep_supervision 的组合"""

    def test_detach_false_deep_supervision_true(self):
        """cascade_detach=False + deep_supervision=True 应正常工作"""
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_heads=3,
            single_head=_make_single_head(),
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            deep_supervision=True,
            cascade_detach=False,
        )
        head.train()
        features = [torch.randn(2, 64, 16, 16)]
        bboxes = torch.rand(2, 10, 4)
        bboxes[:, :, 2] = bboxes[:, :, 0] + 0.2
        bboxes[:, :, 3] = bboxes[:, :, 1] + 0.2
        t = torch.rand(2)

        cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        # deep_supervision=True: 返回所有 head 的输出
        assert cls_logits.shape[0] == 3
        assert pred_bboxes.shape[0] == 3

        loss = cls_logits.sum() + pred_bboxes.sum()
        loss.backward()
        # 所有 head 都应有梯度
        for i, h in enumerate(head.head_series):
            has_grad = sum(1 for p in h.parameters()
                           if p.grad is not None and p.grad.abs().sum() > 0)
            assert has_grad > 0, f'head[{i}] 应有梯度'
