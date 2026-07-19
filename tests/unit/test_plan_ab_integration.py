"""方案 A/B 集成前向测试 — 验证训练/推理流程正确性.

验证:
  1. 方案 A (RandomMatcher): 所有 slot label>=0, loss 有梯度
  2. 方案 B (Hungarian + random_gt): 所有 slot label>=0, loss 有梯度
  3. 推理前向正常 (输出 shape 正确)
  4. loss_bbox / loss_giou 对所有 slot 生效 (不是只对 matched slot)
"""

import pytest
import torch

from setdiff.models.set_head import JointDiffusionHead


def _build_head(**kwargs):
    """构建小型 JointDiffusionHead 用于测试."""
    defaults = dict(
        num_queries=20,
        feat_channels=32,
        num_heads=4,
        num_layers=2,
        dim_feedforward=64,
        num_classes=24,
        snr_scale=2.0,
        num_sample_steps=4,
    )
    defaults.update(kwargs)
    return JointDiffusionHead(**defaults)


def _make_batch(B=2, N=20, M=5, num_classes=24):
    """构建测试 batch."""
    torch.manual_seed(42)
    image_features = torch.randn(B, 64, 32)  # [B, HW, C]
    gt_boxes_list = [torch.randn(M, 4) * 0.5 for _ in range(B)]
    gt_labels_list = [
        torch.randint(0, num_classes, (M,)) for _ in range(B)
    ]
    return image_features, gt_boxes_list, gt_labels_list


class TestPlanAIntegration:
    """方案 A 集成前向测试."""

    def test_training_forward_all_slots_supervised(self):
        """方案 A: 训练前向, 所有 slot label>=0."""
        torch.manual_seed(42)
        head = _build_head(matcher_type='random')
        image_features, gt_boxes_list, gt_labels_list = _make_batch()

        # 通过 monkey-patch 检查 matcher 输出
        original_match_batch = head.matcher.match_batch
        captured = {}

        def capture_match_batch(noise, gt_boxes, gt_labels):
            mb, ml = original_match_batch(noise, gt_boxes, gt_labels)
            captured['matched_boxes'] = mb
            captured['matched_labels'] = ml
            return mb, ml

        head.matcher.match_batch = capture_match_batch

        loss_dict = head(image_features, gt_boxes_list, gt_labels_list)

        # 验证: 所有 slot label >= 0
        matched_labels = captured['matched_labels']
        assert (matched_labels >= 0).all(), (
            "方案 A: 所有 slot 必须有 GT label (无 -1)"
        )

        # 验证: loss_dict 包含三项
        assert 'loss_cls' in loss_dict
        assert 'loss_bbox' in loss_dict
        assert 'loss_giou' in loss_dict

    def test_training_forward_loss_has_gradient(self):
        """方案 A: loss 有梯度, 可反向传播."""
        torch.manual_seed(42)
        head = _build_head(matcher_type='random')
        image_features, gt_boxes_list, gt_labels_list = _make_batch()

        loss_dict = head(image_features, gt_boxes_list, gt_labels_list)
        total_loss = sum(loss_dict.values())

        total_loss.backward()

        # 验证: 至少有一些参数有梯度
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in head.parameters()
            if p.requires_grad
        )
        assert has_grad, "方案 A: loss 必须有梯度"

    def test_training_forward_loss_finite(self):
        """方案 A: loss 有限 (无 NaN/Inf)."""
        torch.manual_seed(42)
        head = _build_head(matcher_type='random')
        image_features, gt_boxes_list, gt_labels_list = _make_batch()

        loss_dict = head(image_features, gt_boxes_list, gt_labels_list)

        for k, v in loss_dict.items():
            assert torch.isfinite(v).item(), (
                f"方案 A: {k} 不是有限值: {v.item()}"
            )

    def test_inference_forward(self):
        """方案 A: 推理前向输出 shape 正确."""
        torch.manual_seed(42)
        head = _build_head(matcher_type='random')
        head.eval()
        image_features, _, _ = _make_batch()

        with torch.no_grad():
            outputs = head(image_features)

        assert 'pred_logits' in outputs
        assert 'pred_boxes' in outputs
        B, N = 2, 20
        assert outputs['pred_logits'].shape == (B, N, 24)
        assert outputs['pred_boxes'].shape == (B, N, 4)

    def test_loss_bbox_giou_only_matched_slots(self):
        """方案 A: loss_bbox / loss_giou 只对 matched slot 生效 (num_pos=M).

        2026-07-19 修复 (对齐 LDMDet criterion.py): criterion 内部用
        Hungarian 重新匹配 pred_boxes 和 GT, num_pos=M (不是 N=300),
        避免梯度稀释 37.5 倍.

        验证: criterion 的 matcher.match_indices 被调用, 返回 M 对匹配
        (而不是用 coupling 阶段的 N 个 matched_labels).
        """
        torch.manual_seed(42)
        head = _build_head(matcher_type='random')
        image_features, gt_boxes_list, gt_labels_list = _make_batch(M=5)

        # 捕获 criterion 内部 matcher 的 match_indices 调用
        captured = {}
        original_match_indices = head.criterion.matcher.match_indices

        def capture(pred, gt):
            result = original_match_indices(pred, gt)
            captured['src_len'] = len(result[0])
            captured['tgt_len'] = len(result[1])
            return result

        head.criterion.matcher.match_indices = capture

        loss_dict = head(image_features, gt_boxes_list, gt_labels_list)

        # criterion 应做 M=5 对匹配 (不是 N=20)
        assert captured['src_len'] == 5, (
            f"方案 A: criterion 应做 M=5 对匹配 (num_pos=M), "
            f"实际 {captured['src_len']} (旧稀释行为 num_pos=N)"
        )
        assert captured['tgt_len'] == 5


class TestPlanBIntegration:
    """方案 B 集成前向测试."""

    def test_training_forward_all_slots_supervised(self):
        """方案 B: 训练前向, 所有 slot label>=0."""
        torch.manual_seed(42)
        head = _build_head(
            matcher_type='hungarian', unmatched_strategy='random_gt'
        )
        image_features, gt_boxes_list, gt_labels_list = _make_batch()

        original_match_batch = head.matcher.match_batch
        captured = {}

        def capture_match_batch(noise, gt_boxes, gt_labels):
            mb, ml = original_match_batch(noise, gt_boxes, gt_labels)
            captured['matched_boxes'] = mb
            captured['matched_labels'] = ml
            return mb, ml

        head.matcher.match_batch = capture_match_batch

        loss_dict = head(image_features, gt_boxes_list, gt_labels_list)

        # 验证: 所有 slot label >= 0
        matched_labels = captured['matched_labels']
        assert (matched_labels >= 0).all(), (
            "方案 B: 所有 slot 必须有 GT label (无 -1)"
        )

        assert 'loss_cls' in loss_dict
        assert 'loss_bbox' in loss_dict
        assert 'loss_giou' in loss_dict

    def test_training_forward_loss_has_gradient(self):
        """方案 B: loss 有梯度."""
        torch.manual_seed(42)
        head = _build_head(
            matcher_type='hungarian', unmatched_strategy='random_gt'
        )
        image_features, gt_boxes_list, gt_labels_list = _make_batch()

        loss_dict = head(image_features, gt_boxes_list, gt_labels_list)
        total_loss = sum(loss_dict.values())

        total_loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in head.parameters()
            if p.requires_grad
        )
        assert has_grad, "方案 B: loss 必须有梯度"

    def test_training_forward_loss_finite(self):
        """方案 B: loss 有限."""
        torch.manual_seed(42)
        head = _build_head(
            matcher_type='hungarian', unmatched_strategy='random_gt'
        )
        image_features, gt_boxes_list, gt_labels_list = _make_batch()

        loss_dict = head(image_features, gt_boxes_list, gt_labels_list)

        for k, v in loss_dict.items():
            assert torch.isfinite(v).item(), (
                f"方案 B: {k} 不是有限值: {v.item()}"
            )

    def test_inference_forward(self):
        """方案 B: 推理前向输出 shape 正确."""
        torch.manual_seed(42)
        head = _build_head(
            matcher_type='hungarian', unmatched_strategy='random_gt'
        )
        head.eval()
        image_features, _, _ = _make_batch()

        with torch.no_grad():
            outputs = head(image_features)

        assert 'pred_logits' in outputs
        assert 'pred_boxes' in outputs
        B, N = 2, 20
        assert outputs['pred_logits'].shape == (B, N, 24)
        assert outputs['pred_boxes'].shape == (B, N, 4)

    def test_loss_bbox_giou_only_matched_slots(self):
        """方案 B: loss_bbox / loss_giou 只对 matched slot 生效 (num_pos=M).

        2026-07-19 修复 (对齐 LDMDet criterion.py): criterion 内部用
        Hungarian 重新匹配 pred_boxes 和 GT, num_pos=M (不是 N=300),
        避免梯度稀释 37.5 倍.

        与方案 A 的区别: 方案 B coupling 阶段用 Hungarian 一对一 + unmatched
        随机 GT; 但 loss 阶段两者相同 (criterion 内部重新匹配).
        """
        torch.manual_seed(42)
        head = _build_head(
            matcher_type='hungarian', unmatched_strategy='random_gt'
        )
        image_features, gt_boxes_list, gt_labels_list = _make_batch(M=5)

        captured = {}
        original_match_indices = head.criterion.matcher.match_indices

        def capture(pred, gt):
            result = original_match_indices(pred, gt)
            captured['src_len'] = len(result[0])
            captured['tgt_len'] = len(result[1])
            return result

        head.criterion.matcher.match_indices = capture

        loss_dict = head(image_features, gt_boxes_list, gt_labels_list)

        assert captured['src_len'] == 5, (
            f"方案 B: criterion 应做 M=5 对匹配 (num_pos=M), "
            f"实际 {captured['src_len']} (旧稀释行为 num_pos=N)"
        )
        assert captured['tgt_len'] == 5
