"""SetDiff loss 监控插桩测试 — 验证 double-counting 修复 + L1/GIoU 梯度尺度对齐.

测试覆盖:
1. verify_no_double_counting: head 返回 loss_dict 不含 'loss' key (修复生效).
2. measure_gradient_ratio (head 层面): 诊断用, 验证梯度有限/非零 (几何依赖).
3. measure_criterion_gradient_ratio (controlled): pred 接近 tgt 时
   L1:GIoU 加权梯度比 ≈ 2.5 (修复后), 不是 5+ (修复前).
4. measure_loss_space_consistency: consistency_ratio ≈ 2*s = 4.0 (s=2).
5. run_quick_check: 整体流程无异常.
"""

import pytest
import torch

from setdiff.criterion.set_loss import SetCriterion
from setdiff.diagnostics.loss_monitor import (
    LossMonitor,
    LossMonitorHook,
)
from setdiff.models.set_head import JointDiffusionHead

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def small_head():
    """小规模 JointDiffusionHead for testing."""
    torch.manual_seed(42)
    return JointDiffusionHead(
        num_queries=10,
        feat_channels=64,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
        num_classes=24,
        snr_scale=2.0,
    )


@pytest.fixture
def small_inputs():
    """小规模输入 (GT 在扩散空间 [-2, 2], 小目标 w,h 为负)."""
    torch.manual_seed(42)
    B, HW, C = 1, 100, 64
    image_features = torch.randn(B, HW, C)
    gt_boxes = [torch.tensor([[0.0, 0.0, -1.6, -1.6]])]
    gt_labels = [torch.tensor([0])]
    return image_features, gt_boxes, gt_labels


# ──────────────────────────────────────────────
# 1. verify_no_double_counting
# ──────────────────────────────────────────────


class TestVerifyNoDoubleCounting:
    """验证 double-counting 修复 (head 返回 loss_dict 不含 'loss' key)."""

    def test_returns_true_after_fix(self, small_head, small_inputs):
        """修复后 verify_no_double_counting 应返回 True."""
        image_features, gt_boxes, gt_labels = small_inputs
        result = LossMonitor.verify_no_double_counting(
            small_head, image_features, gt_boxes, gt_labels
        )
        assert result is True

    def test_loss_dict_has_no_loss_key(self, small_head, small_inputs):
        """loss_dict 不应包含 'loss' key (mmengine parse_losses 会 double-count)."""
        image_features, gt_boxes, gt_labels = small_inputs
        loss_dict = small_head(image_features, gt_boxes, gt_labels)
        assert 'loss' not in loss_dict, (
            f"loss_dict 不应包含 'loss' key (double-counting bug). "
            f'实际 keys: {list(loss_dict.keys())}'
        )

    def test_loss_dict_keys_are_cls_bbox_giou(self, small_head, small_inputs):
        """loss_dict 应只含 loss_cls, loss_bbox, loss_giou."""
        image_features, gt_boxes, gt_labels = small_inputs
        loss_dict = small_head(image_features, gt_boxes, gt_labels)
        assert set(loss_dict.keys()) == {
            'loss_cls',
            'loss_bbox',
            'loss_giou',
        }


# ──────────────────────────────────────────────
# 2. measure_gradient_ratio (head 层面, 诊断用)
# ──────────────────────────────────────────────


class TestMeasureGradientRatio:
    """验证 head 层面梯度测量 (诊断用, 几何依赖).

    注: head 层面的加权梯度比取决于 pred 与 tgt 的几何关系:
    - 训练初期 (pred 随机, 远离 tgt): GIoU 饱和, ratio 可达 40-60+ (非 bug).
    - 训练后期 (pred 接近 tgt): GIoU 不饱和, ratio ≈ 2.5 (权重 5:2).
    因此本类只验证梯度有限/非零, 严格 ratio 验证见
    TestMeasureCriterionGradientRatio.
    """

    def test_ratio_finite_and_positive(self, small_head, small_inputs):
        """加权梯度比应为正有限值 (诊断用, 不强约束具体范围).

        head 层面 ratio 受 GIoU 几何影响:
        - pred 远离 tgt (随机初始化): GIoU 饱和, ratio 可达 40-60+.
        - pred 接近 tgt: ratio ≈ 2.5.
        这里只验证梯度有限且非零 (有梯度信号).
        """
        image_features, gt_boxes, gt_labels = small_inputs
        small_head.zero_grad()
        grad_norms = LossMonitor.measure_gradient_ratio(
            small_head, image_features, gt_boxes, gt_labels
        )
        ratio = grad_norms['ratio_bbox_giou']
        # 只验证有限正数 (不强约束范围, 因 GIoU 几何依赖)
        assert torch.isfinite(torch.tensor(ratio)), (
            f'加权梯度比应为有限值, 实际: {ratio}'
        )
        assert ratio > 0, f'加权梯度比应 > 0, 实际: {ratio}'
        # 宽松上界: 100 (避免极异常场景, 如 NaN/Inf 已被 finite 检查排除)
        assert ratio < 100, (
            f'加权梯度比 {ratio:.2f} >= 100, 异常偏大 (可能 GIoU 完全饱和 '
            f'或 L1 仍在扩散空间). 建议检查 measure_criterion_gradient_ratio.'
        )

    def test_grad_norms_positive(self, small_head, small_inputs):
        """各 loss 项梯度 norm 应为正数 (有梯度信号)."""
        image_features, gt_boxes, gt_labels = small_inputs
        small_head.zero_grad()
        grad_norms = LossMonitor.measure_gradient_ratio(
            small_head, image_features, gt_boxes, gt_labels
        )
        for k in ['loss_cls', 'loss_bbox', 'loss_giou']:
            assert grad_norms[k] > 0, (
                f'{k} 梯度 norm 应 > 0, 实际: {grad_norms[k]}'
            )

    def test_grad_norms_finite(self, small_head, small_inputs):
        """各 loss 项梯度 norm 应为有限值 (无 NaN/Inf)."""
        image_features, gt_boxes, gt_labels = small_inputs
        small_head.zero_grad()
        grad_norms = LossMonitor.measure_gradient_ratio(
            small_head, image_features, gt_boxes, gt_labels
        )
        for k in ['loss_cls', 'loss_bbox', 'loss_giou']:
            assert torch.isfinite(torch.tensor(grad_norms[k])), (
                f'{k} 梯度 norm 应为有限值, 实际: {grad_norms[k]}'
            )

    def test_ratio_bbox_giou_key_present(self, small_head, small_inputs):
        """返回 dict 应包含 'ratio_bbox_giou' key."""
        image_features, gt_boxes, gt_labels = small_inputs
        small_head.zero_grad()
        grad_norms = LossMonitor.measure_gradient_ratio(
            small_head, image_features, gt_boxes, gt_labels
        )
        assert 'ratio_bbox_giou' in grad_norms


# ──────────────────────────────────────────────
# 3. measure_criterion_gradient_ratio (controlled, 严格验证)
# ──────────────────────────────────────────────


class TestMeasureCriterionGradientRatio:
    """验证 criterion 层面梯度尺度对齐 (controlled, pred 接近 tgt).

    通过将 pred_boxes 作为可学习参数直接测量, 避免 head 几何干扰.
    pred 接近 tgt (GIoU 不饱和) 时:
    - 修复后: L1 和 GIoU 都在 [0,1] 空间, ratio ≈ 2.5 (权重 5:2).
    - 修复前: L1 在扩散空间, ratio ≈ 5s:2 = 5 (s=2).
    """

    def _make_close_pred_tgt(self, s=2.0):
        """构造 pred 接近 tgt 的场景 (GIoU 不饱和).

        [0,1] 空间: pred=[0.5,0.5,0.3,0.3] vs tgt=[0.5,0.5,0.2,0.2]
        (同中心, GIoU 不饱和)
        返回扩散空间 [-s, s] 的值, 以及 GT list 形式 (适配新 criterion).
        """
        pred_norm = torch.tensor([[[0.5, 0.5, 0.3, 0.3]]])
        tgt_norm = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
        # 转换到扩散空间 [-s, s]
        pred_diff = (pred_norm * 2.0 - 1.0) * s
        tgt_diff = (tgt_norm * 2.0 - 1.0) * s
        # 2026-07-19: criterion 内部 Hungarian 重新匹配, 传原始 GT list
        # gt_boxes_list: List[Tensor[M,4]], gt_labels_list: List[Tensor[M]]
        gt_boxes_list = [tgt_diff[0]]  # M=1 GT
        gt_labels_list = [torch.tensor([0])]
        return pred_diff, gt_boxes_list, gt_labels_list

    def test_ratio_in_healthy_range_when_pred_close_to_tgt(self):
        """pred 接近 tgt 时, 加权梯度比应在 [0.5, 6.0] (修复后 ≈ 2.5).

        修复后: L1 和 GIoU 都在 [0,1] 空间, ratio 由权重 5:2 决定 ≈ 2.5.
        修复前: L1 在扩散空间, ratio ≈ 5s:2 = 5 (s=2), 接近上界.
        """
        criterion = SetCriterion(num_classes=24, snr_scale=2.0)
        pred_diff, gt_boxes_list, gt_labels_list = self._make_close_pred_tgt(
            s=2.0
        )
        grad_norms = LossMonitor.measure_criterion_gradient_ratio(
            criterion,
            pred_diff,
            gt_boxes_list,
            gt_labels_list,
        )
        ratio = grad_norms['ratio_bbox_giou']
        # 修复后 ≈ 2.5 (权重 5:2); 给出宽松范围 [0.5, 6.0] 容忍数值波动
        assert 0.5 <= ratio <= 6.0, (
            f'pred 接近 tgt 时, criterion 加权梯度比应在 [0.5, 6.0] '
            f'(修复后 ≈ 2.5), 实际: {ratio:.2f}. '
            f'如果 > 6.0, 说明 L1 仍在扩散空间计算 (修复未生效).'
        )

    def test_ratio_not_in_bug_range_when_pred_close_to_tgt(self):
        """pred 接近 tgt 时, ratio 不应在 bug 范围 (>= 8.0 表示空间尺度失衡).

        修复前 bug: ratio ≈ 5s ≈ 10 (L1 在扩散空间, GIoU 梯度被 1/(2s) 衰减).
        修复后: ratio ≈ 2.5 (远低于 8.0).
        """
        criterion = SetCriterion(num_classes=24, snr_scale=2.0)
        pred_diff, gt_boxes_list, gt_labels_list = self._make_close_pred_tgt(
            s=2.0
        )
        grad_norms = LossMonitor.measure_criterion_gradient_ratio(
            criterion,
            pred_diff,
            gt_boxes_list,
            gt_labels_list,
        )
        ratio = grad_norms['ratio_bbox_giou']
        assert ratio < 8.0, (
            f'pred 接近 tgt 时, criterion 加权梯度比 {ratio:.2f} >= 8.0, '
            f'疑似 L1 仍在扩散空间计算 (修复未生效). 修复后应 ≈ 2.5.'
        )

    def test_grad_norms_positive(self):
        """criterion 各 loss 项梯度 norm 应为正数."""
        criterion = SetCriterion(num_classes=24, snr_scale=2.0)
        pred_diff, gt_boxes_list, gt_labels_list = self._make_close_pred_tgt(
            s=2.0
        )
        grad_norms = LossMonitor.measure_criterion_gradient_ratio(
            criterion,
            pred_diff,
            gt_boxes_list,
            gt_labels_list,
        )
        for k in ['loss_bbox', 'loss_giou']:
            assert grad_norms[k] > 0, (
                f'{k} 梯度 norm 应 > 0, 实际: {grad_norms[k]}'
            )

    def test_grad_norms_finite(self):
        """criterion 各 loss 项梯度 norm 应为有限值."""
        criterion = SetCriterion(num_classes=24, snr_scale=2.0)
        pred_diff, gt_boxes_list, gt_labels_list = self._make_close_pred_tgt(
            s=2.0
        )
        grad_norms = LossMonitor.measure_criterion_gradient_ratio(
            criterion,
            pred_diff,
            gt_boxes_list,
            gt_labels_list,
        )
        for k in ['loss_bbox', 'loss_giou']:
            assert torch.isfinite(torch.tensor(grad_norms[k])), (
                f'{k} 梯度 norm 应为有限值, 实际: {grad_norms[k]}'
            )

    def test_ratio_bbox_giou_key_present(self):
        """返回 dict 应包含 'ratio_bbox_giou' key."""
        criterion = SetCriterion(num_classes=24, snr_scale=2.0)
        pred_diff, gt_boxes_list, gt_labels_list = self._make_close_pred_tgt(
            s=2.0
        )
        grad_norms = LossMonitor.measure_criterion_gradient_ratio(
            criterion,
            pred_diff,
            gt_boxes_list,
            gt_labels_list,
        )
        assert 'ratio_bbox_giou' in grad_norms

    def test_ratio_scales_with_weights(self):
        """验证 ratio 确实由 weight_dict 决定 (而非空间尺度)."""
        # 默认 weight_dict: loss_bbox=5.0, loss_giou=2.0 → ratio ≈ 5:2 = 2.5
        criterion = SetCriterion(num_classes=24, snr_scale=2.0)
        pred_diff, gt_boxes_list, gt_labels_list = self._make_close_pred_tgt(
            s=2.0
        )
        grad_norms = LossMonitor.measure_criterion_gradient_ratio(
            criterion,
            pred_diff,
            gt_boxes_list,
            gt_labels_list,
        )
        default_ratio = grad_norms['ratio_bbox_giou']

        # 修改权重: loss_bbox=2.0, loss_giou=5.0 → ratio 应反转 ≈ 2:5 = 0.4
        criterion_flipped = SetCriterion(num_classes=24, snr_scale=2.0)
        criterion_flipped.weight_dict = {
            'loss_cls': 2.0,
            'loss_bbox': 2.0,
            'loss_giou': 5.0,
        }
        grad_norms_flipped = LossMonitor.measure_criterion_gradient_ratio(
            criterion_flipped,
            pred_diff,
            gt_boxes_list,
            gt_labels_list,
        )
        flipped_ratio = grad_norms_flipped['ratio_bbox_giou']

        # 翻转后 ratio 应明显小于默认 (2:5=0.4 vs 5:2=2.5)
        assert flipped_ratio < default_ratio, (
            f'翻转 weight_dict 后 ratio 应变小, '
            f'默认: {default_ratio:.2f}, 翻转: {flipped_ratio:.2f}'
        )


# ──────────────────────────────────────────────
# 4. measure_loss_space_consistency
# ──────────────────────────────────────────────


class TestMeasureLossSpaceConsistency:
    """验证 loss_bbox 和 loss_giou 在同一空间 ([0,1]) 计算."""

    def test_consistency_ratio_near_2s(self):
        """consistency_ratio 应 ≈ 2*s = 4.0 (s=2).

        扩散空间 L1 是 [0,1] 空间 L1 的 2*s 倍 (s=2 → 4x).
        """
        criterion = SetCriterion(num_classes=24, snr_scale=2.0)
        # 所有值在 [-2, 2] 内 (无 clamp 影响)
        pred_boxes = torch.tensor(
            [[[0.0, 0.0, -1.0, -1.0], [0.5, 0.5, 0.5, 0.5]]]
        )
        matched_boxes = torch.tensor(
            [[[0.2, 0.2, -1.4, -1.4], [0.3, 0.3, 0.3, 0.3]]]
        )
        matched_labels = torch.tensor([[0, 0]])
        result = LossMonitor.measure_loss_space_consistency(
            criterion, pred_boxes, matched_boxes, matched_labels
        )
        # 2*s = 4.0 (s=2)
        assert abs(result['consistency_ratio'] - 4.0) < 0.1, (
            f'consistency_ratio 应 ≈ 4.0 (2*s, s=2), '
            f'实际: {result["consistency_ratio"]:.4f}'
        )

    def test_diffusion_space_l1_larger_than_norm_space(self):
        """扩散空间 L1 应大于 [0,1] 空间 L1 (4x scale, s=2)."""
        criterion = SetCriterion(num_classes=24, snr_scale=2.0)
        pred_boxes = torch.tensor(
            [[[0.0, 0.0, -1.0, -1.0], [0.5, 0.5, 0.5, 0.5]]]
        )
        matched_boxes = torch.tensor(
            [[[0.2, 0.2, -1.4, -1.4], [0.3, 0.3, 0.3, 0.3]]]
        )
        matched_labels = torch.tensor([[0, 0]])
        result = LossMonitor.measure_loss_space_consistency(
            criterion, pred_boxes, matched_boxes, matched_labels
        )
        assert (
            result['loss_bbox_diffusion_space']
            > result['loss_bbox_norm_space']
        )

    def test_returns_expected_keys(self):
        """返回 dict 应包含三个关键 key."""
        criterion = SetCriterion(num_classes=24, snr_scale=2.0)
        pred_boxes = torch.tensor([[[0.0, 0.0, -1.0, -1.0]]])
        matched_boxes = torch.tensor([[[0.2, 0.2, -1.4, -1.4]]])
        matched_labels = torch.tensor([[0]])
        result = LossMonitor.measure_loss_space_consistency(
            criterion, pred_boxes, matched_boxes, matched_labels
        )
        assert 'loss_bbox_diffusion_space' in result
        assert 'loss_bbox_norm_space' in result
        assert 'consistency_ratio' in result

    def test_consistency_ratio_scales_with_snr(self):
        """consistency_ratio 应随 snr_scale 线性变化 (2*s)."""
        for s in [1.0, 2.0, 3.0]:
            criterion = SetCriterion(num_classes=24, snr_scale=s)
            # 值在 [-s, s] 内
            pred_boxes = torch.tensor([[[0.0, 0.0, -0.5, -0.5]]])
            matched_boxes = torch.tensor([[[0.2, 0.2, -0.4, -0.4]]])
            matched_labels = torch.tensor([[0]])
            result = LossMonitor.measure_loss_space_consistency(
                criterion, pred_boxes, matched_boxes, matched_labels
            )
            assert abs(result['consistency_ratio'] - 2 * s) < 0.1, (
                f's={s}: consistency_ratio 应 ≈ {2 * s}, '
                f'实际: {result["consistency_ratio"]:.4f}'
            )


# ──────────────────────────────────────────────
# 5. run_quick_check
# ──────────────────────────────────────────────


class TestRunQuickCheck:
    """验证 run_quick_check 能正常运行无异常."""

    def test_runs_without_exception(self):
        """run_quick_check 应正常运行, 不抛异常."""
        LossMonitor.run_quick_check()

    def test_returns_none(self):
        """run_quick_check 应返回 None."""
        result = LossMonitor.run_quick_check()
        assert result is None


# ──────────────────────────────────────────────
# 6. LossMonitorHook
# ──────────────────────────────────────────────


class TestLossMonitorHook:
    """验证 LossMonitorHook 基础行为."""

    def test_hook_init_default_interval(self):
        """默认 interval=50."""
        hook = LossMonitorHook()
        assert hook.interval == 50

    def test_hook_init_custom_interval(self):
        """自定义 interval."""
        hook = LossMonitorHook(interval=10)
        assert hook.interval == 10

    def test_after_train_iter_skips_non_interval(self):
        """非 interval 步应跳过 (不输出)."""
        from unittest.mock import MagicMock

        hook = LossMonitorHook(interval=50)
        runner = MagicMock()
        # batch_idx=10 不是 50 的倍数 (interval=50)
        hook.after_train_iter(runner, batch_idx=10, outputs={})
        # runner.logger 不应被调用
        runner.logger.info.assert_not_called()

    def test_after_train_iter_handles_no_log_vars(self):
        """outputs 无 log_vars 时应安全返回."""
        from unittest.mock import MagicMock

        hook = LossMonitorHook(interval=50)
        runner = MagicMock()
        # batch_idx=0 是 50 的倍数 (0 % 50 == 0)
        # outputs 不含 log_vars
        hook.after_train_iter(runner, batch_idx=0, outputs={})
        runner.logger.info.assert_not_called()

    def test_after_train_iter_logs_when_ratio_available(self):
        """有 log_vars 且 giou>0 时应输出监控日志."""
        from unittest.mock import MagicMock

        hook = LossMonitorHook(interval=50)
        runner = MagicMock()

        class _Outputs:
            def __init__(self, log_vars):
                self.log_vars = log_vars

        outputs = _Outputs(
            {
                'loss_cls': 1.0,
                'loss_bbox': 2.5,
                'loss_giou': 1.0,
            }
        )
        hook.after_train_iter(runner, batch_idx=0, outputs=outputs)
        runner.logger.info.assert_called_once()
        log_msg = runner.logger.info.call_args[0][0]
        assert 'bbox/giou=2.50' in log_msg
