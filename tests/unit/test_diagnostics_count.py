"""方向二计数诊断测试 — 计数分支预测分布、计数约束 NMS 触发率.

验证计数先验是否有效约束检测头输出.
若方向二废弃, 删除 count_diag.py + 本测试文件即可清理.

测试内容:
- 计数分支预测统计 (预测计数 vs 真实计数偏差)
- 计数约束 NMS 触发率 (约束生效比例)
- CountDiagnosticsCallback (集成到 TrainingDiagnosticsHook)
"""

import torch
import pytest

from ldmdet.diagnostics.count_diag import (
    compute_count_prediction_stats,
    compute_count_constraint_stats,
    CountDiagnosticsCallback,
)


# ──────────────────────────────────────────────
# compute_count_prediction_stats: 计数预测统计
# ──────────────────────────────────────────────

class TestCountPredictionStats:
    """测试计数分支预测统计."""

    def test_perfect_prediction(self):
        """完美预测: 偏差为 0."""
        pred_counts = torch.tensor([46, 46, 46, 46])
        gt_counts = torch.tensor([46, 46, 46, 46])
        stats = compute_count_prediction_stats(pred_counts, gt_counts)

        assert stats['mae'] == pytest.approx(0.0, abs=1e-4)
        assert stats['mse'] == pytest.approx(0.0, abs=1e-4)
        assert stats['bias'] == pytest.approx(0.0, abs=1e-4)
        assert stats['accuracy_0'] == 1.0  # 完全准确
        assert stats['accuracy_2'] == 1.0  # 容差 2 内全准

    def test_overestimation(self):
        """系统性高估."""
        pred_counts = torch.tensor([50, 48, 52, 49])
        gt_counts = torch.tensor([46, 46, 46, 46])
        stats = compute_count_prediction_stats(pred_counts, gt_counts)

        assert stats['bias'] > 0  # 高估
        assert stats['mae'] > 0
        assert stats['accuracy_0'] == 0.0  # 无完全准确

    def test_underestimation(self):
        """系统性低估."""
        pred_counts = torch.tensor([40, 42, 38, 41])
        gt_counts = torch.tensor([46, 46, 46, 46])
        stats = compute_count_prediction_stats(pred_counts, gt_counts)

        assert stats['bias'] < 0  # 低估
        assert stats['mae'] > 0

    def test_accuracy_with_tolerance(self):
        """容差准确率."""
        pred_counts = torch.tensor([46, 47, 48, 50])  # 偏差 0, 1, 2, 4
        gt_counts = torch.tensor([46, 46, 46, 46])
        stats = compute_count_prediction_stats(pred_counts, gt_counts)

        assert stats['accuracy_0'] == 0.25  # 1/4 完全准确
        assert stats['accuracy_2'] == 0.75  # 3/4 在容差 2 内
        assert stats['accuracy_5'] == 1.0  # 4/4 在容差 5 内


# ──────────────────────────────────────────────
# compute_count_constraint_stats: 计数约束 NMS 统计
# ──────────────────────────────────────────────

class TestCountConstraintStats:
    """测试计数约束 NMS 触发统计."""

    def test_no_constraint_triggered(self):
        """无约束触发 (预测数 <= 目标数)."""
        # 4 张图, 每张 NMS 后 40 个框, 目标 46
        nms_counts = torch.tensor([40, 40, 40, 40])
        target_counts = torch.tensor([46, 46, 46, 46])
        stats = compute_count_constraint_stats(nms_counts, target_counts)

        assert stats['constraint_trigger_ratio'] == 0.0  # 无触发
        assert stats['over_count_ratio'] == 0.0

    def test_all_constraint_triggered(self):
        """全部触发约束 (预测数 > 目标数)."""
        nms_counts = torch.tensor([60, 55, 50, 48])
        target_counts = torch.tensor([46, 46, 46, 46])
        stats = compute_count_constraint_stats(nms_counts, target_counts)

        assert stats['constraint_trigger_ratio'] == 1.0  # 全触发
        assert stats['over_count_ratio'] > 0
        # 平均超出比例 = (14+9+4+2)/46/4
        assert stats['over_count_ratio'] == pytest.approx((14 + 9 + 4 + 2) / 46 / 4, abs=1e-3)

    def test_partial_constraint_triggered(self):
        """部分触发约束."""
        nms_counts = torch.tensor([40, 50, 46, 60])  # 2 张触发
        target_counts = torch.tensor([46, 46, 46, 46])
        stats = compute_count_constraint_stats(nms_counts, target_counts)

        assert stats['constraint_trigger_ratio'] == 0.5  # 2/4 触发

    def test_threshold_adjustment_stats(self):
        """阈值调整幅度统计."""
        # 模拟 NMS 阈值从 0.5 调整到更高值以满足计数约束
        original_thr = 0.5
        adjusted_thrs = torch.tensor([0.5, 0.6, 0.7, 0.5])  # 2 张调整了阈值
        stats = compute_count_constraint_stats(
            nms_counts=torch.tensor([40, 50, 60, 40]),
            target_counts=torch.tensor([46, 46, 46, 46]),
            original_thr=original_thr,
            adjusted_thrs=adjusted_thrs,
        )

        assert 'thr_adjustment_mean' in stats
        assert stats['thr_adjustment_mean'] > 0  # 平均调整幅度 > 0
        assert stats['thr_adjustment_ratio'] == 0.5  # 2/4 调整了


# ──────────────────────────────────────────────
# CountDiagnosticsCallback: 计数诊断回调
# ──────────────────────────────────────────────

class TestCountDiagnosticsCallback:
    """测试计数诊断回调."""

    def test_initialization(self):
        """初始化."""
        callback = CountDiagnosticsCallback(interval=100)
        assert callback.interval == 100

    def test_update_and_collect(self):
        """更新和采集."""
        callback = CountDiagnosticsCallback(interval=100)
        callback.update(
            pred_counts=torch.tensor([46, 47, 48, 50]),
            gt_counts=torch.tensor([46, 46, 46, 46]),
        )

        data = callback.collect(step=100)
        assert 'count/mae' in data
        assert 'count/mse' in data
        assert 'count/bias' in data
        assert 'count/accuracy_2' in data

    def test_interval_control(self):
        """采样频率控制."""
        callback = CountDiagnosticsCallback(interval=100)
        callback.update(
            pred_counts=torch.tensor([46]),
            gt_counts=torch.tensor([46]),
        )

        # step=50, 不采集
        data = callback.collect(step=50)
        assert data == {}

        # step=100, 采集
        data = callback.collect(step=100)
        assert len(data) > 0

    def test_update_constraint_stats(self):
        """更新约束 NMS 统计."""
        callback = CountDiagnosticsCallback(interval=100)
        callback.update_constraint(
            nms_counts=torch.tensor([40, 50, 60, 40]),
            target_counts=torch.tensor([46, 46, 46, 46]),
        )

        data = callback.collect(step=100)
        assert 'count/constraint_trigger_ratio' in data
        assert 'count/over_count_ratio' in data
