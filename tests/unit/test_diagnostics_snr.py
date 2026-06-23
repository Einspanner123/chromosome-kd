"""方向三 SNR 诊断测试 — SNR 权重分布、动态匹配统计、损失加权效果.

验证 SNR 感知机制是否按预期工作.
若方向三废弃, 删除 snr_diag.py + 本测试文件即可清理.

测试内容:
- SNR 权重分布统计 (不同 t 下的权重分布)
- 动态匹配统计 (k 值分布、匹配质量)
- 损失加权效果 (高/低 SNR 样本的损失比)
- SNRDiagnosticsCallback (集成到 TrainingDiagnosticsHook)
"""

import torch
import pytest

from ldmdet.diagnostics.snr_diag import (
    compute_snr_weight_stats,
    compute_dynamic_match_stats,
    compute_loss_weighting_effect,
    SNRDiagnosticsCallback,
)


# ──────────────────────────────────────────────
# compute_snr_weight_stats: SNR 权重分布统计
# ──────────────────────────────────────────────

class TestSNRWeightStats:
    """测试 SNR 权重分布统计."""

    def test_basic_stats(self):
        """基本统计."""
        # 模拟 SNR 权重 [bs]
        snr_weights = torch.tensor([0.1, 0.5, 0.9, 0.3])
        t = torch.tensor([0.1, 0.5, 0.9, 0.3])
        stats = compute_snr_weight_stats(snr_weights, t)

        assert 'weight_mean' in stats
        assert 'weight_std' in stats
        assert 'weight_min' in stats
        assert 'weight_max' in stats
        assert stats['weight_mean'] == pytest.approx(0.45, abs=1e-4)

    def test_snr_t_correlation(self):
        """SNR 权重与 t 的相关性."""
        # t 越小 (早期), SNR 越高, 权重越大 (假设 logistic 权重)
        t = torch.linspace(0, 1, 100)
        # 模拟 logistic SNR 权重: t 小时权重大
        snr_weights = 1.0 / (1.0 + torch.exp(5 * (t - 0.5)))
        stats = compute_snr_weight_stats(snr_weights, t)

        # 应记录相关性
        assert 'weight_t_correlation' in stats
        # 负相关 (t 大, 权重小)
        assert stats['weight_t_correlation'] < -0.5

    def test_extreme_weights(self):
        """极端权重 (全零或全一)."""
        # 全零
        stats = compute_snr_weight_stats(torch.zeros(10), torch.linspace(0, 1, 10))
        assert stats['weight_mean'] == 0.0
        assert stats['weight_zero_ratio'] == 1.0

        # 全一
        stats = compute_snr_weight_stats(torch.ones(10), torch.linspace(0, 1, 10))
        assert stats['weight_mean'] == 1.0
        assert stats['weight_zero_ratio'] == 0.0

    def test_weight_distribution_by_t_bin(self):
        """按 t 分箱的权重分布."""
        t = torch.tensor([0.1, 0.1, 0.5, 0.5, 0.9, 0.9])
        snr_weights = torch.tensor([0.9, 0.8, 0.5, 0.4, 0.1, 0.2])
        stats = compute_snr_weight_stats(snr_weights, t, num_bins=3)

        # 应记录每个 t 箱的平均权重
        assert 'weight_bin_0_mean' in stats  # t ∈ [0, 1/3)
        assert 'weight_bin_1_mean' in stats  # t ∈ [1/3, 2/3)
        assert 'weight_bin_2_mean' in stats  # t ∈ [2/3, 1]
        # t 小的箱权重大
        assert stats['weight_bin_0_mean'] > stats['weight_bin_2_mean']


# ──────────────────────────────────────────────
# compute_dynamic_match_stats: 动态匹配统计
# ──────────────────────────────────────────────

class TestDynamicMatchStats:
    """测试动态匹配统计."""

    def test_basic_k_stats(self):
        """基本 k 值统计."""
        # 模拟每张图的动态 k 值
        k_values = torch.tensor([5, 10, 15, 20])
        stats = compute_dynamic_match_stats(k_values)

        assert 'k_mean' in stats
        assert 'k_std' in stats
        assert 'k_min' in stats
        assert 'k_max' in stats
        assert stats['k_mean'] == pytest.approx(12.5, abs=1e-4)
        assert stats['k_min'] == 5
        assert stats['k_max'] == 20

    def test_match_quality(self):
        """匹配质量 (IoU 分布)."""
        k_values = torch.tensor([10, 10])
        match_ious = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        stats = compute_dynamic_match_stats(k_values, match_ious=match_ious)

        assert 'match_iou_mean' in stats
        assert 'match_iou_std' in stats
        assert 'match_iou_high_ratio' in stats  # IoU > 0.5 比例
        assert stats['match_iou_mean'] == pytest.approx(0.45, abs=1e-4)
        # IoU > 0.5: 0.9, 0.8, 0.7, 0.6 共 4 个, 4/10 = 0.4
        assert stats['match_iou_high_ratio'] == pytest.approx(0.4, abs=1e-4)

    def test_empty_matches(self):
        """空匹配场景."""
        k_values = torch.tensor([0, 0])
        stats = compute_dynamic_match_stats(k_values)
        assert stats['k_mean'] == 0.0


# ──────────────────────────────────────────────
# compute_loss_weighting_effect: 损失加权效果
# ──────────────────────────────────────────────

class TestLossWeightingEffect:
    """测试损失加权效果统计."""

    def test_basic_effect(self):
        """基本加权效果."""
        # 原始损失 [bs]
        original_losses = torch.tensor([1.0, 1.0, 1.0, 1.0])
        # SNR 权重 [bs]
        snr_weights = torch.tensor([2.0, 1.0, 0.5, 0.1])
        # 加权后损失
        weighted_losses = original_losses * snr_weights

        stats = compute_loss_weighting_effect(original_losses, weighted_losses, snr_weights)

        assert 'loss_scale_mean' in stats
        assert 'loss_scale_std' in stats
        assert 'high_snr_loss_ratio' in stats  # 高 SNR 样本损失占比
        assert 'low_snr_loss_ratio' in stats  # 低 SNR 样本损失占比
        # 高 SNR (权重 > 1) 样本损失占比应 > 低 SNR
        assert stats['high_snr_loss_ratio'] > stats['low_snr_loss_ratio']

    def test_uniform_weights(self):
        """均匀权重 (无加权效果)."""
        original_losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
        snr_weights = torch.ones(4)
        weighted_losses = original_losses * snr_weights

        stats = compute_loss_weighting_effect(original_losses, weighted_losses, snr_weights)

        assert stats['loss_scale_mean'] == pytest.approx(1.0, abs=1e-4)
        assert stats['high_snr_loss_ratio'] == pytest.approx(0.5, abs=1e-4)
        assert stats['low_snr_loss_ratio'] == pytest.approx(0.5, abs=1e-4)


# ──────────────────────────────────────────────
# SNRDiagnosticsCallback: SNR 诊断回调
# ──────────────────────────────────────────────

class TestSNRDiagnosticsCallback:
    """测试 SNR 诊断回调."""

    def test_initialization(self):
        """初始化."""
        callback = SNRDiagnosticsCallback(interval=100)
        assert callback.interval == 100

    def test_update_and_collect(self):
        """更新和采集."""
        callback = SNRDiagnosticsCallback(interval=100)
        callback.update(
            snr_weights=torch.tensor([0.1, 0.5, 0.9]),
            t=torch.tensor([0.1, 0.5, 0.9]),
        )

        data = callback.collect(step=100)
        assert 'snr/weight_mean' in data
        assert 'snr/weight_std' in data
        assert 'snr/weight_t_correlation' in data

    def test_interval_control(self):
        """采样频率控制."""
        callback = SNRDiagnosticsCallback(interval=100)
        callback.update(
            snr_weights=torch.tensor([0.5]),
            t=torch.tensor([0.5]),
        )

        # step=50, 不采集
        data = callback.collect(step=50)
        assert data == {}

        # step=100, 采集
        data = callback.collect(step=100)
        assert len(data) > 0

    def test_update_match_stats(self):
        """更新匹配统计."""
        callback = SNRDiagnosticsCallback(interval=100)
        callback.update_match(
            k_values=torch.tensor([5, 10, 15]),
            match_ious=torch.tensor([0.9, 0.5, 0.3]),
        )

        data = callback.collect(step=100)
        assert 'snr/k_mean' in data
        assert 'snr/match_iou_mean' in data
