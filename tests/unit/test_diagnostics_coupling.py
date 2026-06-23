"""方向一耦合诊断测试 — matched_gt_idx 分布、proposal 梯度 norm.

验证耦合层信息是否传播到 Transformer 表示层.
若方向一废弃, 删除 coupling_diag.py + 本测试文件即可清理.

测试内容:
- matched_gt_idx 分布统计 (均匀性、熵、组间均衡)
- proposal 特征梯度 norm (耦合信息传播指标)
- 耦合诊断回调 (集成到 TrainingDiagnosticsHook)
"""

import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock, patch

from ldmdet.diagnostics.coupling_diag import (
    compute_coupling_stats,
    compute_proposal_grad_norm,
    CouplingDiagnosticsCallback,
)


# ──────────────────────────────────────────────
# compute_coupling_stats: 耦合分布统计
# ──────────────────────────────────────────────

class TestCouplingStats:
    """测试耦合分布统计."""

    def test_uniform_distribution(self):
        """均匀分布: 熵高, 方差低."""
        # 100 个 proposal, 5 个 GT, 每个被匹配 20 次
        matched_gt_idx = torch.tensor([0, 1, 2, 3, 4] * 20)
        stats = compute_coupling_stats(matched_gt_idx, num_gt=5)

        assert stats['num_proposals'] == 100
        assert stats['num_gt'] == 5
        assert stats['mean_per_gt'] == pytest.approx(20.0, abs=1e-4)
        assert stats['std_per_gt'] == pytest.approx(0.0, abs=1e-4)
        assert stats['min_per_gt'] == 20
        assert stats['max_per_gt'] == 20
        # 均匀分布熵最大
        assert stats['entropy'] == pytest.approx(torch.log(torch.tensor(5.0)).item(), abs=1e-3)

    def test_skewed_distribution(self):
        """偏斜分布: 熵低, 方差高."""
        # 100 个 proposal, 5 个 GT, 但 GT 0 被匹配 60 次, 其他各 10 次
        matched_gt_idx = torch.tensor([0] * 60 + [1, 2, 3, 4] * 10)
        stats = compute_coupling_stats(matched_gt_idx, num_gt=5)

        assert stats['mean_per_gt'] == pytest.approx(20.0, abs=1e-4)
        assert stats['std_per_gt'] > 15.0  # 高方差
        assert stats['min_per_gt'] == 10
        assert stats['max_per_gt'] == 60
        # 偏斜分布熵低
        assert stats['entropy'] < torch.log(torch.tensor(5.0)).item()

    def test_empty_gt(self):
        """空 GT 场景."""
        matched_gt_idx = torch.zeros(100, dtype=torch.long)
        stats = compute_coupling_stats(matched_gt_idx, num_gt=0)
        assert stats['num_gt'] == 0
        assert stats['entropy'] == 0.0

    def test_group_balance(self):
        """组间均衡度 (染色体组间 proposal 分配)."""
        # 模拟 3 个组, 每组不同 GT 数
        # 组 A: 2 GT, 组 B: 5 GT, 组 C: 3 GT
        # proposal 分配: 组 A 40, 组 B 50, 组 C 30 (理想: 按比例 24:60:36)
        group_counts = torch.tensor([40, 50, 30])
        group_gt_counts = torch.tensor([2, 5, 3])
        stats = compute_coupling_stats(
            matched_gt_idx=torch.zeros(1),  # 占位
            num_gt=10,
            group_counts=group_counts,
            group_gt_counts=group_gt_counts,
        )

        assert 'group_balance_ratio' in stats
        # group_balance_ratio = 实际比例 vs 理想比例的偏差
        # 理想: 24:60:36, 实际: 40:50:30
        # 偏差应 > 0
        assert stats['group_balance_ratio'] > 0


# ──────────────────────────────────────────────
# compute_proposal_grad_norm: proposal 梯度 norm
# ──────────────────────────────────────────────

class TestProposalGradNorm:
    """测试 proposal 特征梯度 norm 计算."""

    def test_basic_grad_norm(self):
        """基本梯度 norm 计算."""
        # 模拟 proposal 特征 [bs, num_proposals, feat_dim]
        proposals = torch.randn(2, 100, 256, requires_grad=True)
        loss = proposals.sum()
        loss.backward()

        stats = compute_proposal_grad_norm(proposals)

        assert 'grad_norm_mean' in stats
        assert 'grad_norm_std' in stats
        assert 'grad_norm_min' in stats
        assert 'grad_norm_max' in stats
        assert stats['grad_norm_mean'] > 0
        # 每个 proposal 的梯度为 ones(256), norm = sqrt(256) = 16
        assert stats['grad_norm_mean'] == pytest.approx(16.0, abs=1e-4)

    def test_zero_grad(self):
        """零梯度场景."""
        proposals = torch.randn(2, 100, 256, requires_grad=True)
        # 不反向传播, 梯度为 None
        stats = compute_proposal_grad_norm(proposals)
        assert stats['grad_norm_mean'] == 0.0
        assert stats['grad_norm_zero_ratio'] == 1.0

    def test_dead_proposals(self):
        """死 proposal 检测 (梯度为零的 proposal)."""
        proposals = torch.randn(2, 100, 256, requires_grad=True)
        # 只让 batch 0 的前 50 个 proposal 有梯度
        # 总共 200 个 proposal, 50 个有梯度, 150 个无梯度
        loss = proposals[0, :50].sum()
        loss.backward()

        stats = compute_proposal_grad_norm(proposals)
        # 150/200 = 0.75 proposal 梯度为零
        assert stats['grad_norm_zero_ratio'] == pytest.approx(0.75, abs=1e-2)


# ──────────────────────────────────────────────
# CouplingDiagnosticsCallback: 耦合诊断回调
# ──────────────────────────────────────────────

class TestCouplingDiagnosticsCallback:
    """测试耦合诊断回调 (集成到 TrainingDiagnosticsHook)."""

    def test_callback_initialization(self):
        """回调初始化."""
        callback = CouplingDiagnosticsCallback(interval=100)
        assert callback.interval == 100
        assert callback.last_matched_gt_idx is None

    def test_callback_update(self):
        """更新耦合数据."""
        callback = CouplingDiagnosticsCallback(interval=100)
        matched_gt_idx = torch.tensor([0, 1, 2, 0, 1, 2])
        callback.update(matched_gt_idx, num_gt=3)

        assert callback.last_matched_gt_idx is not None
        assert callback.last_num_gt == 3

    def test_callback_collect(self):
        """采集诊断数据."""
        callback = CouplingDiagnosticsCallback(interval=100)
        matched_gt_idx = torch.tensor([0, 0, 1, 1, 2, 2])
        callback.update(matched_gt_idx, num_gt=3)

        data = callback.collect(step=100)
        assert 'coupling/entropy' in data
        assert 'coupling/std_per_gt' in data
        assert 'coupling/mean_per_gt' in data

    def test_callback_interval_control(self):
        """采样频率控制."""
        callback = CouplingDiagnosticsCallback(interval=100)
        callback.update(torch.tensor([0, 1, 2]), num_gt=3)

        # step=50, 不是 100 的倍数, 不采集
        data = callback.collect(step=50)
        assert data == {}

        # step=100, 采集
        data = callback.collect(step=100)
        assert len(data) > 0
