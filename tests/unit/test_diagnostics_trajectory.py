"""方向四轨迹诊断测试 — 验证 trajectory_diag.py 的统计函数.

测试覆盖:
1. compute_scale_conditioned_stats: 尺度条件化统计
2. compute_ot_coupling_stats: OT 耦合质量统计
3. compute_path_curvature: 路径曲率统计
4. compute_velocity_stats: 速度场统计
5. TrajectoryDiagnosticsCallback: 回调机制
"""

import torch
import pytest

from ldmdet.diagnostics.trajectory_diag import (
    TrajectoryDiagnosticsCallback,
    compute_scale_conditioned_stats,
    compute_ot_coupling_stats,
    compute_path_curvature,
    compute_velocity_stats,
)


# ================================================================
# 1. compute_scale_conditioned_stats
# ================================================================


class TestScaleConditionedStats:
    """测试尺度条件化统计"""

    def test_basic_stats(self):
        """基本统计: 应返回所有字段"""
        scales = torch.rand(2, 10) * 0.2
        t = torch.tensor([0.3, 0.7])
        t_eff = torch.rand(2, 10)
        stats = compute_scale_conditioned_stats(scales, t, t_eff)
        assert 'scale_mean' in stats
        assert 'scale_std' in stats
        assert 'scale_min' in stats
        assert 'scale_max' in stats
        assert 't_eff_mean' in stats
        assert 't_eff_std' in stats
        assert 't_eff_t_correlation' in stats
        assert 'scale_t_eff_correlation' in stats

    def test_with_kappa(self):
        """提供 kappa: 应返回 kappa 统计"""
        scales = torch.rand(2, 10) * 0.2
        t = torch.tensor([0.3, 0.7])
        t_eff = torch.rand(2, 10)
        kappa = torch.full_like(scales, 1.5)
        stats = compute_scale_conditioned_stats(scales, t, t_eff, kappa)
        assert 'kappa_mean' in stats
        assert 'kappa_std' in stats
        assert 'kappa_min' in stats
        assert 'kappa_max' in stats
        assert stats['kappa_mean'] == pytest.approx(1.5, abs=1e-6)

    def test_t_eff_t_correlation_linear(self):
        """线性情况 (t_eff = t): 相关性应接近 1"""
        # 当 κ=1 时 t_eff = t, 相关性应为 1
        scales = torch.full((2, 10), 0.15)  # s_max → κ=1
        t = torch.tensor([0.3, 0.7])
        # t_eff = t (因为 κ=1)
        t_eff = t.unsqueeze(1).expand(2, 10)
        stats = compute_scale_conditioned_stats(scales, t, t_eff)
        assert stats['t_eff_t_correlation'] > 0.99

    def test_scale_t_eff_correlation_negative(self):
        """小尺度 → 小 t_eff: 负相关"""
        # 构造: 小尺度对应小 t_eff
        scales = torch.tensor([[0.01, 0.15], [0.01, 0.15]])
        t = torch.tensor([0.5, 0.5])
        # 小尺度 t_eff 小, 大尺度 t_eff 大
        t_eff = torch.tensor([[0.3, 0.5], [0.3, 0.5]])
        stats = compute_scale_conditioned_stats(scales, t, t_eff)
        # 应正相关 (尺度大 → t_eff 大)
        assert stats['scale_t_eff_correlation'] > 0.5

    def test_single_element(self):
        """单元素: 应不报错"""
        scales = torch.tensor([[0.1]])
        t = torch.tensor([0.5])
        t_eff = torch.tensor([[0.4]])
        stats = compute_scale_conditioned_stats(scales, t, t_eff)
        assert stats['scale_mean'] == pytest.approx(0.1, abs=1e-6)


# ================================================================
# 2. compute_ot_coupling_stats
# ================================================================


class TestOTCouplingStats:
    """测试 OT 耦合质量统计"""

    def test_basic_stats(self):
        """基本统计: 应返回所有字段"""
        transport = torch.rand(20, 10)
        transport = transport / transport.sum()
        stats = compute_ot_coupling_stats(transport)
        assert 'transport_entropy' in stats
        assert 'transport_concentration' in stats
        assert 'row_marginal_std' in stats
        assert 'col_marginal_std' in stats

    def test_with_cost(self):
        """提供 cost: 应返回配对代价"""
        transport = torch.rand(20, 10)
        transport = transport / transport.sum()
        cost = torch.rand(20, 10)
        stats = compute_ot_coupling_stats(transport, cost)
        assert 'paired_cost_mean' in stats
        assert 'paired_cost_max' in stats
        assert stats['paired_cost_mean'] > 0

    def test_uniform_transport_high_entropy(self):
        """均匀传输矩阵: 熵应接近 1"""
        N, K = 20, 10
        transport = torch.ones(N, K) / (N * K)
        stats = compute_ot_coupling_stats(transport)
        assert stats['transport_entropy'] > 0.99

    def test_sparse_transport_low_entropy(self):
        """稀疏传输矩阵: 熵应较低 (vs 均匀矩阵)"""
        N, K = 20, 10
        # 稀疏矩阵: 每行只有一个非零
        transport_sparse = torch.zeros(N, K)
        for i in range(N):
            transport_sparse[i, i % K] = 1.0
        stats_sparse = compute_ot_coupling_stats(transport_sparse)

        # 均匀矩阵作为对照
        transport_uniform = torch.ones(N, K) / (N * K)
        stats_uniform = compute_ot_coupling_stats(transport_uniform)

        # 稀疏矩阵的熵应 < 均匀矩阵
        assert stats_sparse['transport_entropy'] < stats_uniform['transport_entropy']

    def test_uniform_marginal_zero_std(self):
        """均匀边缘: 标准差应为 0"""
        N, K = 20, 10
        transport = torch.ones(N, K) / (N * K)
        stats = compute_ot_coupling_stats(transport)
        # 行边缘应都是 1/N, 标准差为 0
        assert stats['row_marginal_std'] < 1e-6
        # 列边缘应都是 1/K, 标准差为 0
        assert stats['col_marginal_std'] < 1e-6


# ================================================================
# 3. compute_path_curvature
# ================================================================


class TestPathCurvature:
    """测试路径曲率统计"""

    def test_zero_curvature(self):
        """相同速度: 曲率应为 0"""
        velocity = torch.randn(2, 10, 4)
        stats = compute_path_curvature(velocity, velocity, dt=1.0)
        assert stats['curvature_mean'] == pytest.approx(0.0, abs=1e-6)
        assert stats['curvature_max'] == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_curvature(self):
        """不同速度: 曲率应 > 0"""
        v1 = torch.randn(2, 10, 4)
        v2 = v1 + 0.5 * torch.randn_like(v1)  # 加扰动
        stats = compute_path_curvature(v1, v2, dt=1.0)
        assert stats['curvature_mean'] > 0
        assert stats['curvature_max'] > 0

    def test_dt_effect(self):
        """dt 越大, 曲率越小 (除以 dt)"""
        v1 = torch.randn(2, 10, 4)
        v2 = v1 + 1.0 * torch.ones_like(v1)
        stats_dt1 = compute_path_curvature(v1, v2, dt=1.0)
        stats_dt2 = compute_path_curvature(v1, v2, dt=2.0)
        assert stats_dt2['curvature_mean'] == pytest.approx(
            stats_dt1['curvature_mean'] / 2, abs=1e-6
        )

    def test_curvature_relative(self):
        """相对曲率: 应为正"""
        v1 = torch.randn(2, 10, 4)
        v2 = v1 + 0.5 * torch.randn_like(v1)
        stats = compute_path_curvature(v1, v2, dt=1.0)
        assert stats['curvature_relative'] > 0

    def test_shape_handling(self):
        """形状处理: 应支持不同形状"""
        v1 = torch.randn(5, 20, 8)
        v2 = torch.randn(5, 20, 8)
        stats = compute_path_curvature(v1, v2, dt=0.5)
        assert 'curvature_mean' in stats
        assert 'curvature_std' in stats


# ================================================================
# 4. compute_velocity_stats
# ================================================================


class TestVelocityStats:
    """测试速度场统计"""

    def test_basic_stats(self):
        """基本统计: 应返回所有字段"""
        velocity = torch.randn(2, 10, 4)
        stats = compute_velocity_stats(velocity)
        assert 'velocity_norm_mean' in stats
        assert 'velocity_norm_std' in stats
        assert 'velocity_norm_min' in stats
        assert 'velocity_norm_max' in stats
        assert 'velocity_direction_consistency' in stats

    def test_consistent_direction(self):
        """一致方向: 一致性应接近 1"""
        # 所有速度同方向
        direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
        velocity = direction.unsqueeze(0).unsqueeze(0).expand(2, 10, -1) * torch.rand(2, 10, 1)
        stats = compute_velocity_stats(velocity)
        assert stats['velocity_direction_consistency'] > 0.99

    def test_random_direction(self):
        """随机方向: 一致性应较低"""
        torch.manual_seed(42)
        velocity = torch.randn(2, 100, 4)  # 大量随机方向
        stats = compute_velocity_stats(velocity)
        assert stats['velocity_direction_consistency'] < 0.5

    def test_zero_velocity(self):
        """零速度: 应不报错 (clamp 避免 div by 0)"""
        velocity = torch.zeros(2, 10, 4)
        stats = compute_velocity_stats(velocity)
        assert stats['velocity_norm_mean'] == 0.0


# ================================================================
# 5. TrajectoryDiagnosticsCallback
# ================================================================


class TestTrajectoryDiagnosticsCallback:
    """测试轨迹诊断回调"""

    def test_init_defaults(self):
        """默认初始化"""
        cb = TrajectoryDiagnosticsCallback()
        assert cb.interval == 100
        assert cb.last_scales is None

    def test_update_scale(self):
        """update_scale: 应存储数据"""
        cb = TrajectoryDiagnosticsCallback()
        scales = torch.rand(2, 10)
        t = torch.tensor([0.3, 0.7])
        t_eff = torch.rand(2, 10)
        cb.update_scale(scales, t, t_eff)
        assert cb.last_scales is not None
        assert cb.last_t is not None
        assert cb.last_t_eff is not None

    def test_update_ot(self):
        """update_ot: 应存储数据"""
        cb = TrajectoryDiagnosticsCallback()
        transport = torch.rand(20, 10)
        cost = torch.rand(20, 10)
        cb.update_ot(transport, cost)
        assert cb.last_transport is not None
        assert cb.last_cost is not None

    def test_update_curvature(self):
        """update_curvature: 应存储数据"""
        cb = TrajectoryDiagnosticsCallback()
        v1 = torch.randn(2, 10, 4)
        v2 = torch.randn(2, 10, 4)
        cb.update_curvature(v1, v2, dt=0.5)
        assert cb.last_velocity is not None
        assert cb.last_velocity_next is not None

    def test_collect_non_sampling_step(self):
        """非采样点: 应返回空字典"""
        cb = TrajectoryDiagnosticsCallback(interval=100)
        cb.update_scale(torch.rand(2, 10), torch.tensor([0.5, 0.5]), torch.rand(2, 10))
        # step=50 不是采样点 (interval=100)
        data = cb.collect(step=50)
        assert data == {}

    def test_collect_zero_step(self):
        """step=0: 应返回空字典"""
        cb = TrajectoryDiagnosticsCallback(interval=100)
        cb.update_scale(torch.rand(2, 10), torch.tensor([0.5, 0.5]), torch.rand(2, 10))
        data = cb.collect(step=0)
        assert data == {}

    def test_collect_sampling_step_scale(self):
        """采样点 (尺度数据): 应返回统计"""
        cb = TrajectoryDiagnosticsCallback(interval=100)
        scales = torch.rand(2, 10) * 0.2
        t = torch.tensor([0.3, 0.7])
        t_eff = torch.rand(2, 10)
        cb.update_scale(scales, t, t_eff)
        data = cb.collect(step=100)
        assert 'trajectory/scale_mean' in data
        assert 'trajectory/t_eff_mean' in data

    def test_collect_sampling_step_ot(self):
        """采样点 (OT 数据): 应返回统计"""
        cb = TrajectoryDiagnosticsCallback(interval=100)
        transport = torch.rand(20, 10)
        transport = transport / transport.sum()
        cost = torch.rand(20, 10)
        cb.update_ot(transport, cost)
        data = cb.collect(step=100)
        assert 'trajectory/ot_transport_entropy' in data
        assert 'trajectory/ot_paired_cost_mean' in data

    def test_collect_sampling_step_curvature(self):
        """采样点 (曲率数据): 应返回统计"""
        cb = TrajectoryDiagnosticsCallback(interval=100)
        v1 = torch.randn(2, 10, 4)
        v2 = v1 + 0.5 * torch.randn_like(v1)
        cb.update_curvature(v1, v2, dt=1.0)
        data = cb.collect(step=100)
        assert 'trajectory/curvature_mean' in data
        assert 'trajectory/velocity_norm_mean' in data

    def test_collect_empty(self):
        """无数据: 应返回空字典"""
        cb = TrajectoryDiagnosticsCallback(interval=100)
        data = cb.collect(step=100)
        assert data == {}

    def test_collect_with_kappa(self):
        """提供 kappa: 应返回 kappa 统计"""
        cb = TrajectoryDiagnosticsCallback(interval=100)
        scales = torch.rand(2, 10) * 0.2
        t = torch.tensor([0.3, 0.7])
        t_eff = torch.rand(2, 10)
        kappa = torch.full_like(scales, 1.5)
        cb.update_scale(scales, t, t_eff, kappa)
        data = cb.collect(step=100)
        assert 'trajectory/kappa_mean' in data
        assert data['trajectory/kappa_mean'] == pytest.approx(1.5, abs=1e-6)
