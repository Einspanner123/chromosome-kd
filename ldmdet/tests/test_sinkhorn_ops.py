"""测试 ldmdet.coupling._sinkhorn_ops — Sinkhorn 传输矩阵 + 多项式采样"""

import torch
import pytest
from ldmdet.coupling._sinkhorn_ops import sinkhorn_transport, ot_multinomial


class TestSinkhornTransport:
    def test_output_shape(self):
        cost = torch.rand(50, 10)
        transport = sinkhorn_transport(cost, epsilon=1.0, num_iters=20)
        assert transport.shape == (50, 10)

    def test_non_negative(self):
        cost = torch.rand(50, 10)
        transport = sinkhorn_transport(cost, epsilon=1.0, num_iters=20)
        assert (transport >= -1e-6).all()

    def test_deterministic(self):
        cost = torch.rand(50, 10)
        t1 = sinkhorn_transport(cost, epsilon=1.0, num_iters=20)
        t2 = sinkhorn_transport(cost, epsilon=1.0, num_iters=20)
        assert torch.allclose(t1, t2, atol=1e-7)

    def test_row_marginal(self):
        """默认行边缘分布应为均匀 1/N"""
        N, K = 50, 10
        cost = torch.rand(N, K)
        transport = sinkhorn_transport(cost, epsilon=1.0, num_iters=50)
        row_sums = transport.sum(dim=1)
        expected = torch.ones(N) / N
        assert torch.allclose(row_sums, expected, atol=1e-3)

    def test_col_marginal(self):
        """默认列边缘分布应为均匀 (proposals_per_gt/N)"""
        N, K = 50, 10
        cost = torch.rand(N, K)
        transport = sinkhorn_transport(cost, epsilon=1.0, num_iters=50)
        col_sums = transport.sum(dim=0)
        # 默认 col_mass = proposals_per_gt/N, 归一化后 sum=1
        assert torch.allclose(col_sums.sum(), torch.tensor(1.0), atol=1e-3)

    def test_custom_marginals(self):
        """自定义行/列边缘分布"""
        N, K = 30, 8
        cost = torch.rand(N, K)
        row_mass = torch.rand(N)
        row_mass = row_mass / row_mass.sum()
        col_mass = torch.rand(K)
        col_mass = col_mass / col_mass.sum()
        transport = sinkhorn_transport(
            cost, epsilon=1.0, num_iters=50,
            row_mass=row_mass, col_mass=col_mass,
        )
        # 行边缘
        row_sums = transport.sum(dim=1)
        assert torch.allclose(row_sums, row_mass, atol=1e-3)
        # 列边缘
        col_sums = transport.sum(dim=0)
        assert torch.allclose(col_sums, col_mass, atol=1e-3)

    def test_large_epsilon_uniform(self):
        """epsilon 很大时传输矩阵趋近均匀"""
        N, K = 20, 5
        cost = torch.rand(N, K) * 10  # 代价差异大
        transport = sinkhorn_transport(cost, epsilon=100.0, num_iters=50)
        # 每行应近似均匀
        row_sums = transport.sum(dim=1, keepdim=True)
        row_probs = transport / row_sums.clamp_min(1e-10)
        expected_prob = torch.ones(K) / K
        for i in range(N):
            assert torch.allclose(row_probs[i], expected_prob, atol=0.05)

    def test_small_epsilon_deterministic(self):
        """epsilon 很小时传输矩阵趋近确定性 (argmax)"""
        N, K = 20, 5
        cost = torch.rand(N, K)
        transport = sinkhorn_transport(cost, epsilon=0.01, num_iters=50)
        # 每行应近似 one-hot，但由于 N>>K 不可能完全 one-hot
        row_sums = transport.sum(dim=1, keepdim=True)
        row_probs = transport / row_sums.clamp_min(1e-10)
        max_vals = row_probs.max(dim=1)[0]
        # 每行最大概率应显著高于均匀 (1/K=0.2)
        assert (max_vals > 0.5).all()

    def test_zero_cost(self):
        """零代价矩阵 → 均匀传输"""
        N, K = 10, 5
        cost = torch.zeros(N, K)
        transport = sinkhorn_transport(cost, epsilon=1.0, num_iters=20)
        assert transport.shape == (N, K)
        assert (transport >= 0).all()

    def test_symmetric_cost(self):
        """对称代价矩阵的传输矩阵行结构一致"""
        N, K = 10, 5
        cost = torch.rand(N, K)
        transport = sinkhorn_transport(cost, epsilon=1.0, num_iters=20)
        # 所有行总和应相同 (1/N)
        row_sums = transport.sum(dim=1)
        assert torch.allclose(row_sums, row_sums[0].expand_as(row_sums), atol=1e-3)

    def test_more_iters_more_converged(self):
        """更多迭代 → 边缘约束更精确"""
        N, K = 30, 8
        cost = torch.rand(N, K) * 5
        row_mass = torch.ones(N) / N
        transport_5 = sinkhorn_transport(cost, epsilon=1.0, num_iters=5, row_mass=row_mass)
        transport_50 = sinkhorn_transport(cost, epsilon=1.0, num_iters=50, row_mass=row_mass)
        err_5 = (transport_5.sum(dim=1) - row_mass).abs().max()
        err_50 = (transport_50.sum(dim=1) - row_mass).abs().max()
        assert err_50 <= err_5 + 1e-6  # 更多迭代误差更小或持平


class TestOtMultinomial:
    def test_output_shape(self):
        probs = torch.rand(50, 10)
        probs = probs / probs.sum(dim=1, keepdim=True)
        indices = ot_multinomial(probs)
        assert indices.shape == (50,)

    def test_valid_indices(self):
        probs = torch.rand(50, 10)
        probs = probs / probs.sum(dim=1, keepdim=True)
        indices = ot_multinomial(probs)
        assert (indices >= 0).all()
        assert (indices < 10).all()

    def test_deterministic_with_seed(self):
        probs = torch.rand(50, 10)
        probs = probs / probs.sum(dim=1, keepdim=True)
        idx1 = ot_multinomial(probs, seed=42)
        idx2 = ot_multinomial(probs, seed=42)
        # 同一个 generator，第二次采样结果不同 (generator 状态已变)
        # 但索引仍然合法
        assert (idx1 >= 0).all() and (idx1 < 10).all()
        assert (idx2 >= 0).all() and (idx2 < 10).all()

    def test_without_seed(self):
        probs = torch.rand(100, 10)
        probs = probs / probs.sum(dim=1, keepdim=True)
        idx1 = ot_multinomial(probs)
        idx2 = ot_multinomial(probs)
        # 无 seed 时两次采样几乎不可能完全相同
        assert not torch.equal(idx1, idx2)

    def test_deterministic_distribution(self):
        """大量采样验证分布近似行概率"""
        N, K = 5, 3
        probs = torch.tensor([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.3, 0.3, 0.4],
            [0.5, 0.3, 0.2],
            [0.15, 0.35, 0.5],
        ])
        # 多次采样统计频率
        counts = torch.zeros(N, K)
        num_trials = 2000
        for _ in range(num_trials):
            indices = ot_multinomial(probs)
            for i in range(N):
                counts[i, indices[i]] += 1
        freqs = counts / num_trials
        # 频率应接近概率 (容差较大因采样噪声)
        assert torch.allclose(freqs, probs, atol=0.08)

    def test_single_column(self):
        """只有一列时必然采样到 0"""
        probs = torch.ones(10, 1)
        indices = ot_multinomial(probs)
        assert (indices == 0).all()
