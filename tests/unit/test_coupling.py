"""测试耦合策略 (random + ot_flow)"""

import torch
import pytest
from ldmdet.coupling import build_coupling
from ldmdet.coupling._sinkhorn_ops import sinkhorn_transport


class TestSinkhornOps:
    def test_transport_matrix_shape(self):
        cost = torch.rand(10, 5)
        transport = sinkhorn_transport(cost, epsilon=1.0, num_iters=20)
        assert transport.shape == (10, 5)
        # 行和 ≈ 1/N
        assert torch.allclose(transport.sum(dim=1), torch.ones(10) / 10, atol=1e-3)
        # 列和 ≈ 1/K
        assert torch.allclose(transport.sum(dim=0), torch.ones(5) / 5, atol=1e-3)

    def test_small_epsilon_near_hard(self):
        # 固定种子: 避免 torch.rand 受全局随机状态影响导致测试不稳定
        torch.manual_seed(42)
        cost = torch.rand(10, 5)
        transport = sinkhorn_transport(cost, epsilon=1e-6, num_iters=500)
        # 小 ε 下传输矩阵应该接近排列 (虽然均匀 marginal 限制每行≤0.1)
        # 验证: 大部分概率集中在最小值位置
        min_cost_idx = cost.argmin(dim=1)
        for i in range(10):
            max_val = transport[i].max()
            max_idx = transport[i].argmax()
            # 大概率 argmax 在最小 cost 位置
            if max_idx != min_cost_idx[i]:
                # 如果不在, 说明次优 cost 非常接近
                # 阈值 0.1: Sinkhorn 在 ε=1e-6 下受 marginal 约束影响,
                # 某些行会被强制分配到次优列 (cost 差异可达 10%)
                assert torch.abs(cost[i, max_idx] - cost[i, min_cost_idx[i]]) < 0.1

    def test_large_epsilon_near_uniform(self):
        cost = torch.rand(10, 5)
        transport = sinkhorn_transport(cost, epsilon=100.0, num_iters=20)
        # 大 ε 下每行 1/N=0.1 均匀分配到 5 列 → 每列 ~0.02
        uniform_row = torch.ones(10, 5) * 0.1 / 5
        assert torch.allclose(transport, uniform_row, atol=0.01)


class TestRandomCoupling:
    def test_build(self):
        c = build_coupling('random')
        assert c is not None

    def test_couple_shape(self):
        c = build_coupling('random')
        noise = torch.randn(10, 4)
        gt = torch.randn(3, 4)
        labels = torch.tensor([0, 1, 2])
        x_start, idx = c.couple(noise, gt, labels, torch.device('cpu'))
        assert x_start.shape == (10, 4)
        assert idx.shape == (10,)
        assert idx.max() < 3

    def test_empty_gt(self):
        c = build_coupling('random')
        noise = torch.randn(10, 4)
        gt = torch.randn(0, 4)
        x_start, idx = c.couple(noise, gt, torch.tensor([]), torch.device('cpu'))
        assert torch.equal(x_start, noise)
        assert (idx == 0).all()


class TestOTFlowCoupling:
    """测试 OT Flow Matching 耦合策略 (方向四)"""

    def test_build(self):
        c = build_coupling('ot_flow', epsilon=1.0, num_iters=10)
        assert c is not None
        assert c.epsilon == 1.0
        assert c.num_iters == 10

    def test_build_defaults(self):
        c = build_coupling('ot_flow')
        assert c.epsilon == 1.0
        assert c.num_iters == 10

    def test_couple_shape(self):
        c = build_coupling('ot_flow', epsilon=1.0)
        noise = torch.randn(500, 4)
        gt = torch.randn(46, 4)
        labels = torch.randint(0, 24, (46,))
        x_start, idx = c.couple(noise, gt, labels, torch.device('cpu'))
        assert x_start.shape == (500, 4)
        assert idx.shape == (500,)
        assert idx.max() < 46

    def test_couple_matches_gt(self):
        """couple 返回的 x_start 应来自 GT 框"""
        c = build_coupling('ot_flow', epsilon=1.0, coupling_mode='argmax')
        noise = torch.randn(100, 4)
        gt = torch.randn(10, 4)
        labels = torch.arange(10)
        x_start, idx = c.couple(noise, gt, labels, torch.device('cpu'))
        # 每个 x_start[i] 应等于 gt[idx[i]]
        for i in range(100):
            assert torch.allclose(x_start[i], gt[idx[i]], atol=1e-6)

    def test_couple_multinomial_mode(self):
        """multinomial 模式应正常工作"""
        torch.manual_seed(42)
        c = build_coupling('ot_flow', epsilon=5.0, coupling_mode='multinomial')
        noise = torch.randn(500, 4)
        gt = torch.randn(46, 4)
        labels = torch.randint(0, 24, (46,))
        x_start, idx = c.couple(noise, gt, labels, torch.device('cpu'))
        assert x_start.shape == (500, 4)
        # multinomial 应匹配到不同 GT
        unique = torch.unique(idx)
        assert len(unique) > 1

    def test_empty_gt(self):
        """空 GT: 应原样返回 noise, idx 全 0"""
        c = build_coupling('ot_flow', epsilon=1.0)
        noise = torch.randn(10, 4)
        gt = torch.randn(0, 4)
        x_start, idx = c.couple(noise, gt, torch.tensor([]), torch.device('cpu'))
        assert torch.equal(x_start, noise)
        assert (idx == 0).all()

    def test_single_gt(self):
        """单个 GT: 所有 proposal 应匹配到它"""
        c = build_coupling('ot_flow', epsilon=1.0)
        noise = torch.randn(50, 4)
        gt = torch.randn(1, 4)
        labels = torch.tensor([0])
        x_start, idx = c.couple(noise, gt, labels, torch.device('cpu'))
        assert (idx == 0).all()
        for i in range(50):
            assert torch.allclose(x_start[i], gt[0], atol=1e-6)

    def test_ot_epsilon_property(self):
        """ot_epsilon 属性应兼容诊断器"""
        c = build_coupling('ot_flow', epsilon=3.0)
        assert c.ot_epsilon == 3.0

    def test_ot_module_exposed(self):
        """ot_module 应暴露内部 OTFlowMatching"""
        c = build_coupling('ot_flow', epsilon=2.0)
        assert c.ot_module is not None
        assert hasattr(c.ot_module, 'compute_ot_coupling')
        assert hasattr(c.ot_module, 'compute_coupling_cost')
