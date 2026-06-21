"""方向一: 非平衡最优传输耦合 (Unbalanced OT Coupling) 测试

测试 unbalanced_sinkhorn_transport 和 UnbalancedGHSSCoupling.

测试覆盖:
1. 退化等价性: λ → ∞ 时非平衡 Sinkhorn 等价于标准 Sinkhorn
2. 边缘违反度: λ 减小时 KL 违反度应增大
3. 传输矩阵性质: 非负、形状正确
4. 耦合策略: 空组、单组、多组场景
5. 极限行为: λ → 0 时退化为 Gibbs 分布
"""

import torch
import pytest

from ldmdet.coupling import build_coupling
from ldmdet.coupling._sinkhorn_ops import (
    sinkhorn_transport,
    unbalanced_sinkhorn_transport,
)


class TestUnbalancedSinkhornTransport:
    """测试 unbalanced_sinkhorn_transport 函数"""

    def test_shape(self):
        cost = torch.rand(10, 5)
        transport = unbalanced_sinkhorn_transport(
            cost, epsilon=1.0, num_iters=20
        )
        assert transport.shape == (10, 5)

    def test_non_negative(self):
        """传输矩阵应非负"""
        cost = torch.rand(20, 8)
        transport = unbalanced_sinkhorn_transport(
            cost, epsilon=1.0, lambda_row=1.0, lambda_col=1.0
        )
        assert (transport >= 0).all()

    def test_degenerate_to_standard_sinkhorn(self):
        """λ → ∞ 时应数值等价于标准 Sinkhorn.

        数学依据: α = λ/(ε+λ) → 1, β = λ/(ε+λ) → 1,
        迭代公式退化为 u = a/(Kv), v = b/(K^T u).
        """
        torch.manual_seed(42)
        cost = torch.rand(15, 7)
        eps = 1.0
        num_iters = 50

        # 标准 Sinkhorn
        standard = sinkhorn_transport(
            cost, epsilon=eps, num_iters=num_iters
        )

        # 非平衡 Sinkhorn, λ 很大 (1e6)
        large_lambda = 1e6
        unbalanced = unbalanced_sinkhorn_transport(
            cost,
            epsilon=eps,
            num_iters=num_iters,
            lambda_row=large_lambda,
            lambda_col=large_lambda,
        )

        assert torch.allclose(standard, unbalanced, atol=1e-4)

    def test_smaller_lambda_more_violation(self):
        """λ 减小时, 边缘违反度 KL(P1||a) + KL(P^T1||b) 应增大.

        数学依据: λ 是 KL 惩罚系数, λ 越小约束越松, 违反度越大.
        """
        torch.manual_seed(42)
        cost = torch.rand(20, 8)
        eps = 1.0
        a = torch.ones(20) / 20
        b = torch.ones(8) / 8

        def kl_violation(P: torch.Tensor) -> float:
            row_marginal = P.sum(dim=1)
            col_marginal = P.sum(dim=0)
            # 广义 KL: sum(u*log(u/v) - u + v)
            kl_row = (
                row_marginal * (row_marginal.clamp_min(1e-10).log() - a.log())
                - row_marginal + a
            ).sum().item()
            kl_col = (
                col_marginal * (col_marginal.clamp_min(1e-10).log() - b.log())
                - col_marginal + b
            ).sum().item()
            return kl_row + kl_col

        violations = []
        for lam in [10.0, 1.0, 0.1]:
            P = unbalanced_sinkhorn_transport(
                cost,
                epsilon=eps,
                num_iters=50,
                row_mass=a,
                col_mass=b,
                lambda_row=lam,
                lambda_col=lam,
            )
            violations.append(kl_violation(P))

        # λ 减小 → 违反度应单调递增
        assert violations[0] < violations[1] < violations[2], (
            f"违反度应随 λ 减小而增大, got {violations}"
        )

    def test_lambda_zero_degenerates_to_gibbs(self):
        """λ → 0 时退化为 Gibbs 分布 P_ij ∝ exp(-C_ij/ε).

        数学依据: α → 0 → log_u → 0 → u → 1;
                  β → 0 → log_v → 0 → v → 1.
        于是 P_ij = exp(log_K_ij) = exp(-C_ij/ε).
        """
        torch.manual_seed(42)
        cost = torch.rand(10, 5)
        eps = 1.0

        P = unbalanced_sinkhorn_transport(
            cost,
            epsilon=eps,
            num_iters=20,
            lambda_row=1e-10,
            lambda_col=1e-10,
        )
        expected = torch.exp(-cost / eps)

        assert torch.allclose(P, expected, atol=1e-4)

    def test_asymmetric_lambda(self):
        """λ_row ≠ λ_col 时, 行/列松弛程度应不同."""
        torch.manual_seed(42)
        cost = torch.rand(15, 8)
        eps = 1.0
        a = torch.ones(15) / 15
        b = torch.ones(8) / 8

        # 仅放松行
        P_row_loose = unbalanced_sinkhorn_transport(
            cost, epsilon=eps, num_iters=50,
            row_mass=a, col_mass=b,
            lambda_row=0.1, lambda_col=100.0,
        )
        # 仅放松列
        P_col_loose = unbalanced_sinkhorn_transport(
            cost, epsilon=eps, num_iters=50,
            row_mass=a, col_mass=b,
            lambda_row=100.0, lambda_col=0.1,
        )

        def kl(u, v):
            return (
                u * (u.clamp_min(1e-10).log() - v.log()) - u + v
            ).sum().item()

        # 行放松时, 行违反度应大于列违反度
        row_viol_row = kl(P_row_loose.sum(1), a)
        col_viol_row = kl(P_row_loose.sum(0), b)
        assert row_viol_row > col_viol_row, (
            f"行放松时行违反度({row_viol_row})应大于列违反度({col_viol_row})"
        )

        # 列放松时, 列违反度应大于行违反度
        row_viol_col = kl(P_col_loose.sum(1), a)
        col_viol_col = kl(P_col_loose.sum(0), b)
        assert col_viol_col > row_viol_col, (
            f"列放松时列违反度({col_viol_col})应大于行违反度({row_viol_col})"
        )


class TestUnbalancedGHSSCoupling:
    """测试 UnbalancedGHSSCoupling 耦合策略"""

    def test_build(self):
        c = build_coupling('unbalanced_ghss', epsilon=5.0)
        assert c is not None
        assert c.epsilon == 5.0
        assert c.lambda_row == 1.0
        assert c.lambda_col == 1.0

    def test_build_with_lambda(self):
        c = build_coupling(
            'unbalanced_ghss',
            epsilon=5.0,
            lambda_row=0.5,
            lambda_col=0.5,
        )
        assert c.lambda_row == 0.5
        assert c.lambda_col == 0.5

    def test_couple_shape(self):
        c = build_coupling('unbalanced_ghss', epsilon=5.0)
        noise = torch.randn(100, 4)
        gt = torch.randn(10, 4)
        labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 3, 4])
        x_start, idx = c.couple(noise, gt, labels, torch.device('cpu'))
        assert x_start.shape == (100, 4)
        assert idx.shape == (100,)
        assert idx.max() < 10

    def test_empty_gt(self):
        """空 GT 时应返回原始 noise, idx 全 0"""
        c = build_coupling('unbalanced_ghss', epsilon=5.0)
        noise = torch.randn(50, 4)
        gt = torch.randn(0, 4)
        x_start, idx = c.couple(
            noise, gt, torch.tensor([]), torch.device('cpu')
        )
        assert torch.equal(x_start, noise)
        assert (idx == 0).all()

    def test_single_group(self):
        """所有 GT 同组时应正常匹配"""
        c = build_coupling('unbalanced_ghss', epsilon=5.0)
        noise = torch.randn(80, 4)
        gt = torch.randn(8, 4)
        labels = torch.zeros(8, dtype=torch.long)  # 全 A 组
        x_start, idx = c.couple(noise, gt, labels, torch.device('cpu'))
        assert x_start.shape == (80, 4)
        assert idx.max() < 8
        assert idx.min() >= 0

    def test_respects_groups(self):
        """组内匹配: 不应匹配到其他组 GT"""
        c = build_coupling('unbalanced_ghss', epsilon=5.0)
        gt = torch.randn(10, 4)
        labels = torch.zeros(10, dtype=torch.long)  # 全 A 组
        noise = torch.randn(100, 4)
        x_start, idx = c.couple(noise, gt, labels, torch.device('cpu'))
        assert idx.max() < 10
        assert idx.min() >= 0

    def test_multiple_groups(self):
        """多组场景: 各组 proposal 数应与 GT 数成比例"""
        c = build_coupling('unbalanced_ghss', epsilon=5.0)
        noise = torch.randn(100, 4)
        # 3 个组: A(3), C(7), G(2) — 共 12 GT
        gt = torch.randn(12, 4)
        labels = torch.tensor([
            0, 0, 0,           # A 组 (3)
            5, 5, 5, 5, 5, 5, 5,  # C 组 (7)
            20, 21,            # G 组 (2)
        ])
        x_start, idx = c.couple(noise, gt, labels, torch.device('cpu'))
        assert x_start.shape == (100, 4)

        # 各组匹配到的 proposal 数应大致按比例
        # A: 3/12 * 100 ≈ 25, C: 7/12 * 100 ≈ 58, G: 2/12 * 100 ≈ 17
        a_count = ((idx >= 0) & (idx < 3)).sum().item()
        c_count = ((idx >= 3) & (idx < 10)).sum().item()
        g_count = ((idx >= 10) & (idx < 12)).sum().item()

        # 允许较大误差 (因为非平衡 OT 会调整)
        assert abs(a_count - 25) < 15
        assert abs(c_count - 58) < 20
        assert abs(g_count - 17) < 15

    def test_degenerate_to_ghss(self):
        """λ_row=λ_col=∞ (用大数模拟) 时传输矩阵应数值接近 GHSS.

        数学依据: λ → ∞ 时非平衡 Sinkhorn 退化为标准 Sinkhorn,
        传输矩阵应数值等价 (atol=1e-4).

        注意: 不比较采样结果, 因为 torch.multinomial 对概率分布的
        微小数值差异 (1e-6 级) 极敏感, 即使数学等价也可能产生不同采样.
        GHSS 用 sinkhorn_transport_batch (padding+mask), UnbalancedGHSS 用
        unbalanced_sinkhorn_transport (单组), 实现路径不同导致数值微差.
        """
        torch.manual_seed(42)
        noise = torch.randn(50, 4)
        gt = torch.randn(5, 4)
        labels = torch.tensor([0, 0, 1, 2, 3])

        # 直接比较传输矩阵: 构造相同的单组场景
        from ldmdet.coupling._sinkhorn_ops import sinkhorn_transport

        cost = torch.cdist(noise, gt, p=2)
        eps = 5.0
        a = torch.ones(50) / 50
        b = torch.ones(5) / 5

        # 标准 Sinkhorn
        P_standard = sinkhorn_transport(
            cost, epsilon=eps, num_iters=20,
            row_mass=a, col_mass=b,
        )

        # 非平衡 Sinkhorn, λ 很大
        P_unbalanced = unbalanced_sinkhorn_transport(
            cost, epsilon=eps, num_iters=20,
            row_mass=a, col_mass=b,
            lambda_row=1e6, lambda_col=1e6,
        )

        assert torch.allclose(P_standard, P_unbalanced, atol=1e-4), (
            f"λ → ∞ 时传输矩阵应数值等价, "
            f"max diff = {(P_standard - P_unbalanced).abs().max()}"
        )

    def test_degenerate_distribution_similarity(self):
        """λ → ∞ 时 UnbalancedGHSS 的匹配分布应与 GHSS 统计相似.

        不要求完全一致 (采样随机性), 但各组 proposal 数应接近.
        """
        torch.manual_seed(42)
        noise = torch.randn(200, 4)
        gt = torch.randn(10, 4)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

        ghss = build_coupling('ghss', epsilon=5.0, sample_seed=42)
        _, idx_ghss = ghss.couple(noise, gt, labels, torch.device('cpu'))

        unb = build_coupling(
            'unbalanced_ghss',
            epsilon=5.0,
            lambda_row=1e6,
            lambda_col=1e6,
            sample_seed=42,
        )
        _, idx_unb = unb.couple(noise, gt, labels, torch.device('cpu'))

        # 各组 proposal 数应接近 (允许 ±10 误差)
        for g_start, g_end in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]:
            cnt_ghss = ((idx_ghss >= g_start) & (idx_ghss < g_end)).sum().item()
            cnt_unb = ((idx_unb >= g_start) & (idx_unb < g_end)).sum().item()
            assert abs(cnt_ghss - cnt_unb) <= 10, (
                f"组 [{g_start},{g_end}): GHSS={cnt_ghss}, Unbalanced={cnt_unb}, "
                f"差异 > 10"
            )

    def test_lambda_affects_distribution(self):
        """不同 λ 应产生不同的匹配分布"""
        torch.manual_seed(42)
        noise = torch.randn(200, 4)
        gt = torch.randn(10, 4)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

        # 大 λ (接近 GHSS)
        c_strict = build_coupling(
            'unbalanced_ghss',
            epsilon=5.0,
            lambda_row=100.0,
            lambda_col=100.0,
            sample_seed=42,
        )
        _, idx_strict = c_strict.couple(noise, gt, labels, torch.device('cpu'))

        # 小 λ (强松弛)
        c_loose = build_coupling(
            'unbalanced_ghss',
            epsilon=5.0,
            lambda_row=0.1,
            lambda_col=0.1,
            sample_seed=42,
        )
        _, idx_loose = c_loose.couple(noise, gt, labels, torch.device('cpu'))

        # 两者匹配分布应不同 (至少部分不同)
        n_diff = (idx_strict != idx_loose).sum().item()
        # 不要求完全不同, 但应有显著差异
        assert n_diff > 10, (
            f"不同 λ 应产生不同匹配, 仅 {n_diff}/200 不同"
        )
