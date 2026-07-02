"""方向四: 流匹配的非线性轨迹 (Nonlinear Trajectories for Flow Matching) 测试

测试覆盖:
1. OTFlowMatching: OT 耦合正确性、速度形状、耦合质量

注意: ScaleConditionedRF 已证伪 (0.741 < 0.746 baseline) 并移除,
      相关测试类已删除. 详见 docs/EXPERIMENT_LINEAGE.md 第十一节.
"""

import torch
import pytest

from ldmdet.diffusion.ot_flow_matching import OTFlowMatching


# ================================================================
# OTFlowMatching
# ================================================================


class TestOTFlowMatching:
    """测试 Mini-batch OT 流匹配"""

    def test_init_defaults(self):
        """默认参数初始化"""
        otfm = OTFlowMatching()
        assert otfm.epsilon == 1.0
        assert otfm.num_iters == 10
        assert otfm.ot_every_step is False
        assert otfm.coupling_mode == 'argmax'

    def test_init_invalid_coupling_mode(self):
        """无效 coupling_mode 应报错"""
        with pytest.raises(ValueError):
            OTFlowMatching(coupling_mode='invalid')

    def test_compute_ot_coupling_shape(self):
        """OT 耦合: 输出形状正确"""
        torch.manual_seed(42)
        otfm = OTFlowMatching(epsilon=1.0)
        x_start = torch.randn(20, 4)
        x_noise = torch.randn(20, 4)
        s_c, n_c = otfm.compute_ot_coupling(x_start, x_noise)
        assert s_c.shape == x_start.shape
        assert n_c.shape == x_noise.shape

    def test_compute_ot_coupling_x_start_unchanged(self):
        """OT 耦合: x_start 不重排 (只重排 x_noise)"""
        torch.manual_seed(42)
        otfm = OTFlowMatching(epsilon=1.0)
        x_start = torch.randn(20, 4)
        x_noise = torch.randn(20, 4)
        s_c, _ = otfm.compute_ot_coupling(x_start, x_noise)
        assert torch.equal(s_c, x_start)

    def test_compute_ot_coupling_single_sample(self):
        """单样本: 应原样返回"""
        otfm = OTFlowMatching()
        x_start = torch.randn(1, 4)
        x_noise = torch.randn(1, 4)
        s_c, n_c = otfm.compute_ot_coupling(x_start, x_noise)
        assert torch.equal(s_c, x_start)
        assert torch.equal(n_c, x_noise)

    def test_compute_ot_coupling_reduces_cost(self):
        """OT 耦合后, 配对代价应 <= 随机配对代价"""
        torch.manual_seed(42)
        N = 50
        x_start = torch.randn(N, 4)
        x_noise = torch.randn(N, 4)

        otfm = OTFlowMatching(epsilon=0.1, num_iters=50)
        _, x_noise_coupled = otfm.compute_ot_coupling(x_start, x_noise)

        # OT 配对代价
        ot_cost = (x_start - x_noise_coupled).norm(2, dim=1).mean().item()
        # 随机配对代价
        rand_perm = torch.randperm(N)
        rand_cost = (x_start - x_noise[rand_perm]).norm(2, dim=1).mean().item()

        assert ot_cost < rand_cost, (
            f"OT 配对代价 ({ot_cost:.4f}) 应小于随机配对 ({rand_cost:.4f})"
        )

    def test_q_sample_shape(self):
        """q_sample: 输出形状正确"""
        torch.manual_seed(42)
        otfm = OTFlowMatching(epsilon=1.0)
        x_start = torch.randn(2, 20, 4)
        x_t, velocity = otfm.q_sample(x_start)
        assert x_t.shape == x_start.shape
        assert velocity.shape == x_start.shape

    def test_q_sample_boundary_t_zero(self):
        """t=0: x_t = x_start (OT 耦合后)"""
        torch.manual_seed(42)
        otfm = OTFlowMatching(epsilon=1.0)
        x_start = torch.randn(2, 20, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.zeros(2)
        x_t, _ = otfm.q_sample(x_start, x_noise, t)
        # t=0 时 x_t = x_start (耦合后 x_start 不重排)
        assert torch.allclose(x_t, x_start, atol=1e-6)

    def test_q_sample_boundary_t_one(self):
        """t=1: x_t = x_noise (OT 耦合后, x_noise 被重排)

        注意: argmax 耦合可能将多行映射到同一列 (非严格排列),
        所以只验证 x_t 来自 x_noise 的某个子集 (值域一致).
        """
        torch.manual_seed(42)
        otfm = OTFlowMatching(epsilon=1.0)
        x_start = torch.randn(2, 20, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.ones(2)
        x_t, _ = otfm.q_sample(x_start, x_noise, t)
        # t=1 时 x_t = x_noise_coupled, 是 x_noise 的重排 (可能有重复)
        # 验证 x_t 的每个元素都来自 x_noise 的某一行
        for b in range(2):
            # x_t[b] 的每一行应能在 x_noise[b] 中找到匹配
            for i in range(x_t.shape[1]):
                # 计算与所有 x_noise 行的距离
                dist = (x_noise[b] - x_t[b, i]).norm(2, dim=1)
                assert dist.min() < 1e-6, (
                    f"x_t[{b}, {i}] 不在 x_noise[{b}] 中"
                )

    def test_q_sample_velocity(self):
        """velocity = x_noise_coupled - x_start_coupled"""
        torch.manual_seed(42)
        otfm = OTFlowMatching(epsilon=1.0)
        x_start = torch.randn(2, 20, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.rand(2)
        x_t, velocity = otfm.q_sample(x_start, x_noise, t)

        # 重新计算耦合
        s_c, n_c = otfm.compute_ot_coupling(x_start[0], x_noise[0])
        expected_v = n_c - s_c
        assert torch.allclose(velocity[0], expected_v, atol=1e-6)

    def test_coupling_mode_multinomial(self):
        """multinomial 模式: 应正常工作"""
        torch.manual_seed(42)
        otfm = OTFlowMatching(epsilon=1.0, coupling_mode='multinomial')
        x_start = torch.randn(20, 4)
        x_noise = torch.randn(20, 4)
        s_c, n_c = otfm.compute_ot_coupling(x_start, x_noise)
        assert s_c.shape == x_start.shape
        assert n_c.shape == x_noise.shape

    def test_compute_coupling_quality(self):
        """compute_coupling_quality: 应返回统计字典"""
        torch.manual_seed(42)
        otfm = OTFlowMatching(epsilon=1.0)
        x_start = torch.randn(20, 4)
        x_noise = torch.randn(20, 4)
        quality = otfm.compute_coupling_quality(x_start, x_noise)
        assert 'mean_cost' in quality
        assert 'max_cost' in quality
        assert 'transport_marginal_std' in quality
        assert quality['mean_cost'] > 0

    def test_compute_coupling_cost(self):
        """compute_coupling_cost: 应返回代价矩阵"""
        torch.manual_seed(42)
        otfm = OTFlowMatching(epsilon=1.0)
        x_start = torch.randn(20, 4)
        x_noise = torch.randn(20, 4)
        cost = otfm.compute_coupling_cost(x_start, x_noise)
        assert cost.shape == (20, 20)
        # 代价矩阵应非负
        assert (cost >= 0).all()

    def test_ot_every_step_flag(self):
        """ot_every_step: 标志应正确设置"""
        otfm = OTFlowMatching(ot_every_step=True)
        assert otfm.ot_every_step is True

    def test_epsilon_effect(self):
        """ε 越小, OT 耦合越确定 (传输矩阵越稀疏)

        用传输矩阵的稀疏度 (max 比例) 验证, 而非配对代价,
        因为 argmax 耦合在小 ε 时可能产生重复映射, 配对代价不一定单调.
        """
        torch.manual_seed(42)
        N = 30
        x_start = torch.randn(N, 4)
        x_noise = torch.randn(N, 4)

        from ldmdet.coupling._sinkhorn_ops import sinkhorn_transport

        # 小 ε: 传输矩阵更稀疏 (max 更大)
        transport_small = sinkhorn_transport(
            torch.cdist(x_start, x_noise, p=2),
            epsilon=0.1, num_iters=50,
        )
        sparsity_small = transport_small.max(dim=1).values.mean().item()

        # 大 ε: 传输矩阵更均匀 (max 更小)
        transport_large = sinkhorn_transport(
            torch.cdist(x_start, x_noise, p=2),
            epsilon=5.0, num_iters=50,
        )
        sparsity_large = transport_large.max(dim=1).values.mean().item()

        # 小 ε 的稀疏度应 > 大 ε (更集中)
        assert sparsity_small > sparsity_large, (
            f"小 ε 稀疏度 ({sparsity_small:.4f}) 应 > 大 ε ({sparsity_large:.4f})"
        )
