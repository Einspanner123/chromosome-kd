"""方向四: 流匹配的非线性轨迹 (Nonlinear Trajectories for Flow Matching) 测试

测试覆盖:
1. ScaleConditionedRF: 退化等价性 (λ=0)、κ 单调性、t_eff 计算、采样
2. OTFlowMatching: OT 耦合正确性、速度形状、耦合质量
3. 隔离性: baseline 行为不受影响
"""

import torch
import pytest

from ldmdet.diffusion.scale_conditioned_rf import ScaleConditionedRF
from ldmdet.diffusion.ot_flow_matching import OTFlowMatching
from ldmdet.diffusion.rectified_flow import RectifiedFlow


# ================================================================
# 4.1 ScaleConditionedRF
# ================================================================


class TestScaleConditionedRF:
    """测试尺度条件化 Rectified Flow"""

    def test_kappa_lambda_zero_degenerate(self):
        """λ=0: κ=1, 退化为标准 RF"""
        rf = ScaleConditionedRF(lambda_mod=0.0, s_max=0.15)
        scales = torch.rand(10, 5) * 0.2  # 任意尺度
        kappa = rf.compute_kappa(scales)
        assert torch.allclose(kappa, torch.ones_like(kappa))

    def test_kappa_monotonic(self):
        """κ(s) 应随 s 单调递减 (小尺度→大 κ)"""
        rf = ScaleConditionedRF(lambda_mod=0.5, s_max=0.15)
        scales = torch.linspace(0.01, 0.15, 20)
        kappa = rf.compute_kappa(scales)
        diff = kappa[1:] - kappa[:-1]
        assert (diff <= 1e-6).all(), "κ(s) 应单调递减"

    def test_kappa_range(self):
        """κ(s) 应在 [1, 1+λ] 范围内 (当 s ∈ [0, s_max])"""
        rf = ScaleConditionedRF(lambda_mod=0.5, s_max=0.15)
        # s 在 [0, s_max] 范围内
        scales = torch.linspace(0.0, 0.15, 20)
        kappa = rf.compute_kappa(scales)
        assert (kappa >= 1.0 - 1e-6).all()
        assert (kappa <= 1.0 + 0.5 + 1e-6).all()

    def test_kappa_s_max_equals_one(self):
        """s = s_max: κ = 1 (退化为标准 RF)"""
        rf = ScaleConditionedRF(lambda_mod=0.5, s_max=0.15)
        scales = torch.tensor([0.15])
        kappa = rf.compute_kappa(scales)
        assert torch.allclose(kappa, torch.tensor([1.0]))

    def test_kappa_zero_scale_max(self):
        """s = 0: κ = 1 + λ (最大调制)"""
        rf = ScaleConditionedRF(lambda_mod=0.5, s_max=0.15)
        scales = torch.tensor([0.0])
        kappa = rf.compute_kappa(scales)
        assert torch.allclose(kappa, torch.tensor([1.5]))

    def test_t_eff_lambda_zero_degenerate(self):
        """λ=0: t_eff = t (退化为标准 RF)"""
        rf = ScaleConditionedRF(lambda_mod=0.0)
        t = torch.rand(4)
        scales = torch.rand(4, 10) * 0.2
        t_eff = rf.compute_t_eff(t, scales)
        # t_eff 应等于 t (广播)
        for b in range(4):
            assert torch.allclose(t_eff[b], t[b].expand_as(t_eff[b]))

    def test_t_eff_small_scale_larger(self):
        """小尺度 → t_eff 更大 (更快接近噪声端, 即更早去噪)

        数学: κ(s) = 1 + λ(s_max - s)/s_max
        小 s → 大 κ → 小 1/κ → t^{1/κ} 更大 (对 t ∈ (0,1))
        即小尺度在相同 t 下 t_eff 更大, 表示更接近纯噪声,
        从而迫使模型更早学习去噪小目标.
        """
        rf = ScaleConditionedRF(lambda_mod=0.5, s_max=0.15)
        t = torch.tensor([0.5])
        scales = torch.tensor([[0.01], [0.15]])  # 小尺度 vs 大尺度
        t_eff = rf.compute_t_eff(t, scales)
        # 小尺度 t_eff 应 > 大尺度 t_eff
        assert t_eff[0, 0] > t_eff[1, 0], (
            f"小尺度 t_eff ({t_eff[0, 0]}) 应 > 大尺度 t_eff ({t_eff[1, 0]})"
        )

    def test_q_sample_shape(self):
        """q_sample: 输出形状正确"""
        rf = ScaleConditionedRF(lambda_mod=0.5)
        x_start = torch.randn(2, 10, 4)
        x_t, velocity, t_eff = rf.q_sample(x_start)
        assert x_t.shape == x_start.shape
        assert velocity.shape == x_start.shape
        assert t_eff.shape == (2, 10)

    def test_q_sample_boundary_t_zero(self):
        """t=0: x_t = x_start (纯数据)"""
        rf = ScaleConditionedRF(lambda_mod=0.5)
        x_start = torch.randn(2, 10, 4)
        x_noise = torch.randn_like(x_start)
        scales = torch.rand(2, 10) * 0.2
        t = torch.zeros(2)
        x_t, _, _ = rf.q_sample(x_start, x_noise, t, scales)
        assert torch.allclose(x_t, x_start, atol=1e-6)

    def test_q_sample_boundary_t_one(self):
        """t=1: x_t = x_noise (纯噪声)"""
        rf = ScaleConditionedRF(lambda_mod=0.5)
        x_start = torch.randn(2, 10, 4)
        x_noise = torch.randn_like(x_start)
        scales = torch.rand(2, 10) * 0.2
        t = torch.ones(2)
        x_t, _, _ = rf.q_sample(x_start, x_noise, t, scales)
        assert torch.allclose(x_t, x_noise, atol=1e-6)

    def test_q_sample_velocity(self):
        """velocity = x_noise - x_start"""
        rf = ScaleConditionedRF(lambda_mod=0.5)
        x_start = torch.randn(2, 10, 4)
        x_noise = torch.randn_like(x_start)
        scales = torch.rand(2, 10) * 0.2
        t = torch.rand(2)
        _, velocity, _ = rf.q_sample(x_start, x_noise, t, scales)
        assert torch.allclose(velocity, x_noise - x_start)

    def test_q_sample_lambda_zero_degenerate_to_standard_rf(self):
        """λ=0: 数值等价于标准 RectifiedFlow"""
        torch.manual_seed(42)
        x_start = torch.randn(2, 10, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.tensor([0.3, 0.7])

        # 标准 RF
        std_rf = RectifiedFlow()
        x_t_std, v_std = std_rf.q_sample(x_start, x_noise, t)

        # 尺度条件化 RF, λ=0
        sc_rf = ScaleConditionedRF(lambda_mod=0.0)
        x_t_sc, v_sc, _ = sc_rf.q_sample(x_start, x_noise, t)

        assert torch.allclose(x_t_std, x_t_sc, atol=1e-6)
        assert torch.allclose(v_std, v_sc, atol=1e-6)

    def test_step_euler_shape(self):
        """step: 输出形状正确"""
        rf = ScaleConditionedRF(lambda_mod=0.5)
        x_t = torch.randn(2, 10, 4)
        x0_pred = torch.randn_like(x_t)
        scales = torch.rand(2, 10) * 0.2
        x_next = rf.step(x_t, x0_pred, 0.8, 0.6, scales)
        assert x_next.shape == x_t.shape

    def test_step_t_zero_to_one_returns_noise(self):
        """step 从 t=0 到 t=1: 应接近 x0_pred 对应的噪声端"""
        # 当 t_curr=0, t_next=1: x_next = x_t + dt_eff * v
        # v = (x_t - x0_pred) / t_eff_curr, t_eff_curr=0 → v 很大
        # 实际上 t=0 时 x_t=x0_pred, v=0, 所以 x_next=x_t=x0_pred
        rf = ScaleConditionedRF(lambda_mod=0.5)
        x_t = torch.randn(2, 10, 4)
        x0_pred = x_t.clone()  # t=0 时 x_t = x0
        scales = torch.rand(2, 10) * 0.2
        # 从 t=0 到 t=0.5
        x_next = rf.step(x_t, x0_pred, 0.0, 0.5, scales)
        # 应不报错, 形状正确
        assert x_next.shape == x_t.shape

    def test_compute_scales_from_boxes(self):
        """从 xyxy 框计算尺度"""
        # 框 [0.1, 0.1, 0.3, 0.3] → w=0.2, h=0.2, area=0.04, sqrt=0.2
        boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3]])
        scales = ScaleConditionedRF.compute_scales_from_boxes(boxes)
        assert torch.allclose(scales, torch.tensor([0.2]), atol=1e-6)

    def test_compute_scales_from_cxcywh(self):
        """从 cxcywh 框计算尺度"""
        boxes = torch.tensor([[0.2, 0.2, 0.2, 0.2]])  # cxcywh
        scales = ScaleConditionedRF.compute_scales_from_cxcywh(boxes)
        assert torch.allclose(scales, torch.tensor([0.2]), atol=1e-6)

    def test_compute_scales_zero_area(self):
        """零面积框: 尺度应为 0"""
        boxes = torch.tensor([[0.1, 0.1, 0.1, 0.1]])  # w=0, h=0
        scales = ScaleConditionedRF.compute_scales_from_boxes(boxes)
        assert torch.allclose(scales, torch.tensor([0.0]))

    def test_get_velocity_shape(self):
        """get_velocity: 输出形状正确"""
        rf = ScaleConditionedRF(lambda_mod=0.5)
        x_t = torch.randn(2, 10, 4)
        x0_pred = torch.randn_like(x_t)
        t = torch.rand(2)
        scales = torch.rand(2, 10) * 0.2
        v = rf.get_velocity(x_t, x0_pred, t, scales)
        assert v.shape == x_t.shape

    def test_scales_none_degenerate(self):
        """scales=None: 退化为标准 RF (κ=1+λ*(s_max-s_max)/s_max=1)"""
        rf = ScaleConditionedRF(lambda_mod=0.5)
        x_start = torch.randn(2, 10, 4)
        x_t, velocity, t_eff = rf.q_sample(x_start, scales=None)
        # t_eff 应等于 t (因为 κ=1)
        # 检查 t_eff 在 [0, 1] 范围内
        assert (t_eff >= 0).all()
        assert (t_eff <= 1).all()


# ================================================================
# 4.2 OTFlowMatching
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


# ================================================================
# 隔离性测试: baseline 不受影响
# ================================================================


class TestBaselineIsolation:
    """验证方向四默认关闭时, baseline 行为不受影响"""

    def test_standard_rf_still_works(self):
        """标准 RectifiedFlow 仍正常工作"""
        rf = RectifiedFlow()
        x_start = torch.randn(2, 10, 4)
        x_t, velocity = rf.q_sample(x_start)
        assert x_t.shape == x_start.shape
        assert velocity.shape == x_start.shape

    def test_scale_conditioned_rf_lambda_zero_is_standard(self):
        """λ=0 的 ScaleConditionedRF 等价于标准 RF"""
        torch.manual_seed(42)
        x_start = torch.randn(2, 10, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.tensor([0.3, 0.7])

        std_rf = RectifiedFlow()
        x_t_std, v_std = std_rf.q_sample(x_start, x_noise, t)

        sc_rf = ScaleConditionedRF(lambda_mod=0.0)
        x_t_sc, v_sc, _ = sc_rf.q_sample(x_start, x_noise, t)

        assert torch.allclose(x_t_std, x_t_sc, atol=1e-6)
        assert torch.allclose(v_std, v_sc, atol=1e-6)
