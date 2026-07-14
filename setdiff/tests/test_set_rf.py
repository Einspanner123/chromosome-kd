"""测试 setdiff.diffusion.set_rf — SetRectifiedFlow"""

import torch
import pytest
from setdiff.diffusion.set_rf import SetRectifiedFlow


class TestSetRectifiedFlow:
    @pytest.fixture
    def rf(self):
        return SetRectifiedFlow(snr_scale=2.0)

    def test_q_sample_shapes(self, rf):
        """q_sample 输出形状正确"""
        B, N = 2, 10
        x_start = torch.randn(B, N, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.rand(B)
        x_t, velocity = rf.q_sample(x_start, x_noise, t)
        assert x_t.shape == x_start.shape
        assert velocity.shape == x_start.shape

    def test_q_sample_t0(self, rf):
        """t=0 时 x_t == x_start"""
        x_start = torch.randn(2, 10, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.zeros(2)
        x_t, _ = rf.q_sample(x_start, x_noise, t)
        assert torch.allclose(x_t, x_start)

    def test_q_sample_t1(self, rf):
        """t=1 时 x_t == x_noise"""
        x_start = torch.randn(2, 10, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.ones(2)
        x_t, _ = rf.q_sample(x_start, x_noise, t)
        assert torch.allclose(x_t, x_noise)

    def test_velocity_formula(self, rf):
        """velocity = x_noise - x_start"""
        x_start = torch.randn(2, 10, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.rand(2)
        _, velocity = rf.q_sample(x_start, x_noise, t)
        expected = x_noise - x_start
        assert torch.allclose(velocity, expected, atol=1e-5)

    def test_q_sample_interpolation(self, rf):
        """x_t = (1-t)*x_start + t*x_noise (中间 t)"""
        x_start = torch.randn(2, 10, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.tensor([0.5, 0.25])
        x_t, _ = rf.q_sample(x_start, x_noise, t)
        t_view = t.view(-1, 1, 1)
        expected = (1.0 - t_view) * x_start + t_view * x_noise
        assert torch.allclose(x_t, expected, atol=1e-5)

    def test_step_shape(self, rf):
        """Euler step 输出形状正确"""
        B, N = 2, 10
        x_t = torch.randn(B, N, 4)
        x_0_pred = torch.randn_like(x_t)
        x_next = rf.step(x_t, x_0_pred, t_curr=0.8, t_next=0.6)
        assert x_next.shape == x_t.shape

    def test_step_finite(self, rf):
        """Euler step 输出应有限"""
        x_t = torch.randn(2, 10, 4)
        x_0_pred = torch.randn_like(x_t)
        x_next = rf.step(x_t, x_0_pred, t_curr=0.8, t_next=0.4)
        assert torch.isfinite(x_next).all()

    def test_step_advances_state(self, rf):
        """Euler step 应使状态向 x_0_pred 方向移动"""
        x_t = torch.randn(2, 10, 4)
        x_0_pred = torch.zeros_like(x_t)  # 目标是 0
        # t_curr=1.0, t_next=0.0 → 一步到 x_0_pred
        x_next = rf.step(x_t, x_0_pred, t_curr=1.0, t_next=0.0)
        # t_next=0 时应等于 x_0_pred
        assert torch.allclose(x_next, x_0_pred, atol=1e-5)

    def test_heun_step_shape(self, rf):
        """Heun step 输出形状正确"""
        B, N = 2, 10
        x_t = torch.randn(B, N, 4)
        x_0_pred = torch.randn_like(x_t)

        def model_fn(x, t):
            return torch.randn_like(x), None

        x_next = rf.heun_step(
            x_t, x_0_pred, t_curr=0.8, t_next=0.6, model_fn=model_fn
        )
        assert x_next.shape == x_t.shape

    def test_joint_state_consistency(self, rf):
        """q_sample 在 [B, N, 4] 上等价于在 [B, N*4] 上"""
        B, N = 2, 10
        x_start = torch.randn(B, N, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.rand(B)

        # Set-level (joint)
        x_t_set, v_set = rf.q_sample(x_start, x_noise, t)

        # 底层 RF 直接在扁平化上 (数值应一致)
        from ldmdet.diffusion.rectified_flow import RectifiedFlow
        rf_base = RectifiedFlow(snr_scale=2.0)
        x_t_flat, v_flat = rf_base.q_sample(
            x_start.reshape(B, N * 4), x_noise.reshape(B, N * 4), t
        )

        assert torch.allclose(x_t_set, x_t_flat.reshape(B, N, 4))
        assert torch.allclose(v_set, v_flat.reshape(B, N, 4))

    def test_different_snr_scale(self):
        """不同 snr_scale 不影响 q_sample 数值 (snr_scale 仅用于数据归一化)"""
        rf_a = SetRectifiedFlow(snr_scale=1.0)
        rf_b = SetRectifiedFlow(snr_scale=4.0)
        x_start = torch.randn(2, 10, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.rand(2)
        x_t_a, v_a = rf_a.q_sample(x_start, x_noise, t)
        x_t_b, v_b = rf_b.q_sample(x_start, x_noise, t)
        # snr_scale 不参与 RF 的 q_sample 公式 (仅影响数据预处理)
        assert torch.allclose(x_t_a, x_t_b)
        assert torch.allclose(v_a, v_b)
