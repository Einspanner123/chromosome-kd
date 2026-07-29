"""DiffusionScheduler 单元测试.

对照原仓库 (MBadran2000/DiffuDETR) dino_diffu_det_noise.py 的 register_schedule_old
+ q_sample + DDIM 逻辑, 验证:
  - cosine_beta_schedule 公式与单调性 (对齐原仓库 line 82-92)
  - alphas_cumprod 单调递减, alphas_cumprod[0] ≈ 1
  - loss_weight (lvlb_weights) 公式: 0.5*sqrt(acp)/(2-acp), lvlb[0]=lvlb[1] (line 278-279)
  - loss_weight 值域与趋势: 高 timestep (低 acp) → 高权重 (SNR 加权语义)
  - q_sample: x_t = sqrt(acp)*x0 + sqrt(1-acp)*noise (line 351-358)
    - t=0 → x_t ≈ x_start
    - t=T-1 → x_t ≈ noise (噪声主导)
  - predict_noise_from_start: 与 q_sample 的 round-trip (line 362-367)
  - extract: 形状广播正确
  - timestep_embedding: [B, dim], 不同时间步 → 不同嵌入
  - ddim_step: eta=0 确定性, t_next<0 返回 x0
  - get_time_pairs: 降序, 数量 = sampling_timesteps
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import math

import torch

from projects.diffudetr.models.diffusion_scheduler import (
    DiffusionScheduler,
    cosine_beta_schedule,
    extract,
    make_ddim_timesteps,
    timestep_embedding,
)

# ========== cosine_beta_schedule ==========


class TestCosineBetaSchedule:
    """验证 cosine beta schedule (对齐原仓库 line 82-92)."""

    def test_betas_in_valid_range(self):
        """betas ∈ [0, 0.999] (clip)."""
        betas = cosine_beta_schedule(1000)
        assert betas.shape == (1000,)
        assert (betas >= 0).all()
        assert (betas <= 0.999).all()

    def test_betas_monotonically_increasing(self):
        """cosine schedule 的 betas 单调非递减."""
        betas = cosine_beta_schedule(1000)
        diffs = betas[1:] - betas[:-1]
        # 允许极小数值波动, 整体应非递减
        assert (diffs >= -1e-6).all(), (
            f'betas not non-decreasing, min diff={diffs.min()}'
        )

    def test_alphas_cumprod_decreasing_to_zero(self):
        """alphas_cumprod 从 ~1 单调递减到 ~0."""
        betas = cosine_beta_schedule(1000)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        assert abs(alphas_cumprod[0].item() - 1.0) < 1e-4, (
            'alphas_cumprod[0] should ≈ 1'
        )
        # 末端应接近 0 (完全噪声)
        assert alphas_cumprod[-1].item() < 0.01, (
            f'alphas_cumprod[-1]={alphas_cumprod[-1].item():.4f} should be < 0.01'
        )
        # 单调递减
        diffs = alphas_cumprod[1:] - alphas_cumprod[:-1]
        assert (diffs <= 1e-6).all()

    def test_matches_formula(self):
        """直接核对公式: alphas_cumprod = cos(((x/T)+s)/(1+s)*pi/2)^2."""
        T, s = 1000, 0.008
        steps = T + 1
        x = torch.linspace(0, T, steps, dtype=torch.float64)
        expected_acp = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        expected_acp = expected_acp / expected_acp[0]
        expected_betas = 1 - (expected_acp[1:] / expected_acp[:-1])
        expected_betas = torch.clip(expected_betas, 0, 0.999)

        betas = cosine_beta_schedule(T)
        assert torch.allclose(betas.double(), expected_betas, atol=1e-6)


# ========== loss_weight (SNR 加权) ==========


class TestLossWeight:
    """验证 SNR loss_weight 公式 (对齐原仓库 register_schedule_old line 278-279).

    原仓库: lvlb_weights = 0.5 * sqrt(acp) / (2 - acp); lvlb_weights[0] = lvlb_weights[1]
    """

    def test_formula_matches(self):
        """loss_weight 数值与公式 0.5*sqrt(acp)/(2-acp) 一致."""
        sched = DiffusionScheduler(
            timesteps=1000, sampling_timesteps=25, scale=2.0
        )
        acp = sched.alphas_cumprod.double()
        expected = 0.5 * torch.sqrt(acp) / (2.0 - acp)
        expected[0] = expected[1]
        assert torch.allclose(
            sched.loss_weight.double(), expected, atol=1e-6
        ), 'loss_weight formula mismatch'

    def test_loss_weight_0_equals_1(self):
        """lvlb_weights[0] = lvlb_weights[1] (原仓库 line 279 特殊处理)."""
        sched = DiffusionScheduler(timesteps=1000)
        assert torch.allclose(sched.loss_weight[0], sched.loss_weight[1])

    def test_value_range(self):
        """loss_weight ∈ [0, 0.5] (公式上界 0.5 当 acp→1)."""
        sched = DiffusionScheduler(timesteps=1000)
        lw = sched.loss_weight
        assert (lw >= 0).all(), f'loss_weight has negative: {lw.min()}'
        assert (lw <= 0.5 + 1e-5).all(), f'loss_weight exceeds 0.5: {lw.max()}'

    def test_high_t_lower_weight(self):
        """高 timestep (低 alphas_cumprod) → 低 loss_weight (高噪声降权语义).

        公式 0.5*sqrt(acp)/(2-acp): acp↓ → w↓.
        语义: 高噪声时 x0 预测不可靠 → 降权.
        注: 原仓库 50ep use_vlb=False 实际不使用此权重, 但公式仍注册.
        """
        sched = DiffusionScheduler(timesteps=1000)
        # t=10 (低噪声, acp 高) vs t=999 (高噪声, acp 低)
        w_low_t = sched.loss_weight[10].item()
        w_high_t = sched.loss_weight[999].item()
        assert w_low_t > w_high_t, (
            f'低 timestep 应有更高权重: w(10)={w_low_t:.4f} vs w(999)={w_high_t:.4f}'
        )
        # 端点: w[0]≈0.5, w[999]≈0
        assert sched.loss_weight[0].item() > 0.49
        assert sched.loss_weight[-1].item() < 0.01


# ========== q_sample ==========


class TestQSample:
    """验证前向扩散 q_sample (对齐原仓库 line 351-358)."""

    def test_t_zero_returns_x_start(self):
        """t=0 时 x_t ≈ x_start (acp[0]≈0.99996, 近似无噪声)."""
        sched = DiffusionScheduler(timesteps=1000)
        x_start = torch.randn(2, 10, 4)
        t = torch.zeros(2, dtype=torch.long)
        x_t = sched.q_sample(x_start, t, noise=torch.zeros_like(x_start))
        # acp[0]=0.99996 (cosine 归一化 + float32), sqrt(acp[0])≈0.99998
        # 故 x_t = sqrt(acp[0])*x_start ≈ x_start (误差 ~3e-5)
        assert torch.allclose(x_t, x_start, atol=1e-3), (
            't=0 应近似返回 x_start'
        )

    def test_t_max_noise_dominates(self):
        """t=T-1 时 x_t ≈ noise (噪声主导, acp≈0)."""
        sched = DiffusionScheduler(timesteps=1000)
        x_start = torch.randn(2, 10, 4) * 5  # 任意 x_start
        noise = torch.randn(2, 10, 4)
        t = torch.full((2,), 999, dtype=torch.long)
        x_t = sched.q_sample(x_start, t, noise=noise)
        # acp[999] ≈ 0 → x_t ≈ noise
        assert torch.allclose(x_t, noise, atol=0.05), (
            f't=T-1 应噪声主导, max diff={(x_t - noise).abs().max():.4f}'
        )

    def test_formula_correctness(self):
        """直接核对 x_t = sqrt(acp)*x0 + sqrt(1-acp)*noise."""
        sched = DiffusionScheduler(timesteps=1000)
        torch.manual_seed(0)
        x_start = torch.randn(1, 5, 4)
        noise = torch.randn(1, 5, 4)
        t = torch.tensor([500])
        x_t = sched.q_sample(x_start, t, noise=noise)
        acp_t = sched.alphas_cumprod[500]
        expected = math.sqrt(acp_t) * x_start + math.sqrt(1 - acp_t) * noise
        assert torch.allclose(x_t, expected, atol=1e-5)

    def test_deterministic_with_fixed_noise(self):
        """相同 noise + 相同 t → 相同 x_t (确定性)."""
        sched = DiffusionScheduler(timesteps=1000)
        x_start = torch.randn(1, 5, 4)
        noise = torch.randn(1, 5, 4)
        t = torch.tensor([300])
        x_t1 = sched.q_sample(x_start, t, noise=noise)
        x_t2 = sched.q_sample(x_start, t, noise=noise)
        assert torch.allclose(x_t1, x_t2)

    def test_scale_independence(self):
        """q_sample 在扩散空间操作, 与 scale 无关 (scale 仅用于空间转换)."""
        s1 = DiffusionScheduler(timesteps=1000, scale=2.0)
        s2 = DiffusionScheduler(timesteps=1000, scale=5.0)
        x_start = torch.randn(1, 5, 4)
        noise = torch.randn(1, 5, 4)
        t = torch.tensor([400])
        # q_sample 不依赖 scale
        assert torch.allclose(
            s1.q_sample(x_start, t, noise), s2.q_sample(x_start, t, noise)
        )


# ========== predict_noise_from_start (round-trip) ==========


class TestPredictNoise:
    """验证 predict_noise_from_start 与 q_sample 的可逆性."""

    def test_round_trip(self):
        """q_sample(x0, t, noise) → predict_noise_from_start(x_t, t, x0) ≈ noise."""
        sched = DiffusionScheduler(timesteps=1000)
        torch.manual_seed(42)
        x_start = torch.randn(2, 8, 4)
        noise = torch.randn(2, 8, 4)
        t = torch.tensor([100, 500])
        x_t = sched.q_sample(x_start, t, noise=noise)
        recovered_noise = sched.predict_noise_from_start(x_t, t, x_start)
        assert torch.allclose(recovered_noise, noise, atol=1e-4), (
            f'round-trip failed, max diff={(recovered_noise - noise).abs().max():.6f}'
        )

    def test_formula(self):
        """noise = (sqrt(1/acp)*x_t - x0) / sqrt(1/acp - 1)."""
        sched = DiffusionScheduler(timesteps=1000)
        x0 = torch.randn(1, 4, 4)
        noise = torch.randn(1, 4, 4)
        t = torch.tensor([250])
        x_t = sched.q_sample(x0, t, noise=noise)
        acp = sched.alphas_cumprod[250].item()
        expected = (math.sqrt(1 / acp) * x_t - x0) / math.sqrt(1 / acp - 1)
        got = sched.predict_noise_from_start(x_t, t, x0)
        assert torch.allclose(got, expected, atol=1e-5)


# ========== extract ==========


class TestExtract:
    """验证 extract (索引 + 广播)."""

    def test_shape_broadcasting_3d(self):
        """extract 对 3D 张量 [B, N, 4] 广播到 [B, 1, 1]."""
        a = torch.arange(1000).float()
        t = torch.tensor([3, 7])
        out = extract(a, t, (2, 10, 4))
        assert out.shape == (2, 1, 1)
        assert out[0].item() == 3.0
        assert out[1].item() == 7.0

    def test_gather_correct_values(self):
        """extract 取出正确的时间步值."""
        a = torch.tensor([10.0, 20.0, 30.0, 40.0])
        t = torch.tensor([0, 2, 3])
        out = extract(a, t, (3, 5))
        assert out.shape == (3, 1)
        assert out[0].item() == 10.0
        assert out[1].item() == 30.0
        assert out[2].item() == 40.0


# ========== timestep_embedding ==========


class TestTimestepEmbedbing:
    """验证正弦时间步嵌入."""

    def test_output_shape(self):
        """输出 [B, dim]."""
        t = torch.tensor([0, 500, 999])
        emb = timestep_embedding(t, dim=256)
        assert emb.shape == (3, 256)

    def test_odd_dim_padded(self):
        """奇数 dim 末尾补 0."""
        t = torch.tensor([100])
        emb = timestep_embedding(t, dim=257)
        assert emb.shape == (1, 257)
        assert emb[0, -1].item() == 0.0

    def test_different_timesteps_different_embeddings(self):
        """不同时间步 → 不同嵌入."""
        t = torch.tensor([0, 500, 999])
        emb = timestep_embedding(t, dim=256)
        assert not torch.allclose(emb[0], emb[1])
        assert not torch.allclose(emb[1], emb[2])

    def test_repeat_only_mode(self):
        """repeat_only=True → 只重复时间步."""
        t = torch.tensor([5.0, 10.0])
        emb = timestep_embedding(t, dim=8, repeat_only=True)
        assert emb.shape == (2, 8)
        assert torch.allclose(emb[0], torch.full((8,), 5.0))
        assert torch.allclose(emb[1], torch.full((8,), 10.0))


# ========== DDIM 采样 ==========


class TestDDIMStep:
    """验证 DDIM 单步采样."""

    def test_t_next_negative_returns_x0(self):
        """t_next<0 (最后一步) 直接返回 x0."""
        sched = DiffusionScheduler(
            timesteps=1000, sampling_timesteps=25, ddim_eta=0.0
        )
        x_t = torch.randn(1, 5, 4)
        x0 = torch.randn(1, 5, 4)
        out = sched.ddim_step(x_t, t=0, t_next=-1, x0=x0)
        assert torch.allclose(out, x0)

    def test_eta_zero_deterministic(self):
        """eta=0 → 确定性 (相同输入相同输出, 不依赖随机噪声)."""
        sched = DiffusionScheduler(timesteps=1000, ddim_eta=0.0)
        x_t = torch.randn(1, 5, 4)
        x0 = torch.randn(1, 5, 4)
        out1 = sched.ddim_step(x_t, t=500, t_next=400, x0=x0)
        out2 = sched.ddim_step(x_t, t=500, t_next=400, x0=x0)
        assert torch.allclose(out1, out2)

    def test_eta_zero_formula(self):
        """eta=0: x_{t-1} = sqrt(alpha_next)*x0 + sqrt(1-alpha_next)*pred_noise."""
        sched = DiffusionScheduler(timesteps=1000, ddim_eta=0.0)
        x_t = torch.randn(1, 5, 4)
        x0 = torch.randn(1, 5, 4)
        t, t_next = 500, 400
        out = sched.ddim_step(x_t, t, t_next, x0)
        # 手算
        alpha = sched.alphas_cumprod[t]
        alpha_next = sched.alphas_cumprod[t_next]
        t_batch = torch.full((1,), t, dtype=torch.long)
        pred_noise = sched.predict_noise_from_start(x_t, t_batch, x0)
        expected = (
            x0 * alpha_next.sqrt() + (1 - alpha_next).sqrt() * pred_noise
        )
        assert torch.allclose(out, expected, atol=1e-5)


class TestGetTimePairs:
    """验证 DDIM 时间对生成."""

    def test_count_equals_sampling_timesteps(self):
        """时间对数量 = sampling_timesteps."""
        sched = DiffusionScheduler(timesteps=1000, sampling_timesteps=25)
        pairs = sched.get_time_pairs()
        assert len(pairs) == 25

    def test_descending_order(self):
        """时间对从大到小 (反向扩散)."""
        sched = DiffusionScheduler(timesteps=1000, sampling_timesteps=10)
        pairs = sched.get_time_pairs()
        starts = [p[0] for p in pairs]
        assert starts == sorted(starts, reverse=True)

    def test_last_pair_negative_next(self):
        """最后一对的 time_next < 0 (标记最后一步)."""
        sched = DiffusionScheduler(timesteps=1000, sampling_timesteps=25)
        pairs = sched.get_time_pairs()
        assert pairs[-1][1] < 0


# ========== make_ddim_timesteps ==========


class TestMakeDDimTimesteps:
    """验证 DDIM 时间步索引生成."""

    def test_count(self):
        ts = make_ddim_timesteps(25, 1000)
        assert len(ts) == 25

    def test_descending(self):
        ts = make_ddim_timesteps(25, 1000)
        assert all(ts[i] >= ts[i + 1] for i in range(len(ts) - 1))


# ========== register_schedule 一致性 ==========


class TestRegisterSchedule:
    """验证缓冲区注册一致性."""

    def test_buffers_registered(self):
        """所有必要缓冲区已注册 (随模型迁移设备)."""
        sched = DiffusionScheduler(timesteps=1000)
        for name in [
            'betas',
            'alphas_cumprod',
            'alphas_cumprod_prev',
            'sqrt_alphas_cumprod',
            'sqrt_one_minus_alphas_cumprod',
            'sqrt_recip_alphas_cumprod',
            'sqrt_recipm1_alphas_cumprod',
            'loss_weight',
        ]:
            assert hasattr(sched, name), f'missing buffer {name}'

    def test_alphas_cumprod_prev_shifted(self):
        """alphas_cumprod_prev = pad(alphas_cumprod[:-1], (1,0), value=1)."""
        sched = DiffusionScheduler(timesteps=1000)
        assert sched.alphas_cumprod_prev[0].item() == 1.0
        assert torch.allclose(
            sched.alphas_cumprod_prev[1:], sched.alphas_cumprod[:-1]
        )

    def test_parameterization_x0(self):
        """parameterization = 'x0' (直接预测干净框)."""
        sched = DiffusionScheduler(timesteps=1000)
        assert sched.parameterization == 'x0'
