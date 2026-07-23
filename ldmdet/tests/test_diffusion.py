"""测试 ldmdet.diffusion — 位置编码、噪声调度、Rectified Flow、采样器"""

import math
import torch
import pytest
from ldmdet.diffusion.embeddings import SinusoidalPositionEmbeddings
from ldmdet.diffusion.noise_schedule import cosine_noise_schedule, load_buffer
from ldmdet.diffusion.rectified_flow import RectifiedFlow, RFDPMSolverMultistep, RFDPMSolverPerDim
from ldmdet.diffusion.sampling import DiffusionSampler
from ldmdet.data.structures import ImageMeta


class TestSinusoidalPositionEmbeddings:
    def test_output_shape(self):
        emb = SinusoidalPositionEmbeddings(dim=128)
        t = torch.randint(0, 1000, (8,))
        out = emb(t)
        assert out.shape == (8, 128)

    def test_single_timestep(self):
        emb = SinusoidalPositionEmbeddings(dim=64)
        t = torch.tensor([0])
        out = emb(t)
        assert out.shape == (1, 64)

    def test_deterministic(self):
        emb = SinusoidalPositionEmbeddings(dim=64)
        t = torch.tensor([42])
        out1 = emb(t)
        out2 = emb(t)
        assert torch.allclose(out1, out2, atol=1e-7)

    def test_different_timesteps_different_embeddings(self):
        emb = SinusoidalPositionEmbeddings(dim=64)
        t = torch.tensor([0, 1, 500, 999])
        out = emb(t)
        for i in range(4):
            for j in range(i + 1, 4):
                assert not torch.allclose(out[i], out[j], atol=1e-5)

    def test_zero_timestep(self):
        emb = SinusoidalPositionEmbeddings(dim=32)
        out = emb(torch.tensor([0]))
        assert torch.isfinite(out).all()


class TestCosineNoiseSchedule:
    def test_output_shape(self):
        T = 1000
        betas = cosine_noise_schedule(T)
        assert betas.shape == (T,)

    def test_range(self):
        betas = cosine_noise_schedule(1000)
        assert (betas >= 0).all() and (betas <= 0.999).all()

    def test_deterministic(self):
        b1 = cosine_noise_schedule(1000)
        b2 = cosine_noise_schedule(1000)
        assert torch.allclose(b1, b2, atol=1e-7)


class TestLoadBuffer:
    def test_basic(self):
        arr = torch.randn(100)
        steps = torch.tensor([0, 10, 50, 99])
        x_shape = [4, 100, 4]
        result = load_buffer(arr, steps, x_shape)
        assert result.shape == (4, 1, 1)
        # 值应与 arr[steps] 一致
        assert torch.allclose(result.squeeze(), arr[steps], atol=1e-6)

    def test_broadcast_shape(self):
        arr = torch.randn(100)
        steps = torch.tensor([5, 10])
        x_shape = [2, 100, 4, 4]
        result = load_buffer(arr, steps, x_shape)
        assert result.shape == (2, 1, 1, 1)


class TestRectifiedFlow:
    @pytest.fixture
    def rf(self):
        return RectifiedFlow(snr_scale=2.0)

    def test_q_sample_shape(self, rf):
        x_start = torch.randn(8, 500, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.rand(8)
        x_t, velocity = rf.q_sample(x_start, x_noise, t)
        assert x_t.shape == x_start.shape
        assert velocity.shape == x_start.shape

    def test_q_sample_t0(self, rf):
        """t=0 时 x_t == x_start (精确等式)"""
        x_start = torch.randn(2, 10, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.zeros(2)
        x_t, _ = rf.q_sample(x_start, x_noise, t)
        assert torch.equal(x_t, x_start)

    def test_q_sample_t1(self, rf):
        """t=1 时 x_t == x_noise (精确等式)"""
        x_start = torch.randn(2, 10, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.ones(2)
        x_t, _ = rf.q_sample(x_start, x_noise, t)
        assert torch.equal(x_t, x_noise)

    def test_velocity(self, rf):
        """velocity = x_noise - x_start"""
        x_start = torch.randn(2, 10, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.rand(2)
        _, velocity = rf.q_sample(x_start, x_noise, t)
        expected = x_noise - x_start
        assert torch.allclose(velocity, expected, atol=1e-5)

    def test_deterministic(self, rf):
        x_start = torch.randn(2, 10, 4)
        x_noise = torch.randn_like(x_start)
        t = torch.rand(2)
        x1, v1 = rf.q_sample(x_start, x_noise, t)
        x2, v2 = rf.q_sample(x_start, x_noise, t)
        assert torch.allclose(x1, x2, atol=1e-7)
        assert torch.allclose(v1, v2, atol=1e-7)

    def test_step(self, rf):
        x_t = torch.randn(2, 10, 4)
        x0_pred = torch.randn_like(x_t)
        x_next = rf.step(x_t, x0_pred, t_curr=0.8, t_next=0.6)
        assert x_next.shape == x_t.shape

    def test_heun_step(self, rf):
        x_t = torch.randn(2, 10, 4)
        x0_pred = torch.randn_like(x_t)
        def model_fn(x, t):
            return torch.randn_like(x), None
        x_next = rf.heun_step(x_t, x0_pred, t_curr=0.8, t_next=0.6, model_fn=model_fn)
        assert x_next.shape == x_t.shape

    def test_get_velocity(self, rf):
        x_t = torch.randn(2, 10, 4)
        x0_pred = torch.randn_like(x_t)
        t = torch.tensor([0.5])
        v = rf.get_velocity(x_t, x0_pred, t)
        assert v.shape == x_t.shape

    def test_get_velocity_t0_clamp(self, rf):
        """t=0 时 clamp(min=1e-5) 防止除零，结果应有限"""
        x_t = torch.randn(2, 10, 4)
        x0_pred = torch.randn_like(x_t)
        t = torch.zeros(2)
        v = rf.get_velocity(x_t, x0_pred, t)
        assert torch.isfinite(v).all()

    def test_get_velocity_t_small(self, rf):
        """t 极小时 velocity 仍有限"""
        x_t = torch.randn(2, 10, 4)
        x0_pred = torch.randn_like(x_t)
        t = torch.tensor([1e-6])
        v = rf.get_velocity(x_t, x0_pred, t)
        assert torch.isfinite(v).all()

    def test_get_velocity_formula(self, rf):
        """v = (x_t - x0_pred) / t 验证"""
        x_t = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        x0_pred = torch.tensor([[[0.5, 1.0, 1.5, 2.0]]])
        t = torch.tensor([0.5])
        v = rf.get_velocity(x_t, x0_pred, t)
        expected = (x_t - x0_pred) / 0.5
        assert torch.allclose(v, expected, atol=1e-5)


class TestRFDPMSolverMultistep:
    def test_init(self):
        solver = RFDPMSolverMultistep(num_steps=6, solver_order=2)
        assert len(solver.timesteps) == 7  # num_steps + 1
        assert solver.timesteps[0] == 1.0
        assert solver.timesteps[-1] == 0.0

    def test_timesteps_decreasing(self):
        solver = RFDPMSolverMultistep(num_steps=6, solver_order=2)
        ts = solver.timesteps
        for i in range(len(ts) - 1):
            assert ts[i] >= ts[i + 1]

    def test_step_output_shape(self):
        solver = RFDPMSolverMultistep(num_steps=4, solver_order=2)
        x = torch.randn(2, 100, 4)
        x0_pred = torch.randn_like(x)
        x_next = solver.step(x, x0_pred, t_n=1.0, step_idx=0)
        assert x_next.shape == x.shape

    def test_reset(self):
        solver = RFDPMSolverMultistep(num_steps=4, solver_order=2)
        x = torch.randn(2, 100, 4)
        x0_pred = torch.randn_like(x)
        solver.step(x, x0_pred, t_n=1.0, step_idx=0)
        assert len(solver.x0_history) > 0
        solver.reset()
        assert len(solver.x0_history) == 0


class TestRFDPMSolverPerDim:
    """方向 A Phase 2: per-dim 阶数分配 DPM-Solver++"""

    def test_init_default_dims(self):
        """默认: h (index 3) 用 1 阶, cx/cy/w (index 0,1,2) 用 2 阶."""
        solver = RFDPMSolverPerDim(num_steps=4)
        assert solver.euler_dims == (3,)
        assert solver.dpm_dims == (0, 1, 2)
        assert solver.solver_order == 2

    def test_init_custom_dims(self):
        solver = RFDPMSolverPerDim(
            num_steps=4, euler_dims=(2, 3), dpm_dims=(0, 1)
        )
        assert solver.euler_dims == (2, 3)
        assert solver.dpm_dims == (0, 1)

    def test_step_output_shape(self):
        solver = RFDPMSolverPerDim(num_steps=4)
        x = torch.randn(2, 100, 4)
        x0_pred = torch.randn_like(x)
        x_next = solver.step(x, x0_pred, t_n=1.0, step_idx=0)
        assert x_next.shape == x.shape

    def test_first_step_linear_only(self):
        """step_idx=0 (历史不足 2) 时所有维度退化为 linear (1 阶)."""
        solver = RFDPMSolverPerDim(num_steps=4)
        x = torch.randn(2, 10, 4)
        x0_pred = torch.randn_like(x)
        x_next = solver.step(x, x0_pred, t_n=1.0, step_idx=0)
        # linear = (t_next/t_n)*x + (1 - t_next/t_n)*x0_pred
        t_next = solver.timesteps[1]
        expected = (t_next / 1.0) * x + (1.0 - t_next / 1.0) * x0_pred
        assert torch.allclose(x_next, expected, atol=1e-6)

    def test_h_dim_uses_euler(self):
        """h 维度 (index 3) 在 step_idx>=1 时仅用 linear (无 correction)."""
        solver = RFDPMSolverPerDim(num_steps=4)
        x = torch.randn(2, 10, 4)
        x0_pred = torch.randn_like(x)
        # step 0: 填充历史
        solver.step(x, x0_pred, t_n=1.0, step_idx=0)
        # step 1: 有历史, 应对 dpm_dims 应用 correction, euler_dims 不应用
        x0_pred_2 = torch.randn_like(x)
        x_next = solver.step(x, x0_pred_2, t_n=solver.timesteps[1], step_idx=1)

        # 计算 linear (所有维度)
        t_n = solver.timesteps[1]
        t_next = solver.timesteps[2]
        linear = (t_next / t_n) * x + (1.0 - t_next / t_n) * x0_pred_2
        # h 维度 (index 3) 应等于 linear (无 correction)
        assert torch.allclose(x_next[..., 3], linear[..., 3], atol=1e-6)

    def test_cxcy_dims_use_dpm(self):
        """cx/cy/w 维度 (index 0,1,2) 在 step_idx>=1 时应用 correction (2 阶)."""
        solver = RFDPMSolverPerDim(num_steps=4)
        x = torch.randn(2, 10, 4)
        x0_pred = torch.randn_like(x)
        solver.step(x, x0_pred, t_n=1.0, step_idx=0)
        x0_pred_2 = torch.randn_like(x)
        x_next = solver.step(x, x0_pred_2, t_n=solver.timesteps[1], step_idx=1)

        # 计算 linear + correction (DPM-Solver++ 2 阶)
        t_n = solver.timesteps[1]
        t_next = solver.timesteps[2]
        linear = (t_next / t_n) * x + (1.0 - t_next / t_n) * x0_pred_2
        D1 = (x0_pred_2 - x0_pred) / (t_n - 1.0)
        phi1 = t_next * math.log(t_n / t_next) - t_n + t_next
        correction = phi1 * D1
        expected_dpm = linear + correction

        # cx/cy/w 维度 (index 0,1,2) 应等于 linear + correction
        for d in (0, 1, 2):
            assert torch.allclose(x_next[..., d], expected_dpm[..., d], atol=1e-6)

    def test_all_dpm_equals_base_solver(self):
        """当 euler_dims=() (所有维度用 2 阶) 时, 应等价于基类 RFDPMSolverMultistep."""
        solver_per = RFDPMSolverPerDim(num_steps=4, euler_dims=(), dpm_dims=(0, 1, 2, 3))
        solver_base = RFDPMSolverMultistep(num_steps=4, solver_order=2)
        x = torch.randn(2, 10, 4)
        x0_1 = torch.randn_like(x)
        x0_2 = torch.randn_like(x)
        # step 0
        out_per_0 = solver_per.step(x, x0_1, t_n=1.0, step_idx=0)
        out_base_0 = solver_base.step(x, x0_1, t_n=1.0, step_idx=0)
        assert torch.allclose(out_per_0, out_base_0, atol=1e-6)
        # step 1
        out_per_1 = solver_per.step(x, x0_2, t_n=solver_per.timesteps[1], step_idx=1)
        out_base_1 = solver_base.step(x, x0_2, t_n=solver_base.timesteps[1], step_idx=1)
        assert torch.allclose(out_per_1, out_base_1, atol=1e-6)

    def test_all_euler_equals_linear(self):
        """当 dpm_dims=() (所有维度用 1 阶) 时, 应等于纯 linear (Euler)."""
        solver = RFDPMSolverPerDim(num_steps=4, euler_dims=(0, 1, 2, 3), dpm_dims=())
        x = torch.randn(2, 10, 4)
        x0_1 = torch.randn_like(x)
        x0_2 = torch.randn_like(x)
        solver.step(x, x0_1, t_n=1.0, step_idx=0)
        x_next = solver.step(x, x0_2, t_n=solver.timesteps[1], step_idx=1)
        t_n = solver.timesteps[1]
        t_next = solver.timesteps[2]
        expected = (t_next / t_n) * x + (1.0 - t_next / t_n) * x0_2
        assert torch.allclose(x_next, expected, atol=1e-6)

    def test_reset(self):
        solver = RFDPMSolverPerDim(num_steps=4)
        x = torch.randn(2, 10, 4)
        x0_pred = torch.randn_like(x)
        solver.step(x, x0_pred, t_n=1.0, step_idx=0)
        assert len(solver.x0_history) > 0
        solver.reset()
        assert len(solver.x0_history) == 0

    def test_eta_str_history_populated(self):
        """step_idx>=1 应记录 eta_str 和 per_dim 诊断."""
        solver = RFDPMSolverPerDim(num_steps=4)
        x = torch.randn(2, 10, 4)
        x0_pred = torch.randn_like(x)
        solver.step(x, x0_pred, t_n=1.0, step_idx=0)
        assert len(solver.eta_str_history) == 1  # step 0 记录 0.0
        solver.step(x, x0_pred, t_n=solver.timesteps[1], step_idx=1)
        assert len(solver.eta_str_history) == 2
        assert len(solver.eta_str_per_dim_history) == 2
        assert len(solver.eta_str_per_dim_history[-1]) == 4  # 4 个维度


class TestDiffusionSampler:
    @pytest.fixture
    def sampler(self):
        return DiffusionSampler(
            diffusion_type='rectified_flow', timesteps=1000,
            sampling_timesteps=4, solver_type='euler',
            ddim_sampling_eta=1.0, rf_schedule='shifted',
            rf_power=1.0, rf_shift=2.0, snr_scale=2.0,
            box_renewal=True, use_ensemble=True,
            use_nms=True, nms_thr=0.5, score_thr=0.05, min_keep=10,
        )

    def test_raw_to_xyxy_shape(self, sampler):
        raw = torch.randn(2, 100, 4)
        img_metas = [ImageMeta(img_shape=(512, 512)), ImageMeta(img_shape=(512, 512))]
        xyxy = sampler.raw_to_xyxy(raw, img_metas)
        assert xyxy.shape == raw.shape

    def test_xyxy_to_raw_shape(self, sampler):
        xyxy = torch.rand(2, 100, 4) * 400 + 50
        xyxy[:, :, 2:] += xyxy[:, :, :2]
        img_metas = [ImageMeta(img_shape=(512, 512)), ImageMeta(img_shape=(512, 512))]
        raw = sampler.xyxy_to_raw(xyxy, img_metas)
        assert raw.shape == xyxy.shape

    def test_roundtrip(self, sampler):
        """raw -> xyxy -> raw 往返一致"""
        raw = torch.randn(2, 50, 4) * 0.5  # 在 snr_scale 范围内
        img_metas = [ImageMeta(img_shape=(512, 512)), ImageMeta(img_shape=(512, 512))]
        recovered = sampler.xyxy_to_raw(sampler.raw_to_xyxy(raw, img_metas), img_metas)
        assert torch.allclose(raw, recovered, atol=1e-2)

    def test_build_time_pairs(self, sampler):
        pairs = sampler.build_time_pairs(torch.device('cpu'))
        assert len(pairs) > 0
        # 每对 (t_curr, t_next) 应满足 t_curr > t_next
        for t_curr, t_next in pairs:
            assert t_curr > t_next

    def test_create_dpm_solver_euler(self, sampler):
        """euler solver 不创建 DPM solver"""
        solver = sampler.create_dpm_solver()
        assert solver is None

    def test_create_dpm_solver(self):
        sampler = DiffusionSampler(
            diffusion_type='rectified_flow', timesteps=1000,
            sampling_timesteps=4, solver_type='dpm_solver_pp',
            ddim_sampling_eta=1.0, rf_schedule='shifted',
            rf_power=1.0, rf_shift=2.0, snr_scale=2.0,
            box_renewal=False, use_ensemble=False,
            use_nms=True, nms_thr=0.5, score_thr=0.05, min_keep=10,
        )
        solver = sampler.create_dpm_solver()
        assert solver is not None
        assert isinstance(solver, RFDPMSolverMultistep)

    def test_apply_box_renewal(self, sampler):
        raw = torch.randn(2, 100, 4)
        cls_logits = torch.randn(2, 100, 24)
        renewed = sampler.apply_box_renewal(raw, cls_logits)
        assert renewed.shape == raw.shape

    def test_ddpm_sampler(self):
        sampler = DiffusionSampler(
            diffusion_type='ddpm', timesteps=1000,
            sampling_timesteps=4, solver_type='euler',
            ddim_sampling_eta=1.0, rf_schedule='linear',
            rf_power=1.0, rf_shift=1.0, snr_scale=2.0,
            box_renewal=False, use_ensemble=False,
            use_nms=True, nms_thr=0.5, score_thr=0.05, min_keep=10,
        )
        pairs = sampler.build_time_pairs(torch.device('cpu'))
        assert len(pairs) > 0

    def test_rf_schedule_power(self):
        """rf_schedule='power' 时间对"""
        sampler = DiffusionSampler(
            diffusion_type='rectified_flow', timesteps=1000,
            sampling_timesteps=4, solver_type='euler',
            ddim_sampling_eta=1.0, rf_schedule='power',
            rf_power=2.0, rf_shift=1.0, snr_scale=2.0,
            box_renewal=False, use_ensemble=False,
            use_nms=True, nms_thr=0.5, score_thr=0.05, min_keep=10,
        )
        pairs = sampler.build_time_pairs(torch.device('cpu'))
        assert len(pairs) == 4
        # t_curr > t_next
        for t_curr, t_next in pairs:
            assert t_curr > t_next
        # 首个 t_curr 应为 1.0^2 = 1.0
        assert abs(pairs[0][0] - 1.0) < 1e-5

    def test_rf_schedule_linear(self):
        """rf_schedule='linear' 时间对均匀分布"""
        sampler = DiffusionSampler(
            diffusion_type='rectified_flow', timesteps=1000,
            sampling_timesteps=4, solver_type='euler',
            ddim_sampling_eta=1.0, rf_schedule='linear',
            rf_power=1.0, rf_shift=1.0, snr_scale=2.0,
            box_renewal=False, use_ensemble=False,
            use_nms=True, nms_thr=0.5, score_thr=0.05, min_keep=10,
        )
        pairs = sampler.build_time_pairs(torch.device('cpu'))
        # 线性: 1.0, 0.75, 0.5, 0.25, 0.0
        assert abs(pairs[0][0] - 1.0) < 1e-5
        assert abs(pairs[-1][1] - 0.0) < 1e-5


class TestDiffusionSamplerPostProcess:
    """测试 DiffusionSampler.post_process — NMS、集成、缩放"""

    @pytest.fixture
    def sampler(self):
        return DiffusionSampler(
            diffusion_type='rectified_flow', timesteps=1000,
            sampling_timesteps=4, solver_type='euler',
            ddim_sampling_eta=1.0, rf_schedule='linear',
            rf_power=1.0, rf_shift=1.0, snr_scale=2.0,
            box_renewal=False, use_ensemble=True,
            use_nms=True, nms_thr=0.5, score_thr=0.05, min_keep=10,
        )

    def _make_ensemble(self, bs=2, num_proposals=50, num_classes=24):
        """构造多步集成结果"""
        results = []
        for _ in range(3):
            cls_logits = torch.randn(bs, num_proposals, num_classes)
            pred_bboxes = torch.rand(bs, num_proposals, 4) * 400 + 50
            pred_bboxes[:, :, 2:] += pred_bboxes[:, :, :2]
            results.append((cls_logits, pred_bboxes))
        return results

    def test_output_length(self, sampler):
        ensemble = self._make_ensemble()
        img_metas = [ImageMeta(img_shape=(512, 512)) for _ in range(2)]
        results = sampler.post_process(ensemble, img_metas, rescale=False)
        assert len(results) == 2

    def test_result_fields(self, sampler):
        ensemble = self._make_ensemble()
        img_metas = [ImageMeta(img_shape=(512, 512)) for _ in range(2)]
        results = sampler.post_process(ensemble, img_metas, rescale=False)
        for r in results:
            assert r.bboxes.shape[1] == 4
            assert r.scores.shape[0] == r.bboxes.shape[0]
            assert r.labels.shape[0] == r.bboxes.shape[0]

    def test_nms_reduces_boxes(self):
        """NMS 应减少重叠框数量"""
        sampler = DiffusionSampler(
            diffusion_type='rectified_flow', timesteps=1000,
            sampling_timesteps=1, solver_type='euler',
            ddim_sampling_eta=1.0, rf_schedule='linear',
            rf_power=1.0, rf_shift=1.0, snr_scale=2.0,
            box_renewal=False, use_ensemble=True,
            use_nms=True, nms_thr=0.1, score_thr=0.01, min_keep=1,
        )
        # 大量重叠框
        base_box = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        bboxes = base_box.repeat(1, 50, 1) + torch.randn(1, 50, 4) * 2
        cls_logits = torch.ones(1, 50, 24) * 5.0  # 高置信度
        ensemble = [(cls_logits, bboxes)]
        img_metas = [ImageMeta(img_shape=(512, 512))]
        results = sampler.post_process(ensemble, img_metas, rescale=False)
        # NMS 后框数应远少于 50
        assert results[0].bboxes.shape[0] < 50

    def test_no_nms(self):
        """关闭 NMS 时保留所有框"""
        sampler = DiffusionSampler(
            diffusion_type='rectified_flow', timesteps=1000,
            sampling_timesteps=1, solver_type='euler',
            ddim_sampling_eta=1.0, rf_schedule='linear',
            rf_power=1.0, rf_shift=1.0, snr_scale=2.0,
            box_renewal=False, use_ensemble=True,
            use_nms=False, nms_thr=0.5, score_thr=0.0, min_keep=1,
        )
        ensemble = self._make_ensemble(bs=1, num_proposals=20)
        img_metas = [ImageMeta(img_shape=(512, 512))]
        results = sampler.post_process(ensemble, img_metas, rescale=False)
        # 3 步集成 × 20 proposals = 60 (无 NMS)
        assert results[0].bboxes.shape[0] == 60

    def test_rescale(self):
        """rescale=True 应按 scale_factor 缩放框到原始图像尺度"""
        sampler = DiffusionSampler(
            diffusion_type='rectified_flow', timesteps=1000,
            sampling_timesteps=1, solver_type='euler',
            ddim_sampling_eta=1.0, rf_schedule='linear',
            rf_power=1.0, rf_shift=1.0, snr_scale=2.0,
            box_renewal=False, use_ensemble=True,
            use_nms=False, nms_thr=0.5, score_thr=0.0, min_keep=1,
        )
        cls_logits = torch.ones(1, 10, 24) * 3.0
        pred_bboxes = torch.rand(1, 10, 4) * 400 + 50
        pred_bboxes[:, :, 2:] += pred_bboxes[:, :, :2]
        ensemble = [(cls_logits, pred_bboxes)]
        # scale_factor=0.5 表示特征图是原图的 0.5 倍
        # rescale=True 时会把框坐标除以 scale_factor (即 ×2) 还原到原图尺度
        img_metas = [ImageMeta(img_shape=(512, 512), scale_factor=[0.5, 0.5])]
        results_no_rescale = sampler.post_process(ensemble, img_metas, rescale=False)
        results_rescale = sampler.post_process(ensemble, img_metas, rescale=True)
        # rescale 后框坐标应更大 (还原到原图尺度)
        assert results_rescale[0].bboxes.max() >= results_no_rescale[0].bboxes.max() - 1e-3

    def test_single_step_ensemble(self):
        """单步集成 (1 组结果)"""
        sampler = DiffusionSampler(
            diffusion_type='rectified_flow', timesteps=1000,
            sampling_timesteps=1, solver_type='euler',
            ddim_sampling_eta=1.0, rf_schedule='linear',
            rf_power=1.0, rf_shift=1.0, snr_scale=2.0,
            box_renewal=False, use_ensemble=True,
            use_nms=True, nms_thr=0.5, score_thr=0.05, min_keep=10,
        )
        cls_logits = torch.randn(1, 50, 24)
        pred_bboxes = torch.rand(1, 50, 4) * 400 + 50
        pred_bboxes[:, :, 2:] += pred_bboxes[:, :, :2]
        ensemble = [(cls_logits, pred_bboxes)]
        img_metas = [ImageMeta(img_shape=(512, 512))]
        results = sampler.post_process(ensemble, img_metas, rescale=False)
        assert len(results) == 1


class TestPredictNoiseFromStart:
    """测试 sampling.predict_noise_from_start"""

    def test_output_shape(self):
        from ldmdet.diffusion.sampling import predict_noise_from_start
        x_t = torch.randn(2, 50, 4)
        t = torch.tensor([100, 500])
        x0 = torch.randn_like(x_t)
        alphas_cumprod = torch.rand(1000)
        noise = predict_noise_from_start(x_t, t, x0, alphas_cumprod)
        assert noise.shape == x_t.shape

    def test_finite(self):
        from ldmdet.diffusion.sampling import predict_noise_from_start
        x_t = torch.randn(2, 50, 4)
        t = torch.tensor([0, 999])
        x0 = torch.randn_like(x_t)
        alphas_cumprod = cosine_noise_schedule(1000).float()
        # 从 betas 计算 alphas_cumprod
        alphas = 1.0 - alphas_cumprod
        alphas_cumprod_actual = torch.cumprod(alphas, dim=0)
        noise = predict_noise_from_start(x_t, t, x0, alphas_cumprod_actual)
        assert torch.isfinite(noise).all()


class TestDDIMStep:
    """测试 DDIM 采样步骤 (DDPM 基线)"""

    @pytest.fixture
    def ddpm_sampler(self):
        return DiffusionSampler(
            diffusion_type='ddpm', timesteps=1000,
            sampling_timesteps=4, solver_type='euler',
            ddim_sampling_eta=0.0,  # eta=0 → 确定性 DDIM
            rf_schedule='linear', rf_power=1.0, rf_shift=1.0,
            snr_scale=2.0, box_renewal=False, use_ensemble=False,
            use_nms=True, nms_thr=0.5, score_thr=0.05, min_keep=10,
        )

    def _make_alphas_cumprod(self, T=1000):
        """构造 alphas_cumprod (cosine schedule)"""
        betas = cosine_noise_schedule(T).float()
        alphas = 1.0 - betas
        return torch.cumprod(alphas, dim=0)

    def test_output_shape(self, ddpm_sampler):
        """ddim_step 输出形状正确"""
        bs, N = 2, 50
        alphas_cumprod = self._make_alphas_cumprod()
        x_raw = torch.randn(bs, N, 4)
        cls_logits = torch.randn(bs, N, 24)
        pred_bboxes = torch.rand(bs, N, 4) * 400 + 50
        pred_bboxes[:, :, 2:] += pred_bboxes[:, :, :2]
        img_metas = [ImageMeta(img_shape=(512, 512)) for _ in range(bs)]

        xyxy_next, x0 = ddpm_sampler.ddim_step(
            t_curr=999, t_next=749, x_raw=x_raw,
            cls_logits=cls_logits, pred_bboxes=pred_bboxes,
            img_metas=img_metas, alphas_cumprod=alphas_cumprod,
        )
        assert xyxy_next.shape == (bs, N, 4)
        assert x0.shape == (bs, N, 4)

    def test_deterministic_eta_zero(self, ddpm_sampler):
        """eta=0 时 DDIM 确定性: 相同输入 → 相同输出"""
        bs, N = 2, 50
        alphas_cumprod = self._make_alphas_cumprod()
        x_raw = torch.randn(bs, N, 4)
        cls_logits = torch.randn(bs, N, 24)
        pred_bboxes = torch.rand(bs, N, 4) * 400 + 50
        pred_bboxes[:, :, 2:] += pred_bboxes[:, :, :2]
        img_metas = [ImageMeta(img_shape=(512, 512)) for _ in range(bs)]

        torch.manual_seed(0)
        xyxy1, x0_1 = ddpm_sampler.ddim_step(
            t_curr=999, t_next=749, x_raw=x_raw,
            cls_logits=cls_logits, pred_bboxes=pred_bboxes,
            img_metas=img_metas, alphas_cumprod=alphas_cumprod,
        )
        torch.manual_seed(0)
        xyxy2, x0_2 = ddpm_sampler.ddim_step(
            t_curr=999, t_next=749, x_raw=x_raw,
            cls_logits=cls_logits, pred_bboxes=pred_bboxes,
            img_metas=img_metas, alphas_cumprod=alphas_cumprod,
        )
        assert torch.allclose(xyxy1, xyxy2, atol=1e-6)
        assert torch.allclose(x0_1, x0_2, atol=1e-6)

    def test_t_next_negative_returns_x0(self, ddpm_sampler):
        """t_next < 0 (最后一步) 应直接返回 x0"""
        bs, N = 2, 50
        alphas_cumprod = self._make_alphas_cumprod()
        x_raw = torch.randn(bs, N, 4)
        cls_logits = torch.randn(bs, N, 24)
        pred_bboxes = torch.rand(bs, N, 4) * 400 + 50
        pred_bboxes[:, :, 2:] += pred_bboxes[:, :, :2]
        img_metas = [ImageMeta(img_shape=(512, 512)) for _ in range(bs)]

        xyxy_next, x0 = ddpm_sampler.ddim_step(
            t_curr=0, t_next=-1, x_raw=x_raw,
            cls_logits=cls_logits, pred_bboxes=pred_bboxes,
            img_metas=img_metas, alphas_cumprod=alphas_cumprod,
        )
        # t_next < 0 时返回 raw_to_xyxy(x0), x0
        expected_xyxy = ddpm_sampler.raw_to_xyxy(x0, img_metas)
        assert torch.allclose(xyxy_next, expected_xyxy, atol=1e-6)

    def test_finite_output(self, ddpm_sampler):
        """输出有限"""
        bs, N = 2, 50
        alphas_cumprod = self._make_alphas_cumprod()
        x_raw = torch.randn(bs, N, 4)
        cls_logits = torch.randn(bs, N, 24)
        pred_bboxes = torch.rand(bs, N, 4) * 400 + 50
        pred_bboxes[:, :, 2:] += pred_bboxes[:, :, :2]
        img_metas = [ImageMeta(img_shape=(512, 512)) for _ in range(bs)]

        xyxy_next, x0 = ddpm_sampler.ddim_step(
            t_curr=999, t_next=499, x_raw=x_raw,
            cls_logits=cls_logits, pred_bboxes=pred_bboxes,
            img_metas=img_metas, alphas_cumprod=alphas_cumprod,
        )
        assert torch.isfinite(xyxy_next).all()
        assert torch.isfinite(x0).all()


class TestRFDPMSolverMultistepOrder3:
    """测试 DPM-Solver++ 三阶求解器"""

    def test_solver_order3_step(self):
        solver = RFDPMSolverMultistep(num_steps=4, solver_order=3)
        x = torch.randn(2, 100, 4)
        # 模拟 3 步历史
        x0_1 = torch.randn_like(x)
        x0_2 = torch.randn_like(x)
        x0_3 = torch.randn_like(x)
        solver.step(x, x0_1, t_n=1.0, step_idx=0)
        solver.step(x, x0_2, t_n=0.75, step_idx=1)
        x_next = solver.step(x, x0_3, t_n=0.5, step_idx=2)
        assert x_next.shape == x.shape
        assert torch.isfinite(x_next).all()

    def test_solver_order3_full_trajectory(self):
        """完整 4 步求解轨迹"""
        solver = RFDPMSolverMultistep(num_steps=4, solver_order=3)
        x = torch.randn(2, 50, 4)
        for step_idx in range(4):
            x0_pred = torch.randn_like(x)
            t_n = solver.timesteps[step_idx]
            x = solver.step(x, x0_pred, t_n=t_n, step_idx=step_idx)
        assert x.shape == (2, 50, 4)
        assert torch.isfinite(x).all()
