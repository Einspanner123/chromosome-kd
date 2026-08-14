"""测试扩散采样"""

import torch

from ldmdet.diffusion.embeddings import SinusoidalPositionEmbeddings
from ldmdet.diffusion.noise_schedule import cosine_noise_schedule, load_buffer
from ldmdet.diffusion.rectified_flow import RectifiedFlow


class TestEmbeddings:
    def test_shape(self):
        emb = SinusoidalPositionEmbeddings(256)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = emb(t)
        assert out.shape == (3, 256)

    def test_zero_time(self):
        emb = SinusoidalPositionEmbeddings(256)
        out = emb(torch.zeros(1))
        # t=0: sin(0)=0, cos(0)=1 → 前半为 0, 后半为 1
        assert torch.allclose(out[0, :128], torch.zeros(128), atol=1e-6)
        assert torch.allclose(out[0, 128:], torch.ones(128), atol=1e-6)


class TestRectifiedFlow:
    def test_q_sample_shape(self):
        rf = RectifiedFlow(snr_scale=2.0)
        x_start = torch.randn(10, 4)
        x_t, velocity = rf.q_sample(x_start)
        assert x_t.shape == (10, 4)
        assert velocity.shape == (10, 4)

    def test_interpolation(self):
        rf = RectifiedFlow(snr_scale=2.0)
        x_start = torch.randn(3, 4)
        x_noise = torch.randn(3, 4)
        t = torch.tensor([0.5, 0.0, 1.0])
        x_t, v = rf.q_sample(x_start, x_noise, t)
        # t=0: x_t = x_start
        assert torch.allclose(x_t[1], x_start[1])
        # t=1: x_t = x_noise
        assert torch.allclose(x_t[2], x_noise[2])
        # t=0.5: x_t = 0.5*x_start + 0.5*x_noise
        assert torch.allclose(x_t[0], 0.5 * x_start[0] + 0.5 * x_noise[0])

    def test_velocity_sign(self):
        rf = RectifiedFlow(snr_scale=2.0)
        x_start = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        x_noise = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        t = torch.tensor([0.5])
        _, v = rf.q_sample(x_start, x_noise, t)
        # v = noise - start = [-1, -2, -3, -4]
        assert torch.allclose(v, -x_start)

    def test_get_velocity(self):
        rf = RectifiedFlow()
        x_t = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        x_0_pred = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        t = torch.tensor([0.5])
        v = rf.get_velocity(x_t, x_0_pred, t)
        # v = (x_t - x_0) / t = (0.5 - 0) / 0.5 = 1.0
        assert torch.allclose(v, torch.ones(1, 4))


class TestNoiseSchedule:
    def test_cosine_shape(self):
        betas = cosine_noise_schedule(1000, device='cpu')
        assert betas.shape == (1000,)

    def test_load_buffer(self):
        arr = torch.arange(1000, dtype=torch.float32)
        steps = torch.tensor([100, 200])
        out = load_buffer(arr, steps, [2, 5, 4, 4])
        assert out.shape == (2, 1, 1, 1)
        assert out[0, 0, 0, 0] == 100.0
        assert out[1, 0, 0, 0] == 200.0
