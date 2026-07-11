"""Rectified Flow 与 DPM-Solver++

x_t = (1-t)x_0 + t x_1   (直线路径)
其中 x_0 是数据 (GT)，x_1 是噪声。
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor


class RectifiedFlow:
    """1-RectFlow: 直线路径前向扩散与采样。"""

    def __init__(self, snr_scale: float = 2.0):
        self.snr_scale = snr_scale

    def q_sample(
        self,
        x_start: Tensor,
        x_noise: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """前向加噪: x_t = (1-t)x_0 + t x_1

        Returns:
            x_t: 加噪样本
            velocity: 目标速度 x_1 - x_0
        """
        if x_noise is None:
            x_noise = torch.randn_like(x_start)
        if t is None:
            t = torch.rand((x_start.shape[0],), device=x_start.device)

        t_view = t.view(-1, *([1] * (x_start.dim() - 1)))
        x_t = (1.0 - t_view) * x_start + t_view * x_noise
        velocity = x_noise - x_start
        return x_t, velocity

    def get_velocity(self, x_t: Tensor, x_0_pred: Tensor, t: Tensor) -> Tensor:
        """v_t = (x_t - x_0_pred) / t"""
        t_view = t.view(-1, *([1] * (x_t.dim() - 1)))
        return (x_t - x_0_pred) / torch.clamp(t_view, min=1e-5)

    def step(
        self,
        x_t: Tensor,
        x_0_pred: Tensor,
        t_curr: float,
        t_next: float,
        velocity: Optional[Tensor] = None,
    ) -> Tensor:
        """Euler step: x_next = x_t + dt * v_t"""
        dt = t_next - t_curr
        if velocity is None:
            velocity = self.get_velocity(
                x_t, x_0_pred, torch.tensor([t_curr], device=x_t.device)
            )
        return x_t + dt * velocity

    def heun_step(
        self,
        x_t: Tensor,
        x_0_pred: Tensor,
        t_curr: float,
        t_next: float,
        model_fn,
        velocity: Optional[Tensor] = None,
    ) -> Tensor:
        """Heun step (二阶): x_next = x_t + (dt/2)(v_t + v_next)"""
        dt = t_next - t_curr
        device = x_t.device

        if velocity is None:
            velocity = self.get_velocity(
                x_t, x_0_pred, torch.tensor([t_curr], device=device)
            )
        x_next_euler = x_t + dt * velocity

        x_0_pred_next, _ = model_fn(x_next_euler, t_next)
        v_next = self.get_velocity(
            x_next_euler,
            x_0_pred_next,
            torch.tensor([t_next], device=device),
        )
        return x_t + (dt / 2.0) * (velocity + v_next)


class RFDPMSolverMultistep:
    """Rectified Flow 专用 DPM-Solver++ 多步法。

    在 t 空间用历史 x0_pred 做多项式插值，精确积分半线性 ODE。
    """

    def __init__(self, num_steps: int = 6, solver_order: int = 2):
        self.num_steps = num_steps
        self.solver_order = solver_order
        self.timesteps: list[float] = [
            float(t) for t in torch.linspace(1.0, 0.0, num_steps + 1)
        ]
        self.reset()

    def reset(self):
        self.x0_history: list[torch.Tensor] = []
        self.t_history: list[float] = []

    def step(
        self,
        x: torch.Tensor,
        x0_pred: torch.Tensor,
        t_n: float,
        step_idx: int,
    ) -> torch.Tensor:
        """单步积分 x(t_n) → x(t_{n+1})"""
        t_next = self.timesteps[step_idx + 1]

        self.x0_history.append(x0_pred)
        self.t_history.append(t_n)
        if len(self.x0_history) > self.solver_order:
            self.x0_history.pop(0)
            self.t_history.pop(0)

        linear = (t_next / t_n) * x + (1.0 - t_next / t_n) * x0_pred

        if len(self.x0_history) < 2:
            return linear

        x0_n = self.x0_history[-1]
        x0_p = self.x0_history[-2]
        t_p = self.t_history[-2]
        D1 = (x0_n - x0_p) / (t_n - t_p)

        if t_next > 1e-7:
            phi1 = t_next * math.log(t_n / t_next) - t_n + t_next
        else:
            phi1 = -t_n

        correction = phi1 * D1

        if self.solver_order >= 3 and len(self.x0_history) >= 3:
            x0_pp = self.x0_history[-3]
            t_pp = self.t_history[-3]
            D1_p = (x0_p - x0_pp) / (t_p - t_pp)
            D2 = (D1 - D1_p) / (t_n - t_pp)

            if t_next > 1e-7:
                phi2 = (t_next + t_p) * (t_n - t_next) - t_next * (
                    t_n + t_p
                ) * math.log(t_n / t_next)
            else:
                phi2 = t_p * t_n

            correction = correction + phi2 * D2

        return linear + correction
