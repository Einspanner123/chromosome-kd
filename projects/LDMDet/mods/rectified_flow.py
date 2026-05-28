from typing import Optional, Tuple

import torch
from torch import Tensor


class RectifiedFlow:
    """
    Rectified Flow (1-RectFlow) 模块。
    实现了直线路径的前向扩散和采样逻辑。
    $x_t = (1-t)x_0 + t x_1$
    其中 $x_0$ 是数据 (GT)，$x_1$ 是噪声。
    """

    def __init__(self, snr_scale: float = 2.0):
        self.snr_scale = snr_scale

    def q_sample(
        self,
        x_start: Tensor,
        x_noise: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        前向采样 (加噪过程): x_t = (1-t)x_0 + t * x_1

        Args:
            x_start (Tensor): 原始数据 x_0, shape [N, 4]
            x_noise (Tensor, optional): 噪声 x_1. 默认为标准正态分布。
            t (Tensor): 时间步, 范围 [0, 1], shape [N] or [1]

        Returns:
            x_t (Tensor): 加噪后的样本
            velocity (Tensor): 目标速度 x_1 - x_0
        """
        if x_noise is None:
            x_noise = torch.randn_like(x_start)

        if t is None:
            # 随机采样 t
            t = torch.rand((x_start.shape[0],), device=x_start.device)

        # 扩展 t 的维度以便广播
        t_view = t.view(-1, *([1] * (x_start.dim() - 1)))

        x_t = (1.0 - t_view) * x_start + t_view * x_noise
        velocity = x_noise - x_start

        return x_t, velocity

    def get_velocity(self, x_t: Tensor, x_0_pred: Tensor, t: Tensor) -> Tensor:
        """
        根据预测的 x_0 计算当前的速度 v_t。
        在 Rectified Flow 中, v_t = (x_t - x_0_pred) / t (当 t > 0 时)
        或者更直接地，如果模型预测了 x_0，且我们知道当前的 x_t 和 t，
        那么从 x_t = (1-t)x_0 + t x_1 可以推导出 x_1 = (x_t - (1-t)x_0) / t
        则 v = x_1 - x_0 = (x_t - x_0) / t
        """
        t_view = t.view(-1, *([1] * (x_t.dim() - 1)))
        # 避免除以 0
        v_t = (x_t - x_0_pred) / torch.clamp(t_view, min=1e-5)
        return v_t

    def step(
        self, x_t: Tensor, x_0_pred: Tensor, t_curr: float, t_next: float
    ) -> Tensor:
        """
        ODE 采样的一步 (Euler Step): x_{t_next} = x_t + (t_next - t_curr) * v_t
        """
        dt = t_next - t_curr
        v_t = self.get_velocity(
            x_t, x_0_pred, torch.tensor([t_curr], device=x_t.device)
        )
        x_next = x_t + dt * v_t
        return x_next

    def heun_step(
        self,
        x_t: Tensor,
        x_0_pred: Tensor,
        t_curr: float,
        t_next: float,
        model_fn,  # 传入一个函数, 用于在 t_next 处预测 x_0
    ) -> Tensor:
        """
        ODE 采样的一步 (Heun Step, 二阶):
        x_next = x_t + (dt/2) * (v_t + v_next)
        """
        dt = t_next - t_curr
        device = x_t.device

        # --- 1. Euler Step (预估下一步位置) ---
        v_t = self.get_velocity(
            x_t, x_0_pred, torch.tensor([t_curr], device=device)
        )
        x_next_euler = x_t + dt * v_t

        # --- 2. 在 t_next 处进行第二次预测 ---
        x_0_pred_next, _ = model_fn(x_next_euler, t_next)

        # --- 3. 计算 BBox 修正 ---
        v_next = self.get_velocity(
            x_next_euler, x_0_pred_next, torch.tensor([t_next], device=device)
        )
        x_next = x_t + (dt / 2.0) * (v_t + v_next)

        return x_next


class RFDPMSolverMultistep:
    """Rectified Flow 专用 DPM-Solver++ 多步法调度器。

    在 $t$ 空间用历史 $x_0^{pred}$ 做多项式插值，精确积分半线性 ODE。
    $t=0$ 终点的奇点通过 $\\lim_{x\\to0} x\\ln x = 0$ 自然退化处理。

    参数:
        num_steps: 推理时间步数 (default 6)
        solver_order: 多步法阶数 (2 or 3)
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
        """单步积分: x(t_n) → x(t_{n+1})。

        Args:
            x: 当前隐变量 x_t, shape [N, 4]
            x0_pred: model(x, t_n) 的 x0 预测
            t_n: 当前时间步
            step_idx: 当前步索引 (用于查找 t_{n+1})

        Returns:
            x_{n+1}, shape [N, 4]
        """
        import math

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
                phi2 = (
                    t_next * (t_n * math.log(t_n / t_next) + t_next - t_n)
                    - (t_n * t_n - t_next * t_next) / 2.0
                    + t_n * (t_n - t_next)
                )
            else:
                phi2 = (t_n * t_n) / 2.0

            correction = correction + phi2 * D2

        return linear + correction
