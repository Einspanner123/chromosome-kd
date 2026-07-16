import math
from typing import Optional, Tuple, Union

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

    def get_velocity(
        self, x_t: Tensor, x_0_pred: Tensor, t: Union[Tensor, float]
    ) -> Tensor:
        """
        根据预测的 x_0 计算当前的速度 v_t。
        在 Rectified Flow 中, v_t = (x_t - x_0_pred) / t (当 t > 0 时)
        或者更直接地，如果模型预测了 x_0，且我们知道当前的 x_t 和 t，
        那么从 x_t = (1-t)x_0 + t x_1 可以推导出 x_1 = (x_t - (1-t)x_0) / t
        则 v = x_1 - x_0 = (x_t - x_0) / t
        """
        if isinstance(t, (int, float)):
            # 标量路径: 直接用 Python float 广播, 避免创建 timestep tensor
            v_t = (x_t - x_0_pred) / max(float(t), 1e-5)
            return v_t
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
        # 直接传 Python float, 避免创建 timestep tensor 的开销
        v_t = self.get_velocity(x_t, x_0_pred, t_curr)
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

        # --- 1. Euler Step (预估下一步位置) ---
        # 直接传 Python float, 避免创建 timestep tensor 的开销
        v_t = self.get_velocity(x_t, x_0_pred, t_curr)
        x_next_euler = x_t + dt * v_t

        # --- 2. 在 t_next 处进行第二次预测 ---
        x_0_pred_next, _ = model_fn(x_next_euler, t_next)

        # --- 3. 计算 BBox 修正 ---
        v_next = self.get_velocity(x_next_euler, x_0_pred_next, t_next)
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
        self._precompute_phi_coefficients()
        self.reset()

    def _precompute_phi_coefficients(self):
        """在 __init__ 中预计算所有步的 phi1, phi2 系数。

        phi1 依赖于 t_n (当前步) 和 t_next (下一步), 均在 self.timesteps 中。
        phi2 依赖于 t_n, t_next, t_p (前一步), 也在 self.timesteps 中。
        当调用方使用线性 schedule (t_n == self.timesteps[step_idx]) 时可直接查找,
        避免每步重复调用 math.log; 非线性 schedule 时回退到运行时计算以保持精度一致。
        """
        self.phi1_list: list[float] = []
        self.phi2_list: list[float] = []
        for step_idx in range(self.num_steps):
            t_n = self.timesteps[step_idx]
            t_next = self.timesteps[step_idx + 1]
            self.phi1_list.append(self._compute_phi1(t_n, t_next))
            # phi2 需要 t_p (前一步), 仅在 step_idx >= 1 时有意义
            if step_idx >= 1:
                t_p = self.timesteps[step_idx - 1]
                self.phi2_list.append(self._compute_phi2(t_n, t_next, t_p))
            else:
                self.phi2_list.append(0.0)  # placeholder, 不会被使用

    @staticmethod
    def _compute_phi1(t_n: float, t_next: float) -> float:
        if t_next > 1e-7:
            return t_next * math.log(t_n / t_next) - t_n + t_next
        return -t_n

    @staticmethod
    def _compute_phi2(t_n: float, t_next: float, t_p: float) -> float:
        if t_next > 1e-7:
            return (t_next + t_p) * (t_n - t_next) - t_next * (
                t_n + t_p
            ) * math.log(t_n / t_next)
        return t_p * t_n

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

        # 线性 schedule (t_n == self.timesteps[step_idx]) 时直接查找预计算 phi1,
        # 避免 math.log 调用; 非线性 schedule 时回退到运行时计算保持精度一致
        if t_n == self.timesteps[step_idx]:
            phi1 = self.phi1_list[step_idx]
        elif t_next > 1e-7:
            phi1 = t_next * math.log(t_n / t_next) - t_n + t_next
        else:
            phi1 = -t_n

        correction = phi1 * D1

        if self.solver_order >= 3 and len(self.x0_history) >= 3:
            x0_pp = self.x0_history[-3]
            t_pp = self.t_history[-3]
            D1_p = (x0_p - x0_pp) / (t_p - t_pp)
            D2 = (D1 - D1_p) / (t_n - t_pp)

            # phi2 同样优先使用预计算值, 需同时确认 t_p 与线性 schedule 一致
            if (
                t_n == self.timesteps[step_idx]
                and t_p == self.timesteps[step_idx - 1]
            ):
                phi2 = self.phi2_list[step_idx]
            elif t_next > 1e-7:
                phi2 = (t_next + t_p) * (t_n - t_next) - t_next * (
                    t_n + t_p
                ) * math.log(t_n / t_next)
            else:
                phi2 = t_p * t_n

            correction = correction + phi2 * D2

        return linear + correction
