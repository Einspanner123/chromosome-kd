"""Rectified Flow 与 DPM-Solver++

x_t = (1-t)x_0 + t x_1   (直线路径)
其中 x_0 是数据 (GT)，x_1 是噪声。
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from ldmdet.diagnostics.instrumentation import probe


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

        # 探针: RF 前向加噪路径统计 (训练时每 100 步)
        probe.record_tensor_stats('rf/x_t', x_t)
        probe.record_tensor_stats('rf/velocity', velocity)
        probe.record_tensor_stats('rf/x_start', x_start)
        probe.record_tensor_stats('rf/x_noise', x_noise)

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

    def __init__(
        self,
        num_steps: int = 6,
        solver_order: int = 2,
        timesteps: Optional[list[float]] = None,
    ):
        self.num_steps = num_steps
        self.solver_order = solver_order
        if timesteps is not None:
            self.timesteps = timesteps
        else:
            self.timesteps = [
                float(t) for t in torch.linspace(1.0, 0.0, num_steps + 1)
            ]
        self.reset()

    def reset(self):
        self.x0_history: list[torch.Tensor] = []
        self.t_history: list[float] = []
        # R1 诊断: 记录每步的直线度指标 eta_str
        # eta_str = ||D1|| / ||x0||, 理想 RF 下 D1=0, eta_str=0
        self.eta_str_history: list[float] = []
        # 方向 A 诊断: per-dim eta_str = |D1[:,:,k]| / |x0[:,:,k]| for k in [cx,cy,w,h]
        # 用于验证 bbox 4 维度的曲率是否一致 (假设 w,h 曲率 << cx,cy)
        # 每个元素是长度 4 的 list[float], 索引顺序 = (cx, cy, w, h)
        self.eta_str_per_dim_history: list[list[float]] = []
        # 方向 D 诊断: 三阶校正项相对量 eta_3rd = ||D2|| / ||x0||
        # 用于验证自适应阶次假设: 哪些 step 的 D2 项显著 (需要 3 阶), 哪些可降为 2 阶
        # 注意: 仅在 solver_order>=3 且历史足够长时记录, 否则记 0.0 (表示未计算)
        self.eta_3rd_history: list[float] = []

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

        # R1 诊断: 记录 eta_str = ||D1|| / ||x0|| (batch 均值)
        # 理想 RF (直线 ODE, x0(t)=const) 下 D1=0, eta_str=0
        # eta_str 大表示学习轨迹非直线, DPM-Solver++ 二阶校正生效
        with torch.no_grad():
            d1_norm = D1.norm(dim=-1)  # [bs, N]
            x0_norm = x0_n.norm(dim=-1).clamp(min=1e-6)  # [bs, N]
            eta_str = (d1_norm / x0_norm).mean().item()
            self.eta_str_history.append(eta_str)

            # 方向 A 诊断: per-dim eta_str
            # D1, x0_n 形状 [bs, N, 4], 最后一维 = (cx, cy, w, h)
            # 用 abs 而非 norm, 因为单维度是标量; 用 clamp 避免除零
            d1_abs = D1.abs()  # [bs, N, 4]
            x0_abs = x0_n.abs().clamp(min=1e-6)  # [bs, N, 4]
            eta_per_dim = (d1_abs / x0_abs).mean(dim=(0, 1)).tolist()  # len=4
            self.eta_str_per_dim_history.append(eta_per_dim)

        if t_next > 1e-7:
            phi1 = t_next * math.log(t_n / t_next) - t_n + t_next
        else:
            phi1 = -t_n

        correction = phi1 * D1

        # 方向 D 诊断: 默认 eta_3rd = 0.0 (未计算 3 阶)
        eta_3rd = 0.0
        if self.solver_order >= 3 and len(self.x0_history) >= 3:
            x0_pp = self.x0_history[-3]
            t_pp = self.t_history[-3]
            D1_p = (x0_p - x0_pp) / (t_p - t_pp)
            D2 = (D1 - D1_p) / (t_n - t_pp)

            # 方向 D 诊断: 记录 ||D2|| / ||x0|| (batch 均值)
            with torch.no_grad():
                d2_norm = D2.norm(dim=-1)  # [bs, N]
                eta_3rd = (d2_norm / x0_norm).mean().item()

            if t_next > 1e-7:
                phi2 = (t_next + t_p) * (t_n - t_next) - t_next * (
                    t_n + t_p
                ) * math.log(t_n / t_next)
            else:
                phi2 = t_p * t_n

            correction = correction + phi2 * D2

        # 方向 D 诊断: 无论是否计算 3 阶都记录 (0.0 表示该 step 未用 3 阶)
        self.eta_3rd_history.append(eta_3rd)

        return linear + correction


class RFDPMSolverAdaptive(RFDPMSolverMultistep):
    """方向 D: 自适应阶次 DPM-Solver++

    在不同 step 使用不同阶次, 减少高阶校正项的无效计算。

    两种模式:
      - ``static``: 前 ``num_3rd_steps`` 步用 3 阶, 其余用 2 阶
        假设 (R1 数据支持): 早期 step (t 大) 曲率最高, 需要 3 阶校正;
        后期 step (t 小) 曲率低, 2 阶足够。
      - ``eta_threshold``: 在线计算 ||D2||/||x0||, 超阈值则用 3 阶, 否则降为 2 阶
        阈值通过 ``eta_3rd_threshold`` 指定 (典型值 0.1~1.0)

    注意:
      - eta_threshold 模式下, D2 始终计算 (用于阈值判断), 但仅在超阈值时应用。
        因此该模式不省 NFE (网络前向次数), 但通过跳过 phi2*D2 校正项影响数值。
      - static 模式下, D2 仅在 3 阶步计算, 2 阶步不计算 D2, 真正节省计算。
      - 推理时改动, 不需要重训练 (基于已训好的 A4 checkpoint 直接推理)。
    """

    def __init__(
        self,
        num_steps: int = 6,
        timesteps: Optional[list[float]] = None,
        adaptive_mode: str = 'static',
        num_3rd_steps: int = 2,
        eta_3rd_threshold: float = 0.5,
    ):
        # solver_order 设为 3, 让基类 step() 走 3 阶分支;
        # 子类 step() 根据模式决定是否实际应用 D2 校正项
        super().__init__(
            num_steps=num_steps,
            solver_order=3,
            timesteps=timesteps,
        )
        assert adaptive_mode in ('static', 'eta_threshold'), (
            f"adaptive_mode 必须是 'static' 或 'eta_threshold', got {adaptive_mode}"
        )
        self.adaptive_mode = adaptive_mode
        self.num_3rd_steps = num_3rd_steps
        self.eta_3rd_threshold = eta_3rd_threshold
        # 记录每个 step 是否实际应用了 3 阶校正 (供诊断脚本读取)
        self.applied_3rd_history: list[bool] = []

    def reset(self):
        super().reset()
        self.applied_3rd_history = []

    def step(
        self,
        x: torch.Tensor,
        x0_pred: torch.Tensor,
        t_n: float,
        step_idx: int,
    ) -> torch.Tensor:
        """自适应阶次单步积分。

        关键差异 vs 基类 step():
          - static 模式: step_idx < num_3rd_steps 时计算并应用 D2, 否则跳过 D2 计算
          - eta_threshold 模式: 始终计算 D2, 但仅在 ||D2||/||x0|| > threshold 时应用
        """
        t_next = self.timesteps[step_idx + 1]

        self.x0_history.append(x0_pred)
        self.t_history.append(t_n)
        if len(self.x0_history) > self.solver_order:
            self.x0_history.pop(0)
            self.t_history.pop(0)

        linear = (t_next / t_n) * x + (1.0 - t_next / t_n) * x0_pred

        if len(self.x0_history) < 2:
            # 历史不足, 仅 linear (与基类一致, 不记录诊断量)
            return linear

        x0_n = self.x0_history[-1]
        x0_p = self.x0_history[-2]
        t_p = self.t_history[-2]
        D1 = (x0_n - x0_p) / (t_n - t_p)

        with torch.no_grad():
            d1_norm = D1.norm(dim=-1)
            x0_norm = x0_n.norm(dim=-1).clamp(min=1e-6)
            eta_str = (d1_norm / x0_norm).mean().item()
            self.eta_str_history.append(eta_str)

            d1_abs = D1.abs()
            x0_abs = x0_n.abs().clamp(min=1e-6)
            eta_per_dim = (d1_abs / x0_abs).mean(dim=(0, 1)).tolist()
            self.eta_str_per_dim_history.append(eta_per_dim)

        if t_next > 1e-7:
            phi1 = t_next * math.log(t_n / t_next) - t_n + t_next
        else:
            phi1 = -t_n

        correction = phi1 * D1

        # 决定是否计算/应用 3 阶校正项
        # 3 阶需要 history >= 3, 即首次可应用 3 阶的 step_idx = 2
        # (step_idx=0 历史=[], step_idx=1 历史=[x0_n], 都无法算 D2)
        has_3rd_history = len(self.x0_history) >= 3
        if self.adaptive_mode == 'static':
            # static: 前 num_3rd_steps 个 "3 阶可应用 step" 用 3 阶
            # 3 阶可应用 step 的索引: step_idx=2 是第 0 个, step_idx=3 是第 1 个, ...
            # 即 step_idx - 2 < num_3rd_steps 等价于 step_idx < num_3rd_steps + 2
            # 例: num_3rd_steps=1 → 仅 step_idx=2 用 3 阶; num_3rd_steps=2 → step_idx=2,3 用 3 阶
            if has_3rd_history:
                n_3rd_capable_idx = step_idx - 2  # 0-indexed among 3rd-order-capable steps
                should_apply_3rd = n_3rd_capable_idx < self.num_3rd_steps
            else:
                should_apply_3rd = False
        else:
            # eta_threshold: 始终计算 D2 (用于阈值判断), 但仅在超阈值时应用
            should_apply_3rd = has_3rd_history

        eta_3rd = 0.0
        applied_3rd = False
        if has_3rd_history:
            x0_pp = self.x0_history[-3]
            t_pp = self.t_history[-3]
            D1_p = (x0_p - x0_pp) / (t_p - t_pp)
            D2 = (D1 - D1_p) / (t_n - t_pp)

            with torch.no_grad():
                d2_norm = D2.norm(dim=-1)
                eta_3rd = (d2_norm / x0_norm).mean().item()

            # eta_threshold 模式: 仅当超阈值才应用
            if self.adaptive_mode == 'eta_threshold':
                should_apply_3rd = (
                    eta_3rd > self.eta_3rd_threshold
                )

            if should_apply_3rd:
                if t_next > 1e-7:
                    phi2 = (t_next + t_p) * (t_n - t_next) - t_next * (
                        t_n + t_p
                    ) * math.log(t_n / t_next)
                else:
                    phi2 = t_p * t_n
                correction = correction + phi2 * D2
                applied_3rd = True

        self.eta_3rd_history.append(eta_3rd)
        self.applied_3rd_history.append(applied_3rd)

        return linear + correction


class RFDPMSolverPerDim(RFDPMSolverMultistep):
    """方向 A Phase 2: per-dim 阶数分配的 DPM-Solver++

    基于 per-dim eta_str 诊断 (方向 A Phase 1):
      - h 维度 (index 3) eta_str 4-11, 曲率最小
      - cx, cy 维度 (index 0, 1) eta_str 17-50, 曲率最大
      - w 维度 (index 2) eta_str 介于二者之间, 与 cx/cy 接近

    策略:
      - h 维度用 1 阶 (Euler/linear, 仅线性项)
      - cx, cy, w 维度用 2 阶 (DPM-Solver++, linear + D1 校正)

    实现: 在 step() 中, linear 项对所有维度应用, correction 项仅对 dpm_dims 应用。
    bbox 最后一维 = (cx, cy, w, h), 即 index 0/1/2 = dpm, index 3 = euler。

    推理时改动, 不需要重训练 (基于已训好的 A4 checkpoint 直接推理)。
    """

    def __init__(
        self,
        num_steps: int = 6,
        timesteps: Optional[list[float]] = None,
        euler_dims: tuple = (3,),       # h 维度用 1 阶
        dpm_dims: tuple = (0, 1, 2),    # cx, cy, w 维度用 2 阶
    ):
        super().__init__(
            num_steps=num_steps,
            solver_order=2,
            timesteps=timesteps,
        )
        self.euler_dims = tuple(euler_dims)
        self.dpm_dims = tuple(dpm_dims)

    def step(
        self,
        x: torch.Tensor,
        x0_pred: torch.Tensor,
        t_n: float,
        step_idx: int,
    ) -> torch.Tensor:
        """per-dim 阶数分配单步积分。

        linear 项对所有维度应用 (1 阶 Euler);
        correction = phi1 * D1 仅对 dpm_dims 维度应用 (2 阶 DPM-Solver++)。
        """
        t_next = self.timesteps[step_idx + 1]

        self.x0_history.append(x0_pred)
        self.t_history.append(t_n)
        if len(self.x0_history) > self.solver_order:
            self.x0_history.pop(0)
            self.t_history.pop(0)

        linear = (t_next / t_n) * x + (1.0 - t_next / t_n) * x0_pred

        # 历史不足 2 步时, 所有维度退化为 1 阶 (linear)
        if len(self.x0_history) < 2:
            self.eta_str_history.append(0.0)
            self.eta_str_per_dim_history.append([0.0] * x.shape[-1])
            self.eta_3rd_history.append(0.0)
            return linear

        x0_n = self.x0_history[-1]
        x0_p = self.x0_history[-2]
        t_p = self.t_history[-2]
        D1 = (x0_n - x0_p) / (t_n - t_p)

        # 诊断 (与基类一致)
        with torch.no_grad():
            d1_norm = D1.norm(dim=-1)
            x0_norm = x0_n.norm(dim=-1).clamp(min=1e-6)
            eta_str = (d1_norm / x0_norm).mean().item()
            self.eta_str_history.append(eta_str)

            d1_abs = D1.abs()
            x0_abs = x0_n.abs().clamp(min=1e-6)
            eta_per_dim = (d1_abs / x0_abs).mean(dim=(0, 1)).tolist()
            self.eta_str_per_dim_history.append(eta_per_dim)

        if t_next > 1e-7:
            phi1 = t_next * math.log(t_n / t_next) - t_n + t_next
        else:
            phi1 = -t_n

        correction = phi1 * D1

        # per-dim 阶数分配: 仅 dpm_dims 维度应用 correction
        dim_mask = torch.zeros(
            x.shape[-1], device=x.device, dtype=x.dtype
        )
        for d in self.dpm_dims:
            dim_mask[d] = 1.0

        # eta_3rd = 0.0 (per-dim solver 不使用 3 阶)
        self.eta_3rd_history.append(0.0)

        return linear + correction * dim_mask
