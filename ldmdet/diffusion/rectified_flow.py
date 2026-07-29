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

    def __init__(
        self,
        snr_scale: float = 2.0,
        # ReFlow (Standard MSE): 标记用参数, 不改变 q_sample 行为
        # use_reflow_coupling=True 时, head._build_training_targets 从预存 coupling
        # 加载 x_start(=x_0^pred) 和 noise, 跳过在线 OT (详见 REFLOW_HEAD_DISTILL_IMPL_PLAN.md §1)
        use_reflow_coupling: bool = False,
        reflow_dims: str = 'all',
    ):
        self.snr_scale = snr_scale
        self.use_reflow_coupling = use_reflow_coupling
        # reflow_dims: 'all' (全维度拉直) | 'cxcy' (仅 cx/cy 拉直, w/h 仍用 GT)
        # 方向 A 诊断: h 维度曲率小, w 维度差距小, cx/cy 曲率最大
        assert reflow_dims in ('all', 'cxcy'), \
            f"reflow_dims 必须是 'all' 或 'cxcy', got {reflow_dims}"
        self.reflow_dims = reflow_dims

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
        dim_d1_mask: Optional[torch.Tensor] = None,
    ):
        self.num_steps = num_steps
        self.solver_order = solver_order
        self.dim_d1_mask = dim_d1_mask  # [4] mask: True=保留 D1, False=置零
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
        renewal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """单步积分 x(t_n) → x(t_{n+1})

        Args:
            x: [bs, N, 4] 当前状态
            x0_pred: [bs, N, 4] 当前步的 x0 预测
            t_n: 当前时间步
            step_idx: 步索引
            renewal_mask: [bs, N] bool, True 表示该 proposal 在上一步被 box_renewal 重置
                          对被 renewal 的 proposal 置零 D1 校正项 (路径 A),
                          避免 renewal 噪声污染 x0_history 导致 D1 失效
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

        # D3 化解路径 A: 对被 renewal 的 proposal 置零 D1 校正项
        # renewal_mask=True 的 proposal 的 x0_history 跨越了 renewal 断点,
        # D1 不再反映真实轨迹曲率而是 renewal 噪声, 必须屏蔽
        if renewal_mask is not None:
            # renewal_mask: [bs, N] → [bs, N, 1] for broadcast with [bs, N, 4]
            D1 = D1 * (~renewal_mask).unsqueeze(-1).float()

        # 分维度 D1 调制: 在 cxcywh 空间中, dim 0,1 (cx,cy) 曲率大,
        # 保留 DPM++ D1 校正; dim 2,3 (w,h) 曲率小, 可选择性置零 D1
        # 退化为 Euler 以避免可能的过冲
        if self.dim_d1_mask is not None:
            # dim_d1_mask: [4] bool/float, True=保留 D1, False=置零 D1
            D1 = D1 * self.dim_d1_mask.to(D1.device)

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


class RFDPMSolverHybrid(RFDPMSolverMultistep):
    """混合求解器: cx/cy 用 DPM-Solver++ 2阶, w/h 用 Heun 2阶。

    动机 (2026-07-29 per-dim 实验):
      dim_d1_mask=[1,1,0,0] 使 w/h 退化为 Euler 1阶 (仅 linear 项). 但 Heun 是
      真正的 2阶求解器 (速度梯形), 机制不同于 DPM++ (x0 插值). 本类测试 w/h 用
      Heun 2阶是否优于 Euler 1阶.

    三种 2阶机制对比:
      - DPM++ (x0 插值): linear + φ₁·D1, D1=(x0_n-x0_p)/(t_n-t_p), 1 NFE (复用历史)
      - Heun  (速度梯形): x + (dt/2)(v_t+v_next), v=(x-x0)/t, 2 NFE (额外前向)
      - Euler (1阶):     x + dt·v_t = linear, 1 NFE

    对低曲率维度 (w/h eta_str≈0.3-0.9, 近直线), Euler≈Heun (恒定速度下梯形=前向).
    本类实证验证这一理论推断。

    NFE: 2×num_steps (Heun 校正项需额外前向), vs DPM++ 的 1×num_steps。

    推理时改动, 不需重训练 (基于 A4 checkpoint 直接推理)。
    """

    def __init__(
        self,
        num_steps: int = 6,
        timesteps: Optional[list[float]] = None,
        dpm_dims: tuple = (0, 1),    # cx, cy 用 DPM-Solver++ 2阶
        heun_dims: tuple = (2, 3),    # w, h 用 Heun 2阶
    ):
        super().__init__(
            num_steps=num_steps,
            solver_order=2,
            timesteps=timesteps,
        )
        self.dpm_dims = tuple(dpm_dims)
        self.heun_dims = tuple(heun_dims)
        # model_fn 由 predict() 在 step 循环前注入 (闭包捕获 features/img_metas)
        # 签名: model_fn(x_tmp, t_tmp) -> (x0_pred, None)
        self.model_fn = None
        # v_next 范数诊断 (实证 Heun 数值不稳定根因):
        # v_next = (x_euler - x0_next) / t_next, t_next 小时分母小可能爆炸.
        # 每步记录: (t_n, t_next, ||v_t||, ||v_next||, ratio=||v_next||/||v_t||)
        self.v_next_diag_history: list[dict] = []

    def step(
        self,
        x: torch.Tensor,
        x0_pred: torch.Tensor,
        t_n: float,
        step_idx: int,
        renewal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """混合单步积分: dpm_dims 用 DPM++ (linear+D1), heun_dims 用 Heun (梯形).

        last step (t_next≈0) 时 Heun 的 v_next 不稳定 (除以 ~0), 全维度回退 linear。
        """
        t_next = self.timesteps[step_idx + 1]

        # === 共享: x0 历史 (供 DPM++ D1 使用) ===
        self.x0_history.append(x0_pred)
        self.t_history.append(t_n)
        if len(self.x0_history) > self.solver_order:
            self.x0_history.pop(0)
            self.t_history.pop(0)

        # linear = RF Euler step (所有维度共用, 也是 Heun 的预测项)
        linear = (t_next / t_n) * x + (1.0 - t_next / t_n) * x0_pred

        # === DPM++ 部分 (dpm_dims): linear + φ₁·D1 ===
        dpm_correction = torch.zeros_like(x)
        if len(self.x0_history) >= 2:
            x0_n = self.x0_history[-1]
            x0_p = self.x0_history[-2]
            t_p = self.t_history[-2]
            D1 = (x0_n - x0_p) / (t_n - t_p)

            if renewal_mask is not None:
                D1 = D1 * (~renewal_mask).unsqueeze(-1).float()

            if t_next > 1e-7:
                phi1 = t_next * math.log(t_n / t_next) - t_n + t_next
            else:
                phi1 = -t_n
            dpm_correction = phi1 * D1

            # 诊断: eta_str (全维度, 显示曲率; dpm_dims 会应用, heun_dims 不会)
            with torch.no_grad():
                d1_norm = D1.norm(dim=-1)
                x0_norm = x0_n.norm(dim=-1).clamp(min=1e-6)
                eta_str = (d1_norm / x0_norm).mean().item()
                self.eta_str_history.append(eta_str)
                d1_abs = D1.abs()
                x0_abs = x0_n.abs().clamp(min=1e-6)
                eta_per_dim = (d1_abs / x0_abs).mean(dim=(0, 1)).tolist()
                self.eta_str_per_dim_history.append(eta_per_dim)
        else:
            self.eta_str_history.append(0.0)
            self.eta_str_per_dim_history.append([0.0] * x.shape[-1])

        self.eta_3rd_history.append(0.0)

        # === Heun 部分 (heun_dims): x + (dt/2)(v_t + v_next) ===
        # 仅当 t_next 足够大且 model_fn 可用时才做梯形校正, 否则回退 linear
        heun_result = linear
        if t_next > 1e-7 and self.model_fn is not None:
            v_t = (x - x0_pred) / max(t_n, 1e-5)       # [bs, N, 4]
            dt = t_next - t_n
            x_euler = x + dt * v_t                      # Euler 预测项
            x0_next, _ = self.model_fn(x_euler, t_next) # 额外前向 (2nd NFE)
            v_next = (x_euler - x0_next) / max(t_next, 1e-5)
            heun_result = x + (dt / 2.0) * (v_t + v_next)

            # v_next 范数诊断: 实证 Heun 数值不稳定根因
            # v_next = (x_euler - x0_next) / t_next, t_next 小 → 分母小 → v_next 可能爆炸
            with torch.no_grad():
                # heun_dims 上的范数 (仅 w/h, 排除 dpm_dims 干扰)
                vt_heun = v_t[..., self.heun_dims].norm(dim=-1)  # [bs, N]
                vn_heun = v_next[..., self.heun_dims].norm(dim=-1)
                ratio = (vn_heun / vt_heun.clamp(min=1e-8)).mean().item()
                self.v_next_diag_history.append({
                    'step_idx': step_idx,
                    't_n': round(t_n, 6),
                    't_next': round(t_next, 6),
                    'v_t_norm': round(float(vt_heun.mean()), 6),
                    'v_next_norm': round(float(vn_heun.mean()), 6),
                    'ratio_v_next_over_v_t': round(ratio, 4),
                })
        else:
            # t_next≈0, Heun 回退 linear (无 v_next 计算)
            self.v_next_diag_history.append({
                'step_idx': step_idx,
                't_n': round(t_n, 6),
                't_next': round(t_next, 6),
                'v_t_norm': 0.0,
                'v_next_norm': 0.0,
                'ratio_v_next_over_v_t': 0.0,
                'note': 't_next≈0, Heun 回退 linear',
            })

        # === 按维度合并 ===
        result = linear.clone()
        for d in self.dpm_dims:
            result[..., d] = linear[..., d] + dpm_correction[..., d]
        for d in self.heun_dims:
            result[..., d] = heun_result[..., d]

        return result
