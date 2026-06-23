"""尺度条件化 Rectified Flow — 方向四: 流匹配的非线性轨迹

突破 1-RectFlow 直线路径对尺度不敏感的局限, 引入尺度调制噪声调度,
使小目标在更早的 t 处去噪, 提升小目标检测和少步采样精度.

数学推导:
    标准直线 RF:  x_t = (1-t) x_0 + t x_1
    尺度条件化:   x_t = α(t,s) x_0 + σ(t,s) x_1
    其中 s = sqrt(area(x_0)), κ(s) = 1 + λ (s_max - s) / s_max
    α(t,s) = 1 - t^{1/κ(s)},  σ(t,s) = t^{1/κ(s)}

性质:
- s = s_max (大目标): κ = 1, 退化为标准线性 RF
- s → 0 (小目标): κ > 1, t^{1/κ} 增长更慢, α 更快接近 1 (更早去噪)

速度场:
    v(t,s) = (t^{1/κ-1} / κ) (x_1 - x_0)
    非常数, 采样时需数值积分.

若方向四废弃, 删除本文件 + tests/unit/test_nonlinear_trajectory.py 即可回滚.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


class ScaleConditionedRF:
    """尺度条件化 Rectified Flow.

    小目标 (s 小) 用更陡的噪声调度, 使其在更早的 t 处去噪.

    Args:
        lambda_mod: 调制强度 (0=标准 RF, 越大尺度差异越显著)
        s_max: 参考最大尺度 (归一化面积平方根, 默认 0.15)
        snr_scale: 与 RectifiedFlow 一致的 SNR 缩放系数 (用于 x_start 缩放)
    """

    def __init__(
        self,
        lambda_mod: float = 0.5,
        s_max: float = 0.15,
        snr_scale: float = 2.0,
    ):
        self.lambda_mod = float(lambda_mod)
        self.s_max = float(s_max)
        self.snr_scale = snr_scale

    # ──────────────────────────────────────────
    # 核心调度函数
    # ──────────────────────────────────────────

    def compute_kappa(self, scales: Tensor) -> Tensor:
        """计算尺度调制系数 κ(s) = 1 + λ (s_max - s) / s_max.

        Args:
            scales: 任意形状, 每个框的尺度 sqrt(area)

        Returns:
            kappa: 同形状, 范围 [1, 1+λ]
        """
        s_max = max(self.s_max, 1e-6)
        return 1.0 + self.lambda_mod * (s_max - scales) / s_max

    def compute_t_eff(self, t: Tensor, scales: Tensor) -> Tensor:
        """计算有效时间 t_eff = t^{1/κ(s)}.

        Args:
            t: [bs] 或 [bs, N] 原始时间
            scales: [bs, N] 框尺度

        Returns:
            t_eff: [bs, N] 有效时间
        """
        # 广播 t 到 scales 的形状
        if t.dim() == 1 and scales.dim() == 2:
            t = t.unsqueeze(1).expand_as(scales)
        kappa = self.compute_kappa(scales)
        # clamp t 避免数值问题
        t_safe = t.clamp(min=0.0, max=1.0)
        return t_safe.pow(1.0 / kappa)

    # ──────────────────────────────────────────
    # 前向加噪
    # ──────────────────────────────────────────

    def q_sample(
        self,
        x_start: Tensor,
        x_noise: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        scales: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """前向加噪: x_t = α(t,s) x_0 + σ(t,s) x_1.

        Args:
            x_start: [bs, N, D] 数据 (GT 框)
            x_noise: [bs, N, D] 噪声, 若 None 则随机
            t: [bs] 或 [bs, N] 时间, 若 None 则随机
            scales: [bs, N] 框尺度, 若 None 则用 s_max (退化为标准 RF)

        Returns:
            x_t: 加噪样本 [bs, N, D]
            velocity: 目标速度 (x_noise - x_start) [bs, N, D]
            t_eff: 有效时间 [bs, N] (用于损失加权)
        """
        if x_noise is None:
            x_noise = torch.randn_like(x_start)
        bs, N, D = x_start.shape
        device = x_start.device

        if t is None:
            t = torch.rand((bs,), device=device)
        if scales is None:
            # 退化为标准 RF: κ = 1+λ*(s_max-s_max)/s_max = 1
            scales = torch.full((bs, N), self.s_max, device=device)

        # 计算 t_eff: [bs, N]
        t_eff = self.compute_t_eff(t, scales)

        # 广播到 [bs, N, D]
        t_view = t_eff.unsqueeze(-1)
        x_t = (1.0 - t_view) * x_start + t_view * x_noise
        velocity = x_noise - x_start
        return x_t, velocity, t_eff

    # ──────────────────────────────────────────
    # 速度场与采样
    # ──────────────────────────────────────────

    def get_velocity(
        self,
        x_t: Tensor,
        x0_pred: Tensor,
        t: Tensor,
        scales: Tensor,
    ) -> Tensor:
        """计算速度场 v_t.

        v_t = (x_t - x0_pred) / t_eff  (近似, 在 t_eff 空间)

        Args:
            x_t: [bs, N, D]
            x0_pred: [bs, N, D]
            t: [bs] 或 [bs, N]
            scales: [bs, N]

        Returns:
            velocity: [bs, N, D]
        """
        t_eff = self.compute_t_eff(t, scales)
        t_view = t_eff.unsqueeze(-1).clamp(min=1e-5)
        return (x_t - x0_pred) / t_view

    def step(
        self,
        x_t: Tensor,
        x0_pred: Tensor,
        t_curr,
        t_next,
        scales: Tensor,
    ) -> Tensor:
        """Euler step with scale-conditioned schedule.

        在 t_eff 空间做 Euler 积分:
            x_next = x_t + dt_eff * v_t
            v_t = (x_t - x0_pred) / t_eff_curr

        Args:
            x_t: [bs, N, D]
            x0_pred: [bs, N, D]
            t_curr: 当前时间 (float 或 [bs, N] tensor)
            t_next: 下一时间 (float 或 [bs, N] tensor)
            scales: [bs, N]

        Returns:
            x_next: [bs, N, D]
        """
        device = x_t.device
        # 统一 t_curr/t_next 为 [bs, N] tensor
        if isinstance(t_curr, (int, float)):
            t_curr = torch.full_like(scales, float(t_curr))
        if isinstance(t_next, (int, float)):
            t_next = torch.full_like(scales, float(t_next))

        t_eff_curr = self.compute_t_eff(t_curr, scales)
        t_eff_next = self.compute_t_eff(t_next, scales)

        # 速度: v = (x_t - x0_pred) / t_eff_curr
        v = (x_t - x0_pred) / t_eff_curr.unsqueeze(-1).clamp(min=1e-5)
        # Euler 步长 (在 t_eff 空间)
        dt_eff = (t_eff_next - t_eff_curr).unsqueeze(-1)
        return x_t + dt_eff * v

    # ──────────────────────────────────────────
    # 辅助: 从框计算尺度
    # ──────────────────────────────────────────

    @staticmethod
    def compute_scales_from_boxes(boxes: Tensor) -> Tensor:
        """从框计算尺度 s = sqrt(area).

        Args:
            boxes: [..., 4] xyxy 格式框 (归一化坐标)

        Returns:
            scales: [...] 每个框的尺度
        """
        # xyxy → wh
        w = boxes[..., 2] - boxes[..., 0]
        h = boxes[..., 3] - boxes[..., 1]
        w = w.clamp(min=0.0)
        h = h.clamp(min=0.0)
        return (w * h).sqrt()

    @staticmethod
    def compute_scales_from_cxcywh(boxes: Tensor) -> Tensor:
        """从 cxcywh 格式框计算尺度.

        Args:
            boxes: [..., 4] cxcywh 格式 (归一化坐标)

        Returns:
            scales: [...] 每个框的尺度
        """
        w = boxes[..., 2].clamp(min=0.0)
        h = boxes[..., 3].clamp(min=0.0)
        return (w * h).sqrt()
