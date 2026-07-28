"""Anderson Acceleration mixing module for cascade head acceleration.

基于 Walker-Ni (2011) type-I 形式, 有限内存 m=2, 跨 proposal 全局最小二乘。

数学原理 (详见 docs/research/proposals/AAC_DESIGN.md §1):
  - cascade head 形式化为不动点迭代: x_{k+1} = G_k(x_k)
  - 残差 f_k = G_k(x_k) - x_k
  - Anderson 加速: 利用最近 m 步历史构造全局组合, 求解小型最小二乘
  - 更新公式: x_{k+1} = x_k + β [f_k - (ΔX + ΔF) γ]
  - γ = (ΔF^T ΔF + λI)^{-1} ΔF^T f_k  (m×m 线性系统)

设计要点:
  - m=2 (有限内存, 避免在 d=4 空间退化为秩亏)
  - 全局 γ: 跨所有 N 个 proposal 共享, 系统矩阵 [N·d, m] 满秩
  - 阻尼 β: 训练初期 β=0.5, 后期 β=1.0 (Henderson-Varadhan 2019)
  - 正则化 λ: ΔF^T ΔF + λI, 避免 rank-deficient (数值稳定)
  - stop-gradient on history: 仅当前 f_k 参与反传 (Phantom gradient)

形状约定:
  x: [bs, N, d]  (cascade head 输出空间, d=4 for xyxy)
  f = g(x) - x: [bs, N, d]  (残差)
"""

from typing import List

import torch
import torch.nn as nn

from ldmdet.diagnostics.instrumentation import probe


class AndersonMixing(nn.Module):
    """有限内存 Anderson 加速 (type-I), 跨 proposal 全局最小二乘。

    Args:
        mem_depth: Anderson 内存深度 m (默认 2), m=0 退化为 Picard
        damping_beta: 阻尼系数 β (默认 1.0), 训练初期用 0.5
        reg_lambda: Tikhonov 正则化 λ (默认 1e-6)
        stop_grad_history: 历史项是否 stop-gradient (默认 True, Phantom gradient)
        gamma_norm_clip: Anderson 系数 γ 的范数上界 (默认 10.0), 防止爆炸
    """

    def __init__(
        self,
        mem_depth: int = 2,
        damping_beta: float = 1.0,
        reg_lambda: float = 1e-6,
        stop_grad_history: bool = True,
        gamma_norm_clip: float = 10.0,
    ):
        super().__init__()
        self.m = mem_depth
        self.beta = damping_beta
        self.lam = reg_lambda
        self.stop_grad_history = stop_grad_history
        self.gamma_norm_clip = gamma_norm_clip
        # 历史 buffer (非 Parameter, 不进入 optimizer)
        # 存储 x_k 和 f_k = G_k(x_k) - x_k, 用于构造 ΔX, ΔF
        self._x_history: List[torch.Tensor] = []
        self._f_history: List[torch.Tensor] = []
        # 诊断: 记录每步的 Anderson 系数 γ (供探针读取)
        self._last_gamma: torch.Tensor | None = None

    def reset(self):
        """每个 cascade 周期 (每个 solver step) 开始时调用。

        AAC 历史不跨 solver step 累积 (与 DPM-Solver++ 的 x0_history 独立)。
        """
        self._x_history.clear()
        self._f_history.clear()
        self._last_gamma = None

    def forward(
        self,
        x_curr: torch.Tensor,   # [bs, N, d] 当前 iterate x_k
        g_x: torch.Tensor,      # [bs, N, d] G_k(x_curr) 即 head 输出
    ) -> torch.Tensor:
        """Anderson 加速一步。

        Args:
            x_curr: 当前 iterate x_k
            g_x: head 输出 G_k(x_k)
        Returns:
            x_next: 加速后的 x_{k+1}

        数学:
            f_k = g_x - x_curr  (残差)
            若历史不足 (k < 1): 退化为 Picard, x_{k+1} = g_x
            否则:
                ΔF = [f_k - f_{k-1}, ...]  跨 proposal 全局
                ΔX = [x_k - x_{k-1}, ...]
                γ = (ΔF^T ΔF + λI)^{-1} ΔF^T f_k
                x_{k+1} = x_k + β [f_k - (ΔX + ΔF) γ]
        """
        f_curr = g_x - x_curr  # [bs, N, d] 残差

        # m=0 或历史不足: 退化为 Picard (与当前 cascade 一致)
        if self.m == 0 or len(self._f_history) < 1:
            self._push_history(x_curr, f_curr)
            return g_x  # x_{k+1} = G_k(x_k) = Picard

        m_k = min(self.m, len(self._f_history))

        # 构造 ΔF, ΔX (跨 batch+proposal+dim 全局, 系统矩阵 [bs·N·d, m_k])
        # 历史项 stop-gradient (Phantom gradient, 仅当前 f_k 反传)
        delta_F_list = []
        delta_X_list = []
        for i in range(m_k):
            f_prev = self._f_history[-(i + 1)]   # f_{k-1}, f_{k-2}, ...
            x_prev = self._x_history[-(i + 1)]   # x_{k-1}, x_{k-2}, ...
            if self.stop_grad_history:
                f_prev = f_prev.detach()
                x_prev = x_prev.detach()
            # 展平为 [bs*N*d] (全局向量), 作为系统矩阵的一行 (row-form 约定, 见设计文档 §1.4)
            delta_F_list.append((f_curr - f_prev).reshape(-1))
            delta_X_list.append((x_curr - x_prev).reshape(-1))

        # [m_k, bs*N*d] — 每行是一个历史差分 (全局向量)
        Delta_F = torch.stack(delta_F_list, dim=0)  # [m_k, bs*N*d]
        Delta_X = torch.stack(delta_X_list, dim=0)  # [m_k, bs*N*d]
        f_flat = f_curr.reshape(-1)                  # [bs*N*d]

        # 最小二乘: γ = (ΔF ΔF^T + λI)^{-1} ΔF f_k
        # 注意: ΔF 是 [m_k, D], ΔF ΔF^T 是 [m_k, m_k] (小矩阵, m_k ≤ 2)
        gram = Delta_F @ Delta_F.t()  # [m_k, m_k]
        gram = gram + self.lam * torch.eye(
            m_k, device=gram.device, dtype=gram.dtype
        )
        rhs = Delta_F @ f_flat.unsqueeze(-1)  # [m_k, 1]
        gamma = torch.linalg.solve(gram, rhs)  # [m_k, 1]

        # γ 范数裁剪 (防止爆炸, Henderson-Varadhan 2019 风险缓解)
        gamma_norm = gamma.norm()
        if gamma_norm > self.gamma_norm_clip:
            gamma = gamma * (self.gamma_norm_clip / gamma_norm.detach())

        self._last_gamma = gamma.detach()

        # 加速更新: x_{k+1} = x_k + β [f_k - (ΔX + ΔF)^T γ]
        # (ΔX + ΔF)^T γ: [D, m_k] @ [m_k, 1] → [D, 1] → [D]
        correction = (Delta_X + Delta_F).t() @ gamma  # [bs*N*d, 1]
        correction = correction.squeeze(-1).reshape_as(x_curr)  # [bs, N, d]

        x_next = x_curr + self.beta * (f_curr - correction)

        # 诊断探针: Anderson 系数统计 (训练时每 100 步)
        if self.training:
            probe.record_scalar(
                'aac/gamma_norm', gamma.norm().item()
            )
            if m_k >= 1:
                probe.record_scalar(
                    'aac/gamma_0', gamma[0].abs().mean().item()
                )
            if m_k >= 2:
                probe.record_scalar(
                    'aac/gamma_1', gamma[1].abs().mean().item()
                )
            # 残差范数衰减 (验证 Anderson 加速效果)
            probe.record_scalar(
                'aac/f_curr_norm', f_curr.norm(dim=-1).mean().item()
            )

        self._push_history(x_curr, f_curr)
        return x_next

    def _push_history(self, x: torch.Tensor, f: torch.Tensor):
        """存储历史 (detach 以避免保留计算图, 除非 stop_grad_history=False)."""
        if self.stop_grad_history:
            self._x_history.append(x.detach())
            self._f_history.append(f.detach())
        else:
            self._x_history.append(x)
            self._f_history.append(f)
        # 保留最近 m+1 个历史 (m 个差分需要 m+1 个点)
        if len(self._x_history) > self.m + 1:
            self._x_history.pop(0)
            self._f_history.pop(0)
