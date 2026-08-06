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

形状约定 (行式, 见方案 §1.4 注):
  x: [bs, N, d]  (cascade head 输出空间, d=4 for xyxy)
  f = g(x) - x: [bs, N, d]  (残差)
  展平后: ΔF, ΔX ∈ [m_k, bs·N·d]  (每行为一个历史差分, 全局向量)
  Gram 矩阵: ΔF ΔF^T ∈ [m_k, m_k]  (小矩阵, m_k ≤ 2)
  注: 方案 §1.4 数学公式用列式 [bs·N·d, m_k], 代码用行式 (转置), 两者数学等价。
"""

from typing import List, Optional, Tuple

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
        self._last_gamma: Optional[torch.Tensor] = None

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
                γ = (ΔF ΔF^T + λI)^{-1} ΔF f_k
                x_{k+1} = x_k + β [f_k - (ΔX + ΔF)^T γ]
        """
        f_curr = g_x - x_curr  # [bs, N, d] 残差

        # m=0 或历史不足 (k=0): 退化为 Picard (与当前 cascade 一致)
        if self.m == 0 or len(self._f_history) < 1:
            self._push_history(x_curr, f_curr)
            return g_x  # x_{k+1} = G_k(x_k) = Picard

        m_k = min(self.m, len(self._f_history))

        # 构造差分矩阵并求解 Anderson 系数
        Delta_F, Delta_X, f_flat = self._build_difference_matrices(
            x_curr, f_curr, m_k
        )
        gamma = self._solve_gamma(Delta_F, f_flat, m_k)

        # 加速更新: x_{k+1} = x_k + β [f_k - (ΔX + ΔF)^T γ]
        # (ΔX + ΔF)^T γ: [D, m_k] @ [m_k, 1] → [D, 1] → reshape → [bs, N, d]
        correction = (Delta_X + Delta_F).t() @ gamma  # [bs*N*d, 1]
        correction = correction.squeeze(-1).reshape_as(x_curr)  # [bs, N, d]
        x_next = x_curr + self.beta * (f_curr - correction)

        # 诊断探针 (训练时每 100 步, 见 probe.record_scalar 的频率控制)
        if self.training:
            self._record_diagnostics(gamma, f_curr, m_k)

        self._push_history(x_curr, f_curr)
        return x_next

    def _build_difference_matrices(
        self,
        x_curr: torch.Tensor,
        f_curr: torch.Tensor,
        m_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构造差分矩阵 ΔF, ΔX (行式, 跨 proposal 全局展平)。

        方案 §1.4: ΔF 的每行是一个历史残差差分 (f_curr - f_prev) 展平为全局向量,
        系统矩阵行式 [m_k, bs·N·d] (与列式 [bs·N·d, m_k] 数学等价, 见方案 §1.4 注)。

        历史项 stop-gradient (Phantom gradient): 仅当前 f_curr 参与反传,
        历史差分通过 .detach() 切断计算图, 避免跨 head 展开的 O(H²) 内存开销。

        Args:
            x_curr: [bs, N, d] 当前 iterate x_k
            f_curr: [bs, N, d] 当前残差 f_k = G_k(x_k) - x_k
            m_k: 实际使用的历史深度 min(m, len(history))
        Returns:
            Delta_F: [m_k, bs·N·d] 残差差分矩阵 (每行一个历史差分)
            Delta_X: [m_k, bs·N·d] 迭代差分矩阵
            f_flat: [bs·N·d] 当前残差展平 (最小二乘的右端项)
        """
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

        Delta_F = torch.stack(delta_F_list, dim=0)  # [m_k, bs*N*d]
        Delta_X = torch.stack(delta_X_list, dim=0)  # [m_k, bs*N*d]
        f_flat = f_curr.reshape(-1)                  # [bs*N*d]
        return Delta_F, Delta_X, f_flat

    def _solve_gamma(
        self,
        Delta_F: torch.Tensor,
        f_flat: torch.Tensor,
        m_k: int,
    ) -> torch.Tensor:
        """求解 Anderson 系数 γ (含 Tikhonov 正则化与范数裁剪)。

        最小二乘: γ = argmin ||f_k - ΔF^T γ||² + λ||γ||²
        正规方程 (行式): (ΔF ΔF^T + λI) γ = ΔF f_k  (m_k × m_k 系统)

        Args:
            Delta_F: [m_k, D] 残差差分矩阵 (行式)
            f_flat: [D] 当前残差展平
            m_k: 系统维度
        Returns:
            gamma: [m_k, 1] Anderson 系数 (已裁剪)
        """
        # Gram 矩阵: ΔF ΔF^T [m_k, m_k] (小矩阵, m_k ≤ 2, 求解代价可忽略)
        gram = Delta_F @ Delta_F.t()
        gram = gram + self.lam * torch.eye(
            m_k, device=gram.device, dtype=gram.dtype
        )
        rhs = Delta_F @ f_flat.unsqueeze(-1)  # [m_k, 1]
        gamma = torch.linalg.solve(gram, rhs)  # [m_k, 1]

        # γ 范数裁剪 (防止爆炸, Henderson-Varadhan 2019 风险缓解)
        # 缩放因子用 .detach() 避免反传通过裁剪操作
        gamma_norm = gamma.norm()
        if gamma_norm > self.gamma_norm_clip:
            gamma = gamma * (self.gamma_norm_clip / gamma_norm.detach())

        self._last_gamma = gamma.detach()
        return gamma

    def _record_diagnostics(
        self,
        gamma: torch.Tensor,
        f_curr: torch.Tensor,
        m_k: int,
    ) -> None:
        """记录 Anderson 系数与残差统计到诊断探针。

        仅在 self.training=True 时调用 (由 forward 控制)。
        探针内部按 train_interval 频率实际采集 (默认 100 步), 未到频率时为 no-op。

        Args:
            gamma: [m_k, 1] Anderson 系数 (已裁剪)
            f_curr: [bs, N, d] 当前残差
            m_k: 实际历史深度
        """
        probe.record_scalar('aac/gamma_norm', gamma.norm().item())
        if m_k >= 1:
            probe.record_scalar('aac/gamma_0', gamma[0].abs().mean().item())
        if m_k >= 2:
            probe.record_scalar('aac/gamma_1', gamma[1].abs().mean().item())
        # 残差范数衰减 (验证 Anderson 加速效果)
        probe.record_scalar('aac/f_curr_norm', f_curr.norm(dim=-1).mean().item())

    def _push_history(self, x: torch.Tensor, f: torch.Tensor):
        """存储历史 (detach 以避免保留计算图, 除非 stop_grad_history=False)。

        保留最近 m+1 个历史 (m 个差分需要 m+1 个点)。
        注: m=0 时仍存储 1 个历史点 (上限 m+1=1), 但 m=0 的 forward
        始终走 Picard 分支, 不会读取历史, 此存储无副作用。
        """
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
