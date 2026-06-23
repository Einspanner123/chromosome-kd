"""Mini-batch OT 流匹配 — 方向四: 流匹配的非线性轨迹

训练时每个 batch 内动态计算 OT 耦合, 而非固定 (x_0, x_1) 配对.
速度目标为 OT 配对后的 (x_1 - x_0).

与 GHSS 的关系:
- GHSS 是训练前的静态 OT 耦合 (确定 x_0 → x_1 配对)
- Mini-batch OT 是训练中的动态耦合 (每个 batch 重新配对)
- 两者互补: GHSS 提供组内结构先验, mini-batch OT 提供全局最优传输

若方向四废弃, 删除本文件 + tests/unit/test_nonlinear_trajectory.py 即可回滚.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from ldmdet.coupling._sinkhorn_ops import sinkhorn_transport


class OTFlowMatching:
    """Mini-batch OT 流匹配.

    训练时每个 batch 内动态计算 OT 耦合, 而非固定 (x_0, x_1) 配对.
    速度目标为 OT 配对后的 (x_1 - x_0).

    Args:
        epsilon: Sinkhorn 熵正则化 (越大→越均匀, 越小→越接近硬 OT)
        num_iters: Sinkhorn 迭代次数
        ot_every_step: 是否每步重算 OT (True=动态, False=仅初始)
        coupling_mode: 耦合采样模式
            - 'argmax': 确定性 argmax (默认)
            - 'multinomial': 随机采样 (允许探索)
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        num_iters: int = 10,
        ot_every_step: bool = False,
        coupling_mode: str = 'argmax',
    ):
        self.epsilon = float(epsilon)
        self.num_iters = int(num_iters)
        self.ot_every_step = bool(ot_every_step)
        if coupling_mode not in ('argmax', 'multinomial'):
            raise ValueError(
                f"coupling_mode 必须为 'argmax' 或 'multinomial', got {coupling_mode}"
            )
        self.coupling_mode = coupling_mode

    # ──────────────────────────────────────────
    # OT 耦合计算
    # ──────────────────────────────────────────

    def compute_ot_coupling(
        self,
        x_start: Tensor,
        x_noise: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """计算 mini-batch OT 耦合.

        Args:
            x_start: [N, D] 数据 (GT 框)
            x_noise: [N, D] 噪声

        Returns:
            x_start_coupled: [N, D] (与输入相同, x_start 不重排)
            x_noise_coupled: [N, D] 重排后的 x_noise
        """
        N = x_start.shape[0]
        if N <= 1:
            return x_start, x_noise

        # 代价矩阵: [N, N]
        cost = torch.cdist(x_start, x_noise, p=2)

        # Sinkhorn → 传输矩阵
        transport = sinkhorn_transport(
            cost, epsilon=self.epsilon, num_iters=self.num_iters
        )

        # 耦合索引
        if self.coupling_mode == 'argmax':
            coupled_idx = transport.argmax(dim=1)  # [N]
        else:  # multinomial
            # 每行按概率采样一个索引
            coupled_idx = torch.multinomial(
                transport.clamp_min(1e-10), 1
            ).squeeze(-1)  # [N]

        return x_start, x_noise[coupled_idx]

    # ──────────────────────────────────────────
    # 前向加噪
    # ──────────────────────────────────────────

    def q_sample(
        self,
        x_start: Tensor,
        x_noise: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """OT 耦合的前向加噪.

        Args:
            x_start: [bs, N, D]
            x_noise: [bs, N, D]
            t: [bs]

        Returns:
            x_t: 加噪样本 [bs, N, D]
            velocity: OT 耦合速度 [bs, N, D]
        """
        if x_noise is None:
            x_noise = torch.randn_like(x_start)
        bs = x_start.shape[0]
        device = x_start.device

        if t is None:
            t = torch.rand((bs,), device=device)

        # 逐 batch 计算 OT 耦合
        x_start_coupled_list = []
        x_noise_coupled_list = []
        for b in range(bs):
            s_c, n_c = self.compute_ot_coupling(x_start[b], x_noise[b])
            x_start_coupled_list.append(s_c)
            x_noise_coupled_list.append(n_c)

        x_start_c = torch.stack(x_start_coupled_list)
        x_noise_c = torch.stack(x_noise_coupled_list)

        # 线性插值 (在 OT 耦合空间)
        t_view = t.view(-1, 1, 1)
        x_t = (1.0 - t_view) * x_start_c + t_view * x_noise_c
        velocity = x_noise_c - x_start_c
        return x_t, velocity

    # ──────────────────────────────────────────
    # 诊断辅助
    # ──────────────────────────────────────────

    def compute_coupling_cost(
        self,
        x_start: Tensor,
        x_noise: Tensor,
    ) -> Tensor:
        """计算 OT 耦合的代价矩阵 (用于诊断).

        Args:
            x_start: [N, D]
            x_noise: [N, D]

        Returns:
            cost: [N, N] 代价矩阵
        """
        return torch.cdist(x_start, x_noise, p=2)

    def compute_coupling_quality(
        self,
        x_start: Tensor,
        x_noise: Tensor,
    ) -> dict:
        """计算 OT 耦合质量 (用于诊断).

        Args:
            x_start: [N, D]
            x_noise: [N, D]

        Returns:
            dict with:
                - mean_cost: 平均配对代价
                - max_cost: 最大配对代价
                - transport_marginal_std: 传输矩阵行边缘标准差
                  (越接近 0 表示均匀分配)
        """
        N = x_start.shape[0]
        if N <= 1:
            return {
                'mean_cost': 0.0,
                'max_cost': 0.0,
                'transport_marginal_std': 0.0,
            }

        cost = torch.cdist(x_start, x_noise, p=2)
        transport = sinkhorn_transport(
            cost, epsilon=self.epsilon, num_iters=self.num_iters
        )
        coupled_idx = transport.argmax(dim=1)
        paired_cost = cost.gather(1, coupled_idx.unsqueeze(1)).squeeze(1)

        # 行边缘标准差 (理想: 每行 1/N)
        row_marginal = transport.sum(dim=1)
        ideal = 1.0 / N
        marginal_std = float((row_marginal - ideal).std().item())

        return {
            'mean_cost': float(paired_cost.mean().item()),
            'max_cost': float(paired_cost.max().item()),
            'transport_marginal_std': marginal_std,
        }
