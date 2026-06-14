"""Sinkhorn OT + 随机采样 — 论文核心方法

从 Sinkhorn 传输矩阵采样 (而非 argmax)，恢复 ε 的多样性控制。
"""

from typing import Optional

import torch
from torch import Tensor

from ldmdet.coupling._sinkhorn_ops import ot_multinomial, sinkhorn_transport
from ldmdet.coupling.base import CouplingStrategy, register_coupling


@register_coupling('sinkhorn_stochastic')
class SinkhornStochasticCoupling(CouplingStrategy):
    """Sinkhorn OT + 随机采样。

    从传输矩阵 P_ε 的每一行按概率采样 GT 索引。
    恢复 ε 作为多样性控制参数的功能。

    Args:
        epsilon: 熵正则化强度 (论文最佳: ε=5)
        num_iters: Sinkhorn 迭代次数 (默认 20)
        sample_seed: 可选随机种子 (用于可复现多 seed 实验)
    """

    def __init__(
        self,
        epsilon: float = 5.0,
        num_iters: int = 20,
        sample_seed: Optional[int] = None,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.num_iters = num_iters
        self.sample_seed = sample_seed

    def couple(
        self,
        noise: Tensor,
        gt_diffusion: Tensor,
        gt_labels: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        N = noise.shape[0]
        if gt_diffusion.shape[0] == 0:
            return noise, torch.zeros(N, dtype=torch.long, device=device)

        cost = torch.cdist(noise, gt_diffusion, p=2)
        transport = sinkhorn_transport(
            cost, epsilon=self.epsilon, num_iters=self.num_iters
        )
        row_probs = transport / transport.sum(dim=1, keepdim=True).clamp_min(
            1e-10
        )
        idx = ot_multinomial(row_probs, seed=self.sample_seed)
        return gt_diffusion[idx], idx
