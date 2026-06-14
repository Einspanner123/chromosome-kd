"""Sinkhorn 迭代的共享实现

用于 sinkhorn_argmax 和 sinkhorn_stochastic 耦合策略。
"""

from typing import Optional

import torch
from torch import Tensor


def sinkhorn_transport(
    cost: Tensor,
    epsilon: float,
    num_iters: int = 20,
    row_mass: Optional[Tensor] = None,
    col_mass: Optional[Tensor] = None,
) -> Tensor:
    """Entropic OT: 计算 Sinkhorn 传输矩阵。

    P_ε = argmin <P, C> - ε H(P)   subject to row/col marginals

    Args:
        cost: [N, K] 成对代价矩阵 (如 L2 距离)
        epsilon: 熵正则化强度 (越大→越均匀)
        num_iters: Sinkhorn 迭代次数
        row_mass: [N] 行边缘分布 (默认均匀)
        col_mass: [K] 列边缘分布 (默认均匀)

    Returns:
        [N, K] 传输矩阵
    """
    N, K = cost.shape
    device = cost.device

    if row_mass is None:
        row_mass = torch.ones(N, device=device) / max(N, 1)
    if col_mass is None:
        proposals_per_gt = max(N // max(K, 1), 1)
        col_mass = torch.full((K,), proposals_per_gt / N, device=device)
        col_mass = col_mass / col_mass.sum()
    else:
        col_mass = col_mass / col_mass.sum().clamp_min(1e-10)

    eps = max(epsilon, 1e-6)
    log_K_mat = -cost / eps
    log_u = torch.zeros(N, device=device)
    log_v = torch.zeros(K, device=device)

    for _ in range(num_iters):
        log_u = torch.log(row_mass + 1e-10) - torch.logsumexp(
            log_K_mat + log_v.unsqueeze(0), dim=1
        )
        log_v = torch.log(col_mass + 1e-10) - torch.logsumexp(
            log_K_mat + log_u.unsqueeze(1), dim=0
        )

    return torch.exp(log_u.unsqueeze(1) + log_K_mat + log_v.unsqueeze(0))


def ot_multinomial(
    row_probs: Tensor, seed: Optional[int] = None
) -> Tensor:
    """从每行概率分布采样一个列索引。

    Args:
        row_probs: [N, K] 归一化行概率
        seed: 可选随机种子 (用于可复现性)

    Returns:
        [N] 采样得到的列索引
    """
    if seed is not None and not hasattr(ot_multinomial, '_generators'):
        ot_multinomial._generators = {}
    if seed is not None:
        device = str(row_probs.device)
        if device not in ot_multinomial._generators:
            gen = torch.Generator(device=row_probs.device)
            gen.manual_seed(seed)
            ot_multinomial._generators[device] = gen
        return torch.multinomial(
            row_probs, 1, generator=ot_multinomial._generators[device]
        ).squeeze(-1)
    return torch.multinomial(row_probs, 1).squeeze(-1)
