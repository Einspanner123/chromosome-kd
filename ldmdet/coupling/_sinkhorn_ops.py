"""Sinkhorn 迭代的共享实现

用于 sinkhorn_argmax 和 sinkhorn_stochastic 耦合策略。
"""

from typing import Optional

import torch
from torch import Tensor

# 模块级 Generator 缓存，按 device 索引，用于 ot_multinomial 的可复现采样
_OT_GENERATORS: dict[str, torch.Generator] = {}


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

    # 预计算 log marginals
    log_row_mass = torch.log(row_mass + 1e-10)
    log_col_mass = torch.log(col_mass + 1e-10)

    for _ in range(num_iters):
        log_u = log_row_mass - torch.logsumexp(
            log_K_mat + log_v.unsqueeze(0), dim=1
        )
        log_v = log_col_mass - torch.logsumexp(
            log_K_mat + log_u.unsqueeze(1), dim=0
        )

    return torch.exp(log_u.unsqueeze(1) + log_K_mat + log_v.unsqueeze(0))


def sinkhorn_transport_batch(
    costs: list[Tensor],
    epsilon: float,
    num_iters: int = 20,
    row_masses: Optional[list[Tensor]] = None,
    col_masses: Optional[list[Tensor]] = None,
) -> list[Tensor]:
    """批量 Sinkhorn: 对多个不同大小的代价矩阵并行迭代。

    使用 padding + mask 将不同大小的矩阵合并为批量操作，
    减少逐组 Python 循环的 kernel launch 开销。

    Args:
        costs: 多个 [N_g, K_g] 代价矩阵列表
        epsilon: 熵正则化强度
        num_iters: Sinkhorn 迭代次数
        row_masses: 可选的行边缘分布列表
        col_masses: 可选的列边缘分布列表

    Returns:
        传输矩阵列表, 每个与对应 cost 同形状
    """
    G = len(costs)
    if G == 0:
        return []
    if G == 1:
        return [sinkhorn_transport(costs[0], epsilon, num_iters,
                                   row_masses[0] if row_masses else None,
                                   col_masses[0] if col_masses else None)]

    device = costs[0].device
    max_N = max(c.shape[0] for c in costs)
    max_K = max(c.shape[1] for c in costs)
    eps = max(epsilon, 1e-6)

    # Pad cost matrices to [G, max_N, max_K] with +inf (→ exp(-inf) = 0 in transport)
    padded_cost = torch.full((G, max_N, max_K), float('inf'), device=device)
    N_sizes = []
    K_sizes = []
    for g in range(G):
        Ng, Kg = costs[g].shape
        padded_cost[g, :Ng, :Kg] = costs[g]
        N_sizes.append(Ng)
        K_sizes.append(Kg)

    log_K_mat = -padded_cost / eps  # [G, max_N, max_K], padded entries → -inf

    # Prepare log marginals
    log_row_mass = torch.full((G, max_N), float('-inf'), device=device)  # log(0) = -inf
    log_col_mass = torch.full((G, max_K), float('-inf'), device=device)
    for g in range(G):
        Ng, Kg = N_sizes[g], K_sizes[g]
        if row_masses is not None:
            rm = row_masses[g]
            log_row_mass[g, :Ng] = torch.log(rm.clamp_min(1e-10))
        else:
            log_row_mass[g, :Ng] = torch.log(torch.ones(Ng, device=device) / max(Ng, 1))
        if col_masses is not None:
            cm = col_masses[g]
            cm = cm / cm.sum().clamp_min(1e-10)
            log_col_mass[g, :Kg] = torch.log(cm.clamp_min(1e-10))
        else:
            proposals_per_gt = max(Ng // max(Kg, 1), 1)
            cm = torch.full((Kg,), proposals_per_gt / Ng, device=device)
            cm = cm / cm.sum()
            log_col_mass[g, :Kg] = torch.log(cm.clamp_min(1e-10))

    log_u = torch.full((G, max_N), float('-inf'), device=device)
    log_v = torch.full((G, max_K), float('-inf'), device=device)
    # 预计算 boolean mask，用于迭代中高效重置 padded 区域
    row_mask = torch.zeros(G, max_N, dtype=torch.bool, device=device)
    col_mask = torch.zeros(G, max_K, dtype=torch.bool, device=device)
    for g in range(G):
        row_mask[g, :N_sizes[g]] = True
        col_mask[g, :K_sizes[g]] = True
        log_u[g, :N_sizes[g]] = 0.0
        log_v[g, :K_sizes[g]] = 0.0

    for _ in range(num_iters):
        # log_u[g,n] = log_row_mass[g,n] - logsumexp_j(log_K[g,n,j] + log_v[g,j])
        log_u = log_row_mass - torch.logsumexp(
            log_K_mat + log_v.unsqueeze(1), dim=2
        )
        # Reset padded rows to -inf
        log_u.masked_fill_(~row_mask, float('-inf'))

        # log_v[g,k] = log_col_mass[g,k] - logsumexp_i(log_K[g,i,k] + log_u[g,i])
        log_v = log_col_mass - torch.logsumexp(
            log_K_mat + log_u.unsqueeze(2), dim=1
        )
        # Reset padded cols to -inf
        log_v.masked_fill_(~col_mask, float('-inf'))

    # Compute transport and unpad
    transport_full = torch.exp(log_u.unsqueeze(2) + log_K_mat + log_v.unsqueeze(1))
    # Clamp to avoid NaN from -inf + inf
    transport_full = transport_full.clamp_min(0.0)
    results = []
    for g in range(G):
        Ng, Kg = N_sizes[g], K_sizes[g]
        results.append(transport_full[g, :Ng, :Kg])
    return results


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
    if seed is not None:
        device = str(row_probs.device)
        if device not in _OT_GENERATORS:
            gen = torch.Generator(device=row_probs.device)
            gen.manual_seed(seed)
            _OT_GENERATORS[device] = gen
        return torch.multinomial(
            row_probs, 1, generator=_OT_GENERATORS[device]
        ).squeeze(-1)
    return torch.multinomial(row_probs, 1).squeeze(-1)
