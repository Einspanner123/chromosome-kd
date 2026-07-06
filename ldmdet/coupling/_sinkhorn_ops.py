"""Sinkhorn 迭代的共享实现

用于 sinkhorn_argmax 和 sinkhorn_stochastic 耦合策略。
"""

from typing import Optional

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

# 模块级 Generator 缓存，按 device 索引，用于 ot_multinomial 的可复现采样
_OT_GENERATORS: dict[str, torch.Generator] = {}


# ============================================================
# Triton kernels: fused logsumexp + mask for Sinkhorn iterations
# ============================================================
if _HAS_TRITON:

    @triton.jit
    def _row_lse_mask_kernel(
        K_ptr,  # [G, N, K] log_K_mat
        v_ptr,  # [G, K] log_v
        rm_ptr,  # [G, N] log_row_mass
        rmask_ptr,  # [G, N] row_mask (float 1.0/0.0)
        out_ptr,  # [G, N] log_u
        N,
        K,
        sK_g,
        sK_n,
        sK_k,
        BLOCK_K: tl.constexpr,
    ):
        """融合: log_u[g,n] = mask ? (log_row_mass - logsumexp_j(K+v)) : -inf"""
        g = tl.program_id(0)
        n = tl.program_id(1)
        gn = g * N + n

        mask = tl.load(rmask_ptr + gn)
        log_rm = tl.load(rm_ptr + gn)

        offs = tl.arange(0, BLOCK_K)
        k_mask = offs < K

        kv = tl.load(
            K_ptr + g * sK_g + n * sK_n + offs * sK_k,
            mask=k_mask,
            other=float('-inf'),
        )
        vv = tl.load(v_ptr + g * K + offs, mask=k_mask, other=float('-inf'))

        x = kv + vv
        m = tl.max(x, axis=0)
        lse = m + tl.log(tl.sum(tl.exp(x - m), axis=0))

        result = tl.where(mask > 0.5, log_rm - lse, float('-inf'))
        tl.store(out_ptr + gn, result)

    @triton.jit
    def _col_lse_mask_kernel(
        K_ptr,  # [G, N, K] log_K_mat
        u_ptr,  # [G, N] log_u
        cm_ptr,  # [G, K] log_col_mass
        cmask_ptr,  # [G, K] col_mask (float 1.0/0.0)
        out_ptr,  # [G, K] log_v
        N,
        K,
        sK_g,
        sK_n,
        sK_k,
        BLOCK_N: tl.constexpr,
    ):
        """融合: log_v[g,k] = mask ? (log_col_mass - logsumexp_i(K+u)) : -inf"""
        g = tl.program_id(0)
        k = tl.program_id(1)
        gk = g * K + k

        mask = tl.load(cmask_ptr + gk)
        log_cm = tl.load(cm_ptr + gk)

        offs = tl.arange(0, BLOCK_N)
        n_mask = offs < N

        kn = tl.load(
            K_ptr + g * sK_g + offs * sK_n + k * sK_k,
            mask=n_mask,
            other=float('-inf'),
        )
        un = tl.load(u_ptr + g * N + offs, mask=n_mask, other=float('-inf'))

        x = kn + un
        m = tl.max(x, axis=0)
        lse = m + tl.log(tl.sum(tl.exp(x - m), axis=0))

        result = tl.where(mask > 0.5, log_cm - lse, float('-inf'))
        tl.store(out_ptr + gk, result)

    def _triton_fused_row_lse(
        log_K_mat: Tensor,
        log_v: Tensor,
        log_row_mass: Tensor,
        row_mask_float: Tensor,
    ) -> Tensor:
        """row 维度融合 logsumexp + mask"""
        G, N, K = log_K_mat.shape
        log_u = torch.empty(G, N, device=log_K_mat.device, dtype=log_K_mat.dtype)
        sK_g, sK_n, sK_k = log_K_mat.stride()
        BLOCK_K = max(16, min(4096, triton.next_power_of_2(K)))
        grid = (G, N)
        _row_lse_mask_kernel[grid](
            log_K_mat,
            log_v,
            log_row_mass,
            row_mask_float,
            log_u,
            N,
            K,
            sK_g,
            sK_n,
            sK_k,
            BLOCK_K=BLOCK_K,
        )
        return log_u

    def _triton_fused_col_lse(
        log_K_mat: Tensor,
        log_u: Tensor,
        log_col_mass: Tensor,
        col_mask_float: Tensor,
    ) -> Tensor:
        """col 维度融合 logsumexp + mask"""
        G, N, K = log_K_mat.shape
        log_v = torch.empty(G, K, device=log_K_mat.device, dtype=log_K_mat.dtype)
        sK_g, sK_n, sK_k = log_K_mat.stride()
        BLOCK_N = max(16, min(4096, triton.next_power_of_2(N)))
        grid = (G, K)
        _col_lse_mask_kernel[grid](
            log_K_mat,
            log_u,
            log_col_mass,
            col_mask_float,
            log_v,
            N,
            K,
            sK_g,
            sK_n,
            sK_k,
            BLOCK_N=BLOCK_N,
        )
        return log_v


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

    # Triton 优化路径: 融合 logsumexp + masked_fill 为单 kernel
    use_triton = (
        _HAS_TRITON
        and log_K_mat.is_cuda
        and max_N <= 4096
        and max_K <= 4096
    )
    if use_triton:
        # triton kernel 需要 float mask (1.0/0.0) 和 contiguous tensor
        log_K_mat_c = log_K_mat.contiguous()
        row_mask_f = row_mask.float()
        col_mask_f = col_mask.float()
        for _ in range(num_iters):
            log_u = _triton_fused_row_lse(
                log_K_mat_c, log_v, log_row_mass, row_mask_f
            )
            log_v = _triton_fused_col_lse(
                log_K_mat_c, log_u, log_col_mass, col_mask_f
            )
    else:
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


def unbalanced_sinkhorn_transport(
    cost: Tensor,
    epsilon: float,
    num_iters: int = 20,
    row_mass: Optional[Tensor] = None,
    col_mass: Optional[Tensor] = None,
    lambda_row: float = 1.0,
    lambda_col: float = 1.0,
) -> Tensor:
    """非平衡 Sinkhorn 传输 (Chizat et al., 2018).

    P = argmin <P,C> - ε H(P) + λ_row KL(P1||a) + λ_col KL(P^T1||b)

    与标准 Sinkhorn 的区别: 边缘约束通过 KL 散度软化为可调松弛,
    允许传输矩阵的行/列和偏离目标边缘, 适应 GT 分布不均匀的场景.

    Args:
        cost: [N, K] 代价矩阵
        epsilon: 熵正则化强度
        num_iters: Sinkhorn 迭代次数
        row_mass: [N] 行边缘目标 (默认均匀)
        col_mass: [K] 列边缘目标 (默认均匀)
        lambda_row: 行边缘松弛系数
            - λ → ∞: 退化为标准 Sinkhorn (严格行约束)
            - λ → 0: 完全放松行约束 (u → 1)
        lambda_col: 列边缘松弛系数 (同上)

    Returns:
        [N, K] 传输矩阵 (行和不必等于 row_mass)

    Reference:
        Chizat et al., "Scaling Algorithms for Unbalanced Transport Problems",
        Mathematics of Computation, 2018.
    """
    N, K = cost.shape
    device = cost.device

    if row_mass is None:
        row_mass = torch.ones(N, device=device) / max(N, 1)
    if col_mass is None:
        col_mass = torch.ones(K, device=device) / max(K, 1)

    eps = max(epsilon, 1e-6)
    # 软约束指数: λ/(ε+λ)
    #   λ → ∞ → 指数 → 1 (标准 Sinkhorn)
    #   λ → 0 → 指数 → 0 (u 或 v → 1, 约束失效)
    alpha = lambda_row / (eps + lambda_row)
    beta = lambda_col / (eps + lambda_col)

    log_K = -cost / eps  # [N, K]
    log_u = torch.zeros(N, device=device)
    log_v = torch.zeros(K, device=device)
    log_a = torch.log(row_mass.clamp_min(1e-10))
    log_b = torch.log(col_mass.clamp_min(1e-10))

    for _ in range(num_iters):
        # log_u = α * (log_a - logsumexp_j(log_K + log_v))
        log_u = alpha * (
            log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        )
        # log_v = β * (log_b - logsumexp_i(log_K + log_u))
        log_v = beta * (
            log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        )

    return torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))
