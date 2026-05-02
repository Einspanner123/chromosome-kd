"""
Lehmer Code 编解码

Lehmer code 将排列 σ ∈ S_N 编码为向量 L = (l_1, ..., l_N)，
其中 l_i = |{j > i : σ(j) < σ(i)}|，即 σ(i) 右侧比它小的元素个数。

关键性质:
- l_i ∈ {0, 1, ..., N-i}
- 任意 (l_1, ..., l_N) 组合都对应一个合法排列 (无需约束投影)
- 编码/解码都是 O(N²)
"""

from typing import List


def permutation_to_lehmer(perm: List[int]) -> List[int]:
    """排列 → Lehmer code

    Args:
        perm: 排列 [σ(0), σ(1), ..., σ(N-1)]，值为 0 到 N-1 的一个排列

    Returns:
        lehmer: Lehmer code [l_0, l_1, ..., l_{N-1}]，l_i ∈ {0, ..., N-1-i}

    Example:
        perm = [2, 0, 3, 1]
        lehmer = [2, 0, 1, 0]
        # l_0 = 2: 右侧比 2 小的有 {0, 1} = 2 个
        # l_1 = 0: 右侧比 0 小的有 {} = 0 个
        # l_2 = 1: 右侧比 3 小的有 {1} = 1 个
        # l_3 = 0: 右侧无元素 = 0
    """
    n = len(perm)
    lehmer = []
    for i in range(n):
        count = 0
        for j in range(i + 1, n):
            if perm[j] < perm[i]:
                count += 1
        lehmer.append(count)
    return lehmer


def lehmer_to_permutation(lehmer: List[int]) -> List[int]:
    """Lehmer code → 排列

    Args:
        lehmer: Lehmer code [l_0, l_1, ..., l_{N-1}]

    Returns:
        perm: 排列 [σ(0), σ(1), ..., σ(N-1)]

    Example:
        lehmer = [2, 0, 1, 0]
        perm = [2, 0, 3, 1]
    """
    n = len(lehmer)
    # 可用元素列表 (按升序)
    available = list(range(n))
    perm = []
    for i in range(n):
        # l_i 是当前可用元素中的排名 (0-indexed)
        idx = lehmer[i]
        perm.append(available[idx])
        available.pop(idx)
    return perm


def validate_lehmer(lehmer: List[int]) -> bool:
    """验证 Lehmer code 是否合法

    Args:
        lehmer: 待验证的 Lehmer code

    Returns:
        True if valid (l_i ∈ {0, ..., N-1-i} for all i)
    """
    n = len(lehmer)
    for i, val in enumerate(lehmer):
        if not (0 <= val <= n - 1 - i):
            return False
    return True


def validate_permutation(perm: List[int]) -> bool:
    """验证排列是否合法 (0 到 N-1 的一个排列)"""
    n = len(perm)
    return sorted(perm) == list(range(n))


# ============================================================
# 批量操作 (用于 Tensor)
# ============================================================

import torch
from torch import Tensor


def batch_permutation_to_lehmer(perm: Tensor) -> Tensor:
    """批量排列 → Lehmer code

    Args:
        perm: (B, N) 排列矩阵

    Returns:
        lehmer: (B, N) Lehmer code 矩阵
    """
    B, N = perm.shape
    lehmer = torch.zeros_like(perm)
    for i in range(N):
        # 对每个位置 i，计算右侧比 perm[:, i] 小的元素个数
        right = perm[:, i + 1:]  # (B, N-1-i)
        current = perm[:, i : i + 1]  # (B, 1)
        lehmer[:, i] = (right < current).sum(dim=1)
    return lehmer


def batch_lehmer_to_permutation(lehmer: Tensor) -> Tensor:
    """批量 Lehmer code → 排列

    Args:
        lehmer: (B, N) Lehmer code 矩阵

    Returns:
        perm: (B, N) 排列矩阵
    """
    B, N = lehmer.shape
    device = lehmer.device

    perm = torch.zeros(B, N, dtype=torch.long, device=device)

    for b in range(B):
        available = list(range(N))
        for i in range(N):
            idx = lehmer[b, i].item()
            perm[b, i] = available[idx]
            available.pop(idx)

    return perm


def lehmer_vocab_size(position: int, total: int) -> int:
    """位置 i 的 Lehmer code 词表大小 = N - i

    Args:
        position: 位置索引 (0-based)
        total: 排列总长度 N

    Returns:
        该位置的合法值数量
    """
    return total - position
