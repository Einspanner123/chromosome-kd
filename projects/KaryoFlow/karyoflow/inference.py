"""
KaryoFlow 推理: 迭代 unmask 生成排列

从全 [MASK] 开始，每步预测所有 masked 位置的概率分布，
按置信度从高到低逐步 unmask，直到所有位置都被揭示。
"""

import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .constants import NUM_SLOTS


@torch.no_grad()
def karyoflow_inference(
    flow_module,
    chrom_features: Tensor,
    num_steps: int = 10,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> dict:
    """KaryoFlow 迭代 unmask 推理 (直接排列版)

    Args:
        flow_module: KaryoFlowModule 模型 (eval mode)
        chrom_features: (B, N, d) 染色体编码特征
        num_steps: 去噪步数
        temperature: softmax 温度
        device: 设备

    Returns:
        dict:
            'permutation': (B, N) 生成的排列
            'confidence': (B, N) 每个位置的置信度
    """
    B, N, _ = chrom_features.shape
    if device is None:
        device = chrom_features.device

    mask_token = flow_module.mask_token_id

    # 初始化: 全部 masked
    perm = torch.full((B, N), mask_token, dtype=torch.long, device=device)
    confidence = torch.zeros(B, N, device=device)
    unmasked = torch.zeros(B, N, dtype=torch.bool, device=device)

    for step in range(num_steps):
        t = torch.full((B,), (num_steps - step) / num_steps, device=device)

        logits = flow_module(chrom_features, perm, t)  # (B, N, N)
        probs = F.softmax(logits / temperature, dim=-1)
        max_probs, max_indices = probs.max(dim=-1)

        still_masked = ~unmasked
        remaining_steps = num_steps - step
        num_still_masked = still_masked.sum(dim=1)
        num_to_unmask = torch.ceil(num_still_masked.float() / remaining_steps).long()
        num_to_unmask = torch.clamp(num_to_unmask, min=1)

        for b in range(B):
            masked_positions = still_masked[b].nonzero().squeeze(1)
            if len(masked_positions) == 0:
                continue

            pos_confs = max_probs[b, masked_positions]
            k = min(num_to_unmask[b].item(), len(masked_positions))
            _, topk_local_idx = pos_confs.topk(k)
            topk_positions = masked_positions[topk_local_idx]

            perm[b, topk_positions] = max_indices[b, topk_positions]
            unmasked[b, topk_positions] = True
            confidence[b, topk_positions] = max_probs[b, topk_positions]

    # 处理仍然 masked 的位置
    still_masked = ~unmasked
    if still_masked.any():
        t = torch.zeros(B, device=device)
        logits = flow_module(chrom_features, perm, t)
        probs = F.softmax(logits / temperature, dim=-1)
        max_probs, max_indices = probs.max(dim=-1)
        perm[still_masked] = max_indices[still_masked]
        confidence[still_masked] = max_probs[still_masked]

    return {
        "permutation": perm,
        "confidence": confidence,
    }


@torch.no_grad()
def karyoflow_one_shot(
    flow_module,
    chrom_features: Tensor,
    temperature: float = 0.5,
) -> dict:
    """单步推理 (最快但可能不太准)

    所有位置同时从 [MASK] 预测，一步出结果。
    用贪心分配保证排列合法性（无重复）。
    """
    B, N, _ = chrom_features.shape
    device = chrom_features.device

    mask_token = flow_module.mask_token_id
    perm_input = torch.full((B, N), mask_token, dtype=torch.long, device=device)

    t = torch.ones(B, device=device)  # t=1 表示全 masked
    logits = flow_module(chrom_features, perm_input, t)  # (B, N, N)

    probs = F.softmax(logits / temperature, dim=-1)  # (B, N, N)

    # 贪心分配: 按概率从高到低逐个分配，保证排列合法
    permutation = torch.zeros(B, N, dtype=torch.long, device=device)
    confidence = torch.zeros(B, N, device=device)

    for b in range(B):
        flat_probs = probs[b].flatten()  # (N*N,)
        sorted_indices = flat_probs.argsort(descending=True)

        used_slots = set()
        used_dets = set()

        for flat_idx in sorted_indices:
            slot = flat_idx.item() // N
            det = flat_idx.item() % N
            if slot not in used_slots and det not in used_dets:
                permutation[b, slot] = det
                confidence[b, slot] = probs[b, slot, det]
                used_slots.add(slot)
                used_dets.add(det)
            if len(used_slots) == N:
                break

    return {
        "permutation": permutation,
        "confidence": confidence,
    }
