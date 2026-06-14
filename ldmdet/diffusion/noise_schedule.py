"""噪声调度工具 (DDPM 体系，保留用于对比实验)"""

import math
from typing import List

import torch
from torch import Tensor


def cosine_noise_schedule(
    T: int, s: float = 0.008, device: str = 'cpu'
) -> Tensor:
    """余弦噪声调度 (Nichol & Dhariwal, 2021)"""
    t = torch.linspace(0, T, T + 1, dtype=torch.float64, device=device)
    alphas_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    alphas_t = alphas_t / alphas_t[0]
    betas_t = 1 - (alphas_t[1:] / alphas_t[:-1])
    return torch.clamp(betas_t, 0, 0.999)


def load_buffer(arr: Tensor, steps: Tensor, x_shape: List[int]) -> Tensor:
    """从序列中提取步对应值，广播到目标形状"""
    step_arr = arr.gather(-1, steps)
    return step_arr.reshape(-1, *[1] * (len(x_shape) - 1))
