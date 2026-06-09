"""噪声调度工具函数"""

import math
from typing import List

import torch
from torch import Tensor


def cosine_noise_schedule(
    T: int, s: float = 0.008, device: str = 'cpu'
) -> Tensor:
    """余弦噪声调度 (Nichol & Dhariwal, 2021)

    Args:
        T: 总时间步数
        s: 偏移量，防止 t=0 时 beta 过小
        device: 计算设备
    """
    t = torch.linspace(0, T, T + 1, dtype=torch.float64, device=device)
    alphas_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    alphas_t = alphas_t / alphas_t[0]
    betas_t = 1 - (alphas_t[1:] / alphas_t[:-1])
    return torch.clamp(betas_t, 0, 0.999)


def load_buffer(arr: Tensor, steps: Tensor, x_shape: List[int]) -> Tensor:
    """从序列中提取时间步对应的值，并调整形状以广播到 x_shape

    Args:
        arr: 序列参数 (如 alphas_cumprod)，shape: [timesteps]
        steps: 时间步索引，shape: [batch_size]
        x_shape: 目标形状，用于 reshape
    """
    step_arr = arr.gather(-1, steps)
    return step_arr.reshape(-1, *[1] * (len(x_shape) - 1))
