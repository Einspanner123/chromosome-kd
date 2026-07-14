"""SHTS — SNR-Hierarchical Time-step Sampling (SNR 层级时间步采样)

基于信噪比(SNR)导数的层级化时间步网格生成算法。
与 DPM-Solver++ 协同，纯推理优化，无需重训练。
"""

from typing import List, Optional

import numpy as np


def build_shts_grid(
    num_steps: int,
    snr_scale: float = 2.0,
    task_weight_alpha: float = 1.0,
    task_weight_sigma: float = 0.15,
    solver_order: int = 2,
) -> List[float]:
    """生成 SNR 层级化时间步网格。

    基于 d(ln SNR)/dt 的变化率 + 任务感知权重，
    通过 CDF 反演法生成非均匀网格。

    Args:
        num_steps: 采样步数
        snr_scale: SNR 缩放因子
        task_weight_alpha: 临界区权重强度 (0=纯SNR, 1=任务感知)
        task_weight_sigma: 临界区高斯宽度
        solver_order: 求解器阶数 (影响步长指数)

    Returns:
        降序时间步列表 [1.0, ..., 0.0]
    """
    # 1. 定义密度函数 rho(t) = |d ln SNR / dt| * w(t)
    t_crit = snr_scale / (1 + snr_scale)  # SNR=1 的 t 值

    def density(t):
        # SNR 变化率: 2 / (t * (1-t))
        snr_rate = 2.0 / (t * (1 - t) + 1e-8)
        # 任务权重: 在临界 SNR 处增强
        w = 1.0 + task_weight_alpha * np.exp(
            -((t - t_crit) ** 2) / (2 * task_weight_sigma ** 2)
        )
        return snr_rate * w

    # 2. 数值积分 CDF
    t_fine = np.linspace(1.0, 1e-4, 10000)
    rho = np.array([density(t) for t in t_fine])
    # 从 t=1 到 t 的积分 (反向)
    t_rev = t_fine[::-1]  # 升序: [0.0001, ..., 1.0]
    # 使用梯形法则: 区间中点代表值
    t_mid = 0.5 * (t_rev[:-1] + t_rev[1:])  # 区间中点
    rho_mid = np.array([density(t) for t in t_mid])
    dt = np.diff(t_rev)
    cdf = np.cumsum(rho_mid * dt)
    cdf = cdf / cdf[-1]

    # 3. 反演 CDF: 均匀分位 -> 非均匀 t
    quantiles = np.linspace(0, 1, num_steps + 1)
    t_grid = np.interp(quantiles, cdf, t_mid)
    t_grid[0] = 1.0
    t_grid[-1] = 0.0

    # 4. 确保严格单调递减 (去除数值误差)
    t_list = sorted(t_grid.tolist(), reverse=True)
    # 去重/去除非单调
    cleaned = [t_list[0]]
    for t in t_list[1:]:
        if t < cleaned[-1]:
            cleaned.append(t)
        else:
            cleaned.append(max(cleaned[-1] - 1e-6, 0.0))
    return cleaned


def build_shts_shifted_grid(
    num_steps: int,
    snr_scale: float = 2.0,
    rf_shift: float = 3.0,
    **kwargs,
) -> List[float]:
    """先 shifted 变换，再 SHTS 分层。

    组合策略: t' = shifted(t), 然后在 t' 空间应用 SHTS。
    反 shifted 变换回原始 t 空间。

    Args:
        num_steps: 采样步数
        snr_scale: SNR 缩放因子
        rf_shift: shifted schedule 参数 s
        **kwargs: 传递给 build_shts_grid 的额外参数

    Returns:
        降序时间步列表 [1.0, ..., 0.0]
    """
    # 1. 生成 SHTS 基网格 (在 t' 空间)
    t_prime_grid = build_shts_grid(num_steps, snr_scale, **kwargs)

    # 2. 反 shifted 变换回原始 t 空间
    # t' = s*t/(1+(s-1)*t) => t = t'/(s-(s-1)*t')
    s = rf_shift
    t_grid = [tp / (s - (s - 1) * tp) for tp in t_prime_grid]

    # 确保严格单调递减
    cleaned = [t_grid[0]]
    for t in t_grid[1:]:
        if t < cleaned[-1]:
            cleaned.append(t)
        else:
            cleaned.append(max(cleaned[-1] - 1e-6, 0.0))
    return cleaned


def get_shts_density_at(t, snr_scale=2.0, task_weight_alpha=1.0, task_weight_sigma=0.15):
    """计算 SHTS 密度函数在 t 处的值 (用于诊断/可视化)"""
    t_crit = snr_scale / (1 + snr_scale)
    snr_rate = 2.0 / (t * (1 - t) + 1e-8)
    w = 1.0 + task_weight_alpha * np.exp(
        -((t - t_crit) ** 2) / (2 * task_weight_sigma ** 2)
    )
    return snr_rate * w