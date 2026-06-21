"""SNR 感知权重计算 — 方向三: SNR 感知的动态匹配

将扩散时间 t (及其对应的信噪比 SNR) 映射为权重 w(t) ∈ [w_min, 1]:
- t→0 (低噪声): w→1, 全权信任匹配代价
- t→1 (高噪声): w→w_min, 抑制不可靠匹配

数学依据: Rectified Flow 的 SNR(t) = (1-t)^2 / t^2

开关: 通过 SNRAwareMatcher 的 snr_mode 参数控制, 默认 'logistic'.
失败时直接删除本文件 + 还原 matcher/criterion 即可回滚.
"""

import torch
from torch import Tensor


def logistic_snr_weight(t: Tensor, w_min: float = 0.0) -> Tensor:
    """Logistic SNR 权重: w(t) = SNR/(1+SNR) = (1-t)^2 / ((1-t)^2 + t^2)

    性质: w(0)=1, w(0.5)=0.5, w(1)=0, 单调递减

    Args:
        t: [bs] 或标量, 扩散时间 ∈ [0, 1]
        w_min: 下界, 避免 w(1)=0 完全抑制 (推荐 0.1)

    Returns:
        w: 同形状, ∈ [w_min, 1]
    """
    w = (1 - t) ** 2 / ((1 - t) ** 2 + t ** 2 + 1e-8)
    return w.clamp(min=w_min, max=1.0)


def exponential_snr_weight(t: Tensor, beta: float = 3.0, w_min: float = 0.0) -> Tensor:
    """指数衰减权重: w(t) = exp(-β·t)

    更激进, β 控制衰减速度.

    Args:
        t: [bs] 或标量
        beta: 衰减速率 (越大越激进)
        w_min: 下界

    Returns:
        w: 同形状, ∈ [w_min, 1]
    """
    w = torch.exp(-beta * t)
    return w.clamp(min=w_min, max=1.0)


def get_snr_weight(
    t: Tensor,
    mode: str = 'logistic',
    beta: float = 3.0,
    w_min: float = 0.0,
) -> Tensor:
    """统一接口

    Args:
        t: [bs] 或标量, 扩散时间 ∈ [0, 1]
        mode: 'logistic' | 'exponential' | 'none'
        beta: exponential 模式的衰减率
        w_min: 权重下界 (避免完全抑制)

    Returns:
        w: 同形状, ∈ [w_min, 1]
    """
    if mode == 'logistic':
        return logistic_snr_weight(t, w_min=w_min)
    elif mode == 'exponential':
        return exponential_snr_weight(t, beta=beta, w_min=w_min)
    elif mode == 'none':
        return torch.ones_like(t)
    else:
        raise ValueError(f"Unknown SNR weight mode: {mode}")
