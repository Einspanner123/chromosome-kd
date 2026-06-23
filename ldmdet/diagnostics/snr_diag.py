"""方向三 SNR 诊断 — 验证 SNR 感知机制是否按预期工作.

若方向三废弃, 删除本文件 + tests/unit/test_diagnostics_snr.py 即可清理.

诊断内容:
1. SNR 权重分布统计 (不同 t 下的权重分布、相关性)
2. 动态匹配统计 (k 值分布、匹配质量)
3. 损失加权效果 (高/低 SNR 样本的损失比)

使用方式:
- 在 criterion.py 中调用 SNRDiagnosticsCallback.update() / update_match()
- TrainingDiagnosticsHook 通过 diagnostics_callback 自动采集
"""

from typing import Dict, Optional

import torch
from torch import Tensor


def compute_snr_weight_stats(
    snr_weights: Tensor,
    t: Tensor,
    num_bins: int = 0,
) -> Dict[str, float]:
    """计算 SNR 权重分布统计.

    Args:
        snr_weights: [bs] SNR 权重
        t: [bs] 扩散时间
        num_bins: 是否按 t 分箱统计 (0=不分箱)

    Returns:
        统计字典:
        - weight_mean/std/min/max: 权重基本统计
        - weight_zero_ratio: 零权重比例
        - weight_t_correlation: 权重与 t 的相关系数
        - weight_bin_{i}_mean: 第 i 个 t 箱的平均权重 (若 num_bins > 0)
    """
    w = snr_weights.float()
    t = t.float()

    stats: Dict[str, float] = {
        'weight_mean': float(w.mean().item()),
        'weight_std': float(w.std().item()) if w.numel() > 1 else 0.0,
        'weight_min': float(w.min().item()),
        'weight_max': float(w.max().item()),
        'weight_zero_ratio': float((w == 0.0).float().mean().item()),
    }

    # 相关系数
    if w.numel() > 1 and w.std() > 0 and t.std() > 0:
        corr = float(torch.corrcoef(torch.stack([w, t]))[0, 1].item())
    else:
        corr = 0.0
    stats['weight_t_correlation'] = corr

    # 按 t 分箱
    if num_bins > 0 and t.numel() > 0:
        t_min, t_max = float(t.min().item()), float(t.max().item())
        if t_max > t_min:
            bin_edges = torch.linspace(t_min, t_max + 1e-6, num_bins + 1)
            for i in range(num_bins):
                mask = (t >= bin_edges[i]) & (t < bin_edges[i + 1])
                if mask.any():
                    stats[f'weight_bin_{i}_mean'] = float(w[mask].mean().item())
                else:
                    stats[f'weight_bin_{i}_mean'] = 0.0

    return stats


def compute_dynamic_match_stats(
    k_values: Tensor,
    match_ious: Optional[Tensor] = None,
) -> Dict[str, float]:
    """计算动态匹配统计.

    Args:
        k_values: [bs] 每张图的动态 k 值
        match_ious: 可选, [N] 所有匹配对的 IoU

    Returns:
        统计字典:
        - k_mean/std/min/max: k 值统计
        - match_iou_mean/std: 匹配 IoU 统计 (若提供)
        - match_iou_high_ratio: IoU > 0.5 比例
    """
    k = k_values.float()
    stats: Dict[str, float] = {
        'k_mean': float(k.mean().item()),
        'k_std': float(k.std().item()) if k.numel() > 1 else 0.0,
        'k_min': float(k.min().item()),
        'k_max': float(k.max().item()),
    }

    if match_ious is not None and match_ious.numel() > 0:
        ious = match_ious.float()
        stats['match_iou_mean'] = float(ious.mean().item())
        stats['match_iou_std'] = float(ious.std().item()) if ious.numel() > 1 else 0.0
        stats['match_iou_high_ratio'] = float((ious > 0.5).float().mean().item())

    return stats


def compute_loss_weighting_effect(
    original_losses: Tensor,
    weighted_losses: Tensor,
    snr_weights: Tensor,
) -> Dict[str, float]:
    """计算损失加权效果统计.

    Args:
        original_losses: [bs] 原始损失
        weighted_losses: [bs] 加权后损失
        snr_weights: [bs] SNR 权重

    Returns:
        统计字典:
        - loss_scale_mean/std: 损失缩放因子统计
        - high_snr_loss_ratio: 高 SNR 样本 (权重 > 中位数) 损失占比
        - low_snr_loss_ratio: 低 SNR 样本损失占比
    """
    orig = original_losses.float()
    weighted = weighted_losses.float()
    w = snr_weights.float()

    # 缩放因子 = weighted / original (避免除零)
    scale = weighted / orig.clamp_min(1e-8)

    stats: Dict[str, float] = {
        'loss_scale_mean': float(scale.mean().item()),
        'loss_scale_std': float(scale.std().item()) if scale.numel() > 1 else 0.0,
    }

    # 高/低 SNR 样本损失占比
    total_weighted = weighted.sum().clamp_min(1e-8)
    if w.numel() > 1 and w.std() > 0:
        median_w = w.median()
        high_mask = w > median_w
        low_mask = w < median_w
        stats['high_snr_loss_ratio'] = float((weighted[high_mask].sum() / total_weighted).item())
        stats['low_snr_loss_ratio'] = float((weighted[low_mask].sum() / total_weighted).item())
    else:
        # 退化情况: 所有权重相同, 高/低各占一半
        stats['high_snr_loss_ratio'] = 0.5
        stats['low_snr_loss_ratio'] = 0.5

    return stats


class SNRDiagnosticsCallback:
    """SNR 诊断回调, 集成到 TrainingDiagnosticsHook.

    用法:
        1. 在 criterion.py 中调用 callback.update(snr_weights, t)
        2. 在 matcher 中调用 callback.update_match(k_values, match_ious)
        3. TrainingDiagnosticsHook 通过 diagnostics_callback 自动调用 collect()

    Args:
        interval: 采样间隔 (iter), 默认 100
    """

    def __init__(self, interval: int = 100):
        self.interval = interval
        self.last_snr_weights: Optional[Tensor] = None
        self.last_t: Optional[Tensor] = None
        self.last_k_values: Optional[Tensor] = None
        self.last_match_ious: Optional[Tensor] = None

    def update(self, snr_weights: Tensor, t: Tensor) -> None:
        """更新 SNR 权重数据 (训练时每个 iter 调用)."""
        self.last_snr_weights = snr_weights.detach().cpu()
        self.last_t = t.detach().cpu()

    def update_match(
        self,
        k_values: Tensor,
        match_ious: Optional[Tensor] = None,
    ) -> None:
        """更新动态匹配数据."""
        self.last_k_values = k_values.detach().cpu()
        if match_ious is not None:
            self.last_match_ious = match_ious.detach().cpu()

    def collect(self, step: int) -> Dict[str, float]:
        """采集诊断数据 (由 TrainingDiagnosticsHook 调用).

        Returns:
            诊断数据字典, 若非采样点则返回空字典
        """
        if step % self.interval != 0 or step == 0:
            return {}

        data: Dict[str, float] = {}

        # SNR 权重统计
        if self.last_snr_weights is not None and self.last_t is not None:
            stats = compute_snr_weight_stats(
                self.last_snr_weights, self.last_t, num_bins=3,
            )
            for k, v in stats.items():
                data[f'snr/{k}'] = v

        # 动态匹配统计
        if self.last_k_values is not None:
            mstats = compute_dynamic_match_stats(
                self.last_k_values, self.last_match_ious,
            )
            for k, v in mstats.items():
                data[f'snr/{k}'] = v

        return data
