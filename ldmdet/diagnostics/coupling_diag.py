"""方向一耦合诊断 — 验证耦合层信息是否传播到 Transformer 表示层.

若方向一废弃, 删除本文件 + tests/unit/test_diagnostics_coupling.py 即可清理.

诊断内容:
1. matched_gt_idx 分布统计 (均匀性、熵、组间均衡)
2. proposal 特征梯度 norm (耦合信息传播指标)

使用方式:
- 在 head.py 的 loss() 中调用 CouplingDiagnosticsCallback.update()
- TrainingDiagnosticsHook 通过 diagnostics_callback 自动采集
"""

from typing import Dict, Optional

import torch
from torch import Tensor


def compute_coupling_stats(
    matched_gt_idx: Tensor,
    num_gt: int,
    group_counts: Optional[Tensor] = None,
    group_gt_counts: Optional[Tensor] = None,
) -> Dict[str, float]:
    """计算耦合分布统计.

    Args:
        matched_gt_idx: [N] 每个 proposal 匹配的 GT 索引
        num_gt: GT 总数
        group_counts: 可选, 各组 proposal 数 (用于组间均衡分析)
        group_gt_counts: 可选, 各组 GT 数

    Returns:
        统计字典:
        - num_proposals: proposal 总数
        - num_gt: GT 总数
        - mean_per_gt: 每个 GT 平均匹配的 proposal 数
        - std_per_gt: 标准差
        - min_per_gt, max_per_gt: 最小/最大
        - entropy: 匹配分布的熵 (越高越均匀)
        - group_balance_ratio: 组间均衡偏差 (仅当 group_counts 提供时)
    """
    N = matched_gt_idx.shape[0]
    stats: Dict[str, float] = {
        'num_proposals': N,
        'num_gt': num_gt,
        'entropy': 0.0,
    }

    if num_gt == 0 or N == 0:
        stats.update({
            'mean_per_gt': 0.0, 'std_per_gt': 0.0,
            'min_per_gt': 0, 'max_per_gt': 0,
        })
        return stats

    # 每个 GT 匹配的 proposal 数 (确保 long 类型)
    idx_long = matched_gt_idx.long()
    counts = torch.bincount(idx_long, minlength=num_gt).float()
    stats['mean_per_gt'] = float(counts.mean().item())
    stats['std_per_gt'] = float(counts.std().item()) if num_gt > 1 else 0.0
    stats['min_per_gt'] = int(counts.min().item())
    stats['max_per_gt'] = int(counts.max().item())

    # 熵 (归一化到 [0, log(num_gt)])
    probs = counts / N
    probs_nonzero = probs[probs > 0]
    entropy = float(-(probs_nonzero * probs_nonzero.log()).sum().item())
    stats['entropy'] = entropy

    # 组间均衡度 (可选)
    if group_counts is not None and group_gt_counts is not None:
        total_proposals = group_counts.sum().clamp_min(1)
        total_gt = group_gt_counts.sum().clamp_min(1)
        # 理想比例 = group_gt / total_gt, 实际比例 = group_proposals / total_proposals
        ideal_ratio = group_gt_counts.float() / total_gt
        actual_ratio = group_counts.float() / total_proposals
        # 偏差 = L1 距离
        balance_ratio = float((actual_ratio - ideal_ratio).abs().sum().item())
        stats['group_balance_ratio'] = balance_ratio

    return stats


def compute_proposal_grad_norm(
    proposals: Tensor,
) -> Dict[str, float]:
    """计算 proposal 特征的梯度 norm 统计.

    用于诊断耦合信息是否传播到 proposal 表示.
    若大量 proposal 梯度为零, 说明耦合层信息被截断.

    Args:
        proposals: [bs, num_proposals, feat_dim] proposal 特征张量

    Returns:
        统计字典:
        - grad_norm_mean: 平均梯度 norm
        - grad_norm_std: 标准差
        - grad_norm_min, grad_norm_max: 极值
        - grad_norm_zero_ratio: 零梯度 proposal 比例
    """
    if proposals.grad is None:
        return {
            'grad_norm_mean': 0.0,
            'grad_norm_std': 0.0,
            'grad_norm_min': 0.0,
            'grad_norm_max': 0.0,
            'grad_norm_zero_ratio': 1.0,
        }

    # [bs, num_proposals, feat_dim] → [bs * num_proposals, feat_dim]
    bs, num_proposals, feat_dim = proposals.shape
    grad_flat = proposals.grad.reshape(bs * num_proposals, feat_dim)

    # 每个 proposal 的梯度 norm: [bs * num_proposals]
    grad_norms = grad_flat.norm(2, dim=1)

    # 零梯度比例
    zero_ratio = float((grad_norms == 0.0).float().mean().item())

    return {
        'grad_norm_mean': float(grad_norms.mean().item()),
        'grad_norm_std': float(grad_norms.std().item()) if grad_norms.numel() > 1 else 0.0,
        'grad_norm_min': float(grad_norms.min().item()),
        'grad_norm_max': float(grad_norms.max().item()),
        'grad_norm_zero_ratio': zero_ratio,
    }


class CouplingDiagnosticsCallback:
    """耦合诊断回调, 集成到 TrainingDiagnosticsHook.

    用法:
        1. 在 head.py 的 loss() 中调用 callback.update(matched_gt_idx, num_gt)
        2. TrainingDiagnosticsHook 通过 diagnostics_callback 自动调用 collect()

    Args:
        interval: 采样间隔 (iter), 默认 100
    """

    def __init__(self, interval: int = 100):
        self.interval = interval
        self.last_matched_gt_idx: Optional[Tensor] = None
        self.last_num_gt: int = 0
        self._last_step: int = -1

    def update(self, matched_gt_idx: Tensor, num_gt: int) -> None:
        """更新耦合数据 (每个 iter 调用)."""
        self.last_matched_gt_idx = matched_gt_idx.detach().cpu()
        self.last_num_gt = num_gt

    def collect(self, step: int) -> Dict[str, float]:
        """采集诊断数据 (由 TrainingDiagnosticsHook 调用).

        Returns:
            诊断数据字典, 若非采样点则返回空字典
        """
        if step % self.interval != 0 or step == 0:
            return {}

        if self.last_matched_gt_idx is None:
            return {}

        stats = compute_coupling_stats(self.last_matched_gt_idx, self.last_num_gt)
        # 加前缀
        return {f'coupling/{k}': v for k, v in stats.items()}
