"""方向二计数诊断 — 验证计数先验是否有效约束检测头输出.

若方向二废弃, 删除本文件 + tests/unit/test_diagnostics_count.py 即可清理.

诊断内容:
1. 计数分支预测统计 (预测计数 vs 真实计数偏差)
2. 计数约束 NMS 触发率 (约束生效比例、阈值调整幅度)

使用方式:
- 在 head.py 的 loss() 中调用 CountDiagnosticsCallback.update()
- 在推理时调用 CountDiagnosticsCallback.update_constraint()
- TrainingDiagnosticsHook 通过 diagnostics_callback 自动采集
"""

from typing import Dict, Optional

import torch
from torch import Tensor


def compute_count_prediction_stats(
    pred_counts: Tensor,
    gt_counts: Tensor,
) -> Dict[str, float]:
    """计算计数分支预测统计.

    Args:
        pred_counts: [N] 预测的计数
        gt_counts: [N] 真实计数

    Returns:
        统计字典:
        - mae: 平均绝对误差
        - mse: 均方误差
        - bias: 偏差 (正=高估, 负=低估)
        - accuracy_0: 完全准确率
        - accuracy_2: 容差 2 内准确率
        - accuracy_5: 容差 5 内准确率
    """
    diff = pred_counts.float() - gt_counts.float()
    abs_diff = diff.abs()

    return {
        'mae': float(abs_diff.mean().item()),
        'mse': float((diff ** 2).mean().item()),
        'bias': float(diff.mean().item()),
        'accuracy_0': float((abs_diff == 0).float().mean().item()),
        'accuracy_2': float((abs_diff <= 2).float().mean().item()),
        'accuracy_5': float((abs_diff <= 5).float().mean().item()),
    }


def compute_count_constraint_stats(
    nms_counts: Tensor,
    target_counts: Tensor,
    original_thr: Optional[float] = None,
    adjusted_thrs: Optional[Tensor] = None,
) -> Dict[str, float]:
    """计算计数约束 NMS 触发统计.

    Args:
        nms_counts: [N] NMS 后的检测框数
        target_counts: [N] 目标计数 (来自计数分支或默认值)
        original_thr: 原始 NMS 阈值 (可选)
        adjusted_thrs: [N] 调整后的 NMS 阈值 (可选)

    Returns:
        统计字典:
        - constraint_trigger_ratio: 触发约束的比例 (nms > target)
        - over_count_ratio: 平均超出比例 (仅触发样本)
        - thr_adjustment_mean: 阈值平均调整幅度 (若提供 adjusted_thrs)
        - thr_adjustment_ratio: 阈值调整的样本比例
    """
    nms_counts = nms_counts.float()
    target_counts = target_counts.float()

    triggered = nms_counts > target_counts
    trigger_ratio = float(triggered.float().mean().item())

    # 超出比例 (仅触发样本)
    if triggered.any():
        over_counts = nms_counts[triggered] - target_counts[triggered]
        over_ratio = float((over_counts / target_counts[triggered].clamp_min(1)).mean().item())
    else:
        over_ratio = 0.0

    stats: Dict[str, float] = {
        'constraint_trigger_ratio': trigger_ratio,
        'over_count_ratio': over_ratio,
    }

    # 阈值调整统计 (可选)
    if original_thr is not None and adjusted_thrs is not None:
        adjustments = (adjusted_thrs.float() - original_thr).abs()
        stats['thr_adjustment_mean'] = float(adjustments.mean().item())
        stats['thr_adjustment_ratio'] = float((adjustments > 0).float().mean().item())

    return stats


class CountDiagnosticsCallback:
    """计数诊断回调, 集成到 TrainingDiagnosticsHook.

    用法:
        1. 在 head.py 的 loss() 中调用 callback.update(pred_counts, gt_counts)
        2. 在推理时调用 callback.update_constraint(nms_counts, target_counts)
        3. TrainingDiagnosticsHook 通过 diagnostics_callback 自动调用 collect()

    Args:
        interval: 采样间隔 (iter), 默认 100
    """

    def __init__(self, interval: int = 100):
        self.interval = interval
        self.last_pred_counts: Optional[Tensor] = None
        self.last_gt_counts: Optional[Tensor] = None
        self.last_nms_counts: Optional[Tensor] = None
        self.last_target_counts: Optional[Tensor] = None

    def update(self, pred_counts: Tensor, gt_counts: Tensor) -> None:
        """更新计数预测数据 (训练时每个 iter 调用)."""
        self.last_pred_counts = pred_counts.detach().cpu()
        self.last_gt_counts = gt_counts.detach().cpu()

    def update_constraint(
        self,
        nms_counts: Tensor,
        target_counts: Tensor,
    ) -> None:
        """更新计数约束 NMS 数据 (推理时调用)."""
        self.last_nms_counts = nms_counts.detach().cpu()
        self.last_target_counts = target_counts.detach().cpu()

    def collect(self, step: int) -> Dict[str, float]:
        """采集诊断数据 (由 TrainingDiagnosticsHook 调用).

        Returns:
            诊断数据字典, 若非采样点则返回空字典
        """
        if step % self.interval != 0 or step == 0:
            return {}

        data: Dict[str, float] = {}

        # 计数预测统计
        if self.last_pred_counts is not None and self.last_gt_counts is not None:
            stats = compute_count_prediction_stats(
                self.last_pred_counts, self.last_gt_counts,
            )
            for k, v in stats.items():
                data[f'count/{k}'] = v

        # 计数约束 NMS 统计
        if self.last_nms_counts is not None and self.last_target_counts is not None:
            cstats = compute_count_constraint_stats(
                self.last_nms_counts, self.last_target_counts,
            )
            for k, v in cstats.items():
                data[f'count/{k}'] = v

        return data
