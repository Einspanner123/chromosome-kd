"""计数约束 NMS (路径 B) — 方向二: 计数先验约束的扩散生成

通过二分搜索置信度阈值, 使 NMS 后检测数尽可能接近目标计数 c.
数学依据: 拉格朗日对偶 (见 docs/research/breakthrough_directions/方向二)

由于 1[conf > τ] 不可微, 用二分搜索求解对偶问题的最优 τ.
复杂度 O(N log(1/ε)), 对 N=500 可忽略.

开关: 通过 head.py 的 use_count_constraint 参数控制, 默认 False.
失败时直接删除本文件 + 还原 head.py 即可回滚.
"""

import torch
from torch import Tensor

from torchvision.ops import batched_nms


def find_threshold_for_count(
    scores: Tensor,
    target_count: int,
    tau_low: float = 0.0,
    tau_high: float = None,
    max_iters: int = 20,
) -> float:
    """二分搜索使 top-k 恰好等于 target_count 的置信度阈值.

    数学推导: count(τ) = sum(1[score_i > τ]) 关于 τ 单调递减,
    因此可用二分搜索求解 count(τ) = target_count.

    Args:
        scores: [N] 置信度
        target_count: 目标检测数
        tau_low: 阈值下界
        tau_high: 阈值上界 (默认 scores.max())
        max_iters: 二分搜索迭代次数 (20 次精度 ~1e-6)

    Returns:
        tau: 使 count(τ) ≤ target_count 的最小阈值
    """
    if tau_high is None:
        tau_high = float(scores.max().item()) if scores.numel() > 0 else 1.0

    if scores.numel() == 0:
        return tau_high

    for _ in range(max_iters):
        tau_mid = (tau_low + tau_high) / 2
        count = (scores > tau_mid).sum().item()
        if count > target_count:
            # 阈值太低, 检测数过多 → 提高阈值
            tau_low = tau_mid
        else:
            # 阈值太高, 检测数过少 → 降低阈值
            tau_high = tau_mid

    return tau_high


def count_constrained_nms(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    target_count: int,
    iou_threshold: float = 0.5,
    min_keep: int = 10,
    max_iters: int = 20,
) -> Tensor:
    """计数约束 NMS.

    通过二分搜索置信度阈值, 使 NMS 后检测数尽可能接近 target_count.
    流程:
        1. 二分搜索 τ 使 sum(score > τ) ≈ target_count
        2. 对通过阈值的框做 batched_nms
        3. 兜底: 至少保留 min_keep 个 (防止过度抑制)

    Args:
        boxes: [N, 4] 预测框 (xyxy)
        scores: [N] 置信度
        labels: [N] 类别
        target_count: 目标检测数 (如 46)
        iou_threshold: NMS IoU 阈值
        min_keep: 最少保留数 (防止过度抑制)
        max_iters: 二分搜索迭代次数

    Returns:
        keep: [K] 保留的索引 (在原 N 中的位置)
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # 步骤 1: 二分搜索阈值
    tau = find_threshold_for_count(
        scores, target_count, max_iters=max_iters
    )

    # 步骤 2: 阈值过滤 + NMS
    mask = scores > tau
    if mask.sum() == 0:
        # 阈值过高, 退化为全量 NMS
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        return keep[:max(target_count, min_keep)]

    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]

    keep_filtered = batched_nms(
        filtered_boxes, filtered_scores, filtered_labels, iou_threshold
    )

    # 映射回原索引
    original_indices = mask.nonzero(as_tuple=True)[0]
    keep = original_indices[keep_filtered]

    # 步骤 3: 兜底 — 至少保留 min_keep 个
    if keep.shape[0] < min_keep:
        all_keep = batched_nms(boxes, scores, labels, iou_threshold)
        keep = all_keep[:max(target_count, min_keep)]

    return keep
