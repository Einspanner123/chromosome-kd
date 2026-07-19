"""Random Matcher — 方案 A: 所有 slot 随机分配 GT.

对齐 LDMDet `_couple_single_image` (head.py:851-853):
    idx = torch.randint(0, num_gt, (num_proposals,))
    return gt_diffusion[idx], idx  # ALL N proposals get a GT!

核心区别 vs HungarianMatcher:
  - HungarianMatcher: global coupled one-to-one matching, unmatched slot
    保留 noise (velocity=0), box_head 对这些 slot 完全无监督.
  - RandomMatcher: 所有 N 个 slot 随机从 M 个 GT 中采样 (允许重复),
    所有 slot 都在 [GT, noise] 插值轨迹上, box_head 对所有 slot 都有监督.

设计动机 (mAP=0 根因):
  HungarianMatcher.match() line 53 `matched_boxes = noise.clone()` 让
  unmatched slot (训练时约 85%) 的 x_t 恒为 noise (与 t 无关),
  pred_boxes 完全无监督; 推理时这些 slot 输出垃圾, Euler step 把 x_t
  推到 OOD, 通过 self-attention 污染 matched slot → mAP=0.

  方案 A 彻底放弃 Hungarian, 所有 slot 都有 GT 监督, 训练-推理分布对齐.
  代价: 失去 global coupled matching (理论 v4 的核心区分点).
"""

from typing import List, Tuple

import torch
from torch import Tensor


class RandomMatcher:
    """方案 A: 所有 slot 随机分配 GT (对齐 LDMDet).

    每个 slot 从 M 个 GT 中独立均匀采样 (允许重复, 当 N >> M 时大部分 GT
    会被采样多次). 这与 LDMDet `_couple_single_image` 的
    `torch.randint(0, num_gt, (num_proposals,))` 完全一致.
    """

    def __init__(self, cost_type: str = 'l2'):
        # cost_type 保留参数兼容性, 但方案 A 不使用 cost matrix
        self.cost_type = cost_type

    @torch.no_grad()
    def match(
        self,
        noise: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """所有 slot 随机分配 GT.

        Args:
            noise: [N, 4] noise proposals in diffusion space (cxcywh).
                方案 A 中 noise 不参与匹配决策, 仅作为扩散终点.
            gt_boxes: [M, 4] GT boxes in diffusion space (cxcywh).
            gt_labels: [M] GT class labels.

        Returns:
            matched_boxes: [N, 4] 每个 slot 分配的 GT box (允许重复).
            matched_labels: [N] 每个 slot 分配的 GT label (>=0, 无 -1).
                M=0 时所有 slot label=-1, matched_boxes=noise.
        """
        N = noise.shape[0]
        M = gt_boxes.shape[0]
        device = noise.device

        # 边界: M=0, 无 GT 可分配, 回退到 noise + label=-1
        if M == 0:
            matched_boxes = noise.clone()
            matched_labels = torch.full(
                (N,), -1, dtype=torch.long, device=device
            )
            return matched_boxes, matched_labels

        # 方案 A 核心: 所有 N 个 slot 从 M 个 GT 中独立均匀采样 (允许重复)
        # 对齐 LDMDet: idx = torch.randint(0, num_gt, (num_proposals,))
        idx = torch.randint(0, M, (N,), device=device)
        matched_boxes = gt_boxes[idx]  # [N, 4]
        matched_labels = gt_labels[idx]  # [N]

        return matched_boxes, matched_labels

    @torch.no_grad()
    def match_batch(
        self,
        noise_batch: Tensor,
        gt_boxes_list: List[Tensor],
        gt_labels_list: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """批量匹配: 每张图独立随机采样.

        Args:
            noise_batch: [B, N, 4]
            gt_boxes_list: list of [M_i, 4] tensors
            gt_labels_list: list of [M_i] tensors

        Returns:
            matched_boxes: [B, N, 4]
            matched_labels: [B, N]
        """
        B, N, _ = noise_batch.shape
        device = noise_batch.device

        matched_boxes = noise_batch.clone()
        matched_labels = torch.full(
            (B, N), -1, dtype=torch.long, device=device
        )

        for i in range(B):
            mb, ml = self.match(
                noise_batch[i], gt_boxes_list[i], gt_labels_list[i]
            )
            matched_boxes[i] = mb
            matched_labels[i] = ml

        return matched_boxes, matched_labels

    # ============================================================
    # 方向 A (DS 路径 A, 2026-07-19): coupling 用 pred_boxes matching
    # ============================================================
    @torch.no_grad()
    def match_coupling_batch(
        self,
        proposals: Tensor,
        noise: Tensor,
        gt_boxes_list: List[Tensor],
        gt_labels_list: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """方向 A 接口兼容: 忽略 proposals, 直接复用 match_batch.

        RandomMatcher 不基于 cost matrix (直接 torch.randint 随机采样),
        proposals 参数无意义. 此方法仅为统一 matcher 接口 (HungarianMatcher
        和 RandomMatcher 都支持 match_coupling_batch), 以便 set_head._forward_train
        能用统一代码路径调用.

        Args:
            proposals: [B, N, 4] 忽略 (RandomMatcher 不基于 cost matrix).
            noise: [B, N, 4] 仅用于 fallback (与 match_batch 的 noise_batch 一致).
            gt_boxes_list: list of [M_i, 4] tensors.
            gt_labels_list: list of [M_i] tensors.

        Returns:
            与 match_batch(noise, ...) 完全一致.
        """
        return self.match_batch(noise, gt_boxes_list, gt_labels_list)
