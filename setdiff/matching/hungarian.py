"""Hungarian Matcher — global coupled matching for joint diffusion.

Finds optimal one-to-one assignment between noise proposals and GT boxes
using ``scipy.optimize.linear_sum_assignment``. The matching depends on ALL
noise vectors (global coupling), not per-slot — this is the key difference
from per-proposal independent diffusion (DiffusionDet/DiffuDETR).

方案 B (unmatched_strategy='random_gt'):
    保留 Hungarian global coupled matching (matched slot 最优一对一),
    unmatched slot 分配随机 GT (保证所有 slot 有监督).
    动机: 原行为 unmatched slot = noise (velocity=0, box_head 无监督),
    导致训练-推理分布不匹配 → mAP=0. 方案 B 让所有 slot 在 [GT, noise]
    插值轨迹上, 同时保留 global coupled matching 的理论区分点.

match_indices (loss 阶段重新匹配, 2026-07-19 新增):
    参考 LDMDet criterion.py 的 loss 阶段重匹配思想 (matcher 不同: Hungarian
    1-to-1 vs LDMDet SimOTA 1-to-many): loss 计算时用 Hungarian 重新匹配
    pred_boxes 和 GT, num_pos=M (不是 N=300), 避免梯度稀释 37.5 倍.
    与 match() 区别: match() 返回 expanded matched_boxes/labels (coupling 阶段);
    match_indices() 返回 (src_idx, tgt_idx) 索引对 (loss 阶段).
"""

from typing import List, Tuple

import torch
from numpy import asarray
from scipy.optimize import linear_sum_assignment
from torch import Tensor


class HungarianMatcher:
    """Global coupled matching via Hungarian algorithm.

    Finds optimal one-to-one assignment between noise proposals and GT boxes.
    Key: the matching depends on ALL noise vectors (global coupling), not
    per-slot.

    Args:
        cost_type: 'l2' (default) or 'l1', cost matrix metric.
        unmatched_strategy: 控制 unmatched slot 的 x_start 分配策略.
            - 'noise' (默认, 向后兼容): unmatched slot = noise (velocity=0).
              ⚠️ 此策略导致 unmatched slot 训练-推理分布不匹配 (mAP=0 根因).
            - 'random_gt' (方案 B): unmatched slot 从 GT 集合随机采样 (允许重复),
              保证所有 slot 有监督. matched slot 仍由 Hungarian 一对一分配.
    """

    def __init__(
        self,
        cost_type: str = 'l2',
        unmatched_strategy: str = 'noise',
    ):
        self.cost_type = cost_type
        if unmatched_strategy not in ('noise', 'random_gt'):
            raise ValueError(
                f"unmatched_strategy 必须是 'noise' 或 'random_gt', "
                f"实际: {unmatched_strategy}"
            )
        self.unmatched_strategy = unmatched_strategy

    @torch.no_grad()
    def match(
        self,
        noise: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Match noise proposals to GT boxes.

        Args:
            noise: [N, 4] noise proposals in diffusion space (cxcywh).
            gt_boxes: [M, 4] GT boxes in diffusion space (cxcywh).
            gt_labels: [M] GT class labels.

        Returns:
            matched_boxes: [N, 4] GT boxes reordered to match noise slots.
                策略 'noise': unmatched slots = noise (velocity=0).
                策略 'random_gt': unmatched slots = 随机 GT (允许重复).
            matched_labels: [N] labels.
                策略 'noise': -1 for unmatched slots.
                策略 'random_gt': 所有 slot >=0 (M=0 时全 -1).
        """
        N = noise.shape[0]
        M = gt_boxes.shape[0]
        device = noise.device

        # 初始化默认值 (策略 'noise' 行为)
        matched_boxes = noise.clone()
        matched_labels = torch.full(
            (N,), -1, dtype=torch.long, device=device
        )

        if M == 0:
            return matched_boxes, matched_labels

        if M > N:
            # More GT than slots: keep only the N cheapest matches.
            gt_boxes = gt_boxes[:N]
            gt_labels = gt_labels[:N]
            M = N

        # Cost matrix: squared Euclidean distance in diffusion space.
        # cost[i, j] = ||noise[i] - gt[j]||^2
        noise_det = noise.detach()
        if self.cost_type == 'l2':
            diff = noise_det.unsqueeze(1) - gt_boxes.unsqueeze(0)  # [N, M, 4]
            cost = diff.pow(2).sum(-1)  # [N, M]
        else:
            cost = torch.cdist(noise_det, gt_boxes, p=1)  # [N, M]

        # scipy linear_sum_assignment operates on CPU.
        cost_cpu = asarray(cost.detach().cpu())
        row_ind, col_ind = linear_sum_assignment(cost_cpu)

        row_ind_t = torch.as_tensor(row_ind, device=device, dtype=torch.long)
        col_ind_t = torch.as_tensor(col_ind, device=device, dtype=torch.long)
        matched_boxes[row_ind_t] = gt_boxes[col_ind_t]
        matched_labels[row_ind_t] = gt_labels[col_ind_t]

        # ============================================================
        # 方案 B: unmatched slot 分配随机 GT
        # ============================================================
        if self.unmatched_strategy == 'random_gt':
            # 找出 unmatched slot (label 仍为 -1)
            unmatched_mask = matched_labels == -1  # [N]
            num_unmatched = unmatched_mask.sum().item()
            if num_unmatched > 0:
                # 从 M 个 GT 中独立均匀采样 num_unmatched 个 (允许重复)
                # 对齐 LDMDet: torch.randint(0, num_gt, (num_proposals,))
                rand_idx = torch.randint(
                    0, M, (num_unmatched,), device=device
                )
                matched_boxes[unmatched_mask] = gt_boxes[rand_idx]
                matched_labels[unmatched_mask] = gt_labels[rand_idx]

        return matched_boxes, matched_labels

    @torch.no_grad()
    def match_batch(
        self,
        noise_batch: Tensor,
        gt_boxes_list: List[Tensor],
        gt_labels_list: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Batch matching.

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
    # loss 阶段: match_indices (参考 LDMDet criterion.py 的重匹配思想)
    # ============================================================
    @torch.no_grad()
    def match_indices(
        self,
        pred_boxes: Tensor,
        gt_boxes: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Loss-stage matching: 找 pred_boxes 与 GT 的最优一对一匹配.

        参考 LDMDet criterion.py line 84 `self.matcher(outputs, targets)` 的
        loss 阶段重匹配思想 (matcher 不同: Hungarian 1-to-1 vs LDMDet SimOTA
        1-to-many): loss 计算时用 Hungarian 重新匹配 pred_boxes 和 GT,
        num_pos=M (不是 N=300), 避免梯度稀释 37.5 倍.

        与 match() 区别:
            - match() (coupling 阶段): 输入是 noise, 返回 expanded
              matched_boxes [N,4] / matched_labels [N] (所有 slot 都有值,
              unmatched_strategy 决定 unmatched slot 行为).
            - match_indices() (loss 阶段): 输入是 pred_boxes, 返回索引对
              (src_idx, tgt_idx), 长度 K=min(N,M). 只 matched slot 参与
              bbox/giou loss, num_pos=K.

        Args:
            pred_boxes: [N, 4] 模型预测的 x_0 (扩散空间 cxcywh).
            gt_boxes: [M, 4] 原始 GT (扩散空间 cxcywh).

        Returns:
            src_idx: [K] long, matched slot 在 N 中的索引.
            tgt_idx: [K] long, 对应 GT 在 M 中的索引.
            K = min(N, M).
        """
        N = pred_boxes.shape[0]
        M = gt_boxes.shape[0]
        device = pred_boxes.device

        if M == 0 or N == 0:
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return empty, empty

        # 截断: M > N 时只取前 N 个 GT (一对一限制)
        if M > N:
            gt_boxes = gt_boxes[:N]
            M = N

        # Cost matrix: squared Euclidean distance in diffusion space.
        pred_det = pred_boxes.detach()
        if self.cost_type == 'l2':
            diff = pred_det.unsqueeze(1) - gt_boxes.unsqueeze(0)  # [N, M, 4]
            cost = diff.pow(2).sum(-1)  # [N, M]
        else:
            cost = torch.cdist(pred_det, gt_boxes, p=1)  # [N, M]

        # scipy linear_sum_assignment operates on CPU.
        cost_cpu = asarray(cost.detach().cpu())
        row_ind, col_ind = linear_sum_assignment(cost_cpu)

        src_idx = torch.as_tensor(row_ind, device=device, dtype=torch.long)
        tgt_idx = torch.as_tensor(col_ind, device=device, dtype=torch.long)
        return src_idx, tgt_idx

    @torch.no_grad()
    def match_indices_batch(
        self,
        pred_boxes_batch: Tensor,
        gt_boxes_list: List[Tensor],
    ) -> List[Tuple[Tensor, Tensor]]:
        """批量 loss-stage matching.

        Args:
            pred_boxes_batch: [B, N, 4] 模型预测的 x_0 (扩散空间).
            gt_boxes_list: list of [M_i, 4] 每张图的 GT (扩散空间).

        Returns:
            results: list of (src_idx, tgt_idx), 长度 B.
        """
        B = pred_boxes_batch.shape[0]
        results = []
        for i in range(B):
            src_idx, tgt_idx = self.match_indices(
                pred_boxes_batch[i], gt_boxes_list[i]
            )
            results.append((src_idx, tgt_idx))
        return results
