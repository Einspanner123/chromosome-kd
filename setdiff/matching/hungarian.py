"""Hungarian Matcher — global coupled matching for joint diffusion.

Finds optimal one-to-one assignment between noise proposals and GT boxes
using ``scipy.optimize.linear_sum_assignment``. The matching depends on ALL
noise vectors (global coupling), not per-slot — this is the key difference
from per-proposal independent diffusion (DiffusionDet/DiffuDETR).
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
    """

    def __init__(self, cost_type: str = 'l2'):
        self.cost_type = cost_type

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
                Unmatched slots receive the noise itself (so velocity = 0).
            matched_labels: [N] labels (-1 for unmatched/padding slots).
        """
        N = noise.shape[0]
        M = gt_boxes.shape[0]
        device = noise.device

        # Default: unmatched slots keep the noise itself (velocity = 0),
        # label = -1 (ignored by loss).
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
