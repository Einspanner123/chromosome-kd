"""SNR 感知动态匹配器 — 方向三: SNR 感知的动态匹配

在标准 SimOTA 基础上, 按扩散时间 t (经 SNR 变换) 加权匹配代价:
- 高噪声 (t→1): 代价权重低, 匹配更保守, dynamic_k 上界小
- 低噪声 (t→0): 代价权重高, 匹配更确定

双重保护:
1. w(t) 乘代价: 高噪声时代价整体降低, 减少错误匹配的梯度污染
2. cost_threshold: w < 0.1 时给代价加常数, 进一步抑制

开关: 通过配置 matcher=dict(type='SNRAwareMatcher', ...) 启用.
失败时直接删除本文件 + 还原 criterion 即可回滚.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ldmdet.criterion.matcher import DiffusionDetMatcher
from ldmdet.criterion.snr_weight import get_snr_weight
from ldmdet.data.structures import InstanceData, ModelOutput


class SNRAwareMatcher(DiffusionDetMatcher):
    """SNR 感知 SimOTA 匹配器.

    在标准 SimOTA 基础上, 按扩散时间 t (经 SNR 变换) 加权匹配代价.

    Args:
        cost_class: 分类代价权重 (继承父类)
        cost_bbox: L1 代价权重 (继承父类)
        cost_giou: GIoU 代价权重 (继承父类)
        center_radius: 中心约束半径 (继承父类)
        candidate_topk: dynamic_k 上界 (继承父类)
        match_costs: 自定义代价函数列表 (继承父类)
        snr_mode: 'logistic' | 'exponential' | 'none'
        snr_beta: exponential 模式的衰减率
        snr_w_min: 权重下界 (避免完全抑制, 推荐 0.1)
        k_max_scale: 是否用 w(t) 缩放 dynamic_k 上界
        cost_threshold: w < 0.1 时给代价加的常数 (高噪声保护)
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        center_radius: float = 2.5,
        candidate_topk: int = 5,
        match_costs: List = None,
        snr_mode: str = 'logistic',
        snr_beta: float = 3.0,
        snr_w_min: float = 0.1,
        k_max_scale: bool = True,
        cost_threshold: float = 50.0,
    ):
        super().__init__(
            cost_class=cost_class,
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
            center_radius=center_radius,
            candidate_topk=candidate_topk,
            match_costs=match_costs,
        )
        self.snr_mode = snr_mode
        self.snr_beta = snr_beta
        self.snr_w_min = snr_w_min
        self.k_max_scale = k_max_scale
        self.cost_threshold = cost_threshold

    @torch.no_grad()
    def forward(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        t: Optional[Tensor] = None,
    ) -> List[Tuple[Tensor, Tensor]]:
        """SNR 感知匹配.

        Args:
            outputs: 模型输出
            targets: GT 列表
            t: [bs] 扩散时间, 若为 None 则退化为标准匹配

        Returns:
            List[(fg_mask, matched_gt_inds)]
        """
        if t is None:
            return super().forward(outputs, targets)

        # 计算 SNR 权重: [bs]
        snr_w = get_snr_weight(
            t, mode=self.snr_mode, beta=self.snr_beta, w_min=self.snr_w_min
        )

        pred_logits = outputs.pred_logits
        pred_bboxes = outputs.pred_boxes
        bs = pred_logits.size(0)
        N = pred_logits.size(1)

        max_gt = max(tgt.bboxes.size(0) for tgt in targets)
        if max_gt == 0:
            return [(
                torch.zeros(N, dtype=torch.bool, device=pred_bboxes.device),
                torch.zeros(N, dtype=torch.long, device=pred_bboxes.device),
            )] * bs

        # 构建 GT padding
        gt_bboxes_padded = pred_bboxes.new_zeros(bs, max_gt, 4)
        gt_labels_padded = pred_logits.new_full((bs, max_gt), 0, dtype=torch.long)
        gt_num = []
        for i, tgt in enumerate(targets):
            n = tgt.bboxes.size(0)
            gt_num.append(n)
            if n > 0:
                gt_bboxes_padded[i, :n] = tgt.bboxes
                gt_labels_padded[i, :n] = tgt.labels

        # 批量代价
        cost_matrix, pairwise_ious = self._batched_cost(
            pred_logits, pred_bboxes, gt_labels_padded, gt_bboxes_padded, gt_num, max_gt
        )

        # SNR 加权代价: [bs] → [bs, 1, 1]
        snr_w_view = snr_w.view(bs, 1, 1)
        cost_matrix = cost_matrix * snr_w_view

        # 高噪声保护: w < 0.1 时给代价加常数, 进一步抑制
        high_noise_mask = snr_w < 0.1  # [bs]
        if high_noise_mask.any():
            cost_matrix[high_noise_mask] += self.cost_threshold

        # 逐图 dynamic_k 匹配
        results = []
        for i in range(bs):
            n = gt_num[i]
            if n == 0:
                results.append((
                    torch.zeros(N, dtype=torch.bool, device=pred_bboxes.device),
                    torch.zeros(N, dtype=torch.long, device=pred_bboxes.device),
                ))
            else:
                # k_max 由 SNR 缩放: 高噪声时减少正样本数
                k_max = self.candidate_topk
                if self.k_max_scale:
                    k_max = max(int(snr_w[i].item() * self.candidate_topk), 1)
                results.append(
                    self._dynamic_k_matching_t(
                        cost_matrix[i, :, :n],
                        pairwise_ious[i, :, :n],
                        n,
                        k_max,
                    )
                )
        return results

    @torch.no_grad()
    def forward_with_gt_cache(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        gt_cache: List[dict] = None,
        t: Optional[Tensor] = None,
    ) -> Tuple[List[Tuple[Tensor, Tensor]], List[dict]]:
        """带 GT 缓存的 SNR 感知匹配.

        t=None 时退化为父类行为 (标准匹配).
        """
        if t is None:
            return super().forward_with_gt_cache(outputs, targets, gt_cache)

        # 计算 SNR 权重
        snr_w = get_snr_weight(
            t, mode=self.snr_mode, beta=self.snr_beta, w_min=self.snr_w_min
        )

        pred_logits = outputs.pred_logits
        pred_bboxes = outputs.pred_boxes
        bs = pred_logits.size(0)
        N = pred_logits.size(1)

        # 构建或复用 GT 缓存
        if gt_cache is None:
            max_gt = max(tgt.bboxes.size(0) for tgt in targets)
            gt_bboxes_padded = pred_bboxes.new_zeros(bs, max_gt, 4)
            gt_labels_padded = pred_logits.new_full((bs, max_gt), 0, dtype=torch.long)
            gt_num = []
            new_cache = []
            for i, tgt in enumerate(targets):
                n = tgt.bboxes.size(0)
                gt_num.append(n)
                if n > 0:
                    gt_bboxes_padded[i, :n] = tgt.bboxes
                    gt_labels_padded[i, :n] = tgt.labels
                gt_ctrs = (tgt.bboxes[:, :2] + tgt.bboxes[:, 2:]) / 2 if n > 0 else tgt.bboxes.new_zeros(0, 2)
                gt_wh = (tgt.bboxes[:, 2:] - tgt.bboxes[:, :2]) if n > 0 else tgt.bboxes.new_zeros(0, 2)
                new_cache.append({
                    'gt_bboxes_padded': gt_bboxes_padded[i],
                    'gt_labels_padded': gt_labels_padded[i],
                    'gt_num': n,
                    'max_gt': max_gt,
                    'gt_ctrs': gt_ctrs,
                    'gt_wh': gt_wh,
                    'center_tl': gt_ctrs - self.center_radius * gt_wh if n > 0 else gt_ctrs,
                    'center_br': gt_ctrs + self.center_radius * gt_wh if n > 0 else gt_ctrs,
                })
        else:
            new_cache = gt_cache
            max_gt = new_cache[0]['max_gt']
            gt_bboxes_padded = torch.stack([c['gt_bboxes_padded'] for c in new_cache])
            gt_labels_padded = torch.stack([c['gt_labels_padded'] for c in new_cache])
            gt_num = [c['gt_num'] for c in new_cache]

        if max_gt == 0:
            empty = (
                torch.zeros(N, dtype=torch.bool, device=pred_bboxes.device),
                torch.zeros(N, dtype=torch.long, device=pred_bboxes.device),
            )
            return [empty] * bs, new_cache

        # 批量代价
        cost_matrix, pairwise_ious = self._batched_cost(
            pred_logits, pred_bboxes, gt_labels_padded, gt_bboxes_padded, gt_num, max_gt
        )

        # SNR 加权代价
        snr_w_view = snr_w.view(bs, 1, 1)
        cost_matrix = cost_matrix * snr_w_view

        # 高噪声保护
        high_noise_mask = snr_w < 0.1
        if high_noise_mask.any():
            cost_matrix[high_noise_mask] += self.cost_threshold

        # 逐图 dynamic_k 匹配
        indices = []
        for i in range(bs):
            n = gt_num[i]
            if n == 0:
                indices.append((
                    torch.zeros(N, dtype=torch.bool, device=pred_bboxes.device),
                    torch.zeros(N, dtype=torch.long, device=pred_bboxes.device),
                ))
            else:
                k_max = self.candidate_topk
                if self.k_max_scale:
                    k_max = max(int(snr_w[i].item() * self.candidate_topk), 1)
                indices.append(
                    self._dynamic_k_matching_t(
                        cost_matrix[i, :, :n],
                        pairwise_ious[i, :, :n],
                        n,
                        k_max,
                    )
                )
        return indices, new_cache

    @torch.no_grad()
    def _dynamic_k_matching_t(
        self,
        cost: Tensor,
        pairwise_ious: Tensor,
        num_gt: int,
        k_max: int,
    ) -> Tuple[Tensor, Tensor]:
        """t 感知的 dynamic k 匹配, k 上界由 SNR 调整.

        与父类 _dynamic_k_matching 的区别:
        - candidate_topk 替换为参数 k_max (受 SNR 缩放)
        """
        matching_matrix = torch.zeros_like(cost)
        pairwise_ious = torch.nan_to_num(pairwise_ious, nan=0.0, posinf=1.0, neginf=0.0)
        candidate_topk = min(k_max, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1, max=candidate_topk)

        # 向量化: 使用 topk 替代完整排序
        _, top_rows = torch.topk(cost, candidate_topk, dim=0, largest=False)
        gt_cols = torch.arange(num_gt, device=cost.device).unsqueeze(0).expand(candidate_topk, -1)
        row_positions = torch.arange(candidate_topk, device=cost.device).unsqueeze(1)
        valid_mask = row_positions < dynamic_ks.unsqueeze(0)
        valid_rows = top_rows[valid_mask]
        valid_cols = gt_cols[valid_mask]
        matching_matrix[valid_rows, valid_cols] = 1.0

        # 消除重复匹配
        duplicate_idx = matching_matrix.sum(1) > 1
        _, cost_argmin = cost.min(1)
        matching_matrix[duplicate_idx] = 0.0
        matching_matrix[duplicate_idx, cost_argmin[duplicate_idx]] = 1.0

        fg_mask = matching_matrix.sum(1) > 0
        matched_gt_inds = matching_matrix.argmax(1)
        return fg_mask, matched_gt_inds
