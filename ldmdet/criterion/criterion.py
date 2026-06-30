"""检测损失计算核心类"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import ops

from ldmdet.criterion.snr_aware_matcher import SNRAwareMatcher
from ldmdet.criterion.snr_weight import get_snr_weight
from ldmdet.data.structures import InstanceData, ModelOutput
from ldmdet.utils.box_ops import bbox_xyxy_to_cxcywh


class DiffusionDetCriterion(nn.Module):
    """DiffusionDet 损失计算核心"""

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        loss_cls: nn.Module,
        loss_bbox: nn.Module,
        loss_giou: nn.Module,
        deep_supervision: bool = True,
        scale_aware: bool = False,
        scale_aware_mode: str = 'inverse',
        scale_aware_min_weight: float = 0.5,
        scale_aware_max_weight: float = 3.0,
        scale_aware_alpha: float = 0.15,
        scale_aware_giou: bool = False,
        bbox_loss_mode: str = 'l1',
        bbox_loss_eps: float = 1e-2,
        # 方向三: SNR 感知损失加权 (默认全关, 不影响 baseline)
        snr_weighted_loss: bool = False,
        snr_mode: str = 'logistic',
        snr_beta: float = 3.0,
        snr_w_min: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.loss_giou = loss_giou
        self.deep_supervision = deep_supervision
        self.scale_aware = scale_aware
        self.scale_aware_mode = scale_aware_mode
        self.scale_aware_min_weight = scale_aware_min_weight
        self.scale_aware_max_weight = scale_aware_max_weight
        self.scale_aware_alpha = scale_aware_alpha
        self.scale_aware_giou = scale_aware_giou
        self.bbox_loss_mode = bbox_loss_mode
        self.bbox_loss_eps = bbox_loss_eps

        # 方向三: SNR 感知损失加权参数
        self.snr_weighted_loss = snr_weighted_loss
        self.snr_mode = snr_mode
        self.snr_beta = snr_beta
        self.snr_w_min = snr_w_min

        # 方向三诊断: SNR 诊断回调 (可选, 默认 None, 不影响 baseline)
        self.snr_diag_callback = None

    def forward(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        t: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # 主输出: 使用 matcher.forward 并建立 GT 缓存
        # 方向三: 若 matcher 是 SNRAwareMatcher 且 t 不为 None, 传 t 给 matcher
        if isinstance(self.matcher, SNRAwareMatcher) and t is not None:
            indices, gt_cache = self.matcher.forward_with_gt_cache(outputs, targets, t=t)
        else:
            indices, gt_cache = self.matcher.forward_with_gt_cache(outputs, targets)
        losses = self._get_loss(outputs, targets, indices, t)

        if self.deep_supervision and outputs.aux_outputs is not None:
            for i, aux_out in enumerate(outputs.aux_outputs):
                # aux_outputs 复用 GT 缓存，避免重复计算 gt_ctrs/gt_wh/center 区域
                if isinstance(self.matcher, SNRAwareMatcher) and t is not None:
                    aux_indices, gt_cache = self.matcher.forward_with_gt_cache(
                        aux_out, targets, gt_cache, t=t
                    )
                else:
                    aux_indices, gt_cache = self.matcher.forward_with_gt_cache(
                        aux_out, targets, gt_cache
                    )
                aux_losses = self._get_loss(aux_out, targets, aux_indices, t)
                for name, val in aux_losses.items():
                    losses[f'aux_{i}_{name}'] = val
        # 方向 D': 保存主输出的 indices 供外部读取 (用于正样本 reg_bias_loss)
        self._last_indices = indices
        return losses

    def _get_loss(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        indices: List[Tuple[Tensor, Tensor]] = None,
        t: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if indices is None:
            if isinstance(self.matcher, SNRAwareMatcher) and t is not None:
                indices = self.matcher(outputs, targets, t=t)
            else:
                indices = self.matcher(outputs, targets)
        loss_cls = self._loss_classification(outputs, targets, indices)
        loss_bbox, loss_giou = self._loss_boxes(outputs, targets, indices)

        # 方向三: SNR 加权损失 (开关控制, 默认不启用)
        # 与匹配代价加权保持一致, 避免匹配与损失不一致
        if self.snr_weighted_loss and t is not None:
            snr_w = get_snr_weight(
                t, mode=self.snr_mode, beta=self.snr_beta, w_min=self.snr_w_min
            )
            # 同 batch 同 t (RF 采样), 取均值作为标量权重
            weight = snr_w.mean()
            loss_cls = loss_cls * weight
            loss_bbox = loss_bbox * weight
            loss_giou = loss_giou * weight

            # 方向三诊断: 更新 SNR 权重统计 (若回调已注入)
            if self.snr_diag_callback is not None:
                self.snr_diag_callback.update(snr_w, t)

        return {'loss_cls': loss_cls, 'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def _loss_classification(self, outputs, targets, indices) -> Tensor:
        src_logits = outputs.pred_logits  # [bs, num_queries, num_classes+1]
        bs, num_queries = src_logits.shape[:2]

        # 构建 padded GT labels: [bs, max_gt]
        max_gt = max(t.labels.shape[0] for t in targets)
        if max_gt == 0:
            # 全部为背景
            target_classes = src_logits.new_full((bs, num_queries), self.num_classes, dtype=torch.long)
            num_pos = src_logits.new_tensor(1, dtype=torch.long)
            loss_cls = self.loss_cls(src_logits.flatten(0, 1), target_classes.flatten(0, 1))
            return loss_cls / num_pos

        gt_labels_padded = src_logits.new_full((bs, max_gt), self.num_classes, dtype=torch.long)
        for i, t in enumerate(targets):
            n = t.labels.shape[0]
            if n > 0:
                gt_labels_padded[i, :n] = t.labels

        # 批量化: 将 indices 堆叠为 [bs, N] 张量
        fg_masks = torch.stack([idx[0] for idx in indices])  # [bs, N] bool
        matched_gt_inds = torch.stack([idx[1] for idx in indices])  # [bs, N] long
        matched_gt_inds_clamped = matched_gt_inds.clamp(min=0)

        # Gather GT labels: [bs, N]
        matched_gt_labels = torch.gather(gt_labels_padded, 1, matched_gt_inds_clamped)
        # 背景位置设为 num_classes
        matched_gt_labels[~fg_masks] = self.num_classes

        target_classes = matched_gt_labels
        # num_pos 保留为张量，避免 .item() 同步
        num_pos = fg_masks.sum().clamp(min=1)

        loss_cls = self.loss_cls(src_logits.flatten(0, 1), target_classes.flatten(0, 1))
        return loss_cls / num_pos

    def _loss_boxes(self, outputs, targets, indices) -> Tuple[Tensor, Tensor]:
        src_boxes = outputs.pred_boxes  # [bs, num_queries, 4]
        bs = src_boxes.shape[0]

        # 构建 padded GT bboxes: [bs, max_gt, 4]
        max_gt = max(t.bboxes.shape[0] for t in targets)
        if max_gt == 0:
            # 无正样本，返回零损失
            return src_boxes.sum() * 0, src_boxes.sum() * 0

        # 批量化: 将 indices 堆叠为 [bs, N] 张量
        fg_masks = torch.stack([idx[0] for idx in indices])  # [bs, N] bool
        matched_gt_inds = torch.stack([idx[1] for idx in indices])  # [bs, N] long

        # 构建 padded GT bboxes: [bs, max_gt, 4]
        max_gt = max(t.bboxes.shape[0] for t in targets)
        gt_bboxes_padded = src_boxes.new_zeros(bs, max_gt, 4)
        for i, t in enumerate(targets):
            n = t.bboxes.shape[0]
            if n > 0:
                gt_bboxes_padded[i, :n] = t.bboxes

        # Gather matched GT bboxes: [bs, N, 4]
        matched_gt_inds_clamped = matched_gt_inds.clamp(min=0)
        tgt_boxes = torch.gather(
            gt_bboxes_padded, 1,
            matched_gt_inds_clamped.unsqueeze(-1).expand(-1, -1, 4)
        )

        # num_pos 保留为张量，避免 .item() 同步
        num_pos = fg_masks.sum().clamp(min=1)

        tgt_cxcywh = bbox_xyxy_to_cxcywh(tgt_boxes)  # [bs, N, 4]
        src_cxcywh = bbox_xyxy_to_cxcywh(src_boxes)  # [bs, N, 4]

        if self.scale_aware:
            tgt_areas = tgt_cxcywh[:, :, 2] * tgt_cxcywh[:, :, 3]  # [bs, N]
            if self.scale_aware_mode == 'log_linear':
                log_areas = torch.log(tgt_areas + 1e-8)
                log_mean = log_areas[fg_masks].mean()
                log_std = log_areas[fg_masks].std() + 1e-8
                z = (log_areas - log_mean) / log_std
                scale_w = 1.0 - self.scale_aware_alpha * z
                scale_w = scale_w.clamp(1.0 - self.scale_aware_alpha * 3, 1.0 + self.scale_aware_alpha * 3)
            elif self.scale_aware_mode == 'sqrt_inverse':
                raw_w = 1.0 / torch.sqrt(tgt_areas + 1e-6)
                scale_w = raw_w / raw_w[fg_masks].mean()
                scale_w = scale_w.clamp(self.scale_aware_min_weight, self.scale_aware_max_weight)
            else:
                raw_w = 1.0 / (tgt_areas + 1e-6)
                scale_w = raw_w / raw_w[fg_masks].mean()
                scale_w = scale_w.clamp(self.scale_aware_min_weight, self.scale_aware_max_weight)
            per_elem_l1 = F.l1_loss(src_cxcywh, tgt_cxcywh, reduction='none')  # [bs, N, 4]
            weighted_l1 = per_elem_l1 * scale_w.unsqueeze(-1)  # [bs, N, 4]
            # 仅对正样本求和
            weighted_sum = (weighted_l1 * fg_masks.unsqueeze(-1).float()).sum()
            loss_bbox = self.loss_bbox.loss_weight * weighted_sum / num_pos
            # GIoU: 逐元素计算后 mask
            per_giou = ops.generalized_box_iou_loss(
                src_boxes.reshape(-1, 4), tgt_boxes.reshape(-1, 4), reduction='none'
            ).reshape(bs, -1)
            per_giou = torch.nan_to_num(per_giou, nan=0.0)
            if self.scale_aware_giou:
                # GIoU 也按 scale_w 加权
                per_giou = per_giou * scale_w
            loss_giou = self.loss_giou.loss_weight * (per_giou * fg_masks.float()).sum() / num_pos
        elif self.bbox_loss_mode == 'relative_l1':
            tgt_w = tgt_cxcywh[:, :, 2].clamp(min=self.bbox_loss_eps)
            tgt_h = tgt_cxcywh[:, :, 3].clamp(min=self.bbox_loss_eps)
            scale = torch.stack([tgt_w, tgt_h, tgt_w, tgt_h], dim=-1)
            per_elem = F.l1_loss(src_cxcywh, tgt_cxcywh, reduction='none')
            masked_rel = (per_elem / scale) * fg_masks.unsqueeze(-1).float()
            loss_bbox = self.loss_bbox.loss_weight * masked_rel.sum() / num_pos
            per_giou = ops.generalized_box_iou_loss(
                src_boxes.reshape(-1, 4), tgt_boxes.reshape(-1, 4), reduction='none'
            ).reshape(bs, -1)
            per_giou = torch.nan_to_num(per_giou, nan=0.0)
            loss_giou = self.loss_giou.loss_weight * (per_giou * fg_masks.float()).sum() / num_pos
        else:
            # L1 loss: 逐元素计算后 mask
            per_elem_l1 = F.l1_loss(src_cxcywh, tgt_cxcywh, reduction='none')  # [bs, N, 4]
            masked_l1 = per_elem_l1 * fg_masks.unsqueeze(-1).float()
            loss_bbox = self.loss_bbox.loss_weight * masked_l1.sum() / num_pos
            # GIoU loss: 逐元素计算后 mask
            per_giou = ops.generalized_box_iou_loss(
                src_boxes.reshape(-1, 4), tgt_boxes.reshape(-1, 4), reduction='none'
            ).reshape(bs, -1)
            per_giou = torch.nan_to_num(per_giou, nan=0.0)
            loss_giou = self.loss_giou.loss_weight * (per_giou * fg_masks.float()).sum() / num_pos

        return loss_bbox, loss_giou
