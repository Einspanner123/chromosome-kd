"""检测损失计算核心类"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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

    def forward(self, outputs: ModelOutput, targets: List[InstanceData]) -> Dict[str, Tensor]:
        losses = self._get_loss(outputs, targets)
        if self.deep_supervision and outputs.aux_outputs is not None:
            for i, aux_out in enumerate(outputs.aux_outputs):
                aux_losses = self._get_loss(aux_out, targets)
                for name, val in aux_losses.items():
                    losses[f'aux_{i}_{name}'] = val
        return losses

    def _get_loss(self, outputs: ModelOutput, targets: List[InstanceData]) -> Dict[str, Tensor]:
        indices = self.matcher(outputs, targets)
        loss_cls = self._loss_classification(outputs, targets, indices)
        loss_bbox, loss_giou = self._loss_boxes(outputs, targets, indices)
        return {'loss_cls': loss_cls, 'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def _loss_classification(self, outputs, targets, indices) -> Tensor:
        src_logits = outputs.pred_logits
        bs, num_queries = src_logits.shape[:2]
        target_classes = src_logits.new_full((bs, num_queries), self.num_classes, dtype=torch.long)
        num_pos = 0
        for i, (src_idx, gt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[i, src_idx] = targets[i].labels[gt_idx]
                num_pos += len(src_idx)
        loss_cls = self.loss_cls(src_logits.flatten(0, 1), target_classes.flatten(0, 1))
        return loss_cls / max(num_pos, 1)

    def _loss_boxes(self, outputs, targets, indices) -> Tuple[Tensor, Tensor]:
        src_boxes = outputs.pred_boxes
        src_list = []
        tgt_list = []
        for i, (src_idx, gt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                src_list.append(src_boxes[i, src_idx])
                tgt_list.append(targets[i].bboxes[gt_idx])
        if len(src_list) == 0:
            return src_boxes.sum() * 0, src_boxes.sum() * 0

        src_boxes_pos = torch.cat(src_list)
        tgt_boxes_pos = torch.cat(tgt_list)
        num_pos = src_boxes_pos.shape[0]

        tgt_cxcywh = bbox_xyxy_to_cxcywh(tgt_boxes_pos)
        src_cxcywh = bbox_xyxy_to_cxcywh(src_boxes_pos)

        if self.scale_aware:
            tgt_areas = tgt_cxcywh[:, 2] * tgt_cxcywh[:, 3]
            if self.scale_aware_mode == 'log_linear':
                log_areas = torch.log(tgt_areas + 1e-8)
                log_mean = log_areas.mean()
                log_std = log_areas.std() + 1e-8
                z = (log_areas - log_mean) / log_std
                scale_w = 1.0 - self.scale_aware_alpha * z
                scale_w = scale_w.clamp(1.0 - self.scale_aware_alpha * 3, 1.0 + self.scale_aware_alpha * 3)
            elif self.scale_aware_mode == 'sqrt_inverse':
                raw_w = 1.0 / torch.sqrt(tgt_areas + 1e-6)
                scale_w = raw_w / raw_w.mean()
                scale_w = scale_w.clamp(self.scale_aware_min_weight, self.scale_aware_max_weight)
            else:
                raw_w = 1.0 / (tgt_areas + 1e-6)
                scale_w = raw_w / raw_w.mean()
                scale_w = scale_w.clamp(self.scale_aware_min_weight, self.scale_aware_max_weight)
            per_elem_l1 = F.l1_loss(src_cxcywh, tgt_cxcywh, reduction='none')
            weighted_sum = (per_elem_l1 * scale_w.unsqueeze(1)).sum()
            loss_bbox = self.loss_bbox.loss_weight * weighted_sum / (4.0 * num_pos * num_pos)
            loss_giou = self.loss_giou(src_boxes_pos, tgt_boxes_pos).sum() / num_pos
        elif self.bbox_loss_mode == 'relative_l1':
            tgt_w = tgt_cxcywh[:, 2].clamp(min=self.bbox_loss_eps)
            tgt_h = tgt_cxcywh[:, 3].clamp(min=self.bbox_loss_eps)
            scale = torch.stack([tgt_w, tgt_h, tgt_w, tgt_h], dim=-1)
            per_elem = F.l1_loss(src_cxcywh, tgt_cxcywh, reduction='none')
            loss_bbox = self.loss_bbox.loss_weight * (per_elem / scale).sum() / num_pos
            loss_giou = self.loss_giou(src_boxes_pos, tgt_boxes_pos).sum() / num_pos
        else:
            loss_bbox = self.loss_bbox(src_cxcywh, tgt_cxcywh).sum() / num_pos
            loss_giou = self.loss_giou(src_boxes_pos, tgt_boxes_pos).sum() / num_pos

        return loss_bbox, loss_giou
