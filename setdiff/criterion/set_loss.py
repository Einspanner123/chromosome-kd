"""SetCriterion — loss for joint diffusion detection.

参考 LDMDet criterion.py 的 loss 阶段重匹配思想 (2026-07-19 修复):
    loss 计算时用 Hungarian 重新匹配 pred_boxes 和 GT, num_pos=M (不是 N=300),
    避免梯度稀释 37.5 倍 (旧 planA/B per-slot 梯度 0.000733 vs LDMDet baseline
    0.025125, 实测 16/600 比值).

    ⚠️ 与 LDMDet 的关键区别 (设计选择, 非完全对齐):
        - LDMDet matcher: SimOTA 动态 Top-K (1-to-many), num_pos = sum(dynamic_k_i),
          通常 > M (每 GT 可匹配多个 proposal).
        - SetDiff matcher: Hungarian 1-to-1 (对齐 DETR 家族 + v4 理论 "global
          coupled matching"), num_pos = sum(M_i) = M.
        - SetDiff 的 num_pos 比 LDMDet 更小, per-slot 梯度更大 (~1-3x, 取决于
          dynamic_k), 这是为了保留 SetDiff v4 理论区分点 (一对一全局耦合匹配).

    coupling 阶段 (matcher.match_batch) 仍负责构造轨迹 x_t = (1-t)*x_0_matched
    + t*noise, 保证所有 slot 在 [GT, noise] 插值轨迹上 (训练-推理分布对齐);
    loss 阶段 (criterion 内部 matcher.match_indices_batch) 独立做 Hungarian
    一对一匹配, 决定哪些 slot 是正样本 (参与 bbox/giou loss).

    两阶段分离的设计 (与 LDMDet 一致):
        - coupling 阶段: 控制 box_head 见过的输入分布 (所有 slot 都见过 GT-noise
          插值, 不再有 unmatched slot 的 OOD 问题).
        - loss 阶段: 控制梯度归一化 (num_pos=M, 梯度集中, 不稀释).
    LDMDet 同时满足两个目标, 此处参考其行为 (matcher 实现不同).

Loss 组成 (方案 A 参考 LDMDet/DiffusionDet):
1. Classification loss (sigmoid focal loss on class logits, w=2).
   - 所有 slot 参与 (matched=正样本, unmatched=背景).
2. Box regression loss (L1 on predicted x_0, cxcywh, w=5).
   - 只对 matched slot 计算, num_pos=M.
3. GIoU loss (GIoU on predicted x_0, xyxy, w=2).
   - 只对 matched slot 计算, num_pos=M.

权重 2:5:2 对齐 DETR 家族 (DETR/Deformable DETR/DINO/LDMDet).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ldmdet.utils.box_ops import bbox_cxcywh_to_xyxy
from setdiff.matching.hungarian import HungarianMatcher


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """GIoU for cxcywh format boxes. Both [N, 4].

    Returns:
        giou: [N] GIoU values in [-1, 1].
    """
    # Convert cxcywh → xyxy
    b1 = bbox_cxcywh_to_xyxy(boxes1)
    b2 = bbox_cxcywh_to_xyxy(boxes2)

    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(
        min=0
    )
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(
        min=0
    )

    # Intersection
    lt = torch.max(b1[:, :2], b2[:, :2])
    rb = torch.min(b1[:, 2:], b2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-8)

    # Enclosing box
    enc_lt = torch.min(b1[:, :2], b2[:, :2])
    enc_rb = torch.max(b1[:, 2:], b2[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[:, 0] * enc_wh[:, 1]

    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-8)
    return giou


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Sigmoid focal loss (sum reduction).

    Args:
        inputs: [N, C] logits.
        targets: [N, C] one-hot targets.
        alpha: balancing factor.
        gamma: focusing parameter.

    Returns:
        loss: scalar (sum over elements).
    """
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    return loss.sum()


class SetCriterion(nn.Module):
    """Loss for joint diffusion detection (对齐 LDMDet criterion.py).

    Combines:
    1. Classification loss (focal loss on class logits, w=2).
       所有 slot 参与 (matched=正样本, unmatched=背景).
    2. Box regression loss (L1 on predicted x_0, w=5).
       只对 matched slot 计算, num_pos=M (对齐 LDMDet, 不稀释).
    3. GIoU loss (GIoU on predicted x_0, w=2).
       只对 matched slot 计算, num_pos=M.

    内置 HungarianMatcher: loss 计算时重新匹配 pred_boxes 和 GT.
    coupling 阶段的 matched_boxes/matched_labels 不再传入 criterion
    (它们仅用于轨迹构造, 由 set_head._forward_train 处理).

    GIoU 计算空间 (关键修复):
        GIoU 必须在 [0,1] 归一化空间计算, 不是扩散空间 [-s, s].
        根因: cxcywh 的 w,h 在扩散空间可能为负 (小目标), 导致框翻转.
    """

    def __init__(
        self,
        num_classes: int,
        weight_dict: Optional[Dict[str, float]] = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
        snr_scale: float = 2.0,
        cost_type: str = 'l2',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        # snr_scale: GT 从 [0,1] 缩放到 [-s, +s] 匹配 N(0,1) 噪声.
        # _loss_giou 需要逆变换回 [0,1] 空间计算 GIoU (修复框翻转 bug).
        self.snr_scale = snr_scale
        # loss 阶段重新匹配的 matcher (对齐 LDMDet criterion.py line 84).
        # 与 set_head 中的 coupling matcher 区分: coupling matcher 决定轨迹,
        # loss matcher 决定哪些 slot 是正样本.
        self.matcher = HungarianMatcher(cost_type=cost_type)
        if weight_dict is None:
            # 方案 A: 2:5:2 对齐 DETR 家族 (cls : bbox : giou)
            weight_dict = {
                'loss_cls': 2.0,
                'loss_bbox': 5.0,
                'loss_giou': 2.0,
            }
        self.weight_dict = weight_dict

    def forward(
        self,
        outputs: Dict[str, Tensor],
        gt_boxes_list: List[Tensor],
        gt_labels_list: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """Args:
            outputs: dict with 'pred_logits' [B, N, C], 'pred_boxes'
                [B, N, 4] (扩散空间 cxcywh).
            gt_boxes_list: list of [M_i, 4] 每张图的原始 GT (扩散空间 cxcywh).
            gt_labels_list: list of [M_i] 每张图的原始 GT label.

        Returns:
            loss_dict: dict of loss terms (loss_cls, loss_bbox, loss_giou).
            loss: scalar total loss (weighted sum).
        """
        pred_logits = outputs['pred_logits']  # [B, N, C]
        pred_boxes = outputs['pred_boxes']  # [B, N, 4]

        # 1. loss 阶段重新匹配 (对齐 LDMDet criterion.py line 84)
        # 返回 List[(src_idx [K_i], tgt_idx [K_i])], K_i = min(N, M_i)
        indices_list = self.matcher.match_indices_batch(
            pred_boxes, gt_boxes_list
        )

        # 2. cls loss: 所有 slot 参与 (matched=正样本, unmatched=背景)
        loss_cls = self._loss_classification(
            pred_logits, gt_labels_list, indices_list
        )

        # 3. bbox/giou loss: 只对 matched slot, num_pos = sum(K_i) = sum(M_i)
        loss_bbox = self._loss_bbox(pred_boxes, gt_boxes_list, indices_list)
        loss_giou = self._loss_giou(pred_boxes, gt_boxes_list, indices_list)

        loss_dict = {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
        }

        total = sum(
            loss_dict[k] * self.weight_dict.get(k, 1.0) for k in loss_dict
        )
        return loss_dict, total

    def _loss_classification(
        self,
        pred_logits: Tensor,
        gt_labels_list: List[Tensor],
        indices_list: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        """Sigmoid focal loss over ALL slots.

        - Matched slots (由 matcher.match_indices 决定): target 是其匹配 GT
          类别的 one-hot.
        - Unmatched slots: target 是全零 (背景).

        所有 slot 都参与 (DETR 标准), unmatched slot 学习抑制所有类别分数.
        num_pos = sum(K_i) (batch 总 matched 数, = sum(M_i) 因为一对一).
        """
        B, N, C = pred_logits.shape
        device = pred_logits.device

        # 构建每张图的 target_labels: [B, N], -1 表示 unmatched (背景)
        target_labels = torch.full(
            (B, N), -1, dtype=torch.long, device=device
        )
        for b, (src_idx, tgt_idx) in enumerate(indices_list):
            if len(src_idx) > 0:
                target_labels[b, src_idx] = gt_labels_list[b][tgt_idx]

        # One-hot targets: [B, N, C]
        targets = torch.zeros_like(pred_logits)
        valid_mask = target_labels >= 0  # [B, N]
        if valid_mask.any():
            valid_labels = target_labels[valid_mask]  # [num_valid]
            targets[valid_mask] = torch.zeros_like(
                targets[valid_mask]
            ).scatter_(1, valid_labels.clamp(0, C - 1).unsqueeze(1), 1.0)

        # num_pos 保留为张量, 避免 .item() 同步 (对齐 LDMDet criterion.py)
        num_pos = valid_mask.sum().clamp(min=1)

        flatten_logits = pred_logits.reshape(-1, C)
        flatten_targets = targets.reshape(-1, C)

        loss = sigmoid_focal_loss(
            flatten_logits,
            flatten_targets,
            alpha=self.alpha,
            gamma=self.gamma,
        )
        # 用 batch 内 matched slot 总数归一化 (对齐 mmdet DETR).
        return loss / num_pos

    def _loss_bbox(
        self,
        pred_boxes: Tensor,
        gt_boxes_list: List[Tensor],
        indices_list: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        """L1 loss on matched slots (x_0 prediction), num_pos = sum(M_i).

        对齐 LDMDet/DiffusionDet: L1 损失承担 x_0 预测回归.
        只对 matcher.match_indices 决定的 matched slot 计算 (一对一),
        num_pos = sum(M_i) 而不是 N, 避免梯度稀释.

        关键修复 — L1 必须在 [0,1] 归一化空间计算, 不是扩散空间 [-s, s]:
            根因: loss_bbox 在扩散空间计算时数值放大 2s 倍 (s=2 → 4x),
                  而 loss_giou 的逆变换引入 1/(2s) 梯度衰减,
                  两者不在同一空间导致有效权重失衡.
            修复: 与 _loss_giou 一致, 先把 pred/tgt 从 [-s, s] 逆变换到 [0, 1]
                  再计算 L1.
        """
        # 收集所有 matched (pred_box, gt_box) 对
        pred_list = []
        tgt_list = []
        for b, (src_idx, tgt_idx) in enumerate(indices_list):
            if len(src_idx) > 0:
                pred_list.append(pred_boxes[b, src_idx])
                tgt_list.append(gt_boxes_list[b][tgt_idx])

        if not pred_list:
            return pred_boxes.sum() * 0.0

        pred = torch.cat(pred_list, dim=0)  # [sum(M_i), 4] 扩散空间 [-s, s]
        tgt = torch.cat(tgt_list, dim=0)
        num_pos = pred.shape[0]  # = sum(M_i)

        # 逆变换: [-s, s] → [0, 1] (与 _loss_giou 一致, 保证梯度尺度对齐)
        s = self.snr_scale
        pred_norm = (pred.clamp(-s, s) / s + 1.0) / 2.0
        tgt_norm = (tgt.clamp(-s, s) / s + 1.0) / 2.0

        return F.l1_loss(pred_norm, tgt_norm, reduction='sum') / max(num_pos, 1)

    def _loss_giou(
        self,
        pred_boxes: Tensor,
        gt_boxes_list: List[Tensor],
        indices_list: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        """GIoU loss on matched slots (x_0 prediction), num_pos = sum(M_i).

        对齐 LDMDet/DiffusionDet: GIoU 提供几何重叠监督, 是检测 AP 的直接代理.
        只对 matcher.match_indices 决定的 matched slot 计算 (一对一),
        num_pos = sum(M_i) 而不是 N.

        关键修复 — GIoU 必须在 [0,1] 归一化空间计算, 不是扩散空间 [-s, s]:
            根因: cxcywh 的 w,h 在扩散空间可能为负 (小目标 GT 经
                  (x*2-1)*snr_scale 变换后). 例如 GT w=0.1 → 扩散空间 w=-1.6.
                  bbox_cxcywh_to_xyxy 后: x1=cx-w/2 > x2=cx+w/2 (框翻转),
                  area=0, giou=0, loss_giou=1.0 恒定.
                  染色体小目标 w,h 通常 0.05-0.3, 全部为负 → loss_giou 完全失效.
            修复: 调用 generalized_box_iou 前, 先把 pred/tgt 从 [-s, s]
                  逆变换到 [0, 1] (与 SetDiffDetector.diffusion_to_norm_space 一致).
        """
        pred_list = []
        tgt_list = []
        for b, (src_idx, tgt_idx) in enumerate(indices_list):
            if len(src_idx) > 0:
                pred_list.append(pred_boxes[b, src_idx])
                tgt_list.append(gt_boxes_list[b][tgt_idx])

        if not pred_list:
            return pred_boxes.sum() * 0.0

        pred = torch.cat(pred_list, dim=0)
        tgt = torch.cat(tgt_list, dim=0)
        num_pos = pred.shape[0]

        # 逆变换: [-s, s] → [0, 1] (GIoU 需要在有效框空间计算, 修复框翻转 bug)
        s = self.snr_scale
        pred_norm = (pred.clamp(-s, s) / s + 1.0) / 2.0
        tgt_norm = (tgt.clamp(-s, s) / s + 1.0) / 2.0

        giou = generalized_box_iou(pred_norm, tgt_norm)
        return (1.0 - giou).sum() / max(num_pos, 1)
