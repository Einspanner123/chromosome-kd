from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import ops

from .structures import InstanceData, ModelOutput
from .utils import bbox_xyxy_to_cxcywh


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
) -> Tensor:
    """
    Focal Loss 纯 PyTorch 实现 (Sigmoid 版本)
    参考: https://arxiv.org/abs/1708.02002
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction='none'
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


class FocalLoss(nn.Module):
    """Focal Loss 封装类"""

    def __init__(
        self,
        use_sigmoid=True,
        alpha=0.25,
        gamma=2.0,
        reduction='sum',
        loss_weight=2.0,
    ):
        super().__init__()
        assert use_sigmoid, 'Currently only supports sigmoid focal loss'
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        pred: [..., num_classes] (logits)
        target: [...] (class indices) or [..., num_classes] (one-hot)
        """
        if target.dim() == pred.dim() - 1:
            num_classes = pred.shape[-1]
            # 创建 one-hot 编码，背景类（通常是 num_classes）设为全 0
            flat_pred = pred.reshape(-1, num_classes)
            flat_target = target.reshape(-1)
            flat_one_hot = torch.zeros_like(flat_pred)

            valid_mask = (flat_target >= 0) & (flat_target < num_classes)
            if valid_mask.any():
                flat_one_hot[valid_mask, flat_target[valid_mask]] = 1.0
            target = flat_one_hot.reshape(pred.shape)

        loss = sigmoid_focal_loss(
            pred, target, self.alpha, self.gamma, self.reduction
        )
        return loss * self.loss_weight


class GIoULoss(nn.Module):
    """GIoU Loss 封装类"""

    def __init__(self, reduction='sum', loss_weight=2.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """pred/target: [N, 4] (xyxy)"""
        loss = ops.generalized_box_iou_loss(
            pred, target, reduction=self.reduction
        )
        return loss * self.loss_weight


class L1Loss(nn.Module):
    """L1 Loss 封装类"""

    def __init__(self, reduction='sum', loss_weight=5.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = F.l1_loss(pred, target, reduction=self.reduction)
        return loss * self.loss_weight


# --- Matching Costs (用于匹配阶段) ---


class FocalLossCost:
    def __init__(self, alpha=0.25, gamma=2.0, weight=2.0, eps=1e-8):
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.eps = eps

    def __call__(
        self,
        pred_logits: Tensor,
        pred_bboxes: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
    ) -> Tensor:
        """
        pred_logits: [N, C]
        gt_labels: [M]
        Returns: [N, M] cost matrix
        """
        num_classes = pred_logits.shape[-1]
        out_prob = pred_logits.sigmoid()

        # 防护: 越界标签会触发 CUDA device-side assert，
        # clamp 到 [0, num_classes-1] 是无奈之举，实际训练中不应出现
        gt_labels = gt_labels.clamp(0, num_classes - 1)

        # 计算所有类别的负样本 cost
        neg_cost_class = (
            -(1 - self.alpha)
            * (out_prob**self.gamma)
            * torch.log(1 - out_prob + self.eps)
        )
        # 计算所有类别的正样本 cost
        pos_cost_class = (
            -self.alpha
            * ((1 - out_prob) ** self.gamma)
            * torch.log(out_prob + self.eps)
        )
        # 为每个 GT 选择对应的类别 cost
        cost_class = (
            pos_cost_class[:, gt_labels] - neg_cost_class[:, gt_labels]
        )
        return cost_class * self.weight


class BBoxL1Cost:
    def __init__(self, weight=5.0):
        self.weight = weight

    def __call__(
        self,
        pred_logits: Tensor,
        pred_bboxes: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
    ) -> Tensor:
        """
        pred_bboxes: [N, 4] (normalized xyxy)
        gt_bboxes: [M, 4] (normalized xyxy)
        """
        # 清洗 NaN/Inf 并 clamp，防止 CUDA device-side assert
        pred_bboxes = torch.nan_to_num(
            pred_bboxes.detach(), nan=0.5, posinf=1.0, neginf=0.0
        ).clamp(0, 1)
        gt_bboxes = torch.nan_to_num(
            gt_bboxes, nan=0.5, posinf=1.0, neginf=0.0
        ).clamp(0, 1)
        return torch.cdist(pred_bboxes, gt_bboxes, p=1) * self.weight


class RelativeL1Cost:
    def __init__(self, weight=5.0, eps=1e-2):
        self.weight = weight
        self.eps = eps

    def __call__(
        self,
        pred_logits: Tensor,
        pred_bboxes: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
    ) -> Tensor:
        """
        pred_bboxes: [N, 4] (normalized xyxy)
        gt_bboxes: [M, 4] (normalized xyxy)
        Returns: [N, M] cost matrix with relative L1 distance
        """
        # GPU 原生 NaN/Inf 清洗: 不触发 CPU 同步, 从根源防止 CUDA device-side assert
        pred_bboxes = torch.nan_to_num(
            pred_bboxes, nan=0.5, posinf=1.0, neginf=0.0
        )
        pred_bboxes = pred_bboxes.detach().clamp(0, 1)
        gt_bboxes = torch.nan_to_num(
            gt_bboxes, nan=0.5, posinf=1.0, neginf=0.0
        )
        # gt_bboxes 也需要 clamp 到 [0,1]，与 pred_bboxes 对等，
        # 否则大值经 cxcywh 转换后直接触发 CUDA device-side assert
        gt_bboxes = gt_bboxes.clamp(0, 1)

        pred_cxcywh = bbox_xyxy_to_cxcywh(pred_bboxes)
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)

        # 二次清洗: bbox_xyxy_to_cxcywh 可能引入 NaN
        pred_cxcywh = torch.nan_to_num(pred_cxcywh, nan=0.5, posinf=1.0, neginf=0.0)
        gt_cxcywh = torch.nan_to_num(gt_cxcywh, nan=0.5, posinf=1.0, neginf=0.0)

        # clamp cxcywh 到合法范围: cx,cy∈[0,1], w,h∈[0,1]
        pred_cxcywh = pred_cxcywh.clamp(0, 1)
        gt_cxcywh = gt_cxcywh.clamp(0, 1)

        gt_w = gt_cxcywh[:, 2].clamp(min=self.eps)
        gt_h = gt_cxcywh[:, 3].clamp(min=self.eps)
        scale = torch.stack([gt_w, gt_h, gt_w, gt_h], dim=-1)
        diff = pred_cxcywh.unsqueeze(1) - gt_cxcywh.unsqueeze(0)
        cost = (diff.abs() / scale.unsqueeze(0)).sum(-1)

        # GPU 原生清洗: 替换可能出现的 NaN/Inf (零开销, 无 CPU 同步)
        cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)

        return cost * self.weight


class IoUCost:
    def __init__(self, iou_mode='giou', weight=2.0):
        self.iou_mode = iou_mode
        self.weight = weight

    def __call__(
        self,
        pred_logits: Tensor,
        pred_bboxes: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
    ) -> Tensor:
        """
        pred_bboxes: [N, 4] (xyxy)
        gt_bboxes: [M, 4] (xyxy)
        """
        # 清洗 NaN/Inf 并 clamp，防止 CUDA device-side assert
        pred_bboxes = torch.nan_to_num(
            pred_bboxes.detach(), nan=0.5, posinf=1.0, neginf=0.0
        ).clamp(0, 1)
        gt_bboxes = torch.nan_to_num(
            gt_bboxes, nan=0.5, posinf=1.0, neginf=0.0
        ).clamp(0, 1)
        if self.iou_mode == 'giou':
            iou = ops.generalized_box_iou(pred_bboxes, gt_bboxes)
        else:
            iou = ops.box_iou(pred_bboxes, gt_bboxes)
        return (1 - iou) * self.weight


class DiffusionDetMatcher(nn.Module):
    """DiffusionDet 动态 Top-K 匹配器 (SimOTA 风格)"""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        center_radius: float = 2.5,
        candidate_topk: int = 5,
        match_costs: List[dict] = None,
    ):
        super().__init__()
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        if match_costs is not None:
            # 如果提供了 match_costs，我们假设它们已经被构建好了或者是配置字典
            # 这里为了简单起见，我们支持传入构建好的对象列表
            self.costs = match_costs
        else:
            self.costs = [
                FocalLossCost(weight=cost_class),
                BBoxL1Cost(weight=cost_bbox),
                IoUCost(iou_mode='giou', weight=cost_giou),
            ]

    @torch.no_grad()
    def forward(
        self, outputs: ModelOutput, targets: List[InstanceData]
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        outputs: ModelOutput containing pred_logits [B, N, C] and pred_boxes [B, N, 4]
        targets: list of InstanceData containing labels [M] and bboxes [M, 4]
        """
        pred_logits = outputs.pred_logits
        pred_bboxes = outputs.pred_boxes
        batch_size = len(targets)

        batch_indices = []
        for i in range(batch_size):
            indices = self._single_assign(
                pred_logits[i], pred_bboxes[i], targets[i]
            )
            batch_indices.append(indices)

        return batch_indices

    def _single_assign(
        self, pred_logits: Tensor, pred_bboxes: Tensor, target: InstanceData
    ) -> Tuple[Tensor, Tensor]:
        gt_bboxes = target.bboxes
        gt_labels = target.labels
        num_gt = gt_bboxes.size(0)

        if num_gt == 0:
            device = pred_bboxes.device
            return torch.zeros(
                0, dtype=torch.long, device=device
            ), torch.zeros(0, dtype=torch.long, device=device)

        # 1. 计算各项 Cost
        cost_list = []
        for cost_fn in self.costs:
            cost = cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
            cost_list.append(cost)

        # 2. 计算是否在 GT 框内或中心范围内
        is_in_boxes_anchor, is_in_boxes_and_center = self._get_in_gt_info(
            pred_bboxes, gt_bboxes
        )

        # 将不在范围内的预测框 Cost 调大
        cost_list.append((~is_in_boxes_and_center) * 100.0)
        cost_matrix = torch.stack(cost_list).sum(0)

        # GPU 原生清洗: 替换 NaN/Inf (零开销, 无 CPU 同步)
        cost_matrix = torch.nan_to_num(
            cost_matrix, nan=1e6, posinf=1e6, neginf=1e6
        )

        cost_matrix[~is_in_boxes_anchor] += 10000.0

        # 3. 动态 K 匹配
        pairwise_ious = ops.box_iou(pred_bboxes, gt_bboxes)
        return self._dynamic_k_matching(cost_matrix, pairwise_ious, num_gt)

    def _get_in_gt_info(
        self, pred_bboxes: Tensor, gt_bboxes: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # pred_bboxes/gt_bboxes 都是 xyxy 归一化格式
        pred_ctrs = (pred_bboxes[:, :2] + pred_bboxes[:, 2:]) / 2
        gt_ctrs = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        gt_wh = gt_bboxes[:, 2:] - gt_bboxes[:, :2]

        # 检查预测框中心是否在 GT 框内
        # pred_ctrs: [N, 2], gt_bboxes: [M, 4]
        lt = pred_ctrs.unsqueeze(1) - gt_bboxes[:, :2].unsqueeze(0)
        rb = gt_bboxes[:, 2:].unsqueeze(0) - pred_ctrs.unsqueeze(1)
        is_in_boxes = torch.cat([lt, rb], dim=-1).min(-1)[0] > 0  # [N, M]

        # 检查预测框中心是否在 GT 中心一定半径内
        lt_c = pred_ctrs.unsqueeze(1) - (
            gt_ctrs - self.center_radius * gt_wh
        ).unsqueeze(0)
        rb_c = (gt_ctrs + self.center_radius * gt_wh).unsqueeze(
            0
        ) - pred_ctrs.unsqueeze(1)
        is_in_centers = (
            torch.cat([lt_c, rb_c], dim=-1).min(-1)[0] > 0
        )  # [N, M]

        is_in_boxes_anchor = is_in_boxes.any(1) | is_in_centers.any(1)
        is_in_boxes_and_center = is_in_boxes & is_in_centers
        return is_in_boxes_anchor, is_in_boxes_and_center

    def _dynamic_k_matching(
        self, cost: Tensor, pairwise_ious: Tensor, num_gt: int
    ) -> Tuple[Tensor, Tensor]:
        matching_matrix = torch.zeros_like(cost)

        # GPU 原生清洗: 处理 pairwise_ious 中的 NaN/Inf (无 CPU 同步)
        pairwise_ious = torch.nan_to_num(
            pairwise_ious, nan=0.0, posinf=1.0, neginf=0.0
        )

        # 为每个 GT 选择动态 K
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # 确保 dynamic_ks 是合法的正整数
        dynamic_ks = torch.clamp(
            topk_ious.sum(0).int(), min=1, max=candidate_topk
        )

        for gt_idx in range(num_gt):
            k = dynamic_ks[gt_idx].item()
            _, pos_idx = torch.topk(cost[:, gt_idx], k=k, largest=False)
            matching_matrix[pos_idx, gt_idx] = 1.0

        # 处理一个预测框匹配多个 GT 的情况：选择 Cost 最小的那个
        duplicate_idx = matching_matrix.sum(1) > 1
        if duplicate_idx.any():
            _, cost_argmin = cost[duplicate_idx].min(1)
            matching_matrix[duplicate_idx] = 0.0
            matching_matrix[duplicate_idx, cost_argmin] = 1.0

        # 最终的正样本索引
        fg_mask = matching_matrix.sum(1) > 0
        matched_gt_inds = matching_matrix[fg_mask].argmax(1)
        return fg_mask.nonzero().squeeze(1), matched_gt_inds


class DiffusionDetCriterion(nn.Module):
    """DiffusionDet 损失计算核心类 (纯 PyTorch)"""

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        loss_cls: nn.Module,
        loss_bbox: nn.Module,
        loss_giou: nn.Module,
        deep_supervision: bool = True,
        loss_objectness_weight: float = 1.0,
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
        self.loss_objectness_weight = loss_objectness_weight
        self.scale_aware = scale_aware
        self.scale_aware_mode = scale_aware_mode
        self.scale_aware_min_weight = scale_aware_min_weight
        self.scale_aware_max_weight = scale_aware_max_weight
        self.scale_aware_alpha = scale_aware_alpha
        self.scale_aware_giou = scale_aware_giou
        self.bbox_loss_mode = bbox_loss_mode
        self.bbox_loss_eps = bbox_loss_eps

    def forward(
        self, outputs: ModelOutput, targets: List[InstanceData]
    ) -> Dict[str, Tensor]:
        """
        outputs: ModelOutput containing pred_logits, pred_boxes and optional aux_outputs
        targets: list of InstanceData containing labels, bboxes and img_shape
        """
        # 1. 计算主输出损失
        losses = self._get_loss(outputs, targets)

        # 2. 计算辅助输出损失 (Deep Supervision)
        if self.deep_supervision and outputs.aux_outputs is not None:
            for i, aux_out in enumerate(outputs.aux_outputs):
                aux_losses = self._get_loss(aux_out, targets)
                for name, val in aux_losses.items():
                    losses[f'aux_{i}_{name}'] = val

        return losses

    def _get_loss(
        self, outputs: ModelOutput, targets: List[InstanceData]
    ) -> Dict[str, Tensor]:
        indices = self.matcher(outputs, targets)

        # 计算分类损失
        loss_cls = self._loss_classification(outputs, targets, indices)

        # 计算回归损失
        loss_bbox, loss_giou = self._loss_boxes(outputs, targets, indices)

        losses = {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
        }

        if outputs.pred_objectness is not None:
            loss_obj = self._loss_objectness(outputs, targets, indices)
            losses['loss_objectness'] = loss_obj

        return losses

    def _loss_classification(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        indices: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        src_logits = outputs.pred_logits  # [B, N, C]
        bs, num_queries = src_logits.shape[:2]

        # 构造目标分类标签
        target_classes = src_logits.new_full(
            (bs, num_queries), self.num_classes, dtype=torch.long
        )
        num_pos = 0
        for i, (src_idx, gt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[i, src_idx] = targets[i].labels[gt_idx]
                num_pos += len(src_idx)

        # 损失计算
        loss_cls = self.loss_cls(
            src_logits.flatten(0, 1), target_classes.flatten(0, 1)
        )
        # 按照正样本数量归一化 (参考原版实现)
        return loss_cls / max(num_pos, 1)

    def _loss_boxes(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        indices: List[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor]:
        src_boxes = outputs.pred_boxes  # [B, N, 4] (normalized xyxy)

        # 提取对应的正样本预测框和目标框
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
                scale_w = scale_w.clamp(
                    1.0 - self.scale_aware_alpha * 3,
                    1.0 + self.scale_aware_alpha * 3,
                )
            elif self.scale_aware_mode == 'sqrt_inverse':
                raw_w = 1.0 / torch.sqrt(tgt_areas + 1e-6)
                scale_w = raw_w / raw_w.mean()
                scale_w = scale_w.clamp(
                    self.scale_aware_min_weight, self.scale_aware_max_weight
                )
            else:
                raw_w = 1.0 / (tgt_areas + 1e-6)
                scale_w = raw_w / raw_w.mean()
                scale_w = scale_w.clamp(
                    self.scale_aware_min_weight, self.scale_aware_max_weight
                )
            per_elem_l1 = F.l1_loss(src_cxcywh, tgt_cxcywh, reduction='none')
            weighted_sum = (per_elem_l1 * scale_w.unsqueeze(1)).sum()
            loss_bbox = (
                self.loss_bbox.loss_weight
                * weighted_sum
                / (4.0 * num_pos * num_pos)
            )
            loss_giou = (
                self.loss_giou(src_boxes_pos, tgt_boxes_pos).sum() / num_pos
            )
        elif self.bbox_loss_mode == 'relative_l1':
            tgt_w = tgt_cxcywh[:, 2].clamp(min=self.bbox_loss_eps)
            tgt_h = tgt_cxcywh[:, 3].clamp(min=self.bbox_loss_eps)
            scale = torch.stack([tgt_w, tgt_h, tgt_w, tgt_h], dim=-1)
            per_elem = F.l1_loss(src_cxcywh, tgt_cxcywh, reduction='none')
            loss_bbox = (
                self.loss_bbox.loss_weight * (per_elem / scale).sum() / num_pos
            )
            loss_giou = (
                self.loss_giou(src_boxes_pos, tgt_boxes_pos).sum() / num_pos
            )
        else:
            loss_bbox = self.loss_bbox(src_cxcywh, tgt_cxcywh)
            loss_giou = self.loss_giou(src_boxes_pos, tgt_boxes_pos)
            loss_bbox = loss_bbox.sum() / num_pos
            loss_giou = loss_giou.sum() / num_pos

        return loss_bbox, loss_giou

    def _loss_objectness(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        indices: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        pred_obj = outputs.pred_objectness  # [B, N, 1]
        bs, num_queries = pred_obj.shape[:2]
        device = pred_obj.device

        target_obj = torch.zeros(bs, num_queries, device=device)
        for i, (src_idx, gt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_obj[i, src_idx] = 1.0

        loss_obj = F.binary_cross_entropy_with_logits(
            pred_obj.squeeze(-1), target_obj, reduction='mean'
        )
        return loss_obj * self.loss_objectness_weight
