from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import ops

from .structures import InstanceData, ModelOutput
from .utils import bbox_xyxy_to_cxcywh, sanitize_bboxes


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
        use_sigmoid: bool = True,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'sum',
        loss_weight: float = 2.0,
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

    def __init__(self, reduction: str = 'sum', loss_weight: float = 2.0):
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

    def __init__(self, reduction: str = 'sum', loss_weight: float = 5.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = F.l1_loss(pred, target, reduction=self.reduction)
        return loss * self.loss_weight


# --- Matching Costs (用于匹配阶段) ---


class FocalLossCost:
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        weight: float = 2.0,
        eps: float = 1e-8,
    ):
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


class RelativeL1Cost:
    def __init__(self, weight: float = 5.0, eps: float = 1e-2):
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
        pred_bboxes = sanitize_bboxes(pred_bboxes.detach())
        gt_bboxes = sanitize_bboxes(gt_bboxes)

        pred_cxcywh = bbox_xyxy_to_cxcywh(pred_bboxes)
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)

        # 二次清洗: bbox_xyxy_to_cxcywh 可能引入 NaN
        pred_cxcywh = sanitize_bboxes(pred_cxcywh)
        gt_cxcywh = sanitize_bboxes(gt_cxcywh)

        gt_w = gt_cxcywh[:, 2].clamp(min=self.eps)
        gt_h = gt_cxcywh[:, 3].clamp(min=self.eps)
        scale = torch.stack([gt_w, gt_h, gt_w, gt_h], dim=-1)
        diff = pred_cxcywh.unsqueeze(1) - gt_cxcywh.unsqueeze(0)
        cost = (diff.abs() / scale.unsqueeze(0)).sum(-1)

        # GPU 原生清洗: 替换可能出现的 NaN/Inf (零开销, 无 CPU 同步)
        cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)

        return cost * self.weight


class IoUCost:
    def __init__(self, iou_mode: str = 'giou', weight: float = 2.0):
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
        pred_bboxes = sanitize_bboxes(pred_bboxes.detach())
        gt_bboxes = sanitize_bboxes(gt_bboxes)
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
        match_costs: Optional[List[dict]] = None,
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
                RelativeL1Cost(weight=cost_bbox),
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
    """DiffusionDet 损失计算核心类 (纯 PyTorch)

    使用 OT (Sinkhorn) 匹配结果作为正例分配，替代 SimOTA 二次匹配。
    OT 保证每个 proposal 都有对应的 GT，100% GT 覆盖率，
    扩散过程天然提供课程学习 (t 大→粗略, t 小→精确)。

    OT 匹配下所有 proposal 都是正例:
    - loss_cls: 分类监督 (Focal Loss，OT 下退化为 CE)
    - loss_giou: GIoU 回归监督 (尺度敏感，补充 displacement MSE)
    - loss_objectness: 已移除 (OT 下全为 1，无信息量)
    - loss_bbox: 已移除 (与 loss_vel/displacement MSE 功能重叠)
    """

    def __init__(
        self,
        num_classes: int,
        matcher: Optional[nn.Module] = None,
        loss_cls: Optional[nn.Module] = None,
        loss_giou: Optional[nn.Module] = None,
        deep_supervision: bool = True,
        bbox_loss_eps: float = 1e-2,
        assigner: Optional[dict] = None,
        ot_pos_ratio: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher  # 保留但不使用
        self.loss_cls = loss_cls
        self.loss_giou = loss_giou
        self.deep_supervision = deep_supervision
        self.bbox_loss_eps = bbox_loss_eps
        self.ot_pos_ratio = ot_pos_ratio

    def forward(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        ot_matched_gt_indices: Optional[List[Tensor]] = None,
        ot_match_probs: Optional[List[Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        outputs: ModelOutput containing pred_logits, pred_boxes and optional aux_outputs
        targets: list of InstanceData containing labels, bboxes and img_shape
        ot_matched_gt_indices: OT 匹配结果, list of (num_proposals,) 每个元素是该 proposal 匹配的 GT 索引
        ot_match_probs: OT 传输概率, list of (num_proposals,) 每个元素是该 proposal 的最大传输概率
        """
        # 1. 计算主输出损失
        losses = self._get_loss(outputs, targets, ot_matched_gt_indices, ot_match_probs)

        # 2. 计算辅助输出损失 (Deep Supervision)
        if self.deep_supervision and outputs.aux_outputs is not None:
            for i, aux_out in enumerate(outputs.aux_outputs):
                aux_losses = self._get_loss(
                    aux_out, targets, ot_matched_gt_indices, ot_match_probs
                )
                for name, val in aux_losses.items():
                    losses[f'aux_{i}_{name}'] = val

        return losses

    def _ot_indices_to_match(
        self,
        ot_matched_gt_indices: List[Tensor],
        num_proposals: int,
        ot_match_probs: Optional[List[Tensor]] = None,
    ) -> List[Tuple[Tensor, Tensor]]:
        """将 OT 匹配结果转换为 criterion 需要的 (src_idx, gt_idx) 格式

        根据 ot_pos_ratio 过滤低质量匹配:
        - 按传输概率排序，只保留 top ot_pos_ratio 的 proposal 作为正例
        - 其余 proposal 标记为背景，不参与回归 loss
        """
        batch_indices = []
        for i, matched_gt in enumerate(ot_matched_gt_indices):
            if ot_match_probs is not None and ot_match_probs[i] is not None:
                probs = ot_match_probs[i]
                # 按概率排序，保留 top-k 作为正例
                k = max(1, int(num_proposals * self.ot_pos_ratio))
                _, topk_idx = probs.topk(k)
                src_idx = topk_idx
                gt_idx = matched_gt[topk_idx]
            else:
                src_idx = torch.arange(num_proposals, device=matched_gt.device)
                gt_idx = matched_gt
            batch_indices.append((src_idx, gt_idx))
        return batch_indices

    def _get_loss(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        ot_matched_gt_indices: Optional[List[Tensor]] = None,
        ot_match_probs: Optional[List[Tensor]] = None,
    ) -> Dict[str, Tensor]:
        if ot_matched_gt_indices is not None:
            indices = self._ot_indices_to_match(
                ot_matched_gt_indices, outputs.pred_logits.shape[1],
                ot_match_probs=ot_match_probs,
            )
        else:
            # 降级到 SimOTA (不应发生，保留兼容)
            indices = self.matcher(outputs, targets)

        # 计算分类损失
        loss_cls = self._loss_classification(outputs, targets, indices)

        # 计算 GIoU 回归损失
        loss_giou = self._loss_giou(outputs, targets, indices)

        losses = {
            'loss_cls': loss_cls,
            'loss_giou': loss_giou,
        }

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
        # OT 匹配: 所有 proposal 都是正例，没有背景类
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
        # 按照正样本数量归一化
        return loss_cls / max(num_pos, 1)

    def _loss_giou(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        indices: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        """GIoU 回归损失 (尺度敏感，补充 displacement MSE)"""
        src_boxes = outputs.pred_boxes  # [B, N, 4] (normalized xyxy)

        # 提取对应的正样本预测框和目标框
        src_list = []
        tgt_list = []
        for i, (src_idx, gt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                src_list.append(src_boxes[i, src_idx])
                tgt_list.append(targets[i].bboxes[gt_idx])

        if len(src_list) == 0:
            return src_boxes.sum() * 0

        src_boxes_pos = torch.cat(src_list)
        tgt_boxes_pos = torch.cat(tgt_list)
        num_pos = src_boxes_pos.shape[0]

        loss_giou = self.loss_giou(src_boxes_pos, tgt_boxes_pos)
        return loss_giou.sum() / num_pos
