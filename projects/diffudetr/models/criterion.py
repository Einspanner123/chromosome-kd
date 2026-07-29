"""SNR 加权损失 + Hungarian 匹配 — DiffuDETR 移植

实现 DETR 风格的 SetCriterion, 带扩散时间步 SNR 加权:
  - HungarianMatcher: cost = cost_class + cost_bbox + cost_giou
  - SetCriterion: 分类 (focal) + 回归 (L1 + GIoU), 按 loss_weight[t] 加权

关键设计决策:
  - 分类用 sigmoid focal loss (num_classes 输出, 不含背景类 logit)
  - 背景类标签 = num_classes (padding 框), focal loss 中不参与正样本
  - SNR 加权: 回归损失 × loss_weight[t], 低 timestep (高 SNR) 权重更高 (高噪声降权)
    对齐原仓库 loss_weight (lvlb_weights) 公式
  - 辅助损失 (aux_loss): 每个 decoder 中间层都计算损失

box 格式约定:
  - 损失计算在归一化 cxcywh 空间 [0,1] (非扩散空间)
  - GIoU 在 xyxy 归一化空间计算
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """cxcywh → xyxy."""
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack(
        [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1
    )


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """xyxy → cxcywh."""
    x1, y1, x2, y2 = x.unbind(-1)
    return torch.stack(
        [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], dim=-1
    )


def generalized_box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> torch.Tensor:
    """广义 GIoU (Generalized IoU).

    对齐 DETR util.generalized_box_iou.
    输入为 xyxy 归一化坐标.

    注意: 不使用 assert 检查 x2>=x1 (原始 DETR 也不检查),
    而是通过 clamp(min=0) 处理无效框, 避免训练早期预测框非法时崩溃.

    Args:
        boxes1: [N, 4] xyxy.
        boxes2: [M, 4] xyxy.

    Returns:
        [N, M] GIoU 矩阵.
    """
    # 不使用 assert — 训练早期预测框可能 x2<x1, 通过 clamp 处理

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    # 最小外接框
    lt_enc = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_enc = (rb_enc - lt_enc).clamp(min=0)
    enclose = wh_enc[..., 0] * wh_enc[..., 1]

    giou = iou - (enclose - union) / enclose.clamp(min=1e-6)
    return giou


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
) -> torch.Tensor:
    """Sigmoid focal loss (对齐 detr sigmoid_focal_loss).

    Args:
        inputs: [N, C] logits.
        targets: [N, C] one-hot.
        alpha: 平衡因子.
        gamma: 聚焦因子.
        reduction: 'none' | 'sum' | 'mean'.
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
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    return loss


class HungarianMatcher(nn.Module):
    """Hungarian 匹配器 (对齐 DETR HungarianMatcher).

    计算 cost = cost_class + cost_bbox + cost_giou, 用 scipy linear_sum_assignment
    求最优匹配.

    Args:
        cost_class: 分类代价权重.
        cost_bbox: L1 回归代价权重.
        cost_giou: GIoU 代价权重.
        num_classes: 类别数 (用于 one-hot 匹配).
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 1.0,
        cost_giou: float = 1.0,
        num_classes: int = 24,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_classes = num_classes
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, (
            'all costs zero'
        )

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """计算 Hungarian 匹配.

        Args:
            pred_logits: [B, N, num_classes] 分类 logits.
            pred_boxes: [B, N, 4] 预测框 (归一化 cxcywh [0,1]).
            targets: 每个元素 dict('labels': [M], 'boxes': [M, 4] 归一化 cxcywh).

        Returns:
            List[(pred_indices, target_indices)] 每张图的匹配结果.
        """
        bs, nq = pred_logits.shape[:2]

        # 展平所有目标的类别和框
        out_prob = pred_logits.flatten(0, 1).sigmoid()  # [B*N, C]
        out_bbox = pred_boxes.flatten(0, 1)  # [B*N, 4]

        tgt_ids = torch.cat([t['labels'] for t in targets])  # [sum(M)]
        tgt_bbox = torch.cat([t['boxes'] for t in targets])  # [sum(M), 4]

        # 分类代价: 1 - prob[target_class] (focal 风格)
        cost_class = 1 - out_prob[:, tgt_ids.long()]

        # L1 代价
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # GIoU 代价
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # 总代价
        C = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, nq, -1).cpu()

        # 每张图分别匹配
        sizes = [len(t['labels']) for t in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.long),
                torch.as_tensor(j, dtype=torch.long),
            )
            for i, j in indices
        ]


class SetCriterion(nn.Module):
    """DETR 风格损失 + SNR 时间步加权.

    对齐原仓库 criterion 逻辑, 增加:
      - loss_weight[t]: SNR 加权回归损失 (低 timestep 权重高, 高噪声降权)
      - aux_loss: 每个 decoder 中间层计算损失

    Args:
        num_classes: 类别数 (24).
        matcher: HungarianMatcher 实例.
        weight_dict: 各损失项权重.
        losses: 计算的损失列表.
        eos_coef: 背景类 (eos) 权重系数.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Optional[Dict[str, float]] = None,
        losses: Optional[List[str]] = None,
        eos_coef: float = 0.1,
        use_vlb: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses or ['labels', 'boxes']
        # use_vlb: 是否启用 SNR 时间步加权 (对齐原仓库 use_vlb, 50ep 默认 False)
        self.use_vlb = use_vlb
        if weight_dict is None:
            weight_dict = {
                'loss_ce': 1.0,
                'loss_bbox': 5.0,
                'loss_giou': 2.0,
            }
        self.weight_dict = weight_dict

        # 背景类权重 (focal loss 中背景降权)
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def _get_src_permutation_idx(
        self, indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """获取匹配后的预测索引 (batch 内展平)."""
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(
        self,
        pred_logits: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """分类损失 (sigmoid focal loss).

        对齐 DETR loss_labels. 背景类标签 = num_classes.
        归一化: ÷ num_boxes (对齐原仓库 diffu_criterion.py, 非 ÷ batch_size).
        """
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            pred_logits.shape[:2],
            self.num_classes,
            dtype=torch.long,
            device=pred_logits.device,
        )
        target_classes_o = torch.cat(
            [t['labels'][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes[idx] = target_classes_o

        # one-hot
        target_classes_onehot = torch.zeros(
            [pred_logits.shape[0], pred_logits.shape[1], self.num_classes + 1],
            dtype=pred_logits.dtype,
            device=pred_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        # 去掉背景类 (focal loss 只对前景类计算)
        target_classes_onehot = target_classes_onehot[..., : self.num_classes]

        loss_ce = sigmoid_focal_loss(
            pred_logits,
            target_classes_onehot,
            alpha=0.25,
            gamma=2.0,
            reduction='none',
        )
        # 归一化: ÷ num_boxes (对齐原仓库), 非 ÷ batch_size
        loss_ce = loss_ce.mean(1).sum() / max(num_boxes, 1.0)
        return {'loss_ce': loss_ce}

    def loss_boxes(
        self,
        pred_boxes: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float = 1.0,
        time_steps: Optional[torch.Tensor] = None,
        loss_weight: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """回归损失 (L1 + GIoU), 可选 SNR 时间步加权.

        对齐原仓库 diffu_criterion.py loss_boxes:
          - 归一化: ÷ num_boxes (GT 总数), 非 ÷ batch_size
          - SNR 加权: 仅 use_vlb=True 时应用 loss_weight[t] (原仓库 50ep use_vlb=False)
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat(
            [t['boxes'][J] for t, (_, J) in zip(targets, indices)], dim=0
        )

        if src_boxes.numel() == 0:
            return {
                'loss_bbox': pred_boxes.sum() * 0,
                'loss_giou': pred_boxes.sum() * 0,
            }

        # L1 损失
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        # GIoU 损失
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
        )

        # SNR 时间步加权: 仅 use_vlb=True 时应用 (对齐原仓库 use_vlb 开关)
        if self.use_vlb and loss_weight is not None and time_steps is not None:
            batch_idx = idx[0]
            if loss_weight.dim() == 1:
                loss_weight = loss_weight.view(-1, 1)
            sample_weight = loss_weight[batch_idx]  # [num_matched, 1]
            loss_bbox = loss_bbox * sample_weight
            loss_giou = loss_giou * sample_weight.squeeze(-1)

        # 归一化: ÷ num_boxes (对齐原仓库), 非 ÷ batch_size
        loss_bbox = loss_bbox.sum() / max(num_boxes, 1.0)
        loss_giou = loss_giou.sum() / max(num_boxes, 1.0)
        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def get_loss(
        self,
        loss: str,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float = 1.0,
        time_steps: Optional[torch.Tensor] = None,
        loss_weight: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """分发到具体损失函数."""
        if loss == 'labels':
            return self.loss_labels(
                outputs['pred_logits'],
                targets,
                indices,
                num_boxes=num_boxes,
                **kwargs,
            )
        elif loss == 'boxes':
            return self.loss_boxes(
                outputs['pred_boxes'],
                targets,
                indices,
                num_boxes=num_boxes,
                time_steps=time_steps,
                loss_weight=loss_weight,
                **kwargs,
            )
        else:
            raise NotImplementedError(f'loss {loss} not implemented')

    def forward(
        self,
        pred_logits_list: List[torch.Tensor],
        pred_boxes_list: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        time_steps: torch.Tensor,
        loss_weight: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算总损失 (含辅助损失).

        Args:
            pred_logits_list: [num_layers] 每层 [B, N, num_classes] 分类 logits.
            pred_boxes_list: [num_layers] 每层 [B, N, 4] 预测框 (归一化 cxcywh [0,1]).
            targets: 每张图 dict('labels', 'boxes').
            time_steps: [B] 时间步.
            loss_weight: [B] SNR 损失权重 (仅 use_vlb=True 时使用).

        Returns:
            loss_dict: 各损失项 (含 aux 前缀的辅助损失), 已乘 weight_dict 权重.
        """
        num_layers = len(pred_logits_list)
        loss_dict = {}

        # num_boxes: batch 内 GT 总数 (对齐原仓库归一化分母)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=pred_logits_list[-1].device
        )
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # 最后一层 (主损失)
        pred_logits = pred_logits_list[-1]
        pred_boxes = pred_boxes_list[-1]
        indices = self.matcher(pred_logits, pred_boxes, targets)
        for loss in self.losses:
            loss_dict.update(
                self.get_loss(
                    loss,
                    {'pred_logits': pred_logits, 'pred_boxes': pred_boxes},
                    targets,
                    indices,
                    num_boxes=num_boxes,
                    time_steps=time_steps,
                    loss_weight=loss_weight,
                )
            )

        # 辅助损失 (中间层)
        for i in range(num_layers - 1):
            pred_logits_i = pred_logits_list[i]
            pred_boxes_i = pred_boxes_list[i]
            indices_i = self.matcher(pred_logits_i, pred_boxes_i, targets)
            for loss in self.losses:
                l_dict = self.get_loss(
                    loss,
                    {'pred_logits': pred_logits_i, 'pred_boxes': pred_boxes_i},
                    targets,
                    indices_i,
                    num_boxes=num_boxes,
                    time_steps=time_steps,
                    loss_weight=loss_weight,
                )
                for k, v in l_dict.items():
                    loss_dict[f'{k}_{i}'] = v

        # 加权 (对齐 DETR weight_dict, aux 层同权重)
        weighted = {}
        for k, v in loss_dict.items():
            # 解析 base key: loss_bbox_0 → loss_bbox, loss_ce → loss_ce
            base_k = k.rsplit('_', 1)[0] if k[-1].isdigit() else k
            weight = self.weight_dict.get(base_k, 0.0)
            if weight > 0:
                weighted[k] = v * weight
        return weighted
