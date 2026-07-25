"""检测损失计算核心类"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import ops

from ldmdet.data.structures import InstanceData, ModelOutput
from ldmdet.diagnostics.instrumentation import probe
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
        # R3: v-prediction 等价的损失重加权
        # 理论 (theory_analysis_RF_DPM.md §4.3): L_v = (1/t²) L_x0 (L2 squared 下)
        # L1 loss 下严格等价应为 1/t, 这里用 1/t² 匹配理论 doc 的梯度放大 claim
        # 并对权重做 batch normalization (均值=1), 避免训练崩溃
        v_prediction: bool = False,
        v_prediction_t_eps: float = 1e-2,
        # ReFlow (Standard MSE): box target 模式
        # 'gt' (默认): box target = GT bboxes (现有行为)
        # 'x0_pred': box target = x_0^pred (A4 推理预测, RF 拉直目标)
        # 详见 REFLOW_HEAD_DISTILL_IMPL_PLAN.md §1.2 混合 target 设计
        box_target_mode: str = 'gt',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.loss_giou = loss_giou
        self.deep_supervision = deep_supervision
        self.v_prediction = v_prediction
        self.v_prediction_t_eps = v_prediction_t_eps
        assert box_target_mode in ('gt', 'x0_pred'), \
            f"box_target_mode 必须是 'gt' 或 'x0_pred', got {box_target_mode}"
        self.box_target_mode = box_target_mode


    def forward(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        t: Optional[Tensor] = None,
        # ReFlow (Standard MSE): box target = A4 预测 (x_0^pred), 替代 GT
        # 仅当 box_target_mode='x0_pred' 时生效; 'gt' 模式忽略此参数
        # 形状: [bs, max_gt, 4] (xyxy, 已 padded) 或 list[[n_i, 4]]
        box_targets: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # 主输出: 使用 matcher.forward 并建立 GT 缓存
        indices, gt_cache = self.matcher.forward_with_gt_cache(
            outputs, targets
        )
        losses = self._get_loss(
            outputs, targets, indices, t, box_targets=box_targets
        )

        if self.deep_supervision and outputs.aux_outputs is not None:
            for i, aux_out in enumerate(outputs.aux_outputs):
                # aux_outputs 复用 GT 缓存，避免重复计算 gt_ctrs/gt_wh/center 区域
                aux_indices, gt_cache = self.matcher.forward_with_gt_cache(
                    aux_out, targets, gt_cache
                )
                aux_losses = self._get_loss(
                    aux_out, targets, aux_indices, t, box_targets=box_targets
                )
                for name, val in aux_losses.items():
                    losses[f'aux_{i}_{name}'] = val
            # 探针: deep_supervision aux loss 数量
            probe.record_scalar('criterion/n_aux_outputs', len(outputs.aux_outputs))
        self._last_indices = indices
        return losses

    def _get_loss(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        indices: List[Tuple[Tensor, Tensor]] = None,
        t: Optional[Tensor] = None,
        box_targets: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if indices is None:
            indices = self.matcher(outputs, targets)
        loss_cls = self._loss_classification(outputs, targets, indices)
        loss_bbox, loss_giou = self._loss_boxes(
            outputs, targets, indices, t, box_targets=box_targets
        )

        # 探针: 损失分量标量 (训练时每 100 步)
        probe.record_scalar('criterion/loss_cls', loss_cls.item())
        probe.record_scalar('criterion/loss_bbox', loss_bbox.item())
        probe.record_scalar('criterion/loss_giou', loss_giou.item())

        return {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
        }

    def _loss_classification(self, outputs, targets, indices) -> Tensor:
        src_logits = outputs.pred_logits  # [bs, num_queries, num_classes+1]
        bs, num_queries = src_logits.shape[:2]

        # 构建 padded GT labels: [bs, max_gt]
        max_gt = max(t.labels.shape[0] for t in targets)
        if max_gt == 0:
            # 全部为背景
            target_classes = src_logits.new_full(
                (bs, num_queries), self.num_classes, dtype=torch.long
            )
            num_pos = src_logits.new_tensor(1, dtype=torch.long)
            loss_cls = self.loss_cls(
                src_logits.flatten(0, 1), target_classes.flatten(0, 1)
            )
            return loss_cls / num_pos

        gt_labels_padded = src_logits.new_full(
            (bs, max_gt), self.num_classes, dtype=torch.long
        )
        for i, t in enumerate(targets):
            n = t.labels.shape[0]
            if n > 0:
                gt_labels_padded[i, :n] = t.labels

        # 批量化: 将 indices 堆叠为 [bs, N] 张量
        fg_masks = torch.stack([idx[0] for idx in indices])  # [bs, N] bool
        matched_gt_inds = torch.stack(
            [idx[1] for idx in indices]
        )  # [bs, N] long
        matched_gt_inds_clamped = matched_gt_inds.clamp(min=0)

        # Gather GT labels: [bs, N]
        matched_gt_labels = torch.gather(
            gt_labels_padded, 1, matched_gt_inds_clamped
        )
        # 背景位置设为 num_classes
        matched_gt_labels[~fg_masks] = self.num_classes

        target_classes = matched_gt_labels
        # num_pos 保留为张量，避免 .item() 同步
        num_pos = fg_masks.sum().clamp(min=1)

        loss_cls = self.loss_cls(
            src_logits.flatten(0, 1), target_classes.flatten(0, 1)
        )
        return loss_cls / num_pos

    def _loss_boxes(
        self,
        outputs,
        targets,
        indices,
        t: Optional[Tensor] = None,
        box_targets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """计算 box L1 + GIoU 损失.

        空间一致性约定 (ReFlow 修复 BUG #1):
          - src_boxes (outputs.pred_boxes): 归一化 xyxy [0,1]
          - targets[*].bboxes (GT): 归一化 xyxy [0,1] (由 head._normalize_targets 转换)
          - box_targets (ReFlow x0_pred): **必须** 归一化 xyxy [0,1]
            (head.loss() 中已将 raw cxcywh scaled 转换为归一化 xyxy,
             切勿直接传 raw 扩散空间的 coupling x0_pred)

        ReFlow 混合 target (box_target_mode='x0_pred'):
          - cls target 始终用 GT (matcher 基于 GT 分配正负样本)
          - box target 切换为 x0_pred (per-proposal 或 per-GT 布局, 按形状自动分流)
        """
        src_boxes = outputs.pred_boxes  # [bs, num_queries, 4] 归一化 xyxy [0,1]
        bs = src_boxes.shape[0]

        # 构建 padded GT bboxes: [bs, max_gt, 4]
        max_gt = max(t_data.bboxes.shape[0] for t_data in targets)
        if max_gt == 0:
            # 无正样本，返回零损失
            return src_boxes.sum() * 0, src_boxes.sum() * 0

        # 批量化: 将 indices 堆叠为 [bs, N] 张量
        fg_masks = torch.stack([idx[0] for idx in indices])  # [bs, N] bool
        matched_gt_inds = torch.stack(
            [idx[1] for idx in indices]
        )  # [bs, N] long

        # 构建 padded GT bboxes: [bs, max_gt, 4]
        gt_bboxes_padded = src_boxes.new_zeros(bs, max_gt, 4)
        for i, t_data in enumerate(targets):
            n = t_data.bboxes.shape[0]
            if n > 0:
                gt_bboxes_padded[i, :n] = t_data.bboxes

        # ReFlow (Standard MSE): box target 来源选择
        # 'gt' (默认): box target = GT bboxes (现有行为)
        # 'x0_pred': box target = A4 预测 (x_0^pred), 仅当传入 box_targets 时生效;
        #            未传则安全降级到 GT (避免训练崩溃)
        # cls target 始终用 GT (matcher 基于 GT 分配正负样本, 不受此处影响)
        use_x0_pred = (
            self.box_target_mode == 'x0_pred' and box_targets is not None
        )
        if use_x0_pred:
            # REFLOW_HEAD_DISTILL_IMPL_PLAN.md §1.2: box target = x_0^pred (RF 拉直目标)
            # 两种布局 (按形状自动分流):
            #   - per-proposal [bs, num_proposals, 4]: 每个 proposal 的 A4 预测 (轨迹端点),
            #     与 pred_boxes 直接对齐, 不经 matcher gather (canonical ReFlow)
            #   - per-GT [bs, max_gt, 4]: 每个 GT 的 A4 预测, 按 matched_gt_inds gather
            is_per_proposal = (
                not isinstance(box_targets, (list, tuple))
                and box_targets.shape[1] == src_boxes.shape[1]
            )
            if is_per_proposal:
                # per-proposal: 直接对齐 (RF 每个提案有独立轨迹端点 x_0^pred)
                tgt_boxes = box_targets.to(src_boxes)  # [bs, num_proposals, 4]
            else:
                # per-GT: 对齐到 [bs, max_gt, 4] 并 gather (与 GT 布局一致)
                if isinstance(box_targets, (list, tuple)):
                    target_boxes_padded = src_boxes.new_zeros(bs, max_gt, 4)
                    for i, bt in enumerate(box_targets):
                        n = bt.shape[0]
                        if n > 0:
                            target_boxes_padded[i, :n] = bt
                else:
                    n_box = box_targets.shape[1]
                    if n_box >= max_gt:
                        target_boxes_padded = box_targets[:, :max_gt].to(src_boxes)
                    else:
                        target_boxes_padded = src_boxes.new_zeros(bs, max_gt, 4)
                        target_boxes_padded[:, :n_box] = box_targets.to(src_boxes)
                matched_gt_inds_clamped = matched_gt_inds.clamp(min=0)
                tgt_boxes = torch.gather(
                    target_boxes_padded,
                    1,
                    matched_gt_inds_clamped.unsqueeze(-1).expand(-1, -1, 4),
                )
        else:
            # 'gt' 模式 或 'x0_pred' 未传 box_targets → 用 GT
            matched_gt_inds_clamped = matched_gt_inds.clamp(min=0)
            tgt_boxes = torch.gather(
                gt_bboxes_padded,
                1,
                matched_gt_inds_clamped.unsqueeze(-1).expand(-1, -1, 4),
            )

        # num_pos 保留为张量，避免 .item() 同步
        num_pos = fg_masks.sum().clamp(min=1)

        tgt_cxcywh = bbox_xyxy_to_cxcywh(tgt_boxes)  # [bs, N, 4]
        src_cxcywh = bbox_xyxy_to_cxcywh(src_boxes)  # [bs, N, 4]

        # L1 loss: 逐元素计算后 mask
        per_elem_l1 = F.l1_loss(
            src_cxcywh, tgt_cxcywh, reduction='none'
        )  # [bs, N, 4]
        if self.v_prediction and t is not None:
            # R3: v-prediction 等价的 1/t² 损失加权
            # t 是 [bs,] 张量 (shifted schedule 后, 范围 [0,1])
            # 理论 (§4.3): v-prediction 在 t→0 时梯度放大 1/t²
            # batch normalization (均值=1) 控制绝对幅度, 避免训练崩溃
            t_clamped = t.clamp(min=self.v_prediction_t_eps)
            v_weight = 1.0 / (t_clamped ** 2)  # [bs,]
            v_weight = v_weight / v_weight.mean().detach()
            v_weight = v_weight.view(-1, 1, 1)  # [bs, 1, 1]
            masked_l1 = (
                per_elem_l1
                * fg_masks.unsqueeze(-1).float()
                * v_weight
            )
        else:
            masked_l1 = per_elem_l1 * fg_masks.unsqueeze(-1).float()

        loss_bbox = self.loss_bbox.loss_weight * masked_l1.sum() / num_pos
        # GIoU loss: 逐元素计算后 mask
        # GIoU 不是 t 的简单函数, v_prediction 下保持不加权
        per_giou = ops.generalized_box_iou_loss(
            src_boxes.reshape(-1, 4),
            tgt_boxes.reshape(-1, 4),
            reduction='none',
        ).reshape(bs, -1)
        per_giou = torch.nan_to_num(per_giou, nan=0.0)

        loss_giou = (
            self.loss_giou.loss_weight
            * (per_giou * fg_masks.float()).sum()
            / num_pos
        )

        # 探针: box 损失详细统计 (正样本数, L1/GIoU per-elem 分布)
        probe.record_scalar('criterion/num_pos', num_pos.item())
        probe.record_scalar('criterion/fg_ratio', fg_masks.float().mean().item())
        # per-elem L1 分布 (仅正样本)
        if per_elem_l1 is not None:
            pos_l1 = per_elem_l1[fg_masks]
            if pos_l1.numel() > 0:
                probe.record_tensor_stats('criterion/l1_per_elem', pos_l1)
        # per-giou 分布 (仅正样本)
        if per_giou is not None:
            pos_giou = per_giou[fg_masks]
            if pos_giou.numel() > 0:
                probe.record_tensor_stats('criterion/giou_per_elem', pos_giou)
        # 预测 box vs GT box 的 cxcywh 差异 (仅正样本, per-dim)
        if src_cxcywh is not None and tgt_cxcywh is not None:
            box_diff = (src_cxcywh - tgt_cxcywh).abs()[fg_masks]
            if box_diff.numel() > 0:
                probe.record_tensor_stats('criterion/box_diff', box_diff)

        return loss_bbox, loss_giou
