from typing import List, Union

import torch
import torch.nn as nn
from mmengine.config import ConfigDict

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import ConfigType

# 导入原有损失组件
from projects.DiffusionDet.diffusiondet.loss import (
    DiffusionDetCriterion,
    DiffusionDetMatcher,
)


@TASK_UTILS.register_module()
class ChromoDetCriterion(DiffusionDetCriterion):
    """
    染色体检测专用损失函数

    在原有损失基础上添加：
    1. 长度预测损失
    2. 形态约束损失
    3. 拓扑关系损失
    """

    def __init__(
        self,
        num_classes=24,
        assigner: Union[ConfigDict, nn.Module] = None,
        deep_supervision=True,
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            alpha=0.25,
            gamma=2.0,
            reduction="sum",
            loss_weight=2.0,
        ),
        loss_bbox=dict(type="L1Loss", reduction="sum", loss_weight=5.0),
        loss_giou=dict(type="GIoULoss", reduction="sum", loss_weight=2.0),
        # 长度先验
        use_length_prior: bool = False,
        loss_length=dict(type="L1Loss", reduction="sum", loss_weight=1.0),
        aspect_ratio_target=10.0,  # 目标长宽比
        # 形态感知
        use_morphology_aware: bool = False,
        loss_aspect_ratio=dict(type="L1Loss", reduction="sum", loss_weight=0.5),
        # 拓扑匹配
        use_topology_pairing: bool = False,
        loss_topology=dict(type="MSELoss", reduction="sum", loss_weight=0.2),
        # 长度排序
        use_length_ordering: bool = False,
        # 数量先验
        use_count_prior: bool = False,
        loss_count=dict(type="MSELoss", reduction="mean", loss_weight=1.0),
    ):
        super().__init__(
            num_classes=num_classes,
            assigner=assigner,
            deep_supervision=deep_supervision,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_giou=loss_giou,
        )

        # 长度先验
        self.use_length_prior = use_length_prior
        if use_length_prior:
            self.loss_length = MODELS.build(loss_length)
            self.aspect_ratio_target = aspect_ratio_target
        # 形态感知
        self.use_morphology_aware = use_morphology_aware
        if use_morphology_aware:
            self.loss_aspect_ratio = MODELS.build(loss_aspect_ratio)
        # 拓扑匹配
        self.use_topology_pairing = use_topology_pairing
        if use_topology_pairing:
            self.loss_topology = MODELS.build(loss_topology)
        # 长度排序
        self.use_length_ordering = use_length_ordering
        # 数量先验
        self.use_count_prior = use_count_prior
        if use_count_prior:
            self.loss_count = MODELS.build(loss_count)

    def forward(self, outputs, batch_gt_instances, batch_img_metas):
        # 完整重写原loss的forward方法
        batch_indices = self.assigner(outputs, batch_gt_instances, batch_img_metas)
        # Compute all the requested losses
        loss_cls = self.loss_classification(outputs, batch_gt_instances, batch_indices)

        loss_bbox, loss_giou = self.loss_boxes(
            outputs, batch_gt_instances, batch_indices
        )

        losses = dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_giou=loss_giou)

        # 长度损失
        if self.use_length_prior:
            loss_length = self.loss_length_computation(
                outputs, batch_gt_instances, batch_indices, batch_img_metas
            )
            losses["loss_length"] = loss_length

        # 形态约束损失
        # if self.use_morphology_aware:
        #     loss_morphology = \
        #         self.loss_morphology_computation(outputs, batch_gt_instances, batch_indices)
        #     losses['loss_aspect_ratio'] = loss_morphology

        # 拓扑损失
        if self.use_topology_pairing:
            loss_topology = self.loss_topology_computation(
                outputs, batch_gt_instances, batch_indices
            )
            losses["loss_topology"] = loss_topology

        # 长度排序
        if self.use_length_ordering:
            loss_ordering = self.loss_ordinal_length_consistency(
                outputs, batch_gt_instances, batch_indices, reduction="mean"
            )
            losses["loss_ordering"] = loss_ordering

        # 数量损失
        if self.use_count_prior:
            loss_count = self.loss_count_computation(outputs, batch_gt_instances)
            losses["loss_count"] = loss_count

        if self.deep_supervision:
            assert "aux_outputs" in outputs
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                batch_indices = self.assigner(
                    aux_outputs, batch_gt_instances, batch_img_metas
                )
                loss_cls = self.loss_classification(
                    aux_outputs, batch_gt_instances, batch_indices
                )
                loss_bbox, loss_giou = self.loss_boxes(
                    aux_outputs, batch_gt_instances, batch_indices
                )
                tmp_losses = dict(
                    loss_cls=loss_cls, loss_bbox=loss_bbox, loss_giou=loss_giou
                )
                for name, value in tmp_losses.items():
                    losses[f"s.{i}.{name}"] = value

                # 长度损失
                if self.use_length_prior:
                    loss_length = self.loss_length_computation(
                        aux_outputs, batch_gt_instances, batch_indices, batch_img_metas
                    )
                    losses[f"s.{i}.loss_length"] = loss_length

                # 拓扑损失
                if self.use_topology_pairing:
                    loss_topology = self.loss_topology_computation(
                        aux_outputs, batch_gt_instances, batch_indices
                    )
                    losses[f"s.{i}.loss_topology"] = loss_topology

                if self.use_length_ordering:
                    loss_ordering = self.loss_ordinal_length_consistency(
                        aux_outputs, batch_gt_instances, batch_indices, reduction="mean"
                    )
                    losses[f"s.{i}.loss_ordering"] = loss_ordering
                # 数量损失
                if self.use_count_prior:
                    loss_count = self.loss_count_computation(
                        aux_outputs, batch_gt_instances
                    )
                    losses[f"s.{i}.loss_count"] = loss_count

        return losses

    def loss_count_computation(self, outputs, batch_gt_instances):
        """计算数量约束损失"""
        pred_logits = outputs["pred_logits"]
        probs = pred_logits.sigmoid()  # [B, N, C]

        # 计算每个类别的预测数量（概率求和）
        pred_counts = probs.sum(dim=1)  # [B, C]

        device = pred_counts.device
        # num_classes = pred_counts.size(1)

        # 构建目标数量
        # 对于常染色体 (0-21)，目标一定是 2
        # 对于性染色体 (22-23)，根据 GT 确定
        target_counts = torch.zeros_like(pred_counts)

        # 常染色体先验：总是 2
        autosome_indices = list(range(22))
        target_counts[:, autosome_indices] = 2.0

        # 性染色体：根据 batch_gt_instances 统计
        # 假设最后两类是 X, Y
        for i, gt_instances in enumerate(batch_gt_instances):
            gt_labels = gt_instances.labels
            # 统计 X (22) 和 Y (23) 的数量
            for cls_id in range(22, 24):
                if cls_id < pred_counts.size(1):
                    count = (gt_labels == cls_id).sum().float()
                    target_counts[i, cls_id] = count

        # 计算 MSE Loss
        loss_count = self.loss_count(pred_counts, target_counts)
        return loss_count


@TASK_UTILS.register_module()
class ChromoDetMatcher(DiffusionDetMatcher):
    """
    染色体检测专用匹配器

    在原有匹配基础上考虑：
    1. 形态相似性
    2. 长度先验
    3. 类别特异性匹配
    """

    def __init__(
        self,
        match_costs: Union[List[Union[dict, ConfigDict]], dict, ConfigDict],
        center_radius: float = 2.5,
        candidate_topk: int = 5,
        iou_calculator: ConfigType = dict(type="BboxOverlaps2D"),
        # 染色体特化参数
        use_length_prior: bool = False,
        length_weight: float = 0.3,
        use_morphology_aware: bool = False,
        morphology_weight: float = 0.5,
        use_topology_pairing: bool = False,
        topology_weight: float = 0.5,
    ):
        super().__init__(
            match_costs=match_costs,
            center_radius=center_radius,
            candidate_topk=candidate_topk,
            iou_calculator=iou_calculator,
        )
        self.use_length_prior = use_length_prior
        self.length_weight = length_weight
        self.use_morphology_aware = use_morphology_aware
        self.morphology_weight = morphology_weight
        self.use_topology_pairing = use_topology_pairing

    def single_assigner(self, pred_instances, gt_instances, img_meta):
        with torch.no_grad():
            gt_bboxes = gt_instances.bboxes
            pred_bboxes = pred_instances.bboxes
            num_gt = gt_bboxes.size(0)

            if num_gt == 0:
                valid_mask = pred_bboxes.new_zeros(
                    (pred_bboxes.shape[0],), dtype=torch.bool
                )
                matched_gt_inds = pred_bboxes.new_zeros(
                    (gt_bboxes.shape[0],), dtype=torch.long
                )
                return valid_mask, matched_gt_inds

            valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
                bbox_xyxy_to_cxcywh(pred_bboxes), bbox_xyxy_to_cxcywh(gt_bboxes)
            )

            # 原有匹配代价
            cost_list = []
            for match_cost in self.match_costs:
                cost = match_cost(
                    pred_instances=pred_instances,
                    gt_instances=gt_instances,
                    img_meta=img_meta,
                )
                cost_list.append(cost)
            pairwise_ious = self.iou_calculator(pred_bboxes, gt_bboxes)

            cost_list.append((~is_in_boxes_and_center) * 100.0)
            cost_matrix = torch.stack(cost_list).sum(0)
            cost_matrix[~valid_mask] = cost_matrix[~valid_mask] + 10000.0

            fg_mask_inboxes, matched_gt_inds = self.dynamic_k_matching(
                cost_matrix, pairwise_ious, num_gt
            )

        return fg_mask_inboxes, matched_gt_inds
