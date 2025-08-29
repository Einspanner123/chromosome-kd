from typing import List, Tuple, Union
import torch
import torch.nn as nn
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import ConfigType

# 导入原有损失组件
from projects.DiffusionDet.diffusiondet.loss import DiffusionDetCriterion, DiffusionDetMatcher


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
                type='FocalLoss',
                use_sigmoid=True,
                alpha=0.25,
                gamma=2.0,
                reduction='sum',
                loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', reduction='sum', loss_weight=5.0),
            loss_giou=dict(type='GIoULoss', reduction='sum', loss_weight=2.0),
            # 长度先验
            use_length_prior:bool=True,
            loss_length=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
            aspect_ratio_target=10.0,  # 目标长宽比
            # 形态感知
            use_morphology_aware:bool=False,
            loss_aspect_ratio=dict(type='L1Loss', reduction='sum', loss_weight=0.5),
            # 拓扑匹配
            use_topology_pairing:bool=False,
            loss_topology=dict(type='MSELoss', reduction='sum', loss_weight=0.2),
            ):
        super().__init__(
            num_classes=num_classes,
            assigner=assigner,
            deep_supervision=deep_supervision,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_giou=loss_giou)
        
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
    
    def forward(self, outputs, batch_gt_instances, batch_img_metas):
        # 完整重写原loss的forward方法
        batch_indices = self.assigner(outputs, batch_gt_instances, batch_img_metas)
        # Compute all the requested losses
        loss_cls = \
            self.loss_classification(outputs, batch_gt_instances, batch_indices)
        
        loss_bbox, loss_giou = \
            self.loss_boxes(outputs, batch_gt_instances, batch_indices)

        losses = dict(
            loss_cls=loss_cls, 
            loss_bbox=loss_bbox, 
            loss_giou=loss_giou)
        
        # 长度损失
        if self.use_length_prior:
            loss_length = \
                self.loss_length_computation(outputs, batch_gt_instances, batch_indices, batch_img_metas)
            losses['loss_length'] = loss_length
        
        # 形态约束损失
        if self.use_morphology_aware:
            loss_morphology = \
                self.loss_morphology_computation(outputs, batch_gt_instances, batch_indices)
            losses['loss_aspect_ratio'] = loss_morphology

        # 拓扑损失
        if self.use_topology_pairing:
            loss_topology = \
                self.loss_topology_computation(outputs, batch_gt_instances, batch_indices)
            losses['loss_topology'] = loss_topology

        if self.deep_supervision:
            assert 'aux_outputs' in outputs
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                batch_indices = \
                    self.assigner(aux_outputs, batch_gt_instances, batch_img_metas)
                loss_cls = \
                    self.loss_classification(aux_outputs, batch_gt_instances, batch_indices)
                loss_bbox, loss_giou = \
                    self.loss_boxes(aux_outputs, batch_gt_instances, batch_indices)
                tmp_losses = dict(
                    loss_cls=loss_cls,
                    loss_bbox=loss_bbox,
                    loss_giou=loss_giou)
                for name, value in tmp_losses.items():
                    losses[f's.{i}.{name}'] = value
                    
                # 长度损失
                if self.use_length_prior:
                    loss_length = self.loss_length_computation(
                        aux_outputs, batch_gt_instances, batch_indices, batch_img_metas)
                    losses[f's.{i}.loss_length'] = loss_length
                
                # 形态约束损失
                if self.use_morphology_aware:
                    loss_morphology = self.loss_morphology_computation(
                        aux_outputs, batch_gt_instances, batch_indices)
                    losses[f's.{i}.loss_aspect_ratio'] = loss_morphology
                
                # 拓扑损失
                if self.use_topology_pairing:
                    loss_topology = self.loss_topology_computation(
                        aux_outputs, batch_gt_instances, batch_indices)
                    losses[f's.{i}.loss_topology'] = loss_topology
                    
        return losses
    
    def loss_length_computation(self, outputs, batch_gt_instances, batch_indices):
        """计算长度预测损失"""
        pred_lengths = outputs['pred_lengths']
        
        target_lengths_list = []
        pred_lengths_matched_list = []
        
        for batch_idx, (gt_instances, (pred_idx, gt_idx), img_meta) in enumerate(zip(batch_gt_instances, batch_indices, batch_img_metas)):
            if len(gt_idx) == 0:
                continue
            
            # 真实框长度
            gt_boxes = gt_instances.bboxes[gt_idx]
            gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
            gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
            gt_lengths = torch.sqrt(gt_w**2 + gt_h**2)
            
            # 匹配的预测长度
            matched_pred_lengths = pred_lengths[batch_idx, gt_idx].squeeze(-1)
            
            # 图像尺寸
            img_h, img_w = img_meta['img_shape']
            img_diagonal = torch.sqrt(torch.tensor(img_h**2 + img_w**2, dtype=torch.float32, device=gt_lengths.device))
            
            # 归一化
            target_lengths_list.append(gt_lengths / img_diagonal)
            pred_lengths_matched_list.append(matched_pred_lengths / img_diagonal)
        
        if len(target_lengths_list) == 0:
            return torch.tensor(0.0, device=pred_lengths.device)
        
        target_lengths_norm = torch.cat(target_lengths_list)
        pred_lengths_norm = torch.cat(pred_lengths_matched_list)
        
        num_instances = target_lengths_norm.shape[0]
        loss_length = self.loss_length(pred_lengths_norm, target_lengths_norm) / num_instances
        
        return loss_length

    def loss_morphology_computation(self, outputs, batch_gt_instances, batch_indices):
        """计算形态约束损失（长宽比）"""
        pred_boxes = outputs['pred_boxes']
        
        target_aspect_ratios_list = []
        pred_aspect_ratios_list = []
        
        for batch_idx, (gt_instances, (pred_idx, gt_idx)) in enumerate(zip(batch_gt_instances, batch_indices)):
            if len(gt_idx) == 0:
                continue
            
            # 真实框长宽比
            gt_boxes = gt_instances.bboxes[gt_idx]
            gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
            gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
            gt_aspect_ratios = gt_h / (gt_w + 1e-6)
            
            # 预测框长宽比
            pred_boxes_matched = pred_boxes[batch_idx, pred_idx]
            pred_w = pred_boxes_matched[:, 2] - pred_boxes_matched[:, 0]
            pred_h = pred_boxes_matched[:, 3] - pred_boxes_matched[:, 1]
            pred_aspect_ratios = pred_h / (pred_w + 1e-6)
            
            target_aspect_ratios_list.append(gt_aspect_ratios)
            pred_aspect_ratios_list.append(pred_aspect_ratios)
        
        if len(target_aspect_ratios_list) == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        
        target_aspect_ratios = torch.cat(target_aspect_ratios_list)
        pred_aspect_ratios = torch.cat(pred_aspect_ratios_list)
        
        # 鼓励细长形态（高长宽比）
        target_aspect_ratios = torch.clamp(target_aspect_ratios, min=2.0, max=20.0)
        pred_aspect_ratios = torch.clamp(pred_aspect_ratios, min=0.1, max=50.0)
        
        num_instances = target_aspect_ratios.shape[0]
        loss_aspect_ratio = self.loss_aspect_ratio(
            pred_aspect_ratios, target_aspect_ratios) / num_instances
        
        return loss_aspect_ratio
    
    def loss_topology_computation(self, outputs, batch_gt_instances, batch_indices):
        """计算拓扑关系损失"""
        pred_boxes = outputs['pred_boxes']
        pred_logits = outputs['pred_logits']


        # pred_idx, gt_idx = indices
        topology_losses = []
        
        for batch_idx, (gt_instances, (pred_idx, gt_idx)) in enumerate(zip(batch_gt_instances, batch_indices)):
            if len(gt_idx) == 0:
                continue
            gt_bboxes_matched = gt_instances.bboxes[gt_idx] # xyxy: [num_target, 4]
            gt_labels_matched = gt_instances.labels[gt_idx] # 0-23: [num_target]

            pred_boxes_matched = pred_boxes[batch_idx, pred_idx] # xyxy: [500, 4]
            pred_logits_matched = pred_logits[batch_idx, pred_idx] # [500, 24]
            
            # pred_idx  # [500], 被选择分配为正样本的预测框
            # gt_idx    # [num_gt], 每个正预测框对应的真实框下标
            # import pdb; pdb.set_trace()
            
            
            # 计算同源染色体配对损失
            pairing_loss = \
                self._compute_pairing_constraint(pred_logits_matched, pred_boxes_matched, gt_labels_matched)
            
            # 计算染色体分布损失（避免过度聚集）
            # distribution_loss = \
            #     self._compute_distribution_constraint(pred_boxes_batch, gt_instances)
            
            # total_topology_loss = pairing_loss + 0.5 * distribution_loss
            topology_losses.append(pairing_loss * 0.1) # 使用小权重，使得不会影响其他指标的梯度
        
        if len(topology_losses) > 0:
            try:
                return torch.stack(topology_losses).mean()
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()
        else:
            return torch.tensor(0.0, device=pred_boxes.device)
    
    def _compute_pairing_constraint(self, pred_logits, pred_bboxes, gt_labels):
        """计算配对约束损失"""
        # TODO: 改为依据分配的真实标签进行约束
        device = pred_logits.device
        # max_score, max_score_label = pred_logits.softmax(1).max(1) # 每个预测的最大分数和对应的类别
        bboxes_center = (pred_bboxes[:, :2] + pred_bboxes[:, 2:]) / 2.0 # 预测框的中心点
        pairing_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        pair_count = 0
        
        
        # 期望距离：不要太近（避免重叠）也不要太远
        # 根据数据集分布得到
        min_distance = 100.0
        max_distance = 350.0
        # 遍历常染色体
        for chr_type in range(22):
            type_idx = torch.where(gt_labels == chr_type)[0] # 同类下标
            num_type_target = type_idx.shape[0]

            if num_type_target == 2:
                # 由于每个框只进行一次分配，这里先只对成对的进行约束
                type_centers = bboxes_center[type_idx]
                distance = torch.norm(type_centers[0] - type_centers[1])
                
                if distance <= min_distance:
                    pairing_loss += min(torch.abs(min_distance - distance), 100)
                elif distance >= max_distance:
                    pairing_loss += min(torch.abs(distance - max_distance), 100)
                else:
                    # TODO: 是否应该对范围内的距离进行约束
                    ...
                pair_count += 1

        if pair_count == 0: # 若没有，返回一个小的基线损失，避免模型无梯度停滞
            pairing_loss += 1.0
        return pairing_loss / max(pair_count, 1)
    
    def _compute_distribution_constraint(self, pred_boxes, gt_instances):
        """计算分布约束损失（避免染色体过度聚集）"""
        if len(gt_instances.bboxes) < 2:
            return torch.tensor(0.0, device=pred_boxes.device)
        
        # gt_boxes = gt_instances.bboxes
        
        # 计算所有染色体的中心点
        centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
        
        # 计算平均距离
        distances = torch.cdist(centers, centers)
        # 排除对角线（自身距离）
        mask = torch.eye(len(centers), device=distances.device).bool()
        distances = distances[~mask]
        
        if len(distances) == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        
        mean_distance = distances.mean()
        
        # 期望的最小平均距离（避免过度聚集）
        expected_min_distance = 30.0
        
        if mean_distance < expected_min_distance:
            distribution_loss = (expected_min_distance - mean_distance) ** 2
        else:
            distribution_loss = torch.tensor(0.0, device=pred_boxes.device)
        
        return distribution_loss
  
  
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
            iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
            # 染色体特化参数
            length_weight: float = 0.3,
            use_morphology_aware:bool=False,
            morphology_weight: float = 0.5,
            use_topology_pairing:bool=False
            ):
        
        super().__init__(
            match_costs=match_costs,
            center_radius=center_radius,
            candidate_topk=candidate_topk,
            iou_calculator=iou_calculator,
            )
        # self.use_length_prior = use_length_prior
        # self.length_priors = torch.tensor(length_priors)
        self.length_weight = length_weight
        self.use_morphology_aware = use_morphology_aware
        self.morphology_weight = morphology_weight
        self.use_topology_pairing = use_topology_pairing
    
    def single_assigner(self, pred_instances, gt_instances, img_meta):
        """单图像匹配，加入形态特征"""
        with torch.no_grad():
            gt_bboxes = gt_instances.bboxes
            pred_bboxes = pred_instances.bboxes
            num_gt = gt_bboxes.size(0)
  
            if num_gt == 0:
                valid_mask = pred_bboxes.new_zeros(
                    (pred_bboxes.shape[0],), dtype=torch.bool)
                matched_gt_inds = pred_bboxes.new_zeros(
                    (gt_bboxes.shape[0],), dtype=torch.long)
                return valid_mask, matched_gt_inds
  
            valid_mask, is_in_boxes_and_center = \
                self.get_in_gt_and_in_center_info(
                    bbox_xyxy_to_cxcywh(pred_bboxes),
                    bbox_xyxy_to_cxcywh(gt_bboxes))
  
            # 原有匹配代价
            cost_list = []
            for match_cost in self.match_costs:
                cost = match_cost(
                    pred_instances=pred_instances,
                    gt_instances=gt_instances,
                    img_meta=img_meta)
                cost_list.append(cost)
            # Change start
            # 添加形态匹配代价
            if self.use_morphology_aware:
                morphology_cost = self._compute_morphology_cost(pred_bboxes, gt_bboxes)
                cost_list.append(morphology_cost * self.morphology_weight)
            
            # 添加长度匹配代价
            if self.use_length_prior:
                length_cost = self._compute_length_cost(pred_bboxes, gt_bboxes, gt_instances.labels)
                cost_list.append(length_cost * self.length_weight)
            # Change end
            pairwise_ious = self.iou_calculator(pred_bboxes, gt_bboxes)
  
            cost_list.append((~is_in_boxes_and_center) * 100.0)
            cost_matrix = torch.stack(cost_list).sum(0)
            cost_matrix[~valid_mask] = cost_matrix[~valid_mask] + 10000.0
  
            fg_mask_inboxes, matched_gt_inds = \
                self.dynamic_k_matching(cost_matrix, pairwise_ious, num_gt)
        
        return fg_mask_inboxes, matched_gt_inds
    
    def _compute_morphology_cost(self, pred_bboxes, gt_bboxes):
        """计算形态匹配代价"""
        # 预测框形态特征
        pred_w = pred_bboxes[:, 2] - pred_bboxes[:, 0]
        pred_h = pred_bboxes[:, 3] - pred_bboxes[:, 1]
        pred_aspect_ratios = pred_h / (pred_w + 1e-6)
        pred_lengths = torch.sqrt(pred_w**2 + pred_h**2)
        
        # 真实框形态特征
        gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        gt_h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        gt_aspect_ratios = gt_h / (gt_w + 1e-6)
        gt_lengths = torch.sqrt(gt_w**2 + gt_h**2)
        
        # 长宽比差异
        aspect_ratio_diff = torch.abs(
            pred_aspect_ratios[:, None] - gt_aspect_ratios[None, :])
        
        # 长度差异（归一化）
        pred_lengths_norm = pred_lengths / pred_lengths.max()
        gt_lengths_norm = gt_lengths / gt_lengths.max()
        length_diff = torch.abs(
            pred_lengths_norm[:, None] - gt_lengths_norm[None, :])
        
        # 综合形态代价
        morphology_cost = aspect_ratio_diff + length_diff
        
        return morphology_cost
    
    def _compute_length_cost(self, pred_bboxes, gt_bboxes, gt_labels):
        """计算长度先验代价"""
        # 长度先验（相对值）
        self.length_priors = self.length_priors.to(pred_bboxes.device)
        # 预测框长度
        diffs = pred_bboxes[:, [2, 3]] - pred_bboxes[:, [0, 1]]
        pred_lengths = torch.norm(diffs, dim=1)
        pred_lengths_norm = pred_lengths / (pred_lengths.max() + 1e-6)
        
        # # 计算与长度先验的差异
        # length_cost = torch.zeros(len(pred_bboxes), len(gt_bboxes), device=pred_bboxes.device)
        
        # for i, gt_label in enumerate(gt_labels):
        #     if gt_label < len(self.length_priors):
        #         expected_length = self.length_priors[gt_label]
        #         length_diff = torch.abs(pred_lengths_norm - expected_length)
        #         length_cost[:, i] = length_diff
        
        # return length_cost
        
        # 获取每个gt对应的长度先验值
        valid_labels = gt_labels < len(self.length_priors)
        expected_lengths = torch.zeros(len(gt_labels), device=pred_bboxes.device)
        expected_lengths[valid_labels] = self.length_priors[gt_labels[valid_labels]]
        
        # 计算与长度先验的差异
        length_cost = torch.abs(
            pred_lengths_norm[:, None] - expected_lengths[None, :]
        )
        
        # 对于超出先验范围的标签，设置较大的代价
        length_cost[:, ~valid_labels] = 1.0
        
        return length_cost
