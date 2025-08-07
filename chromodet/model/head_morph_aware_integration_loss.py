import copy
import math
import random
import warnings
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import (bbox2roi, bbox_cxcywh_to_xyxy,
                                   bbox_xyxy_to_cxcywh, get_box_wh,
                                   scale_boxes)
from mmdet.utils import InstanceList

# 导入原有的基础组件
from projects.DiffusionDet.diffusiondet.head import (
    DynamicDiffusionDetHead, SingleDiffusionDetHead, 
    DynamicConv, SinusoidalPositionEmbeddings,
    cosine_beta_schedule, extract, _DEFAULT_SCALE_CLAMP)

from projects.DiffusionDet.diffusiondet.loss import DiffusionDetCriterion

@MODELS.register_module()
class DynamicHead(DynamicDiffusionDetHead):
    def __init__(self,
                 num_classes=80,  # 类别数
                 feat_channels=256,  # 特征通道数
                 num_proposals=500,  # 建议框数量
                 num_heads=6,  # 注意力头数
                 prior_prob=0.01,  # 先验概率
                 snr_scale=2.0,  # 信噪比缩放因子
                 timesteps=1000,  # 扩散时间步数
                 sampling_timesteps=1,  # 采样时间步数
                 self_condition=False,  # 是否自条件
                 box_renewal=True,  # 是否使用框更新
                 use_ensemble=True,  # 是否使用集成
                 deep_supervision=True,  # 是否使用深度监督
                 ddim_sampling_eta=1.0,  # DDIM采样参数
                 aspect_ratio_gamma=10.0,
                 criterion:dict=None,
                 single_head:dict=None,
                 roi_extractor:dict=None,
                 train_cfg=None,
                 test_cfg=None) -> None:
        super(DynamicHead, self).__init__(
            num_classes=num_classes,
            feat_channels=feat_channels,
            num_proposals=num_proposals,
            num_heads=num_heads,
            prior_prob=prior_prob,
            snr_scale=snr_scale,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            self_condition=self_condition,
            box_renewal=box_renewal,
            use_ensemble=use_ensemble,
            deep_supervision=deep_supervision,
            ddim_sampling_eta=ddim_sampling_eta,
            criterion=criterion,
            single_head=single_head,
            roi_extractor=roi_extractor,
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )
        self._build_morphology_aware_diffusion()
        
        self.aspect_ratio_gamma = aspect_ratio_gamma
        
        
    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        # 准备训练目标
        prepare_outputs = self.prepare_training_targets(batch_data_samples)
        (batch_gt_instances, batch_pred_instances, batch_gt_instances_ignore,
         batch_img_metas) = prepare_outputs

        # 提取噪声边界框和时间步
        batch_diff_bboxes = torch.stack([
            pred_instances.diff_bboxes_abs for pred_instances in batch_pred_instances
        ])  # shape: [batch_size, num_proposals, 4]
        batch_time = torch.stack(
            [pred_instances.time for pred_instances in batch_pred_instances])  # shape: [batch_size]

        # 前向传播
        pred_logits, pred_bboxes = self(x, batch_diff_bboxes, batch_time)

        morphology_features = torch.stack([
            pred_instances.morphology_features for pred_instances in batch_pred_instances
        ])
        # 构建输出字典
        output = {
            'pred_logits': pred_logits[-1],  # 最后一层输出,shape: [batch_size, num_proposals, num_classes]
            'pred_boxes': pred_bboxes[-1],  # 最后一层输出,shape: [batch_size, num_proposals, 4]
            'morphology_features': morphology_features[-1]
        }
        # 如果使用深度监督,添加辅助输出
        if self.deep_supervision:
            output['aux_outputs'] = [{
                'pred_logits': a,
                'pred_boxes': b
            } for a, b in zip(pred_logits[:-1], pred_bboxes[:-1])]

        # 计算损失
        losses = self.criterion(output, batch_gt_instances, batch_img_metas)
        return losses
    def _build_morphology_aware_diffusion(self):
        """构建形态感知的扩散调度"""
        # 基础扩散调度
        betas = cosine_beta_schedule(self.timesteps)
        # 针对细长目标的调整
        # 早期时间步使用更小的噪声（保持形态）
        early_steps = self.timesteps // 4
        betas[:early_steps] *= 0.5
        # 中期时间步正常噪声
        # 后期时间步稍微增大噪声（增强随机性）
        late_steps = 3 * self.timesteps // 4
        betas[late_steps:] *= 1.2
        betas = betas.clamp(min=0.0, max=1-1e-6)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 其他扩散参数计算（与原版相同）
        self.register_buffer(
            'sqrt_alphas_cumprod', 
            torch.sqrt(alphas_cumprod))
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod',
            torch.sqrt(1. - alphas_cumprod))
        self.register_buffer(
            'log_one_minus_alphas_cumprod',
            torch.log(1. - alphas_cumprod))
        self.register_buffer(
            'sqrt_recip_alphas_cumprod',
            torch.sqrt(1. / alphas_cumprod))
        self.register_buffer(
            'sqrt_recipm1_alphas_cumprod',
            torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod)
        self.register_buffer(
            'posterior_variance',
            posterior_variance)
        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) /
                             (1. - alphas_cumprod))

    def prepare_diffusion(self, gt_boxes, image_size):
        """准备扩散过程，加入形态感知"""
        device = gt_boxes.device
        time = torch.randint(
            0, self.timesteps, (1, ), dtype=torch.long, device=device)
        
        noise = torch.randn(self.num_proposals, 4, device=device)
        
        # 形态感知的噪声调整
        # 计算真实框的形态特征
        gt_w = gt_boxes[:, 2]
        gt_h = gt_boxes[:, 3]
        # 统计得到长宽比集中在 w/h=6 以下
        aspect_ratios = gt_w / (gt_h + 1e-6)
        
        # 修改缩放因子的逻辑
        # 改进后的代码
        # 使用更稳定的基准值（如中位数）而不是均值
        aspect_ratio_median = torch.median(aspect_ratios)
        aspect_ratio_deviation = torch.abs(aspect_ratios - aspect_ratio_median)
        
        # 调整gamma值的影响力，使对极端长宽比的目标有更强的噪声抑制
        aspect_ratio_factor = torch.exp(-self.aspect_ratio_gamma * aspect_ratio_deviation / (1.0 + aspect_ratio_deviation))
        
        # 扩大噪声缩放范围，使细长目标的噪声更小，接近正方形的目标噪声相对更大
        factors = 0.3 + 0.7 * torch.exp(-aspect_ratio_factor)
        
        num_gt = gt_boxes.shape[0]
        
        if num_gt < self.num_proposals:
            num_placeholder = self.num_proposals - num_gt
            
            # 2. 生成符合宽高比分布的占位框
            # 2.1 定义宽高比区间和权重（根据直方图分布手动估计）
            bins = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], device=device)
            weights = torch.tensor([
                12000, 11000, 9000, 6000, 4000, 3000,  # 低比值区间（0~6）权重高
                1500, 800, 500, 600                    # 高比值区间（6~10）模拟长尾
            ], dtype=torch.float32, device=device)
            weights = weights / weights.sum()  # 归一化权重
            
            # 2.2 采样宽高比区间并生成具体值
            bin_indices = torch.multinomial(weights, num_placeholder, replacement=True)
            lower = bins[bin_indices]
            upper = bins[bin_indices + 1]
            aspect_ratios_placeholder = lower + (upper - lower) * torch.rand(num_placeholder, device=device)
            
            # 2.3 生成宽（w）、高（h）：控制在合理范围（0.2~0.8，相对值）
            h_placeholder = 0.2 + 0.6 * torch.rand(num_placeholder, device=device)  # 基础高度范围
            w_placeholder = aspect_ratios_placeholder * h_placeholder
            # 限制宽高，避免极端值
            w_placeholder = torch.clamp(w_placeholder, 0.2, 0.8)
            h_placeholder = torch.clamp(w_placeholder / aspect_ratios_placeholder, 0.2, 0.8)
            
            # 2.4 生成中心坐标（cx, cy）：避免过于靠边
            cx_placeholder = 0.1 + 0.8 * torch.rand(num_placeholder, device=device)
            cy_placeholder = 0.1 + 0.8 * torch.rand(num_placeholder, device=device)
            
            # 2.5 组合成cxcywh格式的占位框
            box_placeholder = torch.stack([
                cx_placeholder,
                cy_placeholder,
                w_placeholder,
                h_placeholder
            ], dim=1)
            
            # 2.6 计算占位框的噪声因子
            aspect_ratio_deviation_placeholder = torch.abs(aspect_ratios_placeholder - 1.0)
            aspect_ratio_factor_placeholder = torch.exp(-self.aspect_ratio_gamma * aspect_ratio_deviation_placeholder)
            factors_placeholder = 0.5 + 0.5 * aspect_ratio_factor_placeholder
            
            # 合并因子和框
            factors = torch.cat([aspect_ratio_factor, factors_placeholder], dim=0)
            noise *= factors.unsqueeze(1)
            x_start = torch.cat([gt_boxes, box_placeholder], dim=0)
        else:
            # 采样num_proposals个真实框，保持可复现性
            select_mask = torch.randperm(num_gt)[:self.num_proposals]
            x_start = gt_boxes[select_mask]
            factors = aspect_ratio_factor[select_mask]
            noise *= factors.unsqueeze(1)

        x_start = (x_start * 2. - 1.) * self.snr_scale

        # 噪声采样
        x = self.q_sample(x_start=x_start, time=time, noise=noise)
        x = torch.clamp(x, min=-1 * self.snr_scale, max=self.snr_scale)
        x = ((x / self.snr_scale) + 1) / 2.

        diff_bboxes = bbox_cxcywh_to_xyxy(x)
        diff_bboxes_abs = diff_bboxes * image_size

        metainfo = dict(time=time.squeeze(-1))
        pred_instances = InstanceData(metainfo=metainfo)
        pred_instances.diff_bboxes = diff_bboxes
        pred_instances.diff_bboxes_abs = diff_bboxes_abs
        pred_instances.noise = noise
        
        # 添加形态特征
        pred_instances.morphology_features = self._compute_morphology_features(diff_bboxes_abs)
        
        return pred_instances
    
    def _compute_morphology_features(self, bboxes, image_size=None):
        """计算更丰富的形态特征，增强对细长目标的识别能力"""
        # 计算基本的宽高
        w = bboxes[:, 2] - bboxes[:, 0]  # 宽度
        h = bboxes[:, 3] - bboxes[:, 1]  # 高度
        
        # 防止除零错误
        w_clamped = torch.clamp(w, min=1e-6)
        h_clamped = torch.clamp(h, min=1e-6)
        
        # 1. 长宽比相关特征（对细长目标特别重要）
        aspect_ratio = w / h_clamped  # 宽高比
        aspect_ratio_inv = h / w_clamped  # 高宽比（细长目标此值较大）
        aspect_ratio_log = torch.log(aspect_ratio)  # 对数变换，压缩范围
        aspect_ratio_square = aspect_ratio ** 2  # 放大极端比例的差异
        
        # 2. 大小相关特征
        area = w * h  # 面积
        area_log = torch.log(area + 1e-6)  # 对数面积，适应大尺度变化
        perimeter = 2 * (w + h)  # 周长
        compactness = (perimeter ** 2) / (4 * math.pi * area + 1e-6)  # 紧凑度（圆形为1，越细长越大）
        
        # 3. 对角线特征
        diagonal = torch.sqrt(w **2 + h** 2)  # 对角线长度
        diagonal_normalized = diagonal / (torch.norm(torch.tensor(image_size, device=bboxes.device, dtype=torch.float32)) 
                                        if image_size else 1.0)  # 相对于图像对角线的归一化
        
        # 4. 方向特征
        angle = torch.atan2(h, w)  # 角度（与水平方向的夹角）
        angle_sin = torch.sin(2 * angle)  # 二倍角正弦，增强方向敏感性
        angle_cos = torch.cos(2 * angle)  # 二倍角余弦
        
        top_matches = torch.zeros(bboxes.size(0), 3, device=bboxes.device)
        
        # 6. 宽高的相对大小（相对于图像）
        if image_size is not None:
            w_rel = w / image_size[1]  # 宽度相对图像宽度的比例
            h_rel = h / image_size[0]  # 高度相对图像高度的比例
            size_ratio = torch.stack([w_rel, h_rel], dim=1)
        else:
            size_ratio = torch.zeros(bboxes.size(0), 2, device=bboxes.device)
        
        # 堆叠所有特征
        features = torch.cat([
            # 长宽比特征
            aspect_ratio.unsqueeze(1),
            aspect_ratio_inv.unsqueeze(1),
            aspect_ratio_log.unsqueeze(1),
            aspect_ratio_square.unsqueeze(1),
            
            # 大小特征
            area_log.unsqueeze(1),
            compactness.unsqueeze(1),
            
            # 对角线特征
            diagonal_normalized.unsqueeze(1),
            
            # 方向特征
            angle_sin.unsqueeze(1),
            angle_cos.unsqueeze(1),
            
            # 先验匹配特征
            top_matches,
            
            # 相对大小特征
            size_ratio
        ], dim=1)
        
        # 特征归一化，使各维度尺度一致
        features = F.layer_norm(features, features.size()[1:])
        
        return features
        
@TASK_UTILS.register_module()
class Criterion(DiffusionDetCriterion):
    """
    染色体检测专用损失函数
    
    在原有损失基础上添加：
    1. 长度预测损失
    2. 形态约束损失
    3. 拓扑关系损失
    """
    
    def __init__(self,
                 num_classes=24,
                 assigner=None,
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
                 # 染色体特化损失
                 loss_length=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
                 loss_aspect_ratio=dict(type='L1Loss', reduction='sum', loss_weight=0.5),
                 loss_topology=dict(type='MSELoss', reduction='sum', loss_weight=0.2),
                 # 长度先验
                 length_priors=None,
                 aspect_ratio_target=10.0,  # 目标长宽比
                 **kwargs):
        
        super().__init__(
            num_classes=num_classes,
            assigner=assigner,
            deep_supervision=deep_supervision,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_giou=loss_giou)
        
        # 染色体特化损失
        self.loss_length = MODELS.build(loss_length)
        self.loss_aspect_ratio = MODELS.build(loss_aspect_ratio)
        self.loss_topology = MODELS.build(loss_topology)
        
        # 长度先验
        if length_priors is None:
            self.length_priors = torch.tensor([
                1.0, 0.95, 0.90, 0.85, 0.80, 0.75,
                0.70, 0.65, 0.60, 0.55, 0.50, 0.45,
                0.40, 0.38, 0.36, 0.34, 0.32, 0.30,
                0.28, 0.26, 0.24, 0.22, 0.20, 0.18
            ])
        else:
            self.length_priors = torch.tensor(length_priors)
        
        self.aspect_ratio_target = aspect_ratio_target
    
    def forward(self, outputs, batch_gt_instances, batch_img_metas):
        """前向传播，计算所有损失"""
        # 原有损失
        losses = super().forward(outputs, batch_gt_instances, batch_img_metas)
        
        # 染色体特化损失
        batch_indices = self.assigner(outputs, batch_gt_instances, batch_img_metas)
        
        # 长度损失
        if 'pred_lengths' in outputs:
            loss_length = self.loss_length_computation(
                outputs, batch_gt_instances, batch_indices)
            losses['loss_length'] = loss_length
        
        # 形态约束损失
        loss_morphology = self.loss_morphology_computation(
            outputs, batch_gt_instances, batch_indices)
        losses['loss_aspect_ratio'] = loss_morphology
                    

        # 拓扑损失
        loss_topology = self.loss_topology_computation(
            outputs, batch_gt_instances, batch_indices)
        losses['loss_topology'] = loss_topology
        
        # 深度监督的染色体特化损失
        if self.deep_supervision and 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                batch_indices = self.assigner(aux_outputs, batch_gt_instances, batch_img_metas)
                
                # 长度损失
                if 'pred_lengths' in aux_outputs:
                    loss_length = self.loss_length_computation(
                        aux_outputs, batch_gt_instances, batch_indices)
                    losses[f's.{i}.loss_length'] = loss_length
                
                # 形态约束损失
                loss_morphology = self.loss_morphology_computation(
                    aux_outputs, batch_gt_instances, batch_indices)
                losses[f's.{i}.loss_aspect_ratio'] = loss_morphology
                
                # 拓扑损失
                loss_topology = self.loss_topology_computation(
                    aux_outputs, batch_gt_instances, batch_indices)
                losses[f's.{i}.loss_topology'] = loss_topology
        
        return losses
    
    def loss_length_computation(self, outputs, batch_gt_instances, indices):
        """计算长度预测损失"""
        if 'pred_lengths' not in outputs:
            return torch.tensor(0.0, device=outputs['pred_boxes'].device)
        
        pred_lengths = outputs['pred_lengths']
        pred_boxes = outputs['pred_boxes']
        
        # 计算真实长度
        target_lengths_list = []
        pred_lengths_matched_list = []
        
        for batch_idx, (gt_instances, (pred_idx, gt_idx)) in enumerate(zip(batch_gt_instances, indices)):
            if len(gt_idx) == 0:
                continue
            
            # 真实框长度
            gt_boxes = gt_instances.bboxes[gt_idx]
            gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
            gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
            gt_lengths = torch.sqrt(gt_w**2 + gt_h**2)
            
            # 匹配的预测长度
            matched_pred_lengths = pred_lengths[batch_idx, pred_idx].squeeze(-1)
            
            target_lengths_list.append(gt_lengths)
            pred_lengths_matched_list.append(matched_pred_lengths)
        
        if len(target_lengths_list) == 0:
            return torch.tensor(0.0, device=pred_lengths.device)
        
        target_lengths = torch.cat(target_lengths_list)
        pred_lengths_matched = torch.cat(pred_lengths_matched_list)
        
        # 归一化长度
        target_lengths_norm = target_lengths / target_lengths.max()
        pred_lengths_norm = pred_lengths_matched / pred_lengths_matched.max()
        
        num_instances = target_lengths.shape[0]
        loss_length = self.loss_length(pred_lengths_norm, target_lengths_norm) / num_instances
        
        return loss_length
    
    def loss_morphology_computation(self, outputs, batch_gt_instances, indices):
        """计算形态约束损失（长宽比）"""
        pred_boxes = outputs['pred_boxes']
        
        target_aspect_ratios_list = []
        pred_aspect_ratios_list = []
        
        for batch_idx, (gt_instances, (pred_idx, gt_idx)) in enumerate(zip(batch_gt_instances, indices)):
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
    
    def loss_topology_computation(self, outputs, batch_gt_instances, indices):
        """计算拓扑关系损失"""
        pred_boxes = outputs['pred_boxes']
        batch_size = pred_boxes.shape[0]
        
        topology_losses = []
        
        for batch_idx in range(batch_size):
            gt_instances = batch_gt_instances[batch_idx]
            pred_boxes_single = pred_boxes[batch_idx]
            
            if len(gt_instances.labels) == 0:
                continue
            
            # 计算同源染色体配对损失
            pairing_loss = self._compute_pairing_constraint(
                pred_boxes_single, gt_instances)
            
            # 计算染色体分布损失（避免过度聚集）
            distribution_loss = self._compute_distribution_constraint(
                pred_boxes_single, gt_instances)
            
            total_topology_loss = pairing_loss + 0.5 * distribution_loss
            topology_losses.append(total_topology_loss)
        
        if len(topology_losses) > 0:
            return torch.stack(topology_losses).mean()
        else:
            return torch.tensor(0.0, device=pred_boxes.device)
    
    def _compute_pairing_constraint(self, pred_boxes, gt_instances):
        """计算配对约束损失"""
        gt_labels = gt_instances.labels
        gt_boxes = gt_instances.bboxes
        
        pairing_loss = 0.0
        pair_count = 0
        
        # 遍历常染色体（1-22号）
        for chr_type in range(22):
            # 找到同类型的真实框
            mask = gt_labels == chr_type
            same_type_boxes = gt_boxes[mask]
            
            if len(same_type_boxes) == 2:
                # 计算两个同源染色体的中心距离
                center1 = (same_type_boxes[0][:2] + same_type_boxes[0][2:]) / 2
                center2 = (same_type_boxes[1][:2] + same_type_boxes[1][2:]) / 2
                distance = torch.norm(center1 - center2)
                
                # 期望距离：不要太近（避免重叠）也不要太远
                min_distance = 50.0  # 最小距离
                max_distance = 200.0  # 最大距离
                
                if distance < min_distance:
                    pairing_loss += (min_distance - distance) ** 2
                elif distance > max_distance:
                    pairing_loss += (distance - max_distance) ** 2
                
                pair_count += 1
        
        return pairing_loss / max(pair_count, 1)
    
    def _compute_distribution_constraint(self, pred_boxes, gt_instances):
        """计算分布约束损失（避免染色体过度聚集）"""
        if len(gt_instances.bboxes) < 2:
            return torch.tensor(0.0, device=pred_boxes.device)
        
        gt_boxes = gt_instances.bboxes
        
        # 计算所有染色体的中心点
        centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        
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