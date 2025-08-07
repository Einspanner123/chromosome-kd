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
class Criterion(nn.Module):

    def __init__(
            self,
            num_classes,
            assigner=dict(
                type='DiffusionDetMatcher',
                match_costs=[
                    dict(
                        type='FocalLossCost',
                        alpha=0.25,
                        gamma=2.0,
                        weight=2.0,
                        eps=1e-8),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0),
                    # 新增：形态特征匹配成本
                    dict(type='MSELossCost', weight=1.0)]),
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
            # 新增：形态特征损失配置
            loss_morphology=dict(type='L1Loss', reduction='sum', loss_weight=1.0)
    ):

        super().__init__()
        self.num_classes = num_classes

        if isinstance(assigner, nn.Module):
            self.assigner = assigner
        else:
            self.assigner = TASK_UTILS.build(assigner)

        self.deep_supervision = deep_supervision

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_giou = MODELS.build(loss_giou)
        # 初始化形态特征损失
        self.loss_morphology = MODELS.build(loss_morphology)

    def forward(self, outputs, batch_gt_instances, batch_img_metas):
        # 从outputs中获取预测实例（包含morphology_features）
        pred_instances = outputs['pred_instances']
        batch_indices = self.assigner(outputs, batch_gt_instances,
                                      batch_img_metas)
        
        # 计算所有损失（包含新增的形态损失）
        loss_cls = self.loss_classification(outputs, batch_gt_instances,
                                            batch_indices)
        loss_bbox, loss_giou = self.loss_boxes(outputs, batch_gt_instances,
                                               batch_indices)
        loss_morph = self.loss_morphology_features(
            pred_instances, batch_gt_instances,
            batch_indices, batch_img_metas
        )

        losses = dict(
            loss_cls=loss_cls, 
            loss_bbox=loss_bbox, 
            loss_giou=loss_giou,
            loss_morph=loss_morph  # 新增：形态特征损失
        )

        if self.deep_supervision:
            assert 'aux_outputs' in outputs
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_pred_instances = aux_outputs['pred_instances']
                batch_indices = self.assigner(aux_outputs, batch_gt_instances,
                                              batch_img_metas)
                loss_cls = self.loss_classification(aux_outputs,
                                                    batch_gt_instances,
                                                    batch_indices)
                loss_bbox, loss_giou = self.loss_boxes(aux_outputs,
                                                       batch_gt_instances,
                                                       batch_indices)
                loss_morph = self.loss_morphology_features(
                    aux_pred_instances, batch_gt_instances,
                    batch_indices, batch_img_metas
                )
                
                tmp_losses = dict(
                    loss_cls=loss_cls,
                    loss_bbox=loss_bbox,
                    loss_giou=loss_giou,
                    loss_morph=loss_morph
                )
                for name, value in tmp_losses.items():
                    losses[f's.{i}.{name}'] = value
        return losses

    def loss_classification(self, outputs, batch_gt_instances, indices):
        # 保持原有分类损失逻辑不变
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        target_classes_list = [
            gt.labels[J] for gt, (_, J) in zip(batch_gt_instances, indices)
        ]
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device)
        for idx in range(len(batch_gt_instances)):
            target_classes[idx, indices[idx][0]] = target_classes_list[idx]

        src_logits = src_logits.flatten(0, 1)
        target_classes = target_classes.flatten(0, 1)
        num_instances = max(torch.cat(target_classes_list).shape[0], 1)
        loss_cls = self.loss_cls(
            src_logits,
            target_classes,
        ) / num_instances
        return loss_cls

    def loss_boxes(self, outputs, batch_gt_instances, indices):
        # 保持原有边界框损失逻辑不变
        assert 'pred_boxes' in outputs
        pred_boxes = outputs['pred_boxes']

        target_bboxes_norm_list = [
            gt.norm_bboxes_cxcywh[J]
            for gt, (_, J) in zip(batch_gt_instances, indices)
        ]
        target_bboxes_list = [
            gt.bboxes[J] for gt, (_, J) in zip(batch_gt_instances, indices)
        ]

        pred_bboxes_list = []
        pred_bboxes_norm_list = []
        for idx in range(len(batch_gt_instances)):
            pred_bboxes_list.append(pred_boxes[idx, indices[idx][0]])
            image_size = batch_gt_instances[idx].image_size
            pred_bboxes_norm_list.append(pred_boxes[idx, indices[idx][0]] /
                                         image_size)

        pred_boxes_cat = torch.cat(pred_bboxes_list)
        pred_boxes_norm_cat = torch.cat(pred_bboxes_norm_list)
        target_bboxes_cat = torch.cat(target_bboxes_list)
        target_bboxes_norm_cat = torch.cat(target_bboxes_norm_list)

        if len(pred_boxes_cat) > 0:
            num_instances = pred_boxes_cat.shape[0]

            loss_bbox = self.loss_bbox(
                pred_boxes_norm_cat,
                bbox_cxcywh_to_xyxy(target_bboxes_norm_cat)) / num_instances
            loss_giou = self.loss_giou(pred_boxes_cat,
                                       target_bboxes_cat) / num_instances
        else:
            loss_bbox = pred_boxes.sum() * 0
            loss_giou = pred_boxes.sum() * 0
        return loss_bbox, loss_giou

    def loss_morphology_features(self, pred_instances, batch_gt_instances, indices, batch_img_metas):
        """修改：从实例中提取形态特征计算损失"""
        # 从预测实例中获取形态特征（每个实例的morphology_features属性）
        # pred_instances是列表，每个元素对应一个batch的实例数据
        pred_morph_list = []
        for idx in range(len(batch_gt_instances)):
            i, _ = indices[idx]  # 匹配的预测框索引
            if len(i) == 0:
                continue
            # 从当前batch的预测实例中提取匹配的形态特征
            pred_morph = pred_instances[idx].morphology_features[i]  # (M, C)
            pred_morph_list.append(pred_morph)

        # 计算真实目标的形态特征
        target_morph_list = []
        for idx in range(len(batch_gt_instances)):
            gt = batch_gt_instances[idx]
            _, J = indices[idx]  # 匹配的真实框索引
            if len(J) == 0:
                continue
            
            # 从真实框计算形态特征（与DynamicHead保持一致）
            gt_bboxes = gt.bboxes[J]  # 匹配的真实框 (M, 4) xyxy格式
            image_size = batch_img_metas[idx]['img_shape'][:2]  # (h, w)
            gt_morph = self._compute_morphology_features(gt_bboxes, image_size)
            target_morph_list.append(gt_morph)

        # 计算损失
        if len(pred_morph_list) > 0 and len(target_morph_list) > 0:
            pred_morph_cat = torch.cat(pred_morph_list)
            target_morph_cat = torch.cat(target_morph_list)
            num_instances = pred_morph_cat.shape[0]
            
            # 形态特征损失：预测特征与真实特征的L1距离
            loss_morph = self.loss_morphology(
                pred_morph_cat, 
                target_morph_cat
            ) / num_instances
        else:
            # 无匹配时损失为0（用第一个预测形态特征的设备创建零张量）
            if pred_morph_list:
                loss_morph = torch.tensor(0.0, device=pred_morph_list[0].device)
            else:
                loss_morph = torch.tensor(0.0)
        
        return loss_morph

    def _compute_morphology_features(self, bboxes, image_size):
        """复用形态特征计算逻辑（与DynamicHead完全一致）"""
        w = bboxes[:, 2] - bboxes[:, 0]  # 宽度
        h = bboxes[:, 3] - bboxes[:, 1]  # 高度
        
        # 防止除零错误
        w_clamped = torch.clamp(w, min=1e-6)
        h_clamped = torch.clamp(h, min=1e-6)
        
        # 1. 长宽比相关特征
        aspect_ratio = w / h_clamped
        aspect_ratio_inv = h / w_clamped
        aspect_ratio_log = torch.log(aspect_ratio)
        aspect_ratio_square = aspect_ratio ** 2
        
        # 2. 大小与紧凑度特征
        area = w * h
        area_log = torch.log(area + 1e-6)
        perimeter = 2 * (w + h)
        compactness = (perimeter **2) / (4 * torch.pi * area + 1e-6)
        
        # 3. 对角线特征
        diagonal = torch.sqrt(w** 2 + h **2)
        max_diag = torch.norm(torch.tensor(image_size, device=bboxes.device, dtype=torch.float32))
        diagonal_normalized = diagonal / (max_diag + 1e-6)
        
        # 4. 方向特征
        angle = torch.atan2(h, w)
        angle_sin = torch.sin(2 * angle)
        angle_cos = torch.cos(2 * angle)
        
        # 5. 先验匹配特征（若没有先验则用0填充）
        top_matches = torch.zeros(bboxes.size(0), 3, device=bboxes.device)
        
        # 6. 相对大小特征
        w_rel = w / image_size[1]
        h_rel = h / image_size[0]
        size_ratio = torch.stack([w_rel, h_rel], dim=1)
        
        # 堆叠所有特征并归一化
        features = torch.cat([
            aspect_ratio.unsqueeze(1),
            aspect_ratio_inv.unsqueeze(1),
            aspect_ratio_log.unsqueeze(1),
            aspect_ratio_square.unsqueeze(1),
            area_log.unsqueeze(1),
            compactness.unsqueeze(1),
            diagonal_normalized.unsqueeze(1),
            angle_sin.unsqueeze(1),
            angle_cos.unsqueeze(1),
            top_matches,
            size_ratio
        ], dim=1)
        
        features = F.layer_norm(features, features.size()[1:])
        return features

    def __init__(
            self,
            num_classes,
            assigner=dict(
                type='DiffusionDetMatcher',
                match_costs=[
                    dict(
                        type='FocalLossCost',
                        alpha=0.25,
                        gamma=2.0,
                        weight=2.0,
                        eps=1e-8),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0),
                    # 新增：形态特征匹配成本
                    dict(type='MSELossCost', weight=1.0)]),  # 用于匹配阶段的形态特征成本
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
            # 新增：形态特征损失配置
            loss_morphology=dict(type='L1Loss', reduction='sum', loss_weight=1.0)
    ):

        super().__init__()
        self.num_classes = num_classes

        if isinstance(assigner, nn.Module):
            self.assigner = assigner
        else:
            self.assigner = TASK_UTILS.build(assigner)

        self.deep_supervision = deep_supervision

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_giou = MODELS.build(loss_giou)
        # 初始化形态特征损失
        self.loss_morphology = MODELS.build(loss_morphology)

    def forward(self, outputs, batch_gt_instances, batch_img_metas):
        batch_indices = self.assigner(outputs, batch_gt_instances,
                                      batch_img_metas)
        # 计算所有损失（包含新增的形态损失）
        loss_cls = self.loss_classification(outputs, batch_gt_instances,
                                            batch_indices)
        loss_bbox, loss_giou = self.loss_boxes(outputs, batch_gt_instances,
                                               batch_indices)
        loss_morph = self.loss_morphology_features(outputs, batch_gt_instances,
                                                  batch_indices, batch_img_metas)

        losses = dict(
            loss_cls=loss_cls, 
            loss_bbox=loss_bbox, 
            loss_giou=loss_giou,
            loss_morph=loss_morph  # 新增：形态特征损失
        )

        if self.deep_supervision:
            assert 'aux_outputs' in outputs
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                batch_indices = self.assigner(aux_outputs, batch_gt_instances,
                                              batch_img_metas)
                loss_cls = self.loss_classification(aux_outputs,
                                                    batch_gt_instances,
                                                    batch_indices)
                loss_bbox, loss_giou = self.loss_boxes(aux_outputs,
                                                       batch_gt_instances,
                                                       batch_indices)
                loss_morph = self.loss_morphology_features(aux_outputs, 
                                                          batch_gt_instances,
                                                          batch_indices, 
                                                          batch_img_metas)
                
                tmp_losses = dict(
                    loss_cls=loss_cls,
                    loss_bbox=loss_bbox,
                    loss_giou=loss_giou,
                    loss_morph=loss_morph  # 辅助输出也计算形态损失
                )
                for name, value in tmp_losses.items():
                    losses[f's.{i}.{name}'] = value
        return losses

    def loss_classification(self, outputs, batch_gt_instances, indices):
        # 保持原有分类损失逻辑不变
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        target_classes_list = [
            gt.labels[J] for gt, (_, J) in zip(batch_gt_instances, indices)
        ]
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device)
        for idx in range(len(batch_gt_instances)):
            target_classes[idx, indices[idx][0]] = target_classes_list[idx]

        src_logits = src_logits.flatten(0, 1)
        target_classes = target_classes.flatten(0, 1)
        num_instances = max(torch.cat(target_classes_list).shape[0], 1)
        loss_cls = self.loss_cls(
            src_logits,
            target_classes,
        ) / num_instances
        return loss_cls

    def loss_boxes(self, outputs, batch_gt_instances, indices):
        # 保持原有边界框损失逻辑不变
        assert 'pred_boxes' in outputs
        pred_boxes = outputs['pred_boxes']

        target_bboxes_norm_list = [
            gt.norm_bboxes_cxcywh[J]
            for gt, (_, J) in zip(batch_gt_instances, indices)
        ]
        target_bboxes_list = [
            gt.bboxes[J] for gt, (_, J) in zip(batch_gt_instances, indices)
        ]

        pred_bboxes_list = []
        pred_bboxes_norm_list = []
        for idx in range(len(batch_gt_instances)):
            pred_bboxes_list.append(pred_boxes[idx, indices[idx][0]])
            image_size = batch_gt_instances[idx].image_size
            pred_bboxes_norm_list.append(pred_boxes[idx, indices[idx][0]] /
                                         image_size)

        pred_boxes_cat = torch.cat(pred_bboxes_list)
        pred_boxes_norm_cat = torch.cat(pred_bboxes_norm_list)
        target_bboxes_cat = torch.cat(target_bboxes_list)
        target_bboxes_norm_cat = torch.cat(target_bboxes_norm_list)

        if len(pred_boxes_cat) > 0:
            num_instances = pred_boxes_cat.shape[0]

            loss_bbox = self.loss_bbox(
                pred_boxes_norm_cat,
                bbox_cxcywh_to_xyxy(target_bboxes_norm_cat)) / num_instances
            loss_giou = self.loss_giou(pred_boxes_cat,
                                       target_bboxes_cat) / num_instances
        else:
            loss_bbox = pred_boxes.sum() * 0
            loss_giou = pred_boxes.sum() * 0
        return loss_bbox, loss_giou

    def loss_morphology_features(self, outputs, batch_gt_instances, indices, batch_img_metas):
        """新增：计算形态特征损失"""
        # 确保输出包含形态特征（对应DynamicHead中计算的morphology_features）
        assert 'morphology_features' in outputs
        pred_morph = outputs['morphology_features']  # 预测的形态特征 (B, N, C) 其中C=13

        # 计算真实目标的形态特征
        target_morph_list = []
        for idx in range(len(batch_gt_instances)):
            gt = batch_gt_instances[idx]
            _, J = indices[idx]  # 匹配的真实框索引
            if len(J) == 0:
                continue
            
            # 从真实框计算形态特征（复用之前的特征计算逻辑）
            gt_bboxes = gt.bboxes[J]  # 匹配的真实框 (M, 4) xyxy格式
            image_size = batch_img_metas[idx]['img_shape'][:2]  # (h, w)
            gt_morph = self._compute_morphology_features(gt_bboxes, image_size)
            target_morph_list.append(gt_morph)

        # 提取匹配的预测形态特征
        pred_morph_list = []
        for idx in range(len(batch_gt_instances)):
            i, _ = indices[idx]  # 匹配的预测框索引
            if len(i) == 0:
                continue
            pred_morph_list.append(pred_morph[idx, i])  # (M, C)

        # 计算损失
        if len(pred_morph_list) > 0 and len(target_morph_list) > 0:
            pred_morph_cat = torch.cat(pred_morph_list)
            target_morph_cat = torch.cat(target_morph_list)
            num_instances = pred_morph_cat.shape[0]
            
            # 形态特征损失：预测特征与真实特征的L1距离
            loss_morph = self.loss_morphology(
                pred_morph_cat, 
                target_morph_cat
            ) / num_instances
        else:
            # 无匹配时损失为0
            loss_morph = pred_morph.sum() * 0.0
        
        return loss_morph

    def _compute_morphology_features(self, bboxes, image_size):
        """复用形态特征计算逻辑（与DynamicHead保持一致）"""
        # 计算基本的宽高
        w = bboxes[:, 2] - bboxes[:, 0]  # 宽度
        h = bboxes[:, 3] - bboxes[:, 1]  # 高度
        
        # 防止除零错误
        w_clamped = torch.clamp(w, min=1e-6)
        h_clamped = torch.clamp(h, min=1e-6)
        
        # 1. 长宽比相关特征
        aspect_ratio = w / h_clamped
        aspect_ratio_inv = h / w_clamped
        aspect_ratio_log = torch.log(aspect_ratio)
        aspect_ratio_square = aspect_ratio ** 2
        
        # 2. 大小与紧凑度特征
        area = w * h
        area_log = torch.log(area + 1e-6)
        perimeter = 2 * (w + h)
        compactness = (perimeter **2) / (4 * torch.pi * area + 1e-6)
        
        # 3. 对角线特征
        diagonal = torch.sqrt(w** 2 + h **2)
        max_diag = torch.norm(torch.tensor(image_size, device=bboxes.device, dtype=torch.float32))
        diagonal_normalized = diagonal / (max_diag + 1e-6)
        
        # 4. 方向特征
        angle = torch.atan2(h, w)
        angle_sin = torch.sin(2 * angle)
        angle_cos = torch.cos(2 * angle)
        
        # 5. 先验匹配特征（若没有先验则用0填充）
        top_matches = torch.zeros(bboxes.size(0), 3, device=bboxes.device)
        
        # 6. 相对大小特征
        w_rel = w / image_size[1]
        h_rel = h / image_size[0]
        size_ratio = torch.stack([w_rel, h_rel], dim=1)
        
        # 堆叠所有特征并归一化
        features = torch.cat([
            aspect_ratio.unsqueeze(1),
            aspect_ratio_inv.unsqueeze(1),
            aspect_ratio_log.unsqueeze(1),
            aspect_ratio_square.unsqueeze(1),
            area_log.unsqueeze(1),
            compactness.unsqueeze(1),
            diagonal_normalized.unsqueeze(1),
            angle_sin.unsqueeze(1),
            angle_cos.unsqueeze(1),
            top_matches,
            size_ratio
        ], dim=1)
        
        features = F.layer_norm(features, features.size()[1:])
        return features