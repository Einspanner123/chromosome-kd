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
from projects.DiffusionDet.diffusiondet.head import (DynamicDiffusionDetHead, SingleDiffusionDetHead, 
                   DynamicConv, SinusoidalPositionEmbeddings,
                   cosine_beta_schedule, extract, _DEFAULT_SCALE_CLAMP)

@MODELS.register_module()
class ChromoDetDynamicHead(DynamicDiffusionDetHead):
    """
    染色体检测专用的动态扩散检测头
    
    主要改进：
    1. 形态感知的噪声调度
    2. 长度先验引导的扩散采样
    3. 拓扑约束损失
    4. 重叠敏感的NMS
    """
    
    def __init__(self,
                 num_classes=24,  # 24种染色体类型
                 feat_channels=256,
                 num_proposals=500,
                 num_heads=6,
                 prior_prob=0.01,
                 snr_scale=2.0,
                 timesteps=1000,
                 sampling_timesteps=1,
                 self_condition=False,
                 box_renewal=True,
                 use_ensemble=True,
                 deep_supervision=True,
                 ddim_sampling_eta=1.0,
                 # 染色体特化参数
                 aspect_ratio_gamma=10.0,
                 length_prior_weight=0.1,
                 topology_loss_weight=0.05,
                 morphology_aware_noise=True,
                 length_priors=None,
                 criterion=dict(
                     type='ChromosomeDiffusionDetCriterion',
                     num_classes=24,
                     assigner=dict(
                         type='ChromosomeDiffusionDetMatcher',
                         match_costs=[
                             dict(
                                 type='FocalLossCost',
                                 alpha=2.0,
                                 gamma=0.25,
                                 weight=2.0),
                             dict(
                                 type='BBoxL1Cost',
                                 weight=5.0,
                                 box_format='xyxy'),
                             dict(type='IoUCost', iou_mode='giou', weight=2.0)
                         ],
                         center_radius=2.5,
                         candidate_topk=5),
                 ),
                 single_head=dict(
                     type='ChromosomeSingleDiffusionDetHead',
                     num_cls_convs=1,
                     num_reg_convs=3,
                     dim_feedforward=2048,
                     num_heads=8,
                     dropout=0.0,
                     act_cfg=dict(type='ReLU'),
                     dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)),
                 **kwargs) -> None:
        
        super().__init__(
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
            **kwargs)
        
        # 染色体特化参数
        self.aspect_ratio_gamma = aspect_ratio_gamma
        self.length_prior_weight = length_prior_weight
        self.topology_loss_weight = topology_loss_weight
        self.morphology_aware_noise = morphology_aware_noise
        
        # 染色体长度先验
        if length_priors is None:
            self.length_priors = torch.tensor([
                1.0, 0.95, 0.90, 0.85, 0.80, 0.75,
                0.70, 0.65, 0.60, 0.55, 0.50, 0.45,
                0.40, 0.38, 0.36, 0.34, 0.32, 0.30,
                0.28, 0.26, 0.24, 0.22, 0.20, 0.18
            ])
        else:
            self.length_priors = torch.tensor(length_priors)
        
        
        
        # 形态特征编码器
        self.morphology_encoder = nn.Sequential(
            nn.Linear(4, feat_channels // 2),  # 长宽比、角度等特征
            nn.ReLU(),
            nn.Linear(feat_channels // 2, feat_channels // 4)
        )
        
        # 长度预测头
        self.length_predictor = nn.Linear(feat_channels, 1)
        
        # 拓扑关系建模
        self.topology_encoder = nn.MultiheadAttention(
            feat_channels, num_heads=4, dropout=0.1)
    
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
        
        # 基础噪声
        noise = torch.randn(self.num_proposals, 4, device=device)
        
        # 形态感知的噪声调整
        if self.morphology_aware_noise and len(gt_boxes) > 0:
            # 计算真实框的形态特征
            gt_w = gt_boxes[:, 2] 
            gt_h = gt_boxes[:, 3]
            aspect_ratios = gt_h / (gt_w + 1e-6)
            
            # 对细长目标（高长宽比）减少噪声
            aspect_ratio_factor = torch.exp(-self.aspect_ratio_gamma * 
                                          torch.abs(aspect_ratios - aspect_ratios.mean()))
            
            # 调整噪声强度
            for i, factor in enumerate(aspect_ratio_factor):
                if i < len(noise):
                    noise[i] *= (0.5 + 0.5 * factor)  # 噪声范围：0.5-1.0
        
        num_gt = gt_boxes.shape[0]
        if num_gt < self.num_proposals:
            # 生成占位框，考虑长度先验
            box_placeholder = torch.randn(
                self.num_proposals - num_gt, 4, device=device) / 6. + 0.5
            
            # 应用长度先验
            if len(self.length_priors) >= self.num_proposals - num_gt:
                length_priors_subset = self.length_priors[num_gt:self.num_proposals]
                # 调整宽度和高度以符合长度先验
                for i, length_prior in enumerate(length_priors_subset):
                    # 假设细长形态，高度 > 宽度
                    box_placeholder[i, 2] *= 0.3  # 宽度较小
                    box_placeholder[i, 3] *= length_prior  # 高度按先验调整
            
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        else:
            select_mask = [True] * self.num_proposals + \
                          [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]

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
    
    def _compute_morphology_features(self, bboxes):
        """计算形态特征"""
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        # 长宽比
        aspect_ratios = h / (w + 1e-6)
        # 长度
        lengths = torch.sqrt(w**2 + h**2)
        # 角度（假设长轴方向）
        angles = torch.atan2(h, w)
        # 面积
        areas = w * h
        features = torch.stack([aspect_ratios, lengths, angles, areas], dim=1)
        return features
    
    def forward(self, features, init_bboxes, init_t, init_features=None):
        """前向传播，加入拓扑关系建模"""
        time = self.time_mlp(init_t)

        inter_class_logits = []
        inter_pred_bboxes = []
        inter_pred_lengths = []

        bs = len(features[0])
        bboxes = init_bboxes

        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None

        for head_idx, single_head in enumerate(self.head_series):
            # 单头预测
            class_logits, pred_bboxes, proposal_features = single_head(
                features, bboxes, proposal_features, self.roi_extractor, time)
            
            # 长度预测
            pred_lengths = self.length_predictor(proposal_features.squeeze(0))
            
            # 拓扑关系建模（最后一层）
            if head_idx == len(self.head_series) - 1:
                # 自注意力建模染色体间关系
                topo_features, _ = self.topology_encoder(
                    proposal_features, proposal_features, proposal_features)
                proposal_features = proposal_features + topo_features
                
                # 重新预测
                class_logits, pred_bboxes, _ = single_head(
                    features, bboxes, proposal_features, self.roi_extractor, time)
            
            if self.deep_supervision:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
                inter_pred_lengths.append(pred_lengths)
            
            bboxes = pred_bboxes.detach()

        if self.deep_supervision:
            return (torch.stack(inter_class_logits), 
                   torch.stack(inter_pred_bboxes),
                   torch.stack(inter_pred_lengths))
        else:
            return (class_logits[None, ...], 
                   pred_bboxes[None, ...],
                   pred_lengths[None, ...])
    
    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """损失计算，加入染色体特化损失"""
        prepare_outputs = self.prepare_training_targets(batch_data_samples)
        (batch_gt_instances, batch_pred_instances, batch_gt_instances_ignore,
         batch_img_metas) = prepare_outputs

        batch_diff_bboxes = torch.stack([
            pred_instances.diff_bboxes_abs
            for pred_instances in batch_pred_instances
        ])
        batch_time = torch.stack(
            [pred_instances.time for pred_instances in batch_pred_instances])

        # 前向传播
        pred_results = self(x, batch_diff_bboxes, batch_time)
        pred_logits, pred_bboxes = pred_results[:2]
        
        output = {
            'pred_logits': pred_logits[-1],
            'pred_boxes': pred_bboxes[-1]
        }
        
        # 添加长度预测
        if len(pred_results) > 2:
            pred_lengths = pred_results[2]
            output['pred_lengths'] = pred_lengths[-1]
        
        if self.deep_supervision:
            aux_outputs = []
            for i in range(len(pred_logits) - 1):
                aux_output = {
                    'pred_logits': pred_logits[i],
                    'pred_boxes': pred_bboxes[i]
                }
                if len(pred_results) > 2:
                    aux_output['pred_lengths'] = pred_results[2][i]
                aux_outputs.append(aux_output)
            output['aux_outputs'] = aux_outputs

        losses = self.criterion(output, batch_gt_instances, batch_img_metas)
        
        # 添加拓扑约束损失
        if self.topology_loss_weight > 0:
            topology_loss = self._compute_topology_loss(
                output, batch_gt_instances, batch_img_metas)
            losses['loss_topology'] = topology_loss * self.topology_loss_weight
        
        return losses
    
    def _compute_topology_loss(self, outputs, batch_gt_instances, batch_img_metas):
        """计算拓扑约束损失"""
        pred_bboxes = outputs['pred_boxes']
        batch_size = pred_bboxes.shape[0]
        
        topology_losses = []
        
        for batch_idx in range(batch_size):
            gt_instances = batch_gt_instances[batch_idx]
            pred_boxes_single = pred_bboxes[batch_idx]
            
            if len(gt_instances.labels) == 0:
                continue
            
            # 计算同源染色体配对损失
            pairing_loss = self._compute_pairing_loss(
                pred_boxes_single, gt_instances)
            topology_losses.append(pairing_loss)
        
        if len(topology_losses) > 0:
            return torch.stack(topology_losses).mean()
        else:
            return pred_bboxes.sum() * 0  # 返回零损失
    
    def _compute_pairing_loss(self, pred_boxes, gt_instances):
        """计算配对损失"""
        gt_labels = gt_instances.labels
        gt_boxes = gt_instances.bboxes
        
        pairing_loss = 0.0
        pair_count = 0
        
        # 遍历同源染色体对
        for chr1_type, chr2_type in [(i, i) for i in range(22)]:  # 常染色体
            # 找到对应类型的真实框
            mask1 = gt_labels == chr1_type
            mask2 = gt_labels == chr2_type
            
            if chr1_type == chr2_type:  # 同号染色体
                same_type_boxes = gt_boxes[mask1]
                if len(same_type_boxes) == 2:
                    # 计算两个同源染色体的距离
                    center1 = (same_type_boxes[0][:2] + same_type_boxes[0][2:]) / 2
                    center2 = (same_type_boxes[1][:2] + same_type_boxes[1][2:]) / 2
                    distance = torch.norm(center1 - center2)
                    
                    # 期望距离（可调参数）
                    expected_distance = 100.0  # 像素
                    pairing_loss += F.mse_loss(distance, 
                                             torch.tensor(expected_distance, 
                                                        device=distance.device))
                    pair_count += 1
        
        return pairing_loss / max(pair_count, 1)


@MODELS.register_module()
class ChromoDetSingleHead(SingleDiffusionDetHead):
    
    def __init__(self, 
                 num_classes=24,
                 feat_channels=256,
                 **kwargs):
        super().__init__(
            num_classes=num_classes,
            feat_channels=feat_channels,
            **kwargs)
        
        # 添加长度预测分支
        self.length_predictor = nn.Sequential(
            nn.Linear(feat_channels, feat_channels // 2),
            nn.ReLU(),
            nn.Linear(feat_channels // 2, 1)
        )
    
    def forward(self, features, bboxes, pro_features, pooler, time_emb):
        """前向传播，添加长度预测"""
        # 原有的前向传播
        class_logits, pred_bboxes, obj_features = super().forward(
            features, bboxes, pro_features, pooler, time_emb)
        
        # 长度预测
        N, num_boxes = bboxes.shape[:2]
        fc_feature = obj_features.transpose(0, 1).reshape(N * num_boxes, -1)
        pred_lengths = self.length_predictor(fc_feature)
        pred_lengths = pred_lengths.view(N, num_boxes, -1)
        
        return class_logits, pred_bboxes, obj_features, pred_lengths