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
                 use_morphology_aware:bool=False, # 形态感知
                 aspect_ratio_gamma=10.0,
                 use_length_prior:bool=False, # 长度感知
                 length_prior_weight=0.1,
                 length_priors=[
                    1.0, 0.95, 0.90, 0.85, 0.80, 0.75,
                    0.70, 0.65, 0.60, 0.55, 0.50, 0.45,
                    0.40, 0.38, 0.36, 0.34, 0.32, 0.30,
                    0.28, 0.26, 0.24, 0.22, 0.20, 0.18],
                 use_topology_pairing:bool=False, # 拓扑匹配
                 topology_loss_weight=0.05,
                 criterion=dict(
                     type='ChromoDetCriterion',
                     num_classes=24,
                     assigner=dict(
                         type='ChromoDetMatcher',
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
                     type='ChromoDetSingleHead',
                     num_cls_convs=1,
                     num_reg_convs=3,
                     dim_feedforward=2048,
                     num_heads=8,
                     dropout=0.0,
                     act_cfg=dict(type='ReLU'),
                     dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)),
                 roi_extractor=None,
                 train_cfg=None, test_cfg=None) -> None:
        
        """ 染色体特化参数 """
        self.use_morphology_aware = use_morphology_aware # 形态感知
        self.aspect_ratio_gamma = aspect_ratio_gamma
        self.use_length_prior = use_length_prior # 长度先验
        self.length_priors = length_priors
        if not isinstance(self.length_priors, torch.Tensor):
            self.length_priors = torch.tensor(length_priors)
        self.length_prior_weight = length_prior_weight
        self.use_topology_pairing = use_topology_pairing # 拓扑匹配
        self.topology_loss_weight = topology_loss_weight
        assert topology_loss_weight > 0, "topology_loss_weight should be > 0"
        
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
            roi_extractor=roi_extractor,
            train_cfg=train_cfg,
            test_cfg=test_cfg)
        
        # 形态特征编码器
        if use_morphology_aware:
            self.morphology_encoder = nn.Sequential(
                nn.Linear(4, feat_channels // 2),  # 长宽比、角度等特征
                nn.ReLU(),
                nn.Linear(feat_channels // 2, feat_channels // 4)
            )
            self._build_morphology_aware_diffusion()
        
        # 长度预测头
        # Single-head 中已有长度头
        # if use_length_prior:
        #     self.length_predictor = nn.Linear(feat_channels, 1)
        
        # 拓扑关系建模
        if use_topology_pairing:
            self.topology_encoder = \
                nn.MultiheadAttention(feat_channels, num_heads=4, dropout=0.1)
    
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
        betas = betas.clamp(min=0.0, max=1-1e-6) # 防止超过[0,1]范围
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 其他扩散参数计算（与原版相同）
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
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
        aspect_ratio_factor = torch.exp(
            -self.aspect_ratio_gamma * aspect_ratio_deviation / (1.0 + aspect_ratio_deviation))
        
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
        if self.use_morphology_aware:
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
        inter_pred_lengths = [] if self.use_length_prior else None

        bs = len(features[0])
        bboxes = init_bboxes

        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None

        for head_idx, single_head in enumerate(self.head_series):
            # 单头预测

            if self.use_length_prior:
                class_logits, pred_bboxes, proposal_features, pred_lengths = single_head(
                    features, bboxes, proposal_features, self.roi_extractor, time)
            else:
                class_logits, pred_bboxes, proposal_features = single_head(
                    features, bboxes, proposal_features, self.roi_extractor, time)
            
            # 长度预测
            # 使用 单头的长度预测头
            # pred_lengths = None
            # if self.use_length_prior:
            #     pred_lengths = self.length_predictor(proposal_features.squeeze(0))
            
            # 拓扑关系建模（最后一层）
            if self.use_topology_pairing and head_idx == len(self.head_series) - 1:
                # 自注意力建模染色体间关系
                topo_features, _ = self.topology_encoder(
                    proposal_features, proposal_features, proposal_features)
                proposal_features = proposal_features + topo_features
                
                # 重新预测
                # TODO: 这里是否可以改成多一层head，或者取代原先最后一层head的输入
                class_logits, pred_bboxes, _ = single_head(
                    features, bboxes, proposal_features, self.roi_extractor, time)
            
            if self.deep_supervision:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
                if self.use_length_prior:
                    inter_pred_lengths.append(pred_lengths)
            
            bboxes = pred_bboxes.detach()

        if self.deep_supervision:
            res = [
                torch.stack(inter_class_logits), 
                torch.stack(inter_pred_bboxes),
            ]
            if self.use_length_prior:
                res.append(torch.stack(inter_pred_lengths))
            return tuple(res)
        else:
            res = [
                class_logits[None, ...],
                pred_bboxes[None, ...],
            ]
            if self.use_length_prior:
                res.append(pred_lengths[None, ...])
            return tuple(res)
    
    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """损失计算，加入染色体特化损失"""
        prepare_outputs = self.prepare_training_targets(batch_data_samples)
        batch_gt_instances, batch_pred_instances, _, batch_img_metas = prepare_outputs

        batch_diff_bboxes = torch.stack([
            pred_instances.diff_bboxes_abs
            for pred_instances in batch_pred_instances])
        batch_time = torch.stack(
            [pred_instances.time for pred_instances in batch_pred_instances])

        # 前向传播
        pred_results = self(x, batch_diff_bboxes, batch_time)

        if self.use_length_prior:
            pred_logits, pred_bboxes, pred_lengths = pred_results
            output = {
                'pred_logits': pred_logits[-1],
                'pred_boxes': pred_bboxes[-1],
                'pred_lengths' : pred_lengths[-1]
            }
        else:
            pred_logits, pred_bboxes = pred_results
            output = {
                'pred_logits': pred_logits[-1],
                'pred_boxes': pred_bboxes[-1]
            }
        
        # 深度监督，添加辅助输出
        if self.deep_supervision:
            aux_outputs = []
            if self.use_length_prior:
                for i in range(len(pred_logits) - 1):
                    aux_output = {
                        'pred_logits': pred_logits[i],
                        'pred_boxes': pred_bboxes[i],
                        'pred_lengths': pred_lengths[i]
                    }
                    aux_outputs.append(aux_output)
            else:
                for i in range(len(pred_logits) - 1):
                    aux_output = {
                            'pred_logits': pred_logits[i],
                            'pred_boxes': pred_bboxes[i]
                        }
                    aux_outputs.append(aux_output)
            output['aux_outputs'] = aux_outputs

        losses = self.criterion(output, batch_gt_instances, batch_img_metas)
        
        # 添加拓扑约束损失
        if self.use_topology_pairing:
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

    def predict_by_feat(
            self,
            x,
            time_pairs,
            batch_noise_bboxes,
            batch_noise_bboxes_raw,
            batch_image_size,
            device,
            batch_img_metas=None,
            cfg=None,
            rescale=True):
        """
        根据特征进行预测
        
        Args:
            x: 特征金字塔
            time_pairs: 时间对列表,用于反向扩散
            batch_noise_bboxes: 批次噪声边界框（xyxy格式）
            batch_noise_bboxes_raw: 批次原始噪声边界框（未处理）
            batch_image_size: 批次图像尺寸
            device: 设备
            batch_img_metas: 批次图像元信息
            cfg: 配置
            rescale: 是否缩放
            
        Returns:
            预测结果列表
        """
        batch_size = len(batch_img_metas)  # 批次大小

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []  # 集成预测结果
        # 遍历时间对（反向扩散过程）
        for time, time_next in time_pairs:
            # 构造时间张量
            batch_time = \
                torch.full(
                    (batch_size, ), time, 
                    device=device, dtype=torch.long)  # shape: [batch_size]
            # 前向传播
            pred_results = self(x, batch_noise_bboxes, batch_time)
            
            if self.use_length_prior:
                pred_logits, pred_bboxes, pred_lengths = pred_results
            else:
                pred_logits, pred_bboxes = pred_results

            x_start = pred_bboxes[-1]  # 预测的去噪结果

            # 转换为cx,cy,w,h格式并归一化
            x_start = x_start / batch_image_size[:, None, :]  # shape: [batch_size, num_proposals, 4]
            x_start = bbox_xyxy_to_cxcywh(x_start)  # 转换为cx,cy,w,h格式
            x_start = (x_start * 2 - 1.) * self.snr_scale  # 缩放到[-snr_scale, snr_scale]
            x_start = torch.clamp(
                x_start, min=-1 * self.snr_scale, max=self.snr_scale)  # 限制范围
            # 从去噪结果预测噪声
            pred_noise = self.predict_noise_from_start(
                batch_noise_bboxes_raw, batch_time, x_start)
            
            pred_noise_list, x_start_list = [], []  # 每个图像的预测噪声和去噪结果
            noise_bboxes_list, num_remain_list = [], []  # 噪声框和保留数量
            if self.box_renewal:  # 如果使用框更新
                score_thr = cfg.get('score_thr', 0)  # 置信度阈值
                # 对批次中每个图像处理
                for img_id in range(batch_size):
                    score_per_image = pred_logits[-1][img_id]  # 当前图像的分类得分

                    score_per_image = torch.sigmoid(score_per_image)  # sigmoid激活
                    value, _ = torch.max(score_per_image, -1, keepdim=False)  # 每个框的最大类别得分
                    keep_idx = value > score_thr  # 保留高置信度框

                    num_remain_list.append(torch.sum(keep_idx))  # 保留框数量
                    pred_noise_list.append(pred_noise[img_id, keep_idx, :])  # 保留框对应的预测噪声
                    x_start_list.append(x_start[img_id, keep_idx, :])  # 保留框对应的去噪结果
                    noise_bboxes_list.append(batch_noise_bboxes[img_id, keep_idx, :])  # 保留的噪声框
            
            # 如果是最后一步
            if time_next < 0:
                # 不同于原始DiffusionDet
                if self.use_ensemble and self.sampling_timesteps > 1:
                    # 使用集成预测
                    box_pred_per_image, scores_per_image, labels_per_image = \
                        self.inference(
                            box_cls=pred_logits[-1],
                            box_pred=pred_bboxes[-1],
                            cfg=cfg,
                            device=device)
                    ensemble_score.append(scores_per_image)
                    ensemble_label.append(labels_per_image)
                    ensemble_coord.append(box_pred_per_image)
                continue

            # DDIM采样参数计算
            alpha = self.alphas_cumprod[time]  # 当前时间步alpha累积值
            alpha_next = self.alphas_cumprod[time_next]  # 下一时间步alpha累积值

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) *
                                              (1 - alpha_next) /
                                              (1 - alpha)).sqrt()  # sigma参数
            c = (1 - alpha_next - sigma**2).sqrt()  # c参数

            batch_noise_bboxes_list = []  # 新的噪声框列表
            batch_noise_bboxes_raw_list = []  # 新的原始噪声框列表
            # 对批次中每个图像处理
            for idx in range(batch_size):
                pred_noise = pred_noise_list[idx]  # 预测噪声
                x_start = x_start_list[idx]  # 去噪结果
                noise_bboxes = noise_bboxes_list[idx]  # 当前噪声框
                num_remain = num_remain_list[idx]  # 保留框数量
                noise = torch.randn_like(noise_bboxes)  # 新的随机噪声

                # DDIM采样步骤: 
                # x_{t-1} = 
                #   sqrt(alpha_{t-1}) * x_0 + 
                #   sqrt(1 - alpha_{t-1} - sigma^2) * pred_noise + 
                #   sigma * noise
                noise_bboxes = x_start * alpha_next.sqrt() + \
                    c * pred_noise + sigma * noise

                if self.box_renewal:  # 如果使用框更新
                    # 用随机框补充
                    if num_remain < self.num_proposals:
                        # 如果保留框少于建议框数量,用随机框填充
                        noise_bboxes = torch.cat(
                            (noise_bboxes,
                             torch.randn(
                                 self.num_proposals - num_remain,
                                 4,
                                 device=device)),
                            dim=0)  # shape: [num_proposals, 4]
                    else:
                        # 如果保留框多于建议框数量,随机选择
                        select_mask = [True] * self.num_proposals + \
                                      [False] * (num_remain -
                                                 self.num_proposals)
                        random.shuffle(select_mask)
                        noise_bboxes = noise_bboxes[select_mask]

                    # 保存原始噪声框
                    batch_noise_bboxes_raw_list.append(noise_bboxes)
                    # 处理噪声框: 转换为xyxy格式并缩放到图像尺寸
                    noise_bboxes = torch.clamp(
                        noise_bboxes,
                        min=-1 * self.snr_scale,
                        max=self.snr_scale)  # 限制范围
                    noise_bboxes = ((noise_bboxes / self.snr_scale) + 1) / 2  # 转换到[0,1]
                    noise_bboxes = bbox_cxcywh_to_xyxy(noise_bboxes)  # 转换为xyxy格式
                    noise_bboxes = noise_bboxes * batch_image_size[idx]  # 缩放到绝对坐标

                batch_noise_bboxes_list.append(noise_bboxes)
            # 更新噪声框
            batch_noise_bboxes = torch.stack(batch_noise_bboxes_list)
            batch_noise_bboxes_raw = torch.stack(batch_noise_bboxes_raw_list)
            
            # 如果使用集成预测
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = \
                    self.inference(
                        box_cls=pred_logits[-1],
                        box_pred=pred_bboxes[-1],
                        cfg=cfg,
                        device=device)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)
        
        # 如果使用集成预测
        if self.use_ensemble and self.sampling_timesteps > 1:
            steps = len(ensemble_score)  # 集成步数
            results_list = []
            # 对批次中每个图像处理
            for idx in range(batch_size):
                # 收集所有步的预测结果
                ensemble_score_per_img = [
                    ensemble_score[i][idx] for i in range(steps)
                ]
                ensemble_label_per_img = [
                    ensemble_label[i][idx] for i in range(steps)
                ]
                ensemble_coord_per_img = [
                    ensemble_coord[i][idx] for i in range(steps)
                ]

                # 拼接所有步的结果
                scores_per_image = torch.cat(ensemble_score_per_img, dim=0)
                labels_per_image = torch.cat(ensemble_label_per_img, dim=0)
                box_pred_per_image = torch.cat(ensemble_coord_per_img, dim=0)

                # 如果使用NMS
                if self.use_nms:
                    det_bboxes, keep_idxs = batched_nms(
                        box_pred_per_image, scores_per_image, labels_per_image,
                        cfg.nms)
                    box_pred_per_image = box_pred_per_image[keep_idxs]
                    labels_per_image = labels_per_image[keep_idxs]
                    scores_per_image = det_bboxes[:, -1]  # NMS可能重新加权得分
                # 创建结果对象
                results = InstanceData()
                results.bboxes = box_pred_per_image
                results.scores = scores_per_image
                results.labels = labels_per_image
            results_list.append(results)
        else:
            # 不使用集成预测,直接使用最后一步的结果
            box_cls = pred_logits[-1]
            box_pred = pred_bboxes[-1]
            results_list = self.inference(box_cls, box_pred, cfg, device)
        
        # 如果需要缩放结果
        if rescale:
            results_list = self.do_results_post_process(
                results_list, cfg, batch_img_metas=batch_img_metas)
        return results_list

@MODELS.register_module()
class ChromoDetSingleHead(SingleDiffusionDetHead):
    """染色体专用的单头检测器"""
    
    def __init__(self, 
                 num_classes,
                 feat_channels,
                 num_cls_convs,
                 num_reg_convs,
                 dim_feedforward,
                 num_heads,
                 dropout,
                 pooler_resolution,
                 use_focal_loss,
                 use_fed_loss,
                 use_length_prior:bool=None):
        super().__init__(
            num_classes=num_classes,
            feat_channels=feat_channels,
            dim_feedforward=dim_feedforward,
            num_cls_convs=num_cls_convs,
            num_reg_convs=num_reg_convs,
            num_heads=num_heads,
            dropout=dropout,
            pooler_resolution=pooler_resolution,
            use_focal_loss=use_focal_loss,
            use_fed_loss=use_fed_loss,
            )
        
        self.use_length_prior = use_length_prior
        assert use_length_prior is not None, \
            "Parameter 'use_length_prior' should be set."
        # 添加长度预测分支
        if use_length_prior:
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
        if self.use_length_prior:
            pred_lengths = self.length_predictor(fc_feature)
            pred_lengths = pred_lengths.view(N, num_boxes, -1)
            return class_logits, pred_bboxes, obj_features, pred_lengths
        else:
            return class_logits, pred_bboxes, obj_features