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
        self.aspect_ratio_gamma = aspect_ratio_gamma
        self.length_priors = torch.tensor([
                1.0, 0.95, 0.90, 0.85, 0.80, 0.75,
                0.70, 0.65, 0.60, 0.55, 0.50, 0.45,
                0.40, 0.38, 0.36, 0.34, 0.32, 0.30,
                0.28, 0.26, 0.24, 0.22, 0.20, 0.18
            ])

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
        
        # 对细长目标（高长宽比）减少噪声
        aspect_ratio_factor = torch.exp(
            -self.aspect_ratio_gamma * torch.abs(aspect_ratios - aspect_ratios.mean()))
        
        # 调整噪声强度
        factors = 0.5 + 0.5 * aspect_ratio_factor
        # for i, factor in enumerate(aspect_ratio_factor):
        #     if i < len(noise):
        #         noise[i] *= (0.5 + 0.5 * factor)  # 噪声范围：0.5-1.0
        
        num_gt = gt_boxes.shape[0]
        
        if num_gt < self.num_proposals:
            noise[:len(factors), :] *= factors.unsqueeze(1)
            # 生成占位框，考虑长度先验
            box_placeholder = torch.randn(
                self.num_proposals - num_gt, 4, device=device) / 6. + 0.5
            
            # 应用长度先验
            if len(self.length_priors) >= self.num_proposals - num_gt:
                length_priors_subset = self.length_priors[num_gt:self.num_proposals]
                # 调整宽度和高度以符合长度先验
                for i, length_prior in enumerate(length_priors_subset):
                    # 假设细长形态，根据统计， 宽>高
                    box_placeholder[i, 2] *= length_prior
                    box_placeholder[i, 3] *= 0.3 
            
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        else:
            noise *= factors[:len(noise)]
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
        # pred_instances.morphology_features = self._compute_morphology_features(diff_bboxes_abs)
        
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