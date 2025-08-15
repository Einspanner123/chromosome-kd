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