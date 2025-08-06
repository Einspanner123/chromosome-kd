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