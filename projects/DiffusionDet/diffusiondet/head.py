# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from https://github.com/ShoufaChen/DiffusionDet/blob/main/diffusiondet/detector.py
# Modified from https://github.com/ShoufaChen/DiffusionDet/blob/main/diffusiondet/head.py

# This work is licensed under the CC-BY-NC 4.0 License.
# Users should be careful about adopting these features in any commercial matters.
# For more details, please refer to https://github.com/ShoufaChen/DiffusionDet/blob/main/LICENSE

import copy
import math
import random
import warnings
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import (
    bbox2roi,
    bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh,
    get_box_wh,
    scale_boxes,
)
from mmdet.utils import InstanceList

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in
    https://openreview.net/forum?id=-NEXDKk8gZ.

    使用余弦调度生成beta值序列,控制前向扩散过程中每一步的噪声添加量
    """
    # timesteps+1个点,范围[0, timesteps]
    steps = timesteps + 1
    x = torch.linspace(
        0, timesteps, steps, dtype=torch.float64
    )  # shape: [timesteps+1]
    # 计算累积alpha值,使用余弦函数生成递减序列
    alphas_cumprod = (
        torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    )  # 取值范围[0, 1]
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # 归一化
    # 通过累积alpha值计算beta值: beta = 1 - alpha_t/alpha_{t-1}
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)  # 限制beta值范围


def extract(a, t, x_shape):
    """`extract` the appropriate t index for a batch of indices.

    从序列a中提取时间步t对应的值,并调整形状以匹配x_shape
    a: 序列参数,如alphas_cumprod等,shape: [timesteps]
    t: 时间步索引,shape: [batch_size]
    x_shape: 目标形状,用于reshape
    """
    batch_size = t.shape[0]  # 获取批次大小
    out = a.gather(
        -1, t
    )  # 根据t中的索引从a中收集对应的值,shape: [batch_size], 等价于a[t]
    # reshape为与x_shape相同的维度数,除了第0维为batch_size,其余维都是1
    # 例如x_shape=[8, 500, 4]时,out变为[8, 1, 1]
    return out.reshape(
        batch_size, *((1,) * (len(x_shape) - 1))
    )  # shape: [bs, 1, 1, 1, ..., 1(输入的维度数-1)]


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码模块,用于时间步编码"""

    def __init__(self, dim):
        """
        初始化正弦位置编码
        dim: 编码维度
        """
        super().__init__()
        self.dim = dim  # 位置编码的维度

    def forward(self, time):
        """
        前向传播
        time: 时间步张量,shape: [batch_size]
        返回: 正弦位置编码,shape: [batch_size, dim]
        """
        device = time.device  # 获取设备
        half_dim = self.dim // 2  # 一半维度用于sin,一半用于cos
        embeddings = math.log(10000) / (half_dim - 1)  # 计算频率因子
        # 生成频率参数,shape: [half_dim]
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        # time[:, None]: [batch_size, 1], embeddings[None, :]: [1, half_dim]
        # 相乘后得到: [batch_size, half_dim]
        embeddings = time[:, None] * embeddings[None, :]
        # 拼接sin和cos部分,得到完整的位置编码: [batch_size, dim]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


@MODELS.register_module()
class DynamicDiffusionDetHead(nn.Module):
    """动态DiffusionDet检测头"""

    def __init__(
        self,
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
        criterion=dict(  # 损失函数配置
            type='DiffusionDetCriterion',
            num_classes=80,
            assigner=dict(
                type='DiffusionDetMatcher',
                match_costs=[
                    dict(
                        type='FocalLossCost', alpha=2.0, gamma=0.25, weight=2.0
                    ),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0),
                ],
                center_radius=2.5,
                candidate_topk=5,
            ),
        ),
        single_head=dict(  # 单个检测头配置
            type='DiffusionDetHead',
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            act_cfg=dict(type='ReLU'),
            dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
        ),
        roi_extractor=dict(  # ROI提取器配置
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        test_cfg=None,  # 测试配置
        **kwargs,
    ) -> None:
        super().__init__()
        # 构建ROI特征提取器
        self.roi_extractor = MODELS.build(roi_extractor)

        self.num_classes = num_classes  # 类别数
        self.feat_channels = feat_channels  # 特征通道数
        self.num_proposals = num_proposals  # 建议框数量
        self.num_heads = num_heads  # 注意力头数

        # 构建扩散过程参数
        assert isinstance(timesteps, int), (
            f'The type of `timesteps` should be int but got {type(timesteps)}'
        )
        assert sampling_timesteps <= timesteps
        self.timesteps = timesteps  # 扩散总时间步数
        self.sampling_timesteps = sampling_timesteps  # 采样时间步数
        self.snr_scale = snr_scale  # 信噪比缩放因子

        # 是否使用DDIM采样（加速采样）
        self.ddim_sampling = self.sampling_timesteps < self.timesteps
        self.ddim_sampling_eta = ddim_sampling_eta  # DDIM采样参数
        self.self_condition = self_condition  # 是否使用自条件
        self.box_renewal = box_renewal  # 是否使用框更新策略
        self.use_ensemble = use_ensemble  # 是否使用集成预测

        self._build_diffusion()  # 构建扩散过程所需参数

        # 构建分配器
        assert criterion.get('assigner', None) is not None
        assigner = TASK_UTILS.build(criterion.get('assigner'))
        # 初始化参数
        self.use_focal_loss = assigner.use_focal_loss  # 是否使用focal loss
        self.use_fed_loss = assigner.use_fed_loss  # 是否使用fed loss

        # 构建损失函数
        criterion.update(deep_supervision=deep_supervision)
        self.criterion = TASK_UTILS.build(criterion)

        # 构建动态检测头
        single_head_ = single_head.copy()
        single_head_num_classes = single_head_.get('num_classes', None)
        if single_head_num_classes is None:
            single_head_.update(num_classes=num_classes)
        else:
            if single_head_num_classes != num_classes:
                warnings.warn(
                    'The `num_classes` of `DynamicDiffusionDetHead` and '
                    '`SingleDiffusionDetHead` should be same, changing '
                    f'`single_head.num_classes` to {num_classes}'
                )
                single_head_.update(num_classes=num_classes)

        single_head_feat_channels = single_head_.get('feat_channels', None)
        if single_head_feat_channels is None:
            single_head_.update(feat_channels=feat_channels)
        else:
            if single_head_feat_channels != feat_channels:
                warnings.warn(
                    'The `feat_channels` of `DynamicDiffusionDetHead` and '
                    '`SingleDiffusionDetHead` should be same, changing '
                    f'`single_head.feat_channels` to {feat_channels}'
                )
                single_head_.update(feat_channels=feat_channels)

        default_pooler_resolution = roi_extractor['roi_layer'].get(
            'output_size'
        )
        assert default_pooler_resolution is not None
        single_head_pooler_resolution = single_head_.get('pooler_resolution')
        if single_head_pooler_resolution is None:
            single_head_.update(pooler_resolution=default_pooler_resolution)
        else:
            if single_head_pooler_resolution != default_pooler_resolution:
                warnings.warn(
                    'The `pooler_resolution` of `DynamicDiffusionDetHead` '
                    'and `SingleDiffusionDetHead` should be same, changing '
                    f'`single_head.pooler_resolution` to {num_classes}'
                )
                single_head_.update(
                    pooler_resolution=default_pooler_resolution
                )

        single_head_.update(
            use_focal_loss=self.use_focal_loss, use_fed_loss=self.use_fed_loss
        )
        single_head_module = MODELS.build(single_head_)

        self.num_heads = num_heads
        # 创建多个检测头形成序列,每个头完成一次去噪
        self.head_series = nn.ModuleList(
            [copy.deepcopy(single_head_module) for _ in range(num_heads)]
        )

        self.deep_supervision = deep_supervision  # 是否使用深度监督

        # 时间嵌入MLP: 用于处理时间步信息
        time_dim = feat_channels * 4  # 时间嵌入维度
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(feat_channels),  # 正弦位置编码
            nn.Linear(feat_channels, time_dim),
            nn.GELU(),  # 线性变换 + GELU激活
            nn.Linear(time_dim, time_dim),
        )  # 再次线性变换

        self.prior_prob = prior_prob  # 先验概率
        self.test_cfg = test_cfg  # 测试配置
        self.use_nms = self.test_cfg.get('use_nms', True)  # 是否使用NMS
        self._init_weights()  # 初始化权重

    def _init_weights(self):
        """初始化网络权重"""
        # 根据先验概率计算偏置值
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(
                    p
                )  # 对于维度大于1的参数使用Xavier初始化

            # 为focal loss和fed loss初始化分类层偏置
            if self.use_focal_loss or self.use_fed_loss:
                if p.dim() > 0 and (
                    p.shape[-1] == self.num_classes
                    or p.shape[-1] == self.num_classes + 1
                ):
                    nn.init.constant_(p, bias_value)  # 分类层偏置初始化

    def _build_diffusion(self):
        """构建扩散过程所需参数"""
        betas = cosine_beta_schedule(self.timesteps)  # 生成beta序列
        alphas = 1.0 - betas  # alpha = 1 - beta
        # 累积alpha值: alpha_t_bar = alpha_1 * alpha_2 * ... * alpha_t
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # 前一个时间步的累积alpha值,第一个元素为1
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # 注册为buffer,不会被更新但会随模型移动设备
        self.register_buffer('betas', betas)  # beta值
        self.register_buffer('alphas_cumprod', alphas_cumprod)  # 累积alpha值
        self.register_buffer(
            'alphas_cumprod_prev', alphas_cumprod_prev
        )  # 前一个累积alpha值

        # 前向扩散q(x_t | x_{t-1})所需计算
        self.register_buffer(
            'sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod)
        )  # sqrt(alpha_t_bar)
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod)
        )  # sqrt(1 - alpha_t_bar)
        self.register_buffer(
            'log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod)
        )  # log(1 - alpha_t_bar)
        self.register_buffer(
            'sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod)
        )  # sqrt(1 / alpha_t_bar)
        self.register_buffer(
            'sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1)
        )  # sqrt(1 / alpha_t_bar - 1)

        # 后验分布q(x_{t-1} | x_t, x_0)所需计算
        # 计算后验方差: beta_t * (1 - alpha_{t-1}_bar) / (1 - alpha_t_bar)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            'posterior_variance', posterior_variance
        )  # 后验方差

        # 由于开始时后验方差为0,需要截断log计算
        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(posterior_variance.clamp(min=1e-20)),
        )  # 截断后的后验log方差
        # 后验均值系数1: beta_t * sqrt(alpha_{t-1}_bar) / (1 - alpha_t_bar)
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        # 后验均值系数2: (1 - alpha_{t-1}_bar) * sqrt(alpha_t) / (1 - alpha_t_bar)
        self.register_buffer(
            'posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod),
        )

    def forward(self, features, init_bboxes, init_t, init_features=None):
        """
        前向传播

        Args:
            features: 特征金字塔, 元组形式, 每个元素shape: [batch_size, channels, height, width]
            init_bboxes: 初始边界框, shape: [batch_size, num_proposals, 4]
            init_t: 初始时间步, shape: [batch_size]
            init_features: 初始特征, 可选, shape: [channels]

        Returns:
            pred_logits: 预测分类logits, shape: [num_heads, batch_size, num_proposals, num_classes]
            pred_bboxes: 预测边界框, shape: [num_heads, batch_size, num_proposals, 4]
        """
        # 时间嵌入: 将时间步编码为高维向量
        time = self.time_mlp(init_t)  # shape: [batch_size, time_dim]

        inter_class_logits = []  # 存储中间分类logits
        inter_pred_bboxes = []  # 存储中间预测框

        bs = len(features[0])  # 批次大小
        bboxes = init_bboxes  # 当前边界框,初始为输入的噪声框

        # 处理初始特征
        if init_features is not None:
            # 扩展初始特征以匹配批次大小
            init_features = init_features[None].repeat(
                1, bs, 1
            )  # shape: [1, batch_size, channels]
            proposal_features = init_features.clone()  # 提案特征
        else:
            proposal_features = None

        # 依次通过每个检测头进行去噪
        for head_idx, single_head in enumerate(self.head_series):
            # 单个检测头前向传播
            class_logits, pred_bboxes, proposal_features = single_head(
                features, bboxes, proposal_features, self.roi_extractor, time
            )
            # 如果使用深度监督,保存中间结果
            if self.deep_supervision:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            # 更新边界框（使用detach防止梯度传播）
            bboxes = pred_bboxes.detach()

        # 返回结果
        if self.deep_supervision:
            # 堆叠所有中间结果
            return torch.stack(inter_class_logits), torch.stack(
                inter_pred_bboxes
            )
        else:
            # 只返回最后一个头的结果
            return class_logits[None, ...], pred_bboxes[None, ...]

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """执行前向传播并计算检测头的损失

        Args:
            x (tuple[Tensor]): 上游网络的特征,每个都是4D张量
            batch_data_samples (List[:obj:[DetDataSample](file:///home/linkst/workplace/chromo/chromosome-kd/mmdet/structures/det_data_sample.py#L6-L232)]): 数据样本

        Returns:
            dict: 损失组件字典
        """
        # 准备训练目标
        prepare_outputs = self.prepare_training_targets(batch_data_samples)
        (
            batch_gt_instances,
            batch_pred_instances,
            batch_gt_instances_ignore,
            batch_img_metas,
        ) = prepare_outputs

        # 提取噪声边界框和时间步
        batch_diff_bboxes = torch.stack(
            [
                pred_instances.diff_bboxes_abs
                for pred_instances in batch_pred_instances
            ]
        )  # shape: [batch_size, num_proposals, 4]
        batch_time = torch.stack(
            [pred_instances.time for pred_instances in batch_pred_instances]
        )  # shape: [batch_size]

        # 前向传播
        pred_logits, pred_bboxes = self(x, batch_diff_bboxes, batch_time)

        # 构建输出字典
        output = {
            'pred_logits': pred_logits[
                -1
            ],  # 最后一层输出,shape: [batch_size, num_proposals, num_classes]
            'pred_boxes': pred_bboxes[
                -1
            ],  # 最后一层输出,shape: [batch_size, num_proposals, 4]
        }
        # 如果使用深度监督,添加辅助输出
        if self.deep_supervision:
            output['aux_outputs'] = [
                {'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(pred_logits[:-1], pred_bboxes[:-1])
            ]

        # 计算损失
        losses = self.criterion(output, batch_gt_instances, batch_img_metas)
        return losses

    def prepare_training_targets(self, batch_data_samples):
        """
        准备训练目标

        Args:
            batch_data_samples: 批次数据样本

        Returns:
            tuple: (batch_gt_instances, batch_pred_instances, batch_gt_instances_ignore, batch_img_metas)
        """
        # 可选：设置随机种子以保持结果一致
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        batch_gt_instances = []  # 真实实例
        batch_pred_instances = []  # 预测实例（带噪声的）
        batch_gt_instances_ignore = []  # 忽略的实例
        batch_img_metas = []  # 图像元信息
        # 遍历批次中的每个样本
        for data_sample in batch_data_samples:
            img_meta = data_sample.metainfo  # 图像元信息
            gt_instances = data_sample.gt_instances  # 真实实例

            gt_bboxes = gt_instances.bboxes  # 真实边界框,shape: [num_gts, 4]
            h, w = img_meta['img_shape']  # 图像尺寸
            image_size = gt_bboxes.new_tensor(
                [w, h, w, h]
            )  # 图像大小张量,shape: [4]

            # 归一化边界框到[0,1]范围
            norm_gt_bboxes = gt_bboxes / image_size  # shape: [num_gts, 4]
            # 转换为cx,cy,w,h格式
            norm_gt_bboxes_cxcywh = bbox_xyxy_to_cxcywh(
                norm_gt_bboxes
            )  # shape: [num_gts, 4]
            # 准备扩散训练目标（添加噪声）
            pred_instances = self.prepare_diffusion(
                norm_gt_bboxes_cxcywh, image_size
            )

            # 设置元信息
            gt_instances.set_metainfo(dict(image_size=image_size))
            gt_instances.norm_bboxes_cxcywh = norm_gt_bboxes_cxcywh

            batch_gt_instances.append(gt_instances)
            batch_pred_instances.append(pred_instances)
            batch_img_metas.append(data_sample.metainfo)
            # 处理忽略的实例
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)
        return (
            batch_gt_instances,
            batch_pred_instances,
            batch_gt_instances_ignore,
            batch_img_metas,
        )

    def prepare_diffusion(self, gt_boxes, image_size):
        """
        准备扩散训练目标（添加噪声）

        Args:
            gt_boxes: 真实边界框（cx,cy,w,h格式）,shape: [num_gts, 4]
            image_size: 图像尺寸,shape: [4]

        Returns:
            pred_instances: 带噪声的实例数据
        """
        device = gt_boxes.device  # 获取设备
        # 随机采样一个时间步
        time = torch.randint(
            0, self.timesteps, (1,), dtype=torch.long, device=device
        )  # shape: [1]
        # 生成随机噪声
        noise = torch.randn(
            self.num_proposals, 4, device=device
        )  # shape: [num_proposals, 4]

        num_gt = gt_boxes.shape[0]  # 真实框数量
        if num_gt < self.num_proposals:
            # 如果真实框少于建议框数量,用随机框填充
            # 3 * sigma = 1/2 --> sigma: 1/6
            box_placeholder = (
                torch.randn(self.num_proposals - num_gt, 4, device=device)
                / 6.0
                + 0.5
            )  # shape: [num_proposals-num_gt, 4]
            box_placeholder[:, 2:] = torch.clip(
                box_placeholder[:, 2:], min=1e-4
            )  # 保证宽度和高度为正
            x_start = torch.cat(
                (gt_boxes, box_placeholder), dim=0
            )  # shape: [num_proposals, 4]
        else:
            # 如果真实框多于建议框数量,随机选择
            select_mask = [True] * self.num_proposals + [False] * (
                num_gt - self.num_proposals
            )
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]  # shape: [num_proposals, 4]

        # 缩放处理: x_start范围从[0,1]变为[-snr_scale, snr_scale]
        x_start = (x_start * 2.0 - 1.0) * self.snr_scale

        # 前向扩散采样: 添加噪声
        x = self.q_sample(x_start=x_start, time=time, noise=noise)

        # 限制范围并转换回[0,1]
        x = torch.clamp(x, min=-1 * self.snr_scale, max=self.snr_scale)
        x = ((x / self.snr_scale) + 1) / 2.0

        # 转换为xyxy格式
        diff_bboxes = bbox_cxcywh_to_xyxy(x)  # shape: [num_proposals, 4]
        # 转换为绝对坐标
        diff_bboxes_abs = diff_bboxes * image_size  # shape: [num_proposals, 4]

        # 创建实例数据
        metainfo = dict(time=time.squeeze(-1))  # 时间步信息
        pred_instances = InstanceData(metainfo=metainfo)
        pred_instances.diff_bboxes = diff_bboxes  # 相对坐标边界框
        pred_instances.diff_bboxes_abs = diff_bboxes_abs  # 绝对坐标边界框
        pred_instances.noise = noise  # 噪声
        return pred_instances

    # 前向扩散
    def q_sample(self, x_start, time, noise=None):
        """
        前向扩散采样

        Args:
            x_start: 初始数据（真实框）,shape: [num_proposals, 4]
            time: 时间步,shape: [1]
            noise: 噪声,shape: [num_proposals, 4]

        Returns:
            加噪后的数据,shape: [num_proposals, 4]
        """
        if noise is None:
            noise = torch.randn_like(x_start)  # 如果未提供噪声则生成

        x_start_shape = x_start.shape  # 获取形状

        # 提取对应时间步的参数
        sqrt_alphas_cumprod_t = extract(
            self.sqrt_alphas_cumprod, time, x_start_shape
        )  # shape: [1, 1, 1]
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, time, x_start_shape
        )  # shape: [1, 1, 1]

        # 执行前向扩散: x_t = sqrt(alpha_t_bar) * x_0 + sqrt(1 - alpha_t_bar) * noise
        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        rescale: bool = False,
    ) -> InstanceList:
        """执行前向传播和预测

        Args:
            x (tuple[Tensor]): 上游网络的多层特征
            batch_data_samples (List[:obj:[DetDataSample](file:///home/linkst/workplace/chromo/chromosome-kd/mmdet/structures/det_data_sample.py#L6-L232)]): 数据样本
            rescale (bool, optional): 是否重新缩放结果

        Returns:
            list[obj:`InstanceData`]: 每张图像的检测结果
        """
        # 可选：设置随机种子保持结果一致
        # seed = 0
        # random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        device = x[-1].device  # 获取设备

        # 提取图像元信息
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        # 准备测试目标（初始噪声框）
        (
            time_pairs,
            batch_noise_bboxes,
            batch_noise_bboxes_raw,
            batch_image_size,
        ) = self.prepare_testing_targets(batch_img_metas, device)

        # 执行预测
        predictions = self.predict_by_feat(
            x,
            time_pairs=time_pairs,
            batch_noise_bboxes=batch_noise_bboxes,
            batch_noise_bboxes_raw=batch_noise_bboxes_raw,
            batch_image_size=batch_image_size,
            device=device,
            batch_img_metas=batch_img_metas,
        )
        return predictions

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
        rescale=True,
    ):
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

        # Allow overriding sampling_timesteps from cfg
        sampling_timesteps = cfg.get(
            'sampling_timesteps', self.sampling_timesteps
        )
        cfg['sampling_timesteps'] = sampling_timesteps

        ensemble_score, ensemble_label, ensemble_coord = (
            [],
            [],
            [],
        )  # 集成预测结果

        # Adjust time pairs based on sampling_timesteps
        if sampling_timesteps != self.sampling_timesteps:
            # Re-calculate time pairs for this specific inference
            times = torch.linspace(
                -1, self.timesteps - 1, steps=sampling_timesteps + 1
            )
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))

        # 遍历时间对（反向扩散过程）
        for time, time_next in time_pairs:
            # 构造时间张量
            batch_time = torch.full(
                (batch_size,), time, device=device, dtype=torch.long
            )  # shape: [batch_size]
            # 前向传播
            pred_logits, pred_bboxes = self(x, batch_noise_bboxes, batch_time)

            x_start = pred_bboxes[-1]  # 预测的去噪结果

            # 转换为cx,cy,w,h格式并归一化
            x_start = (
                x_start / batch_image_size[:, None, :]
            )  # shape: [batch_size, num_proposals, 4]
            x_start = bbox_xyxy_to_cxcywh(x_start)  # 转换为cx,cy,w,h格式
            x_start = (
                x_start * 2 - 1.0
            ) * self.snr_scale  # 缩放到[-snr_scale, snr_scale]
            x_start = torch.clamp(
                x_start, min=-1 * self.snr_scale, max=self.snr_scale
            )  # 限制范围
            # 从去噪结果预测噪声
            pred_noise = self.predict_noise_from_start(
                batch_noise_bboxes_raw, batch_time, x_start
            )

            pred_noise_list, x_start_list = (
                [],
                [],
            )  # 每个图像的预测噪声和去噪结果
            noise_bboxes_list, num_remain_list = [], []  # 噪声框和保留数量
            if self.box_renewal:  # 如果使用框更新
                score_thr = cfg.get('score_thr', 0)  # 置信度阈值
                # 对批次中每个图像处理
                for img_id in range(batch_size):
                    score_per_image = pred_logits[-1][
                        img_id
                    ]  # 当前图像的分类得分

                    score_per_image = torch.sigmoid(
                        score_per_image
                    )  # sigmoid激活
                    value, _ = torch.max(
                        score_per_image, -1, keepdim=False
                    )  # 每个框的最大类别得分
                    keep_idx = value > score_thr  # 保留高置信度框

                    # Chromosome Optimization: Count Prior
                    # Ensure we keep at least a certain number of boxes (e.g. 60)
                    # to cover all 46 chromosomes (22 pairs + XY/XX)
                    MIN_KEEP = cfg.get('min_keep', 60)
                    if torch.sum(keep_idx) < MIN_KEEP:
                        k = min(MIN_KEEP, value.shape[0])
                        _, topk_indices = torch.topk(value, k)
                        keep_idx[topk_indices] = True

                    num_remain_list.append(torch.sum(keep_idx))  # 保留框数量
                    pred_noise_list.append(
                        pred_noise[img_id, keep_idx, :]
                    )  # 保留框对应的预测噪声
                    x_start_list.append(
                        x_start[img_id, keep_idx, :]
                    )  # 保留框对应的去噪结果
                    noise_bboxes_list.append(
                        batch_noise_bboxes[img_id, keep_idx, :]
                    )  # 保留的噪声框

            # 如果是最后一步
            if time_next < 0:
                # 不同于原始DiffusionDet
                if self.use_ensemble and sampling_timesteps > 1:
                    # 使用集成预测
                    box_pred_per_image, scores_per_image, labels_per_image = (
                        self.inference(
                            box_cls=pred_logits[-1],
                            box_pred=pred_bboxes[-1],
                            cfg=cfg,
                            device=device,
                        )
                    )
                    ensemble_score.append(scores_per_image)
                    ensemble_label.append(labels_per_image)
                    ensemble_coord.append(box_pred_per_image)
                continue

            # DDIM采样参数计算
            alpha = self.alphas_cumprod[time]  # 当前时间步alpha累积值
            alpha_next = self.alphas_cumprod[
                time_next
            ]  # 下一时间步alpha累积值

            sigma = (
                self.ddim_sampling_eta
                * (
                    (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
                ).sqrt()
            )  # sigma参数
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

                # DDIM采样步骤: x_{t-1} = sqrt(alpha_{t-1}) * x_0 +
                #                         sqrt(1 - alpha_{t-1} - sigma^2) * pred_noise +
                #                         sigma * noise
                noise_bboxes = (
                    x_start * alpha_next.sqrt()
                    + c * pred_noise
                    + sigma * noise
                )

                if self.box_renewal:  # 如果使用框更新
                    # 用随机框补充
                    if num_remain < self.num_proposals:
                        # 如果保留框少于建议框数量,用随机框填充
                        noise_bboxes = torch.cat(
                            (
                                noise_bboxes,
                                torch.randn(
                                    self.num_proposals - num_remain,
                                    4,
                                    device=device,
                                ),
                            ),
                            dim=0,
                        )  # shape: [num_proposals, 4]
                    else:
                        # 如果保留框多于建议框数量,随机选择
                        select_mask = [True] * self.num_proposals + [False] * (
                            num_remain - self.num_proposals
                        )
                        random.shuffle(select_mask)
                        noise_bboxes = noise_bboxes[select_mask]

                    # 保存原始噪声框
                    batch_noise_bboxes_raw_list.append(noise_bboxes)
                    # 处理噪声框: 转换为xyxy格式并缩放到图像尺寸
                    noise_bboxes = torch.clamp(
                        noise_bboxes,
                        min=-1 * self.snr_scale,
                        max=self.snr_scale,
                    )  # 限制范围
                    noise_bboxes = (
                        (noise_bboxes / self.snr_scale) + 1
                    ) / 2  # 转换到[0,1]
                    noise_bboxes = bbox_cxcywh_to_xyxy(
                        noise_bboxes
                    )  # 转换为xyxy格式
                    noise_bboxes = (
                        noise_bboxes * batch_image_size[idx]
                    )  # 缩放到绝对坐标

                batch_noise_bboxes_list.append(noise_bboxes)
            # 更新噪声框
            batch_noise_bboxes = torch.stack(batch_noise_bboxes_list)
            batch_noise_bboxes_raw = torch.stack(batch_noise_bboxes_raw_list)

            # 如果使用集成预测
            if self.use_ensemble and sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = (
                    self.inference(
                        box_cls=pred_logits[-1],
                        box_pred=pred_bboxes[-1],
                        cfg=cfg,
                        device=device,
                    )
                )
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        # 如果使用集成预测
        if self.use_ensemble and sampling_timesteps > 1:
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
                        box_pred_per_image,
                        scores_per_image,
                        labels_per_image,
                        cfg.nms,
                    )
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
                results_list, cfg, batch_img_metas=batch_img_metas
            )
        return results_list

    @staticmethod
    def do_results_post_process(results_list, cfg, batch_img_metas=None):
        """
        后处理预测结果

        Args:
            results_list: 预测结果列表
            cfg: 配置
            batch_img_metas: 批次图像元信息

        Returns:
            处理后的结果列表
        """
        processed_results = []
        # 对每个结果进行后处理
        for results, img_meta in zip(results_list, batch_img_metas):
            assert img_meta.get('scale_factor') is not None
            scale_factor = [
                1 / s for s in img_meta['scale_factor']
            ]  # 缩放因子
            results.bboxes = scale_boxes(
                results.bboxes, scale_factor
            )  # 缩放边界框
            # 裁剪边界框到图像范围内
            h, w = img_meta['ori_shape']
            results.bboxes[:, 0::2] = results.bboxes[:, 0::2].clamp(
                min=0, max=w
            )  # x坐标限制在[0,w]
            results.bboxes[:, 1::2] = results.bboxes[:, 1::2].clamp(
                min=0, max=h
            )  # y坐标限制在[0,h]

            # 过滤小尺寸边界框
            if cfg.get('min_bbox_size', 0) >= 0:
                w, h = get_box_wh(results.bboxes)  # 获取框的宽高
                valid_mask = (w > cfg.min_bbox_size) & (
                    h > cfg.min_bbox_size
                )  # 有效框掩码
                if not valid_mask.all():
                    results = results[valid_mask]  # 只保留有效框
            processed_results.append(results)

        return processed_results

    def prepare_testing_targets(self, batch_img_metas, device):
        """
        准备测试目标（初始噪声）

        Args:
            batch_img_metas: 批次图像元信息
            device: 设备

        Returns:
            (time_pairs, batch_noise_bboxes, batch_noise_bboxes_raw, batch_image_size)
        """
        # 生成时间步序列: [-1, 0, 1, 2, ..., T-1]
        times = torch.linspace(
            -1, self.timesteps - 1, steps=self.sampling_timesteps + 1
        )
        times = list(reversed(times.int().tolist()))
        # 生成时间对: [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        noise_bboxes_list = []  # 噪声框列表
        noise_bboxes_raw_list = []  # 原始噪声框列表
        image_size_list = []  # 图像尺寸列表
        # 对每张图像处理
        for img_meta in batch_img_metas:
            h, w = img_meta['img_shape']  # 图像尺寸
            # 图像尺寸张量
            image_size = torch.tensor(
                [w, h, w, h], dtype=torch.float32, device=device
            )  # shape: [4]
            # 生成随机噪声框
            noise_bboxes_raw = torch.randn(
                (self.num_proposals, 4), device=device
            )  # shape: [num_proposals, 4]
            # 处理噪声框
            noise_bboxes = torch.clamp(
                noise_bboxes_raw, min=-1 * self.snr_scale, max=self.snr_scale
            )  # 限制范围
            noise_bboxes = (
                (noise_bboxes / self.snr_scale) + 1
            ) / 2  # 转换到[0,1]
            noise_bboxes = bbox_cxcywh_to_xyxy(noise_bboxes)  # 转换为xyxy格式
            noise_bboxes = noise_bboxes * image_size

            noise_bboxes_raw_list.append(noise_bboxes_raw)
            noise_bboxes_list.append(noise_bboxes)
            image_size_list.append(image_size[None])  # 添加批次维度
        # 堆叠为批次张量
        batch_noise_bboxes = torch.stack(
            noise_bboxes_list
        )  # shape: [batch_size, num_proposals, 4]
        batch_image_size = torch.cat(image_size_list)  # shape: [batch_size, 4]
        batch_noise_bboxes_raw = torch.stack(
            noise_bboxes_raw_list
        )  # shape: [batch_size, num_proposals, 4]
        return (
            time_pairs,
            batch_noise_bboxes,
            batch_noise_bboxes_raw,
            batch_image_size,
        )

    def predict_noise_from_start(self, x_t, t, x0):
        """
        从去噪结果预测噪声

        Args:
            x_t: 加噪数据,shape: [batch_size, num_proposals, 4]
            t: 时间步,shape: [batch_size]
            x0: 去噪结果,shape: [batch_size, num_proposals, 4]

        Returns:
            预测的噪声,shape: [batch_size, num_proposals, 4]
        """
        # 根据公式: noise = (sqrt(1/alpha_t_bar) * x_t - x_0) / sqrt(1/alpha_t_bar - 1)
        results = (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return results

    def inference(self, box_cls, box_pred, cfg, device):
        """
        推理函数

        Args:
            box_cls (Tensor): 分类概率张量,shape: (batch_size, num_proposals, K)
            box_pred (Tensor): 边界框回归值张量,shape: (batch_size, num_proposals, 4)

        Returns:
            results (List[Instances]): 每张图像的检测结果列表
        """
        results = []

        sampling_timesteps = cfg.get(
            'sampling_timesteps', self.sampling_timesteps
        )

        # 如果使用focal loss或fed loss
        if self.use_focal_loss or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)  # sigmoid激活得到分类得分
            # 生成类别标签
            labels = (
                torch.arange(self.num_classes, device=device)
                .unsqueeze(0)
                .repeat(self.num_proposals, 1)
                .flatten(0, 1)
            )  # shape: [num_proposals * num_classes]
            box_pred_list = []  # 边界框预测列表
            scores_list = []  # 得分列表
            labels_list = []  # 标签列表
            # 对批次中每个图像处理
            for i, (scores_per_image, box_pred_per_image) in enumerate(
                zip(scores, box_pred)
            ):
                # 找到最高得分的预测
                scores_per_image, topk_indices = scores_per_image.flatten(
                    0, 1
                ).topk(
                    self.num_proposals, sorted=False
                )  # 选择前num_proposals个预测
                labels_per_image = labels[topk_indices]  # 对应的标签
                # 重复边界框预测以匹配类别
                box_pred_per_image = (
                    box_pred_per_image.view(-1, 1, 4)
                    .repeat(1, self.num_classes, 1)
                    .view(-1, 4)
                )  # shape: [num_proposals*num_classes, 4]
                box_pred_per_image = box_pred_per_image[
                    topk_indices
                ]  # 选择对应预测

                # 如果使用集成预测且采样步数大于1
                if self.use_ensemble and sampling_timesteps > 1:
                    box_pred_list.append(box_pred_per_image)
                    scores_list.append(scores_per_image)
                    labels_list.append(labels_per_image)
                    continue

                # 如果使用NMS
                if self.use_nms:
                    nms_cfg = cfg.nms
                    if cfg.get('use_soft_nms', False):
                        if nms_cfg.get('type') != 'soft_nms':
                            nms_cfg = dict(
                                type='soft_nms',
                                iou_threshold=0.5,
                                sigma=0.5,
                                min_score=0.001,
                                method='gaussian',
                            )
                            if 'iou_threshold' in cfg.nms:
                                nms_cfg['iou_threshold'] = cfg.nms[
                                    'iou_threshold'
                                ]

                    det_bboxes, keep_idxs = batched_nms(
                        box_pred_per_image,
                        scores_per_image,
                        labels_per_image,
                        nms_cfg,
                    )

                    # Count Constraint Recovery
                    min_num = cfg.get('min_num_bboxes', 0)
                    if min_num > 0 and len(keep_idxs) < min_num:
                        needed = min_num - len(keep_idxs)
                        mask = torch.ones(
                            scores_per_image.shape[0],
                            dtype=torch.bool,
                            device=device,
                        )
                        mask[keep_idxs] = False
                        remaining_scores = scores_per_image[mask]
                        remaining_indices = torch.nonzero(mask).squeeze(1)
                        if len(remaining_indices) > 0:
                            vals, sort_idx = remaining_scores.sort(
                                descending=True
                            )
                            to_add_indices = remaining_indices[
                                sort_idx[:needed]
                            ]
                            keep_idxs = torch.cat([keep_idxs, to_add_indices])
                            added_boxes = box_pred_per_image[to_add_indices]
                            added_scores = scores_per_image[
                                to_add_indices
                            ].unsqueeze(1)
                            added_dets = torch.cat(
                                [added_boxes, added_scores], dim=1
                            )
                            det_bboxes = torch.cat([det_bboxes, added_dets])

                    box_pred_per_image = box_pred_per_image[keep_idxs]
                    labels_per_image = labels_per_image[keep_idxs]
                    scores_per_image = det_bboxes[:, -1]
                # 创建结果对象
                result = InstanceData()
                result.bboxes = box_pred_per_image
                result.scores = scores_per_image
                result.labels = labels_per_image
                results.append(result)

        else:
            # 对于每个框分配最佳类别或第二最佳（如果最佳是`no_object`）
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(
                -1
            )  # softmax激活并获取最大得分和标签

            # 对批次中每个图像处理
            for i, (
                scores_per_image,
                labels_per_image,
                box_pred_per_image,
            ) in enumerate(zip(scores, labels, box_pred)):
                # 如果使用集成预测且采样步数大于1
                if self.use_ensemble and sampling_timesteps > 1:
                    return (
                        box_pred_per_image,
                        scores_per_image,
                        labels_per_image,
                    )

                # 如果使用NMS
                if self.use_nms:
                    nms_cfg = cfg.nms
                    if cfg.get('use_soft_nms', False):
                        if nms_cfg.get('type') != 'soft_nms':
                            nms_cfg = dict(
                                type='soft_nms',
                                iou_threshold=0.5,
                                sigma=0.5,
                                min_score=0.001,
                                method='gaussian',
                            )
                            if 'iou_threshold' in cfg.nms:
                                nms_cfg['iou_threshold'] = cfg.nms[
                                    'iou_threshold'
                                ]

                    det_bboxes, keep_idxs = batched_nms(
                        box_pred_per_image,
                        scores_per_image,
                        labels_per_image,
                        nms_cfg,
                    )

                    # Count Constraint Recovery
                    min_num = cfg.get('min_num_bboxes', 0)
                    if min_num > 0 and len(keep_idxs) < min_num:
                        needed = min_num - len(keep_idxs)
                        mask = torch.ones(
                            scores_per_image.shape[0],
                            dtype=torch.bool,
                            device=device,
                        )
                        mask[keep_idxs] = False
                        remaining_scores = scores_per_image[mask]
                        remaining_indices = torch.nonzero(mask).squeeze(1)
                        if len(remaining_indices) > 0:
                            vals, sort_idx = remaining_scores.sort(
                                descending=True
                            )
                            to_add_indices = remaining_indices[
                                sort_idx[:needed]
                            ]
                            keep_idxs = torch.cat([keep_idxs, to_add_indices])
                            added_boxes = box_pred_per_image[to_add_indices]
                            added_scores = scores_per_image[
                                to_add_indices
                            ].unsqueeze(1)
                            added_dets = torch.cat(
                                [added_boxes, added_scores], dim=1
                            )
                            det_bboxes = torch.cat([det_bboxes, added_dets])

                    box_pred_per_image = box_pred_per_image[keep_idxs]
                    labels_per_image = labels_per_image[keep_idxs]
                    scores_per_image = det_bboxes[:, -1]

                # 创建结果对象
                result = InstanceData()
                result.bboxes = box_pred_per_image
                result.scores = scores_per_image
                result.labels = labels_per_image
                results.append(result)
        # 如果使用集成预测且采样步数大于1
        if self.use_ensemble and sampling_timesteps > 1:
            return box_pred_list, scores_list, labels_list
        else:
            return results


@MODELS.register_module()
class SingleDiffusionDetHead(nn.Module):
    """单个DiffusionDet检测头"""

    def __init__(
        self,
        num_classes=80,  # 类别数
        feat_channels=256,  # 特征通道数
        dim_feedforward=2048,  # 前馈网络维度
        num_cls_convs=1,  # 分类卷积层数
        num_reg_convs=3,  # 回归卷积层数
        num_heads=8,  # 注意力头数
        dropout=0.0,  # dropout率
        pooler_resolution=7,  # 池化分辨率
        scale_clamp=_DEFAULT_SCALE_CLAMP,  # 缩放截断值
        bbox_weights=(2.0, 2.0, 1.0, 1.0),  # 边界框权重
        use_focal_loss=True,  # 是否使用focal loss
        use_fed_loss=False,  # 是否使用fed loss
        act_cfg=dict(type='ReLU', inplace=True),  # 激活函数配置
        dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),  # 动态卷积配置
    ) -> None:
        super().__init__()
        self.feat_channels = feat_channels  # 特征通道数

        # 动态模块
        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(
            feat_channels, num_heads, dropout=dropout
        )
        # 实例交互模块（动态卷积）
        self.inst_interact = DynamicConv(
            feat_channels=feat_channels,
            pooler_resolution=pooler_resolution,
            dynamic_dim=dynamic_conv['dynamic_dim'],
            dynamic_num=dynamic_conv['dynamic_num'],
        )

        # 前馈网络
        self.linear1 = nn.Linear(feat_channels, dim_feedforward)  # 第一线性层
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.linear2 = nn.Linear(dim_feedforward, feat_channels)  # 第二线性层

        # LayerNorm层
        self.norm1 = nn.LayerNorm(feat_channels)  # 自注意力后归一化
        self.norm2 = nn.LayerNorm(feat_channels)  # 实例交互后归一化
        self.norm3 = nn.LayerNorm(feat_channels)  # 前馈网络后归一化
        self.dropout1 = nn.Dropout(dropout)  # 自注意力dropout
        self.dropout2 = nn.Dropout(dropout)  # 实例交互dropout
        self.dropout3 = nn.Dropout(dropout)  # 前馈网络dropout

        # 激活函数
        self.activation = build_activation_layer(act_cfg)

        # 时间嵌入块
        self.block_time_mlp = nn.Sequential(
            nn.SiLU(),  # SiLU激活
            nn.Linear(feat_channels * 4, feat_channels * 2),
        )  # 线性变换

        # 分类模块
        cls_module = list()
        for _ in range(num_cls_convs):
            cls_module.append(
                nn.Linear(feat_channels, feat_channels, False)
            )  # 线性层（无偏置）
            cls_module.append(nn.LayerNorm(feat_channels))  # LayerNorm
            cls_module.append(nn.ReLU(inplace=True))  # ReLU激活
        self.cls_module = nn.ModuleList(cls_module)  # 分类模块

        # 回归模块
        reg_module = list()
        for _ in range(num_reg_convs):
            reg_module.append(
                nn.Linear(feat_channels, feat_channels, False)
            )  # 线性层（无偏置）
            reg_module.append(nn.LayerNorm(feat_channels))  # LayerNorm
            reg_module.append(nn.ReLU(inplace=True))  # ReLU激活
        self.reg_module = nn.ModuleList(reg_module)  # 回归模块

        # 预测层
        self.use_focal_loss = use_focal_loss  # 是否使用focal loss
        self.use_fed_loss = use_fed_loss  # 是否使用fed loss
        if self.use_focal_loss or self.use_fed_loss:
            # 如果使用focal loss或fed loss,输出num_classes维
            self.class_logits = nn.Linear(feat_channels, num_classes)
        else:
            # 否则输出num_classes+1维（包括背景类）
            self.class_logits = nn.Linear(feat_channels, num_classes + 1)
        self.bboxes_delta = nn.Linear(feat_channels, 4)  # 边界框回归层
        self.scale_clamp = scale_clamp  # 缩放截断值
        self.bbox_weights = bbox_weights  # 边界框权重

    def forward(self, features, bboxes, pro_features, pooler, time_emb):
        """
        前向传播

        Args:
            features: 特征金字塔,列表形式
            bboxes: 边界框,shape: (N, num_boxes, 4)
            pro_features: 提案特征,shape: (N, num_boxes, feat_channels)
            pooler: ROI池化器
            time_emb: 时间嵌入,shape: (N, time_dim)

        Returns:
            class_logits: 分类logits,shape: (N, num_boxes, num_classes)
            pred_bboxes: 预测边界框,shape: (N, num_boxes, 4)
            obj_features: 对象特征,shape: (1, N*num_boxes, feat_channels)
        """
        N, num_boxes = bboxes.shape[:2]  # 获取批次大小和框数量

        # ROI特征提取
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(bboxes[b])  # 收集每张图像的边界框
        rois = bbox2roi(proposal_boxes)  # 转换为ROI格式

        roi_features = pooler(features, rois)  # ROI池化提取特征

        # 处理提案特征
        if pro_features is None:
            # 如果没有提供提案特征,则从ROI特征计算均值
            pro_features = roi_features.view(
                N, num_boxes, self.feat_channels, -1
            ).mean(-1)  # shape: (N, num_boxes, feat_channels)

        # 调整ROI特征形状以适应注意力机制
        roi_features = roi_features.view(
            N * num_boxes, self.feat_channels, -1
        ).permute(2, 0, 1)  # shape: (49, N*num_boxes, feat_channels)

        # 自注意力
        pro_features = pro_features.view(
            N, num_boxes, self.feat_channels
        ).permute(1, 0, 2)  # shape: (num_boxes, N, feat_channels)
        # 执行自注意力: Query=Key=Value=pro_features
        pro_features2 = self.self_attn(
            pro_features, pro_features, value=pro_features
        )[0]  # shape: (num_boxes, N, feat_channels)
        # 残差连接和归一化
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # 实例交互
        # 调整形状以进行实例交互
        pro_features = (
            pro_features.view(num_boxes, N, self.feat_channels)
            .permute(1, 0, 2)
            .reshape(1, N * num_boxes, self.feat_channels)
        )  # shape: (1, N*num_boxes, feat_channels)
        # 动态卷积交互
        pro_features2 = self.inst_interact(pro_features, roi_features)
        # 残差连接和归一化
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(
            pro_features
        )  # shape: (1, N*num_boxes, feat_channels)

        # 对象特征处理（前馈网络）
        obj_features2 = self.linear2(
            self.dropout(self.activation(self.linear1(obj_features)))
        )  # 前馈网络
        obj_features = obj_features + self.dropout3(obj_features2)  # 残差连接
        obj_features = self.norm3(obj_features)  # 归一化

        # 时间嵌入条件化
        fc_feature = obj_features.transpose(0, 1).reshape(
            N * num_boxes, -1
        )  # shape: (N*num_boxes, feat_channels)

        scale_shift = self.block_time_mlp(time_emb)  # 时间嵌入MLP
        scale_shift = torch.repeat_interleave(
            scale_shift, num_boxes, dim=0
        )  # 扩展到每个框
        scale, shift = scale_shift.chunk(2, dim=1)  # 分割为缩放和偏移
        # 时间条件化: feature = feature * (scale + 1) + shift
        fc_feature = fc_feature * (scale + 1) + shift

        # 分类和回归分支
        cls_feature = fc_feature.clone()  # 分类特征
        reg_feature = fc_feature.clone()  # 回归特征
        # 通过分类模块
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        # 通过回归模块
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        # 预测分类logits和边界框偏移
        class_logits = self.class_logits(
            cls_feature
        )  # shape: (N*num_boxes, num_classes)
        bboxes_deltas = self.bboxes_delta(
            reg_feature
        )  # shape: (N*num_boxes, 4)
        pred_bboxes = self.apply_deltas(
            bboxes_deltas, bboxes.view(-1, 4)
        )  # 应用偏移到边界框

        # 调整输出形状
        return (
            class_logits.view(N, num_boxes, -1),
            pred_bboxes.view(N, num_boxes, -1),
            obj_features,
        )

    def apply_deltas(self, deltas, boxes):
        """将变换`deltas` (dx, dy, dw, dh) 应用到`boxes`

        Args:
            deltas (Tensor): 变换偏移,shape: (N, k*4), k >= 1
            boxes (Tensor): 要变换的框,shape: (N, 4)

        Returns:
            pred_boxes: 变换后的框,shape: (N, k*4)
        """
        boxes = boxes.to(deltas.dtype)  # 确保数据类型一致

        # 计算框的宽度和高度
        widths = boxes[:, 2] - boxes[:, 0]  # x2 - x1
        heights = boxes[:, 3] - boxes[:, 1]  # y2 - y1
        ctr_x = boxes[:, 0] + 0.5 * widths  # 中心x坐标
        ctr_y = boxes[:, 1] + 0.5 * heights  # 中心y坐标

        wx, wy, ww, wh = self.bbox_weights  # 边界框权重
        dx = deltas[:, 0::4] / wx  # x方向偏移
        dy = deltas[:, 1::4] / wy  # y方向偏移
        dw = deltas[:, 2::4] / ww  # 宽度偏移
        dh = deltas[:, 3::4] / wh  # 高度偏移

        # 防止将过大值送入torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)  # 截断宽度偏移
        dh = torch.clamp(dh, max=self.scale_clamp)  # 截断高度偏移

        # 计算预测的中心坐标和宽高
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]  # 预测中心x
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]  # 预测中心y
        pred_w = torch.exp(dw) * widths[:, None]  # 预测宽度
        pred_h = torch.exp(dh) * heights[:, None]  # 预测高度

        # 构造预测框: (x1, y1, x2, y2)
        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):
    """动态卷积模块"""

    def __init__(
        self,
        feat_channels: int,  # 特征通道数
        dynamic_dim: int = 64,  # 动态维度
        dynamic_num: int = 2,  # 动态层数
        pooler_resolution: int = 7,
    ) -> None:  # 池化分辨率
        super().__init__()

        self.feat_channels = feat_channels  # 特征通道数
        self.dynamic_dim = dynamic_dim  # 动态维度
        self.dynamic_num = dynamic_num  # 动态层数
        self.num_params = self.feat_channels * self.dynamic_dim  # 参数数量
        # 动态层: 生成动态卷积参数
        self.dynamic_layer = nn.Linear(
            self.feat_channels, self.dynamic_num * self.num_params
        )

        # LayerNorm层
        self.norm1 = nn.LayerNorm(self.dynamic_dim)  # 第一归一化层
        self.norm2 = nn.LayerNorm(self.feat_channels)  # 第二归一化层

        # 激活函数
        self.activation = nn.ReLU(inplace=True)

        # 输出层
        num_output = self.feat_channels * pooler_resolution**2  # 输出维度
        self.out_layer = nn.Linear(
            num_output, self.feat_channels
        )  # 输出线性层
        self.norm3 = nn.LayerNorm(self.feat_channels)  # 输出归一化层

    def forward(self, pro_features: Tensor, roi_features: Tensor) -> Tensor:
        """前向传播

        Args:
            pro_features: 提案特征,shape: (1, N * num_boxes, self.feat_channels)
            roi_features: ROI特征,shape: (49, N * num_boxes, self.feat_channels)

        Returns:
            features: 处理后的特征,shape: (1, N * num_boxes, self.feat_channels)
        """
        # 调整特征形状
        features = roi_features.permute(
            1, 0, 2
        )  # shape: (N*num_boxes, 49, feat_channels)
        parameters = self.dynamic_layer(pro_features).permute(
            1, 0, 2
        )  # shape: (N*num_boxes, 1, dynamic_num*num_params)

        # 分割参数为两个卷积层的参数
        param1 = parameters[:, :, : self.num_params].view(
            -1, self.feat_channels, self.dynamic_dim
        )  # shape: (N*num_boxes, feat_channels, dynamic_dim)
        param2 = parameters[:, :, self.num_params :].view(
            -1, self.dynamic_dim, self.feat_channels
        )  # shape: (N*num_boxes, dynamic_dim, feat_channels)

        # 执行动态卷积: 第一层
        features = torch.bmm(
            features, param1
        )  # shape: (N*num_boxes, feat_channels, dynamic_dim)
        features = self.norm1(features)  # 归一化
        features = self.activation(features)  # 激活

        # 执行动态卷积: 第二层
        features = torch.bmm(
            features, param2
        )  # shape: (N*num_boxes, feat_channels, feat_channels)
        features = self.norm2(features)  # 归一化
        features = self.activation(features)  # 激活

        # 展平并通过输出层
        features = features.flatten(
            1
        )  # 展平,shape: (N*num_boxes, dynamic_dim*feat_channels)
        features = self.out_layer(features)  # 输出层
        features = self.norm3(features)  # 归一化
        features = self.activation(features)  # 激活

        return features
