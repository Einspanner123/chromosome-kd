"""UNet包装器

基于diffusers的UNet2DConditionModel，提供bottleneck特征提取接口，
用于BBox扩散头的特征共享。
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

logger = logging.getLogger(__name__)


class ChromoUNet(nn.Module):
    """染色体图像生成UNet

    包装diffusers的UNet2DConditionModel，额外暴露bottleneck特征供BBox头使用。

    特点：
      - 使用cross-attention接收条件embedding
      - forward返回去噪预测 + bottleneck特征
      - 支持梯度检查点节省显存
      - 使用持久hook避免每次forward注册/删除的开销
      - 支持加载SD-1.5等预训练UNet权重做fine-tune初始化
    """

    def __init__(
        self,
        sample_size: int = 96,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str, ...] = (
            'CrossAttnDownBlock2D',
            'CrossAttnDownBlock2D',
            'CrossAttnDownBlock2D',
            'DownBlock2D',
        ),
        up_block_types: Tuple[str, ...] = (
            'UpBlock2D',
            'CrossAttnUpBlock2D',
            'CrossAttnUpBlock2D',
            'CrossAttnUpBlock2D',
        ),
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        attention_head_dim: int = 8,
        cross_attention_dim: int = 768,
        gradient_checkpointing: bool = False,
        pretrained_model: Optional[str] = None,
    ):
        super().__init__()

        if pretrained_model is not None:
            # fine-tune 路线：从预训练模型加载UNet权重
            # 架构与SD-1.5 UNet同构，可无损加载
            logger.info(
                f'Loading pretrained UNet from {pretrained_model}/unet'
            )
            self.unet = UNet2DConditionModel.from_pretrained(
                pretrained_model, subfolder='unet'
            )
        else:
            # from-scratch 路线：随机初始化
            self.unet = UNet2DConditionModel(
                sample_size=sample_size,
                in_channels=in_channels,
                out_channels=out_channels,
                down_block_types=down_block_types,
                up_block_types=up_block_types,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
            )

        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # bottleneck通道数 = 最后一个block_out_channels
        self.bottleneck_channels = block_out_channels[-1]

        # 持久hook：在__init__中注册一次，避免每次forward重复注册/删除
        self._bottleneck_feat = None
        self._hook_handle = self.unet.mid_block.register_forward_hook(
            self._bottleneck_hook
        )

    def _bottleneck_hook(self, module, input, output):
        """持久hook：捕获mid_block输出作为bottleneck特征"""
        self._bottleneck_feat = output

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            sample: 潜空间噪声 [B, 4, H, W]
            timestep: 时间步 [B]
            encoder_hidden_states: 条件embedding [B, seq_len, cross_attention_dim]
            return_dict: 是否返回dict
        Returns:
            dict with:
              - 'sample': 去噪预测 [B, 4, H, W]
              - 'bottleneck_feat': bottleneck特征 [B, C, H/8, W/8]
        """
        # UNet前向传播，hook自动捕获bottleneck特征
        out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True,
        )

        result = {
            'sample': out.sample,
            'bottleneck_feat': self._bottleneck_feat,
        }

        return result

    def num_parameters(self, only_trainable: bool = False) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if not only_trainable or p.requires_grad
        )
