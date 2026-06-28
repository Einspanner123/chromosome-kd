"""ChromoGenUNet — 包装 diffusers UNet2DConditionModel 用于 mmengine 配置构建

方向六 Phase 1: 让 ChromoGen UNet 可通过 MODELS.build 加载,
供 ChromoGenFeatureExtractor 使用。

ChromoGenFeatureExtractor 期望 unet 拥有 down_blocks/mid_block 属性
(diffusers UNet2DConditionModel 的结构), 本 wrapper 透传这些属性。
"""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

from mmdet.registry import MODELS


@MODELS.register_module()
class ChromoGenUNet(nn.Module):
    """ChromoGen UNet wrapper (透传 diffusers UNet2DConditionModel)

    Args:
        sample_size: 潜空间尺寸 (默认 96, 即 768/8)
        in_channels: 输入通道 (默认 4, VAE latent)
        out_channels: 输出通道 (默认 4)
        block_out_channels: 各 block 通道数
        attention_head_dim: attention head 维度
        cross_attention_dim: cross attention 维度 (条件 embedding)
        layers_per_block: 每 block 层数
    """

    def __init__(
        self,
        sample_size: int = 96,
        in_channels: int = 4,
        out_channels: int = 4,
        block_out_channels=(320, 640, 1280, 1280),
        attention_head_dim: int = 8,
        cross_attention_dim: int = 768,
        layers_per_block: int = 2,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=(
                'CrossAttnDownBlock2D',
                'CrossAttnDownBlock2D',
                'CrossAttnDownBlock2D',
                'DownBlock2D',
            ),
            up_block_types=(
                'UpBlock2D',
                'CrossAttnUpBlock2D',
                'CrossAttnUpBlock2D',
                'CrossAttnUpBlock2D',
            ),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
        )

        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

    # 透传 diffusers UNet 的关键属性
    @property
    def down_blocks(self):
        return self.unet.down_blocks

    @property
    def mid_block(self):
        return self.unet.mid_block

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
    ):
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=return_dict,
        )
