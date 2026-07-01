"""ChromoGenVAE — 包装 diffusers AutoencoderKL 用于 image → latent 编码

方向六 Phase 1: 让 VAE 可通过 mmengine MODELS.build 加载。

ChromoGen 使用 SD1.5 预训练 KL-f8 VAE, 将图像 (B,3,H,W) 编码为潜变量 (B,4,H/8,W/8)。
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL

from mmdet.registry import MODELS


@MODELS.register_module()
class ChromoGenVAE(nn.Module):
    """ChromoGen VAE 编码器 wrapper

    Args:
        vae_model: diffusers VAE 模型路径或名称
        vae_subfolder: VAE 子目录 (None 表示直接路径)
        scaling_factor: 潜变量缩放因子 (SD VAE 默认 0.18215)
        freeze: 是否冻结参数 (默认 True)
        input_mean: 输入图像的归一化均值 (若提供, 先反归一化到 [0,255] 再缩放到 [-1,1])
        input_std: 输入图像的归一化标准差
    """

    def __init__(
        self,
        vae_model: str = 'stabilityai/sd-vae-ft-mse',
        vae_subfolder: str = None,
        scaling_factor: float = 0.18215,
        freeze: bool = True,
        input_mean=None,
        input_std=None,
    ):
        super().__init__()

        # 加载 VAE
        if vae_subfolder is not None:
            self.vae = AutoencoderKL.from_pretrained(
                vae_model, subfolder=vae_subfolder
            )
        else:
            self.vae = AutoencoderKL.from_pretrained(vae_model)

        self.scaling_factor = scaling_factor
        self.input_mean = input_mean
        self.input_std = input_std

        if freeze:
            self.set_frozen()
        self.eval()

    @classmethod
    def from_chromogen_checkpoint(
        cls,
        checkpoint: str,
        vae_model: str = 'stabilityai/sd-vae-ft-mse',
        vae_subfolder: str = None,
        scaling_factor: float = 0.18215,
        freeze: bool = True,
    ) -> 'ChromoGenVAE':
        """从 ChromoGen pipeline checkpoint 加载 VAE 权重

        Args:
            checkpoint: ChromoGen pipeline checkpoint 路径
            vae_model: VAE 基础模型路径 (用于加载结构)
            scaling_factor: 潜变量缩放因子

        Returns:
            ChromoGenVAE 实例
        """
        instance = cls(
            vae_model=vae_model,
            vae_subfolder=vae_subfolder,
            scaling_factor=scaling_factor,
            freeze=freeze,
        )

        # 加载 ChromoGen checkpoint 中的 VAE 权重
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', {}))

        # 提取 VAE 权重 (key 以 'vae.' 开头)
        vae_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('vae.'):
                vae_state_dict[key[4:]] = value  # 去除 'vae.' 前缀

        if vae_state_dict:
            missing, unexpected = instance.vae.load_state_dict(
                vae_state_dict, strict=False
            )
            if missing:
                print(f'[ChromoGenVAE] Missing keys: {len(missing)}')
            if unexpected:
                print(f'[ChromoGenVAE] Unexpected keys: {len(unexpected)}')

        return instance

    def set_frozen(self):
        """冻结参数并设为 eval 模式"""
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像为潜变量

        Args:
            images: (B, 3, H, W) 图像
                    - 若 input_mean/input_std 提供: ImageNet 归一化后的图像
                    - 否则: [-1, 1] 范围图像

        Returns:
            latent: (B, 4, H/8, W/8) 缩放后的潜变量
        """
        # 若提供 input_mean/input_std, 先反归一化到 [0, 255] 再缩放到 [-1, 1]
        if self.input_mean is not None and self.input_std is not None:
            mean = torch.tensor(
                self.input_mean, device=images.device, dtype=images.dtype
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                self.input_std, device=images.device, dtype=images.dtype
            ).view(1, 3, 1, 1)
            images = images * std + mean  # [0, 255]
            images = images / 127.5 - 1.0  # [-1, 1]

        latent = self.vae.encode(images).latent_dist.sample()
        latent = latent * self.scaling_factor
        return latent

    def train(self, mode: bool = True):
        """重写 train 方法, 保持 VAE 冻结"""
        super().train(mode)
        # VAE 始终保持 eval 模式
        self.vae.eval()
        return self
