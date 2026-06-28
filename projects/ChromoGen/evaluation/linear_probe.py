"""Phase 0 探针实验: Linear Probe

验证 ChromoGen 特征是否包含检测有用的信息。
在冻结特征上训练线性分类器, 预测 bbox 类别。

实验设计:
  1. 加载 ChromoGen 模型 (VAE + UNet, 冻结)
  2. 对训练集图像提取多尺度特征 (VAE latent, UNet down/mid)
  3. 对每个 bbox, 用 ROI Align 提取对应位置的特征
  4. 训练线性分类器预测类别
  5. 在验证集上评估准确率

预期结果:
  - 如果 ChromoGen 特征包含检测信息, Linear Probe 准确率应显著高于随机 (1/24 ≈ 4.2%)
  - 对比不同层特征: down1 (浅层) vs mid (深层) 的信息量差异
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from projects.ChromoGen.models.chromogen_pipeline import ChromoGenPipeline


class FeatureExtractor:
    """从 ChromoGen 模型提取多尺度特征

    提取层:
      - vae_latent: VAE 编码后的潜变量 (B, 4, H/8, W/8)
      - down1: UNet down_block 0 输出 (B, 320, H/8, W/8)
      - down2: UNet down_block 1 输出 (B, 640, H/16, W/16)
      - down3: UNet down_block 2 输出 (B, 1280, H/32, W/32)
      - mid: UNet mid_block 输出 (B, 1280, H/64, W/64)
    """

    def __init__(
        self,
        checkpoint: str,
        vae_path: str = None,
        device: str = 'cuda',
    ):
        self.device = torch.device(device)
        self._load_model(checkpoint, vae_path)
        self._register_hooks()
        self._feature_buffer = {}

    def _load_model(self, checkpoint: str, vae_path: str = None):
        """加载 ChromoGen 模型"""
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        cfg = ckpt.get('config', {})
        vae_model = vae_path or cfg.get('vae_model', 'stabilityai/sd-vae-ft-mse')

        self.model = ChromoGenPipeline(
            vae_model=vae_model,
            sample_size=cfg.get('sample_size', 96),
            unet_block_out_channels=cfg.get(
                'unet_block_out_channels', (320, 640, 1280, 1280)
            ),
            unet_attention_head_dim=cfg.get('unet_attention_head_dim', 8),
            cross_attention_dim=cfg.get('cross_attention_dim', 768),
            gradient_checkpointing=False,
            condition_embed_dim=cfg.get('condition_embed_dim', 768),
            condition_max_count=cfg.get('condition_max_count', 50),
            condition_dropout=0.0,
            enable_bbox_head=False,
            num_train_timesteps=cfg.get('num_train_timesteps', 1000),
            noise_schedule=cfg.get('noise_schedule', 'linear'),
            prediction_type=cfg.get('prediction_type', 'epsilon'),
            lambda_img=cfg.get('lambda_img', 1.0),
            lambda_bbox=cfg.get('lambda_bbox', 0.5),
            lambda_cls=cfg.get('lambda_cls', 0.5),
            cfg_dropout=0.0,
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

    def _register_hooks(self):
        """注册 forward hook 收集 UNet 中间特征"""
        unet = self.model.unet.unet  # diffusers UNet2DConditionModel

        # down_blocks hook
        for i, block in enumerate(unet.down_blocks):
            block.register_forward_hook(self._make_hook(f'down{i + 1}'))

        # mid_block hook
        unet.mid_block.register_forward_hook(self._make_hook('mid'))

    def _make_hook(self, name: str):
        """创建 forward hook"""

        def hook(module, input, output):
            if isinstance(output, tuple):
                feat = output[0]
            else:
                feat = output
            self._feature_buffer[name] = feat

        return hook

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> dict:
        """提取多尺度特征

        Args:
            images: (B, 3, H, W) 归一化图像

        Returns:
            feats: dict with 'vae_latent', 'down1', 'down2', 'down3', 'mid'
        """
        images = images.to(self.device)
        self._feature_buffer.clear()

        # 1. VAE 编码
        latents = self.model.vae.encode(images).latent_dist.sample()
        latents = latents * self.model.vae.config.scaling_factor

        # 2. UNet 前向 (使用固定 timestep=500 和零条件)
        B = images.shape[0]
        timestep = torch.full((B,), 500, device=self.device, dtype=torch.long)
        # 零条件 (无条件提取)
        condition = torch.zeros(
            B, 25, 768, device=self.device
        )  # NUM_CLASSES + 1 = 25

        _ = self.model.unet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=condition,
        )

        # 3. 收集特征
        feats = {
            'vae_latent': latents.cpu(),
        }
        for key in ['down1', 'down2', 'down3', 'mid']:
            if key in self._feature_buffer:
                feats[key] = self._feature_buffer[key].cpu()

        return feats


class RoIFeatureCollector:
    """从特征图提取 ROI 特征 (使用 ROI Align)"""

    def __init__(self, output_size: int = 7):
        self.output_size = output_size

    def __call__(
        self,
        feats: torch.Tensor,
        bboxes: torch.Tensor,
        image_size: tuple,
    ) -> torch.Tensor:
        """提取 ROI 特征

        Args:
            feats: (B, C, H, W) 特征图
            bboxes: (N, 4) 归一化 bbox [x1, y1, x2, y2] (0-1)
            image_size: (H, W) 原始图像尺寸

        Returns:
            roi_feats: (N, C, output_size, output_size)
        """
        # 将归一化坐标转为像素坐标
        img_h, img_w = image_size
        bboxes_pixel = bboxes.clone()
        bboxes_pixel[:, 0] *= img_w  # x1
        bboxes_pixel[:, 1] *= img_h  # y1
        bboxes_pixel[:, 2] *= img_w  # x2
        bboxes_pixel[:, 3] *= img_h  # y2

        # roi_align 需要 batch_idx: (N, 5) [batch_idx, x1, y1, x2, y2]
        # 假设所有 bbox 来自 batch 0
        n = bboxes.shape[0]
        rois = torch.zeros(n, 5, device=feats.device, dtype=feats.dtype)
        rois[:, 1:] = bboxes_pixel

        # ROI Align
        roi_feats = roi_align(
            input=feats,
            boxes=rois,
            output_size=(self.output_size, self.output_size),
            spatial_scale=1.0,  # 特征图已经是目标尺度
            aligned=True,
        )

        return roi_feats


class LinearProbe(nn.Module):
    """线性探针分类器

    在冻结特征上训练线性分类器, 评估特征的检测信息量。
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 24,
        roi_size: int = 7,
    ):
        super().__init__()
        self.roi_size = roi_size
        self.classifier = nn.Linear(in_channels * roi_size * roi_size, num_classes)

    def forward(self, roi_feats: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            roi_feats: (N, C, roi_size, roi_size) ROI 特征

        Returns:
            logits: (N, num_classes) 类别 logits
        """
        N = roi_feats.shape[0]
        flattened = roi_feats.view(N, -1)
        logits = self.classifier(flattened)
        return logits

    @torch.no_grad()
    def predict(self, roi_feats: torch.Tensor) -> torch.Tensor:
        """预测类别"""
        logits = self.forward(roi_feats)
        return logits.argmax(dim=1)
