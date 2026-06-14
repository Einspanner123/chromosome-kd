"""ChromoGen训练与推理Pipeline

核心模块，整合UNet、条件编码器、BBox头，提供训练和推理接口。
通过 enable_bbox_head 配置控制BBox头是否参与训练。
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler

from .bbox_head import BBoxDiffusionHead
from .condition_encoder import NUM_CLASSES, ChromoConditionEncoder
from .unet import ChromoUNet


class ChromoGenPipeline(nn.Module):
    """ChromoGen多任务扩散生成Pipeline

    组件：
      1. VAE (frozen): SD1.5预训练KL-f8，图像↔潜空间
      2. Condition Encoder: 染色体类别+数量 → 条件embedding
      3. UNet: 潜空间去噪网络
      4. BBox Head (可选): 并行bbox扩散生成头

    配置项：
      - enable_bbox_head: bool, 是否启用BBox头参与训练
      - lambda_img: float, 图像扩散损失权重
      - lambda_bbox: float, bbox损失权重
      - lambda_cls: float, 分类损失权重
    """

    def __init__(
        self,
        # VAE
        vae_model: str = 'stabilityai/sd-vae-ft-mse',
        vae_subfolder: str = None,
        # UNet
        sample_size: int = 96,
        unet_block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        unet_attention_head_dim: int = 8,
        cross_attention_dim: int = 768,
        gradient_checkpointing: bool = False,
        # Condition Encoder
        condition_embed_dim: int = 768,
        condition_max_count: int = 50,
        condition_dropout: float = 0.1,
        # BBox Head
        enable_bbox_head: bool = True,
        bbox_feat_channels: int = 512,
        bbox_num_proposals: int = 100,
        bbox_num_heads: int = 8,
        bbox_num_layers: int = 3,
        bbox_snr_scale: float = 2.0,
        # Noise Scheduler
        num_train_timesteps: int = 1000,
        noise_schedule: str = 'linear',
        prediction_type: str = 'epsilon',
        # Loss weights
        lambda_img: float = 1.0,
        lambda_bbox: float = 0.5,
        lambda_cls: float = 0.5,
        # Classifier-free guidance
        cfg_dropout: float = 0.1,
    ):
        super().__init__()

        self.enable_bbox_head = enable_bbox_head
        self.lambda_img = lambda_img
        self.lambda_bbox = lambda_bbox
        self.lambda_cls = lambda_cls
        self.cfg_dropout = cfg_dropout
        self.prediction_type = prediction_type

        # 1. VAE (frozen)
        self.vae = AutoencoderKL.from_pretrained(
            vae_model,
            subfolder=vae_subfolder,
        )
        self.vae.requires_grad_(False)
        self.vae_scale_factor = 2 ** (
            len(self.vae.config.block_out_channels) - 1
        )

        # 2. Condition Encoder
        self.condition_encoder = ChromoConditionEncoder(
            num_classes=NUM_CLASSES,
            embed_dim=condition_embed_dim,
            max_count=condition_max_count,
            dropout=condition_dropout,
        )

        # 3. UNet
        self.unet = ChromoUNet(
            sample_size=sample_size,
            in_channels=4,
            out_channels=4,
            block_out_channels=unet_block_out_channels,
            attention_head_dim=unet_attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            gradient_checkpointing=gradient_checkpointing,
        )

        # 4. BBox Head (可选)
        if enable_bbox_head:
            self.bbox_head = BBoxDiffusionHead(
                in_channels=self.unet.bottleneck_channels,
                feat_channels=bbox_feat_channels,
                num_classes=NUM_CLASSES,
                num_proposals=bbox_num_proposals,
                num_heads=bbox_num_heads,
                num_layers=bbox_num_layers,
                snr_scale=bbox_snr_scale,
            )
        else:
            self.bbox_head = None

        # 5. Noise Scheduler (训练用)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=noise_schedule,
            prediction_type=prediction_type,
        )

        # 6. 采样Scheduler (推理用，后续可替换)
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=noise_schedule,
            prediction_type=prediction_type,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        class_labels: torch.Tensor,
        counts: torch.Tensor,
        gt_bboxes: Optional[List[torch.Tensor]] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """训练前向传播

        Args:
            pixel_values: [B, 3, H, W] 原始图像
            class_labels: [B, 24] 类别索引
            counts: [B, 24] 各类数量
            gt_bboxes: list of [N_i, 4] (cxcywh归一化)，仅enable_bbox_head时需要
            gt_labels: list of [N_i]，仅enable_bbox_head时需要
        Returns:
            losses dict
        """
        # 1. VAE编码
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        # 2. 采样噪声和时间步
        noise = torch.randn_like(latents)
        B = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=latents.device,
        ).long()

        # 3. 加噪
        noisy_latents = self.noise_scheduler.add_noise(
            latents, noise, timesteps
        )

        # 4. 条件编码 (含classifier-free guidance dropout)
        if self.training and torch.rand(1).item() < self.cfg_dropout:
            # 随机dropout条件 → 无条件生成
            condition = torch.zeros(
                B,
                NUM_CLASSES + 1,
                self.condition_encoder.embed_dim,
                device=latents.device,
            )
        else:
            condition = self.condition_encoder(class_labels, counts)

        # 5. UNet去噪
        unet_out = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=condition,
        )

        # 6. 图像扩散损失
        if self.prediction_type == 'epsilon':
            target = noise
        elif self.prediction_type == 'v_prediction':
            target = self.noise_scheduler.get_velocity(
                latents, noise, timesteps
            )
        else:
            target = latents

        loss_img = F.mse_loss(unet_out['sample'], target)

        # 7. BBox头损失 (可选)
        losses = {'loss_img': loss_img}

        if (
            self.enable_bbox_head
            and self.bbox_head is not None
            and gt_bboxes is not None
        ):
            # BBox头使用归一化的t ∈ [0,1]
            t_rf = (
                timesteps.float()
                / self.noise_scheduler.config.num_train_timesteps
            )

            bbox_out = self.bbox_head(
                bottleneck_feat=unet_out['bottleneck_feat'],
                condition=condition,
                t=t_rf,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
            )

            losses['loss_bbox'] = bbox_out['loss_bbox']
            losses['loss_cls'] = bbox_out['loss_cls']
            losses['loss_total'] = (
                self.lambda_img * loss_img
                + self.lambda_bbox * bbox_out['loss_bbox']
                + self.lambda_cls * bbox_out['loss_cls']
            )
        else:
            losses['loss_total'] = self.lambda_img * loss_img

        return losses

    @torch.no_grad()
    def generate(
        self,
        class_labels: torch.Tensor,
        counts: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_size: Tuple[int, int] = (768, 768),
        return_bboxes: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """推理：生成图像和可选的bbox

        Args:
            class_labels: [B, 24] 类别索引
            counts: [B, 24] 各类数量
            num_inference_steps: DDIM采样步数
            guidance_scale: classifier-free guidance缩放
            image_size: 输出图像尺寸 (H, W)
            return_bboxes: 是否生成bbox
        Returns:
            dict with 'images' [B, 3, H, W] and optionally 'bboxes', 'labels', 'scores'
        """
        B = class_labels.shape[0]
        device = next(self.parameters()).device

        # 潜空间尺寸
        H_lat = image_size[0] // self.vae_scale_factor
        W_lat = image_size[1] // self.vae_scale_factor

        # 设置推理scheduler
        self.inference_scheduler.set_timesteps(
            num_inference_steps, device=device
        )

        # 初始噪声
        latents = torch.randn(B, 4, H_lat, W_lat, device=device)

        # 条件编码
        condition = self.condition_encoder(class_labels, counts)

        # 无条件编码 (for CFG)
        uncond = torch.zeros_like(condition)

        # DDIM采样循环
        for t in self.inference_scheduler.timesteps:
            # CFG: 条件 + 无条件
            latent_input = torch.cat([latents, latents])
            timestep_input = torch.cat(
                [t.unsqueeze(0).expand(B), t.unsqueeze(0).expand(B)]
            )
            cond_input = torch.cat([condition, uncond])

            unet_out = self.unet(
                sample=latent_input,
                timestep=timestep_input,
                encoder_hidden_states=cond_input,
            )

            # CFG缩放
            noise_pred_cond = unet_out['sample'][:B]
            noise_pred_uncond = unet_out['sample'][B:]
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # 获取bottleneck特征 (仅条件分支)
            bottleneck_feat = unet_out['bottleneck_feat'][:B]

            # DDIM step
            latents = self.inference_scheduler.step(
                noise_pred, t, latents
            ).prev_sample

        # VAE解码
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        result = {'images': images}

        # BBox生成 (可选)
        if (
            return_bboxes
            and self.enable_bbox_head
            and self.bbox_head is not None
        ):
            # 使用最后一步的bottleneck特征和t=0
            t_final = torch.zeros(B, device=device)
            bbox_out = self.bbox_head(
                bottleneck_feat=bottleneck_feat,
                condition=condition,
                t=t_final,
            )
            result.update(bbox_out)

        return result

    @classmethod
    def from_config(cls, config) -> 'ChromoGenPipeline':
        """从ChromoGenConfig创建Pipeline实例

        Args:
            config: ChromoGenConfig实例
        Returns:
            ChromoGenPipeline实例
        """
        return cls(
            vae_model=config.vae.model,
            vae_subfolder=config.vae.subfolder,
            sample_size=config.unet.sample_size,
            unet_block_out_channels=config.unet.block_out_channels,
            unet_attention_head_dim=config.unet.attention_head_dim,
            cross_attention_dim=config.unet.cross_attention_dim,
            gradient_checkpointing=config.unet.gradient_checkpointing,
            condition_embed_dim=config.condition_encoder.embed_dim,
            condition_max_count=config.condition_encoder.max_count,
            condition_dropout=config.condition_encoder.dropout,
            enable_bbox_head=config.enable_bbox_head,
            bbox_feat_channels=config.bbox_head.feat_channels,
            bbox_num_proposals=config.bbox_head.num_proposals,
            bbox_num_heads=config.bbox_head.num_heads,
            bbox_num_layers=config.bbox_head.num_layers,
            bbox_snr_scale=config.bbox_head.snr_scale,
            num_train_timesteps=config.diffusion.num_train_timesteps,
            noise_schedule=config.diffusion.noise_schedule,
            prediction_type=config.diffusion.prediction_type,
            lambda_img=config.loss.lambda_img,
            lambda_bbox=config.loss.lambda_bbox,
            lambda_cls=config.loss.lambda_cls,
            cfg_dropout=config.loss.cfg_dropout,
        )

    def get_trainable_params(self):
        """获取可训练参数（排除frozen VAE）"""
        return [p for p in self.parameters() if p.requires_grad]

    def prepare_bboxes_for_training(
        self,
        bboxes_xyxy: List[torch.Tensor],
        labels: List[torch.Tensor],
        img_h: int,
        img_w: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """将xyxy格式的bbox转为cxcywh归一化格式供训练使用

        Args:
            bboxes_xyxy: list of [N_i, 4] xyxy格式
            labels: list of [N_i]
            img_h, img_w: 图像尺寸
        Returns:
            bboxes_cxcywh: list of [N_i, 4] cxcywh归一化
            labels: list of [N_i]
        """
        bboxes_cxcywh = []
        for bbox in bboxes_xyxy:
            if bbox.numel() == 0:
                bboxes_cxcywh.append(bbox)
                continue
            # xyxy → cxcywh归一化
            bbox = bbox.float()
            x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            bboxes_cxcywh.append(torch.stack([cx, cy, w, h], dim=-1))

        return bboxes_cxcywh, labels
