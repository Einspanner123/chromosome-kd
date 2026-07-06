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
        unet_pretrained_model: Optional[str] = None,
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
        # unet_pretrained_model不为None时，从预训练模型加载UNet权重做fine-tune
        # 架构与SD-1.5 UNet同构，可无损加载
        self.unet = ChromoUNet(
            sample_size=sample_size,
            in_channels=4,
            out_channels=4,
            block_out_channels=unet_block_out_channels,
            attention_head_dim=unet_attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            gradient_checkpointing=gradient_checkpointing,
            pretrained_model=unet_pretrained_model,
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
        # karras: 训练时用 DDPMScheduler with scaled_linear（Karras 风格的 beta schedule，
        #         在低 SNR 区域有更高采样密度，对医学图像有改善）
        #         注意：KDPM2DiscreteScheduler 是推理专用，没有 get_velocity/add_noise，
        #               不能用于训练。用 DDPMScheduler with scaled_linear 是社区验证过的
        #               Karras 训练近似方案（SD-2.0/SDXL 默认配置）。
        # 其他: 用 DDPMScheduler with 指定 beta_schedule
        if noise_schedule == 'karras':
            noise_schedule_resolved = 'scaled_linear'
        else:
            noise_schedule_resolved = noise_schedule

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=noise_schedule_resolved,
            prediction_type=prediction_type,
        )
        # 6. 采样Scheduler (推理用，后续可替换)
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=noise_schedule_resolved,
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
        # 重要：始终运行 condition_encoder，再按概率置零（condition * 0.0）。
        # 不能用 torch.zeros() 替换 condition，因为在 Stage 1（frozen UNet +
        # VAE no_grad 编码）下，torch.zeros() 不要求梯度，整个计算图无梯度
        # 入口，loss 会丢失 grad_fn，backward 报错：
        #   "element 0 of tensors does not require grad and does not have a grad_fn"
        # 用 condition * 0.0 保留 grad_fn（MulBackward），梯度可正常回传
        # 到 CondEncoder（梯度值为 0，符合 CFG dropout 语义：dropped-out
        # step 不更新 encoder）。
        condition = self.condition_encoder(class_labels, counts)
        if self.training and torch.rand(1).item() < self.cfg_dropout:
            condition = condition * 0.0

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
            unet_pretrained_model=config.unet.pretrained_model,
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

    def freeze_unet(self):
        """冻结 UNet 参数（分阶段训练 Stage 1 用）

        Condition Encoder 仍可训练，UNet 完全冻结。

        重要：同时关闭 gradient_checkpointing，因为：
        1. UNet 冻结后无需省显存（不存中间激活梯度）
        2. gradient_checkpointing + frozen 参数会导致 autograd 优化掉
           梯度回传路径，使 condition 的 grad_fn 丢失，backward 报错
        """
        for p in self.unet.parameters():
            p.requires_grad = False
        if hasattr(self.unet.unet, 'disable_gradient_checkpointing'):
            self.unet.unet.disable_gradient_checkpointing()

    def unfreeze_unet(self):
        """解冻 UNet 参数（分阶段训练 Stage 2 用）"""
        for p in self.unet.parameters():
            p.requires_grad = True
        if hasattr(self.unet.unet, 'enable_gradient_checkpointing'):
            self.unet.unet.enable_gradient_checkpointing()

    def get_param_groups(self, unet_lr: float, cond_lr: float):
        """返回分组参数列表，支持 UNet 与 Condition Encoder 不同 LR

        Args:
            unet_lr: UNet 学习率
            cond_lr: Condition Encoder 学习率

        Returns:
            list[dict]: [{"params": ..., "lr": unet_lr},
                         {"params": ..., "lr": cond_lr}]
        """
        unet_params = [p for p in self.unet.parameters() if p.requires_grad]
        cond_params = [
            p for p in self.condition_encoder.parameters() if p.requires_grad
        ]
        groups = []
        if unet_params:
            groups.append({'params': unet_params, 'lr': unet_lr})
        if cond_params:
            groups.append({'params': cond_params, 'lr': cond_lr})
        # 如果启用 BBox Head，归入 cond_lr 组（与 CondEncoder 同步）
        if self.enable_bbox_head and self.bbox_head is not None:
            bbox_params = [
                p for p in self.bbox_head.parameters() if p.requires_grad
            ]
            if bbox_params:
                groups.append({'params': bbox_params, 'lr': cond_lr})
        return groups

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
