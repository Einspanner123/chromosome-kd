"""ChromoGen配置dataclass

用结构化配置替代Pipeline.__init__的大量参数，提高可读性和可维护性。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .constants import NUM_CLASSES


@dataclass
class VAEConfig:
    model: str = 'stabilityai/sd-vae-ft-mse'
    subfolder: str = None


@dataclass
class UNetConfig:
    sample_size: int = 96
    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    attention_head_dim: int = 8
    cross_attention_dim: int = 768
    gradient_checkpointing: bool = False
    pretrained_model: Optional[str] = None

    @property
    def bottleneck_channels(self) -> int:
        return self.block_out_channels[-1]


@dataclass
class ConditionEncoderConfig:
    num_classes: int = NUM_CLASSES
    embed_dim: int = 768
    max_count: int = 50
    dropout: float = 0.1


@dataclass
class BBoxHeadConfig:
    in_channels: int = 1280
    feat_channels: int = 512
    num_classes: int = NUM_CLASSES
    num_proposals: int = 100
    num_heads: int = 8
    num_layers: int = 3
    snr_scale: float = 2.0
    num_cls_convs: int = 1
    num_reg_convs: int = 3


@dataclass
class DiffusionConfig:
    num_train_timesteps: int = 1000
    noise_schedule: str = 'linear'
    prediction_type: str = 'epsilon'
    num_inference_steps: int = 50
    guidance_scale: float = 7.5


@dataclass
class LossConfig:
    lambda_img: float = 1.0
    lambda_bbox: float = 0.5
    lambda_cls: float = 0.5
    cfg_dropout: float = 0.1


@dataclass
class ChromoGenConfig:
    """ChromoGen完整配置"""

    vae: VAEConfig = field(default_factory=VAEConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    condition_encoder: ConditionEncoderConfig = field(
        default_factory=ConditionEncoderConfig
    )
    bbox_head: BBoxHeadConfig = field(default_factory=BBoxHeadConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    enable_bbox_head: bool = True
    image_size: int = 768
