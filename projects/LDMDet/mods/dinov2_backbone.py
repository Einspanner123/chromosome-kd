"""DINOv2 Backbone with Simple Feature Pyramid for object detection.

Supports DINOv2 ViT-S/14 and ViT-B/14 from timm, converting single-scale
ViT output into multi-scale feature maps compatible with FPN-based detectors.

Reference:
  - ViTDet: https://arxiv.org/abs/2203.16527 (Simple Feature Pyramid)
  - DINOv2: https://arxiv.org/abs/2304.07193
  - DEIMv2: https://arxiv.org/abs/2509.20787 (Spatial Tuning Adapter)
"""

import math
from typing import ClassVar, List, Tuple

import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS


class SimpleFeaturePyramid(nn.Module):
    """Convert single-scale ViT output to multi-scale feature maps.

    Uses transposed convolutions for upsampling and strided convolutions for
    downsampling, following the ViTDet Simple Feature Pyramid design.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 14,
        target_strides: List[int] = [4, 8, 16, 32],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.target_strides = target_strides

        # Project ViT embed dim to out_channels
        self.proj = nn.Conv2d(in_channels, out_channels, 1)

        # Compute scale factors relative to patch_size stride
        # ViT output stride = patch_size (e.g., 14)
        vit_stride = patch_size

        self.upsample_modules = nn.ModuleDict()
        self.downsample_modules = nn.ModuleDict()
        self.identity_modules = nn.ModuleDict()

        for s in target_strides:
            ratio = vit_stride / s  # >1 means upsample, <1 means downsample
            key = f'stride_{s}'

            if ratio > 1:
                # Need upsampling: use transposed convolutions
                # We use 2x upsampling layers stacked
                num_up_layers = int(math.log2(round(ratio)))
                layers = []
                for _ in range(num_up_layers):
                    layers.append(
                        nn.ConvTranspose2d(
                            out_channels, out_channels, 2, stride=2
                        )
                    )
                    layers.append(nn.GroupNorm(32, out_channels))
                    layers.append(nn.GELU())
                self.upsample_modules[key] = nn.Sequential(*layers)
            elif ratio < 1:
                # Need downsampling
                num_down_layers = int(math.log2(round(1 / ratio)))
                layers = []
                for _ in range(num_down_layers):
                    layers.append(
                        nn.Conv2d(
                            out_channels,
                            out_channels,
                            3,
                            stride=2,
                            padding=1,
                        )
                    )
                    layers.append(nn.GroupNorm(32, out_channels))
                    layers.append(nn.GELU())
                self.downsample_modules[key] = nn.Sequential(*layers)
            else:
                # Same stride, just a conv for refinement
                self.identity_modules[key] = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.GroupNorm(32, out_channels),
                    nn.GELU(),
                )

    def forward(self, vit_features: Tensor) -> List[Tensor]:
        """
        Args:
            vit_features: [B, N, C] from ViT (after removing CLS token)

        Returns:
            List of feature maps at target_strides, from finest to coarsest
        """
        B, N, C = vit_features.shape
        # Infer spatial dims (assume square or near-square)
        H = W = int(math.sqrt(N))
        if H * W != N:
            # Try to find H, W from aspect ratio
            W = N // H
            while H * W != N and H > 1:
                H -= 1
                W = N // H

        x = vit_features.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.proj(x)  # [B, out_channels, H, W]

        results = []
        for s in self.target_strides:
            key = f'stride_{s}'
            if key in self.upsample_modules:
                out = self.upsample_modules[key](x)
            elif key in self.downsample_modules:
                out = self.downsample_modules[key](x)
            else:
                out = self.identity_modules[key](x)
            results.append(out)

        return results


@MODELS.register_module()
class DINOv2Backbone(nn.Module):
    """DINOv2 ViT backbone with Simple Feature Pyramid for detection.

    Uses timm to load DINOv2 pretrained weights and converts the single-scale
    ViT output to multi-scale features suitable for detection heads.

    Args:
        model_name: timm model name, e.g. 'vit_small_patch14_dinov2.lvd142m'
        out_channels: output channels for each pyramid level (default: 256)
        target_strides: target strides for pyramid levels (default: [4,8,16,32])
        freeze_layers: number of ViT blocks to freeze (default: -1, none)
        use_reg_tokens: whether model uses register tokens (default: True)
        num_reg_tokens: number of register tokens (default: 4)
    """

    # Map common aliases to timm model names
    MODEL_ALIASES: ClassVar[dict] = {
        'dinov2_s': 'vit_small_patch14_reg4_dinov2.lvd142m',
        'dinov2_b': 'vit_base_patch14_reg4_dinov2.lvd142m',
        'dinov2_l': 'vit_large_patch14_reg4_dinov2.lvd142m',
        'dinov2_s_noreg': 'vit_small_patch14_dinov2.lvd142m',
        'dinov2_b_noreg': 'vit_base_patch14_dinov2.lvd142m',
    }

    def __init__(
        self,
        model_name: str = 'dinov2_s',
        out_channels: int = 256,
        target_strides: List[int] = None,
        freeze_layers: int = -1,
        use_reg_tokens: bool = True,
        num_reg_tokens: int = 4,
        pretrained: bool = True,
    ):
        super().__init__()

        if target_strides is None:
            target_strides = [4, 8, 16, 32]

        # Resolve alias
        timm_name = self.MODEL_ALIASES.get(model_name, model_name)

        # Build ViT from timm
        self.vit = self._build_vit(timm_name, pretrained)
        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_embed.patch_size[0]
        self.use_reg_tokens = use_reg_tokens
        self.num_reg_tokens = num_reg_tokens

        # Freeze layers if specified
        if freeze_layers > 0:
            self._freeze_vit_layers(freeze_layers)

        # Simple Feature Pyramid
        self.fpn = SimpleFeaturePyramid(
            in_channels=self.embed_dim,
            out_channels=out_channels,
            patch_size=self.patch_size,
            target_strides=target_strides,
        )

        self.out_channels = out_channels
        self.target_strides = target_strides

    def _build_vit(self, timm_name: str, pretrained: bool):
        """Build ViT model from timm with dynamic input size support."""
        import timm

        # dynamic_img_size=True enables position embedding interpolation
        # for variable input resolutions (standard in ViTDet, RF-DETR, etc.)
        model = timm.create_model(
            timm_name, pretrained=pretrained, dynamic_img_size=True
        )

        return model

    def _freeze_vit_layers(self, num_layers: int):
        """Freeze the first num_layers of ViT blocks."""
        # Freeze patch embed
        for param in self.vit.patch_embed.parameters():
            param.requires_grad = False
        # Freeze pos embed
        if hasattr(self.vit, 'pos_embed') and self.vit.pos_embed is not None:
            self.vit.pos_embed.requires_grad = False
        # Freeze specified blocks
        for i in range(min(num_layers, len(self.vit.blocks))):
            for param in self.vit.blocks[i].parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        """Extract multi-scale features.

        Args:
            x: input images [B, 3, H, W]

        Returns:
            Tuple of feature maps at target_strides (finest to coarsest)
        """
        B = x.shape[0]

        # Get ViT features
        # timm's forward_features returns dict with 'x_norm' etc.
        vit_out = self.vit.forward_features(x)

        # Handle different timm output formats
        if isinstance(vit_out, dict):
            # timm >= 1.0 returns dict
            features = vit_out.get('x_norm', vit_out.get('x', None))
            if features is None:
                # Try to get patch tokens
                features = vit_out.get('x_norm_patchtokens', None)
            if features is None:
                raise ValueError(
                    f'Unexpected vit output keys: {vit_out.keys()}'
                )
        else:
            features = vit_out

        # features shape: [B, N + num_prefix_tokens, embed_dim]
        # Remove prefix tokens (CLS + register tokens)
        num_prefix = 1  # CLS token
        if self.use_reg_tokens:
            num_prefix += self.num_reg_tokens

        if features.shape[1] > (x.shape[2] // self.patch_size) * (
            x.shape[3] // self.patch_size
        ):
            features = features[:, num_prefix:]

        # Build feature pyramid
        pyramid_features = self.fpn(features)

        return tuple(pyramid_features)
