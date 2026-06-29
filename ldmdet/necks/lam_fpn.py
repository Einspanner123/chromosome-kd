"""LAMFPN — 方向 G: 局部注意力特征金字塔

用 LAM (Local Attention Module) 替换标准 FPN 的简单相加融合, 并可选附加
DualAttention (通道+空间) 和 CrossLayerAttention (跨层).

设计:
- 继承 mmdet FPN, 接口完全兼容
- top-down 路径在 apply_lam_levels 指定层级用 LAMModule 融合 (softmax 注意力)
- 输出层应用 DualAttention (通道+空间双重增强, 带跳过机制)
- 可选 CrossLayerAttention 跨所有层级加权融合

参考:
- lamfpn.py (项目内部 LAMFPN 实现)
- CBAM (Woo et al., ECCV 2018) — DualAttention 灵感
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from mmdet.models.necks.fpn import FPN
from mmdet.registry import MODELS


class LAMModule(BaseModule):
    """局部注意力融合模块.

    用 softmax 注意力权重代替简单相加, 自适应融合 target + source 特征.

    结构:
        concat(target, source) → conv(C/4) → conv(2) → softmax → 加权融合 → conv

    Args:
        channels: 输入/输出通道数
        channel_reduction: 注意力中间通道压缩比
        use_dw_conv: 是否用深度可分离卷积
        conv_cfg, norm_cfg, act_cfg: 卷积配置
    """

    def __init__(
        self,
        channels: int,
        channel_reduction: int = 4,
        use_dw_conv: bool = True,
        conv_cfg=None,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
        init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'),
    ):
        super().__init__(init_cfg)
        reduced_channels = max(channels // channel_reduction, 1)
        conv_module = DepthwiseSeparableConvModule if use_dw_conv else ConvModule

        # 注意力权重生成: concat → 压缩 → 2 维权重
        self.attention_conv = conv_module(
            channels * 2,
            reduced_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.attention_weights = nn.Conv2d(reduced_channels, 2, kernel_size=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

        # 融合后处理
        self.fusion_conv = conv_module(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, target_feat: torch.Tensor, source_feat: torch.Tensor) -> torch.Tensor:
        """前向.

        Args:
            target_feat: (B, C, H, W) 当前层特征 (高分辨率)
            source_feat: (B, C, H, W) 上采样后的上层特征 (高语义)

        Returns:
            (B, C, H, W) 融合后特征
        """
        concat_feat = torch.cat([target_feat, source_feat], dim=1)
        attention_feat = self.attention_conv(concat_feat)
        weights = self.softmax(self.attention_weights(attention_feat))
        # weights: (B, 2, H, W) — 逐空间位置的 target/source 权重
        fused = weights[:, 0:1] * target_feat + weights[:, 1:2] * source_feat
        return self.fusion_conv(fused)


class DualAttention(BaseModule):
    """通道 + 空间双重注意力 (CBAM 风格).

    Args:
        channels: 输入通道数
        reduction: 通道注意力中间层压缩比
        use_dw_conv: 空间注意力是否用深度卷积
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        use_dw_conv: bool = True,
        init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'),
    ):
        super().__init__(init_cfg)
        # 通道注意力: 共享 MLP 处理 avg+max pool
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        # 空间注意力: 7x7 conv
        if use_dw_conv:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(2, 1, 1, bias=False),
                nn.Conv2d(1, 1, 7, padding=3, groups=1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(2, 1, 7, padding=3, bias=False),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向: 通道注意力 → 空间注意力."""
        # 通道注意力
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        channel_att = torch.sigmoid(self.shared_mlp(avg_pool) + self.shared_mlp(max_pool))
        x = x * channel_att
        # 空间注意力
        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([spatial_avg, spatial_max], dim=1))
        return x * spatial_att


class CrossLayerAttention(BaseModule):
    """跨层注意力: 让每个层级融合其他所有层级的加权特征.

    Args:
        channels: 通道数
        num_levels: 层级数
    """

    def __init__(
        self,
        channels: int,
        num_levels: int,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'),
    ):
        super().__init__(init_cfg)
        self.num_levels = num_levels
        self.shared_transform = ConvModule(
            channels, channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg,
        )
        # 权重生成器: 输出 num_levels-1 个其他层级的权重
        self.shared_weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, num_levels - 1, 1),
            nn.Sigmoid(),
        )
        self.fuse_conv = ConvModule(
            channels, channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg,
        )

    def forward(self, features):
        """前向.

        Args:
            features: list of (B, C, H_i, W_i), 长度 = num_levels

        Returns:
            list of (B, C, H_i, W_i), 跨层融合后
        """
        transformed = [self.shared_transform(f) for f in features]
        outputs = []
        for level_idx, feat in enumerate(features):
            # 收集其他层级 (resize 到当前层尺寸)
            others = []
            for idx in range(self.num_levels):
                if idx != level_idx and idx < len(features):
                    resized = F.interpolate(
                        transformed[idx], size=feat.shape[2:],
                        mode='bilinear', align_corners=False,
                    )
                    others.append(resized)
            if others:
                weights = self.shared_weight_gen(feat)  # (B, num_levels-1, 1, 1)
                weights = weights[:, :len(others)]
                fused = sum(others[i] * weights[:, i:i + 1] for i in range(len(others)))
                output = self.fuse_conv(feat + fused)
            else:
                output = feat
            outputs.append(output)
        return outputs


@MODELS.register_module()
class LAMFPN(FPN):
    """LAMFPN — 局部注意力特征金字塔.

    继承标准 FPN, 在 top-down 路径用 LAMModule 替换简单相加,
    并在输出层应用 DualAttention. 可选 CrossLayerAttention.

    Args:
        in_channels, out_channels, num_outs: 同 FPN
        apply_lam_levels: 应用 LAM 的层级索引 (0-based, 对应 laterals 索引)
        use_dw_conv: 是否用深度可分离卷积
        channel_reduction: LAM 通道压缩比
        skip_attention_thresh: 跳过注意力的能量阈值 (0 则不跳过)
        use_modern_norm: 是否用 GroupNorm 代替 BN
        use_modern_act: 'silu' / 'mish' / None
        attention_type: 'dual' (DualAttention) / None
        use_cross_layer_attention: 是否启用跨层注意力
        use_checkpoint: 是否用梯度检查点
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs=4,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        upsample_cfg=dict(mode='nearest'),
        init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'),
        apply_lam_levels=(1, 2),
        use_dw_conv=True,
        channel_reduction=4,
        skip_attention_thresh=0.0,
        use_modern_norm=True,
        use_modern_act='silu',
        attention_type='dual',
        use_cross_layer_attention=False,
        use_checkpoint=True,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
            init_cfg=init_cfg,
            **kwargs,
        )
        self.use_checkpoint = use_checkpoint
        self.skip_attention_thresh = skip_attention_thresh

        # 现代化 norm/act
        self.modern_norm_cfg = (
            dict(type='GN', num_groups=32, requires_grad=True)
            if use_modern_norm else (norm_cfg or dict(type='BN'))
        )
        if use_modern_act == 'silu':
            self.modern_act_cfg = dict(type='SiLU')
        elif use_modern_act == 'mish':
            self.modern_act_cfg = dict(type='Mish')
        else:
            self.modern_act_cfg = act_cfg or dict(type='ReLU')

        # LAM 模块: 索引对应 laterals 融合 (i-1 与 i 融合 → 索引 i-1)
        self.apply_lam_levels = set(apply_lam_levels)
        self.lam_modules = nn.ModuleList([
            LAMModule(
                channels=out_channels,
                channel_reduction=channel_reduction,
                use_dw_conv=use_dw_conv,
                conv_cfg=conv_cfg,
                norm_cfg=self.modern_norm_cfg,
                act_cfg=self.modern_act_cfg,
            ) if i in self.apply_lam_levels else None
            for i in range(num_outs - 1)
        ])

        # DualAttention
        self.attention_type = attention_type
        attn_levels = {1, 2} if attention_type == 'dual' else set()
        self.feature_attention = nn.ModuleList([
            DualAttention(
                channels=out_channels,
                reduction=16,
                use_dw_conv=use_dw_conv,
            ) if i in attn_levels and attention_type == 'dual' else None
            for i in range(num_outs)
        ])

        # 跨层注意力 (可选)
        self.use_cross_layer_attention = use_cross_layer_attention
        self.cross_attention = (
            CrossLayerAttention(
                channels=out_channels,
                num_levels=num_outs,
                norm_cfg=self.modern_norm_cfg,
                act_cfg=self.modern_act_cfg,
            ) if use_cross_layer_attention else None
        )

    def forward(self, inputs):
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward_impl, inputs, use_reentrant=False)
        return self._forward_impl(inputs)

    def _forward_impl(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # 构建 laterals
        laterals = [
            self.lateral_convs[i](inputs[i + self.start_level])
            for i in range(len(self.lateral_convs))
        ]

        # top-down 路径: i 从高到低, laterals[i-1] 融合 upsample(laterals[i])
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], **self.upsample_cfg,
            )
            if (i - 1) in self.apply_lam_levels and self.lam_modules[i - 1] is not None:
                laterals[i - 1] = self.lam_modules[i - 1](laterals[i - 1], upsampled)
            else:
                laterals[i - 1] = laterals[i - 1] + upsampled

        # 应用 DualAttention (带跳过机制)
        enhanced = []
        for i, lateral in enumerate(laterals):
            attn = self.feature_attention[i]
            if attn is not None:
                if self.skip_attention_thresh > 0:
                    with torch.no_grad():
                        importance = torch.mean(F.adaptive_avg_pool2d(lateral, 1))
                    if importance > self.skip_attention_thresh:
                        enhanced.append(attn(lateral))
                    else:
                        enhanced.append(lateral)
                else:
                    enhanced.append(attn(lateral))
            else:
                enhanced.append(lateral)

        # 跨层注意力
        if self.cross_attention is not None:
            enhanced = self.cross_attention(enhanced)

        # FPN convs 输出
        outs = [self.fpn_convs[i](enhanced[i]) for i in range(used_backbone_levels)]

        # 额外层级 (P6/P7 等)
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                extra_source = (
                    inputs[self.backbone_end_level - 1]
                    if self.add_extra_convs == 'on_input' else laterals[-1]
                )
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)
