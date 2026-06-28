"""FeatureBridgeModule (FBM) — 将 ChromoGen UNet 特征桥接到 LDMDet FPN 特征空间

方向六：生成模型感知迁移。

架构对齐 (ChromoGen UNet 在 VAE latent H/8 空间操作):
    LDMDet FPN Level 0 (stride 4, 256ch)   → 不融合 (无对应 ChromoGen 特征)
    LDMDet FPN Level 1 (stride 8, H/8)     ← 上采样 ChromoGen down1 (320ch, H/16)
    LDMDet FPN Level 2 (stride 16, H/16)   ← 上采样 ChromoGen down2 (640ch, H/32)
    LDMDet FPN Level 3 (stride 32, H/32)   ← 上采样 ChromoGen down3 (1280ch, H/64)
                                           + 上采样 ChromoGen mid (1280ch, H/64)

零初始化: alpha 参数初始化为 0，保证训练初始不破坏 baseline。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureBridgeModule(nn.Module):
    """将 ChromoGen 特征桥接到 LDMDet FPN 特征空间

    Args:
        cg_channels: ChromoGen 各层通道数 [down1, down2, down3, mid]
        ld_channels: LDMDet FPN 输出通道数 (默认 256)
    """

    def __init__(self, cg_channels=(320, 640, 1280, 1280), ld_channels=256):
        super().__init__()
        assert len(cg_channels) == 4, (
            f'cg_channels must have 4 elements [down1, down2, down3, mid], got {len(cg_channels)}'
        )

        self.ld_channels = ld_channels
        self.cg_channels = cg_channels

        # 投影层: ChromoGen 通道 → LDMDet 通道 (1x1 conv)
        self.proj_down1 = nn.Conv2d(cg_channels[0], ld_channels, 1)
        self.proj_down2 = nn.Conv2d(cg_channels[1], ld_channels, 1)
        self.proj_down3 = nn.Conv2d(cg_channels[2], ld_channels, 1)
        self.proj_mid = nn.Conv2d(cg_channels[3], ld_channels, 1)

        # 可学习融合权重 (零初始化, 保证训练初始不破坏 baseline)
        # alpha[0]→level1, alpha[1]→level2, alpha[2]→level3
        self.alpha = nn.Parameter(torch.zeros(3))

    def forward(self, ld_feats, cg_feats):
        """融合 LDMDet FPN 特征与 ChromoGen UNet 特征

        Args:
            ld_feats: LDMDet FPN 4 层特征列表 [P3, P4, P5, P6] (stride 4/8/16/32)
            cg_feats: ChromoGen UNet 编码器特征字典
                      {'down1': ..., 'down2': ..., 'down3': ..., 'mid': ...}

        Returns:
            fused: 融合后的 4 层特征列表 (与 ld_feats 形状一致)
        """
        # ── 输入校验 ──
        if len(ld_feats) != 4:
            raise ValueError(
                f'ld_feats must have 4 levels (stride 4/8/16/32), got {len(ld_feats)}'
            )

        required_keys = ['down1', 'down2', 'down3', 'mid']
        for key in required_keys:
            if key not in cg_feats:
                raise KeyError(f'cg_feats missing required key: {key}')

        bs = ld_feats[0].shape[0]
        for key in required_keys:
            if cg_feats[key].shape[0] != bs:
                raise ValueError(
                    f'Batch size mismatch: ld_feats={bs}, cg_feats[{key}]={cg_feats[key].shape[0]}'
                )

        # ── 融合 ──
        # ChromoGen UNet 在 VAE latent (H/8) 空间操作, 各 down_block 输出:
        #   down1: H/16 (latent/2), down2: H/32 (latent/4), down3: H/64 (latent/8)
        # LDMDet FPN: Level 0 (H/4), Level 1 (H/8), Level 2 (H/16), Level 3 (H/32)
        # 使用自适应上采样匹配各 level 空间尺寸
        fused = list(ld_feats)  # copy (level 0 保持不变)

        # Level 1 (stride 8, H/8) ← 上采样 down1 (H/16, 320ch)
        cg_d1 = self.proj_down1(cg_feats['down1'])
        cg_d1 = F.interpolate(
            cg_d1, size=ld_feats[1].shape[2:], mode='bilinear', align_corners=False
        )
        fused[1] = ld_feats[1] + self.alpha[0] * cg_d1

        # Level 2 (stride 16, H/16) ← 上采样 down2 (H/32, 640ch)
        cg_d2 = self.proj_down2(cg_feats['down2'])
        cg_d2 = F.interpolate(
            cg_d2, size=ld_feats[2].shape[2:], mode='bilinear', align_corners=False
        )
        fused[2] = ld_feats[2] + self.alpha[1] * cg_d2

        # Level 3 (stride 32, H/32) ← 上采样 down3 (H/64, 1280ch) + 上采样 mid (H/64, 1280ch)
        cg_d3 = self.proj_down3(cg_feats['down3'])
        cg_mid = self.proj_mid(cg_feats['mid'])
        cg_d3 = F.interpolate(
            cg_d3, size=ld_feats[3].shape[2:], mode='bilinear', align_corners=False
        )
        cg_mid = F.interpolate(
            cg_mid, size=ld_feats[3].shape[2:], mode='bilinear', align_corners=False
        )
        fused[3] = ld_feats[3] + self.alpha[2] * (cg_d3 + cg_mid)

        return fused

    def extra_repr(self):
        return (
            f'cg_channels={self.cg_channels}, ld_channels={self.ld_channels}, '
            f'alpha_init=zero'
        )
