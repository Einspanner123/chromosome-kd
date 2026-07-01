"""FeatureBridgeModule (FBM) — 将 ChromoGen UNet 特征桥接到 LDMDet FPN 特征空间

方向六：生成模型感知迁移。

架构对齐 (ChromoGen UNet 在 VAE latent H/8 空间操作):
    LDMDet FPN Level 0 (stride 4, 256ch)   → 不融合 (无对应 ChromoGen 特征)
    LDMDet FPN Level 1 (stride 8, H/8)     ← 上采样 ChromoGen down1 (320ch, H/16)
    LDMDet FPN Level 2 (stride 16, H/16)   ← 上采样 ChromoGen down2 (640ch, H/32)
    LDMDet FPN Level 3 (stride 32, H/32)   ← 上采样 ChromoGen down3 (1280ch, H/64)
                                           + 上采样 ChromoGen mid (1280ch, H/64)

E6.3 增强设计 (针对 E6.2 alpha→0 失效问题):
1. per-channel sigmoid gate (替代标量 alpha): 每通道独立门控, 表达力更强
2. gate_init=0.1 (替代零初始化): 初始 10% 融合权重, 提供非零梯度信号
3. 投影后加 GroupNorm + GELU: 增强特征对齐能力
4. residual gating: fused = ld*(1-g) + g*cg_proj, 保证 baseline 路径不被破坏
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureBridgeModule(nn.Module):
    """将 ChromoGen 特征桥接到 LDMDet FPN 特征空间

    Args:
        cg_channels: ChromoGen 各层通道数 [down1, down2, down3, mid]
        ld_channels: LDMDet FPN 输出通道数 (默认 256)
        gate_init: 门控初始值 (默认 0.1, 即初始 10% 融合权重)
        num_groups: GroupNorm 分组数 (默认 32)
    """

    def __init__(
        self,
        cg_channels=(320, 640, 1280, 1280),
        ld_channels=256,
        gate_init: float = 0.1,
        num_groups: int = 32,
    ):
        super().__init__()
        assert len(cg_channels) == 4, (
            f'cg_channels must have 4 elements [down1, down2, down3, mid], got {len(cg_channels)}'
        )
        assert 0.0 < gate_init < 1.0, (
            f'gate_init must be in (0, 1), got {gate_init}'
        )
        assert ld_channels % num_groups == 0, (
            f'ld_channels ({ld_channels}) must be divisible by num_groups ({num_groups})'
        )

        self.ld_channels = ld_channels
        self.cg_channels = cg_channels
        self.gate_init = gate_init
        self.num_groups = num_groups

        # 投影层: ChromoGen 通道 → LDMDet 通道 (1x1 conv + GroupNorm + GELU)
        # 注意: GroupNorm 后接 GELU, 最后无激活以便 residual gating 保持线性
        self.proj_down1 = self._build_proj(cg_channels[0], ld_channels, num_groups)
        self.proj_down2 = self._build_proj(cg_channels[1], ld_channels, num_groups)
        self.proj_down3 = self._build_proj(cg_channels[2], ld_channels, num_groups)
        self.proj_mid = self._build_proj(cg_channels[3], ld_channels, num_groups)

        # per-channel gate (sigmoid 激活, 初始化为 gate_init)
        # gate_logit = log(gate_init / (1 - gate_init))
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.gate_down1 = nn.Parameter(
            torch.full((1, ld_channels, 1, 1), gate_logit)
        )
        self.gate_down2 = nn.Parameter(
            torch.full((1, ld_channels, 1, 1), gate_logit)
        )
        self.gate_down3 = nn.Parameter(
            torch.full((1, ld_channels, 1, 1), gate_logit)
        )
        self.gate_mid = nn.Parameter(
            torch.full((1, ld_channels, 1, 1), gate_logit)
        )

    def _build_proj(self, in_channels: int, out_channels: int, num_groups: int) -> nn.Sequential:
        """构建投影层: 1x1 conv → GroupNorm → GELU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU(),
        )

    def forward(self, ld_feats, cg_feats):
        """融合 LDMDet FPN 特征与 ChromoGen UNet 特征

        采用 residual gating: fused = ld * (1 - g) + g * cg_proj
        - g = sigmoid(gate_logit), 范围 (0, 1), 每通道独立
        - 初始 g ≈ gate_init, 保证 baseline 路径占主导 (1-gate_init)
        - cg_proj 保留梯度, 可反向传播到 ChromoGen UNet (若解冻)

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

        # 缓存中间统计 (用于诊断, 不参与梯度)
        stats = {}

        # Level 1 (stride 8, H/8) ← 上采样 down1 (H/16, 320ch)
        cg_d1 = self.proj_down1(cg_feats['down1'])
        cg_d1 = F.interpolate(
            cg_d1, size=ld_feats[1].shape[2:], mode='bilinear', align_corners=False
        )
        g1 = torch.sigmoid(self.gate_down1)
        fused[1] = ld_feats[1] * (1.0 - g1) + g1 * cg_d1

        # Level 2 (stride 16, H/16) ← 上采样 down2 (H/32, 640ch)
        cg_d2 = self.proj_down2(cg_feats['down2'])
        cg_d2 = F.interpolate(
            cg_d2, size=ld_feats[2].shape[2:], mode='bilinear', align_corners=False
        )
        g2 = torch.sigmoid(self.gate_down2)
        fused[2] = ld_feats[2] * (1.0 - g2) + g2 * cg_d2

        # Level 3 (stride 32, H/32) ← 上采样 down3 (H/64, 1280ch) + 上采样 mid (H/64, 1280ch)
        # 先用 gate_mid 在 down3 和 mid 之间做加权, 再用 gate_down3 与 ld 融合
        cg_d3 = self.proj_down3(cg_feats['down3'])
        cg_d3 = F.interpolate(
            cg_d3, size=ld_feats[3].shape[2:], mode='bilinear', align_corners=False
        )
        cg_mid = self.proj_mid(cg_feats['mid'])
        cg_mid = F.interpolate(
            cg_mid, size=ld_feats[3].shape[2:], mode='bilinear', align_corners=False
        )
        g_mid = torch.sigmoid(self.gate_mid)
        cg_combined = g_mid * cg_d3 + (1.0 - g_mid) * cg_mid

        g3 = torch.sigmoid(self.gate_down3)
        fused[3] = ld_feats[3] * (1.0 - g3) + g3 * cg_combined

        # ── 诊断统计 (detached, 不影响梯度) ──
        with torch.no_grad():
            stats['gate_L1_mean'] = g1.mean().item()
            stats['gate_L1_std'] = g1.std().item()
            stats['gate_L2_mean'] = g2.mean().item()
            stats['gate_L2_std'] = g2.std().item()
            stats['gate_L3_mean'] = g3.mean().item()
            stats['gate_L3_std'] = g3.std().item()
            stats['gate_mid_mean'] = g_mid.mean().item()
            stats['gate_mid_std'] = g_mid.std().item()

            # 特征范数: 判断 CG 和 LD 哪个在贡献
            stats['ld_L1_norm'] = ld_feats[1].norm().item() / bs
            stats['ld_L2_norm'] = ld_feats[2].norm().item() / bs
            stats['ld_L3_norm'] = ld_feats[3].norm().item() / bs
            stats['cg_L1_norm'] = cg_d1.norm().item() / bs
            stats['cg_L2_norm'] = cg_d2.norm().item() / bs
            stats['cg_L3_norm'] = cg_d3.norm().item() / bs
            stats['cg_mid_norm'] = cg_mid.norm().item() / bs

            # 融合后范数
            stats['fused_L1_norm'] = fused[1].norm().item() / bs
            stats['fused_L2_norm'] = fused[2].norm().item() / bs
            stats['fused_L3_norm'] = fused[3].norm().item() / bs

            # CG 原始特征范数 (投影前, 反映 ChromoGen UNet 输出强度)
            stats['cg_raw_down1_norm'] = cg_feats['down1'].norm().item() / bs
            stats['cg_raw_down2_norm'] = cg_feats['down2'].norm().item() / bs
            stats['cg_raw_down3_norm'] = cg_feats['down3'].norm().item() / bs
            stats['cg_raw_mid_norm'] = cg_feats['mid'].norm().item() / bs

        self._last_stats = stats

        return fused

    def extra_repr(self):
        return (
            f'cg_channels={self.cg_channels}, ld_channels={self.ld_channels}, '
            f'gate_init={self.gate_init}, num_groups={self.num_groups}'
        )

    def get_gate_values(self):
        """返回当前各层 sigmoid gate 值 (用于监控)"""
        with torch.no_grad():
            return {
                'gate_down1': torch.sigmoid(self.gate_down1).mean().item(),
                'gate_down2': torch.sigmoid(self.gate_down2).mean().item(),
                'gate_down3': torch.sigmoid(self.gate_down3).mean().item(),
                'gate_mid': torch.sigmoid(self.gate_mid).mean().item(),
            }

    def get_diagnostics(self):
        """返回完整诊断信息 (上次 forward 缓存)

        包含:
        - gate 统计: 各层 sigmoid gate 的 mean/std
        - 特征范数: LD/CG/fused 各层 L2 范数 (判断谁在贡献)
        - CG 原始范数: ChromoGen UNet 各层输出范数 (判断 UNet 解冻效果)
        - proj 层参数统计: 投影层权重 mean/std (判断投影是否在学习)
        - gate 参数梯度: 各层 gate 的梯度范数 (判断 gate 是否在被优化)
        """
        diag = {}

        # 1. 上次 forward 的统计
        if hasattr(self, '_last_stats'):
            diag.update(self._last_stats)

        # 2. proj 层参数统计
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param is None:
                    continue
                # gate 参数
                if name.startswith('gate_'):
                    g = torch.sigmoid(param)
                    diag[f'param_{name}_mean'] = g.mean().item()
                    diag[f'param_{name}_std'] = g.std().item()
                    diag[f'param_{name}_min'] = g.min().item()
                    diag[f'param_{name}_max'] = g.max().item()
                    # 梯度统计
                    if param.grad is not None:
                        diag[f'grad_{name}_norm'] = param.grad.norm().item()
                # proj 权重
                elif 'weight' in name and 'proj' in name:
                    diag[f'param_{name}_mean'] = param.mean().item()
                    diag[f'param_{name}_std'] = param.std().item()

        return diag
