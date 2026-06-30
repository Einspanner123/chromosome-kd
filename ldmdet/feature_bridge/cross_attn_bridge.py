"""CrossAttnFeatureBridgeModule — Cross-Attention 特征桥接

方向六 Phase 1 E6.4: 针对 E6.3 gate 不学习问题的改进方案

E6.3 问题:
- Gate 保持在初始值 0.1 附近, 模型没有主动调整 CG 贡献度
- Simple residual gating (ld*(1-g)+g*cg) 下, 模型要么全盘接受要么全盘拒绝
- mAP=0.703 < E6.2=0.737 < baseline=0.753

E6.4 改进: Cross-Attention 融合
- LD 特征作为 Query, CG 特征作为 Key/Value
- 模型在空间位置级别选择性查询 CG 信息 (而非整体加权)
- Output gamma zero-init (类似 AdaLN-Zero), 保证 baseline 不被破坏

架构设计 (显存与表达力平衡):
- Level 1 (H/8, 96x96=9216 tokens): 简单 zero-init additive gate
  (attention matrix 9216^2 太大, 且低级特征 attention 收益小)
- Level 2 (H/16, 48x48=2304 tokens): 简单 zero-init additive gate
  (attention matrix 2304^2=5.3M, 开销仍偏大)
- Level 3 (H/32, 24x24=576 tokens): Cross-Attention
  (attention matrix 576^2=331K, 可控; 最高语义层 CG 特征最有价值)
  - CG down3 + mid 拼接为 K/V source, 提供更丰富的上下文
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class CrossAttnFeatureBridgeModule(nn.Module):
    """Cross-Attention 特征桥接模块

    Level 1/2: zero-init additive gate (简单融合)
    Level 3: cross-attention (LD=Q, CG=K/V) + zero-init gamma

    Args:
        cg_channels: ChromoGen 各层通道数 [down1, down2, down3, mid]
        ld_channels: LDMDet FPN 输出通道数 (默认 256)
        num_heads: Level 3 cross-attention 的 head 数 (默认 4)
        num_groups: GroupNorm 分组数 (默认 32)
    """

    def __init__(
        self,
        cg_channels=(320, 640, 1280, 1280),
        ld_channels=256,
        num_heads: int = 4,
        num_groups: int = 32,
    ):
        super().__init__()
        assert len(cg_channels) == 4
        assert ld_channels % num_groups == 0
        assert ld_channels % num_heads == 0, (
            f'ld_channels ({ld_channels}) must be divisible by num_heads ({num_heads})'
        )

        self.ld_channels = ld_channels
        self.cg_channels = cg_channels
        self.num_heads = num_heads
        self.num_groups = num_groups

        # ============================================================
        # Level 1 (H/8): 简单 zero-init 标量 gate (无 sigmoid, 梯度不消失)
        # ============================================================
        self.proj_down1 = self._build_proj(cg_channels[0], ld_channels, num_groups)
        self.gate_down1 = nn.Parameter(torch.zeros(1))

        # ============================================================
        # Level 2 (H/16): 简单 zero-init 标量 gate
        # ============================================================
        self.proj_down2 = self._build_proj(cg_channels[1], ld_channels, num_groups)
        self.gate_down2 = nn.Parameter(torch.zeros(1))

        # ============================================================
        # Level 3 (H/32): Cross-Attention
        # K/V source: CG down3 + mid (拼接后提供更丰富上下文)
        # Q: LD Level 3 特征
        # ============================================================
        # K/V projection (CG → ld_channels)
        self.proj_down3_k = nn.Conv2d(cg_channels[2], ld_channels, 1)
        self.proj_down3_v = nn.Conv2d(cg_channels[2], ld_channels, 1)
        self.proj_mid_k = nn.Conv2d(cg_channels[3], ld_channels, 1)
        self.proj_mid_v = nn.Conv2d(cg_channels[3], ld_channels, 1)

        # Q projection (LD → ld_channels, 这里 LD 已经是 ld_channels, 用 1x1 conv 做线性变换)
        self.proj_q_down3 = nn.Conv2d(ld_channels, ld_channels, 1)

        # Cross-attention (batch_first: B, L, C)
        self.attn_down3 = nn.MultiheadAttention(
            embed_dim=ld_channels,
            num_heads=num_heads,
            batch_first=True,
        )

        # Output norm + gamma (zero-init, 类似 AdaLN-Zero)
        self.attn_norm = nn.GroupNorm(num_groups, ld_channels)
        self.gamma_down3 = nn.Parameter(torch.zeros(1))

        # Level 3 也有一个简单 gate 作为 fallback (zero-init 标量)
        self.proj_down3_simple = self._build_proj(cg_channels[2], ld_channels, num_groups)
        self.gate_down3 = nn.Parameter(torch.zeros(1))

    def _build_proj(self, in_channels: int, out_channels: int, num_groups: int) -> nn.Sequential:
        """构建投影层: 1x1 conv → GroupNorm → GELU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU(),
        )

    def forward(self, ld_feats, cg_feats):
        """融合 LDMDet FPN 特征与 ChromoGen UNet 特征

        Args:
            ld_feats: LDMDet FPN 4 层特征 [P3, P4, P5, P6] (stride 4/8/16/32)
            cg_feats: ChromoGen UNet 特征字典 {'down1','down2','down3','mid'}

        Returns:
            fused: 融合后的 4 层特征列表
        """
        # 输入校验
        assert len(ld_feats) == 4
        for key in ['down1', 'down2', 'down3', 'mid']:
            assert key in cg_feats, f'cg_feats missing key: {key}'

        fused = list(ld_feats)
        stats = {}

        # ============================================================
        # Level 1 (H/8): 简单 additive gate (标量, zero-init)
        # ============================================================
        cg_d1 = self.proj_down1(cg_feats['down1'])
        cg_d1 = F.interpolate(
            cg_d1, size=ld_feats[1].shape[2:], mode='bilinear', align_corners=False
        )
        fused[1] = ld_feats[1] + self.gate_down1 * cg_d1

        # ============================================================
        # Level 2 (H/16): 简单 additive gate (标量, zero-init)
        # ============================================================
        cg_d2 = self.proj_down2(cg_feats['down2'])
        cg_d2 = F.interpolate(
            cg_d2, size=ld_feats[2].shape[2:], mode='bilinear', align_corners=False
        )
        fused[2] = ld_feats[2] + self.gate_down2 * cg_d2

        # ============================================================
        # Level 3 (H/32): Cross-Attention
        # Q: LD Level 3, K/V: CG down3 + mid
        # ============================================================
        B, C, H3, W3 = ld_feats[3].shape
        # Q from LD
        q = self.proj_q_down3(ld_feats[3])  # B, C, H3, W3
        q = q.flatten(2).transpose(1, 2)  # B, H3*W3, C

        # K/V from CG down3 + mid (拼接两个源, 提供更丰富上下文)
        cg_d3_k = self.proj_down3_k(cg_feats['down3'])
        cg_d3_k = F.interpolate(cg_d3_k, size=(H3, W3), mode='bilinear', align_corners=False)
        cg_d3_k = cg_d3_k.flatten(2).transpose(1, 2)  # B, H3*W3, C

        cg_d3_v = self.proj_down3_v(cg_feats['down3'])
        cg_d3_v = F.interpolate(cg_d3_v, size=(H3, W3), mode='bilinear', align_corners=False)
        cg_d3_v = cg_d3_v.flatten(2).transpose(1, 2)

        cg_mid_k = self.proj_mid_k(cg_feats['mid'])
        cg_mid_k = F.interpolate(cg_mid_k, size=(H3, W3), mode='bilinear', align_corners=False)
        cg_mid_k = cg_mid_k.flatten(2).transpose(1, 2)

        cg_mid_v = self.proj_mid_v(cg_feats['mid'])
        cg_mid_v = F.interpolate(cg_mid_v, size=(H3, W3), mode='bilinear', align_corners=False)
        cg_mid_v = cg_mid_v.flatten(2).transpose(1, 2)

        # 拼接 down3 和 mid 作为 K/V (序列长度 = 2 * H3*W3)
        k = torch.cat([cg_d3_k, cg_mid_k], dim=1)  # B, 2*H3*W3, C
        v = torch.cat([cg_d3_v, cg_mid_v], dim=1)  # B, 2*H3*W3, C

        # Cross-attention (保留 weights 用于 entropy 诊断)
        attn_out, attn_weights = self.attn_down3(
            q, k, v, need_weights=True
        )  # attn_out: B, H3*W3, C; attn_weights: B, H3*W3, 2*H3*W3
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H3, W3)
        attn_out = self.attn_norm(attn_out)

        # Gamma-controlled residual (zero-init gamma → 初始不改变 baseline)
        gamma_val = self.gamma_down3
        fused[3] = ld_feats[3] + gamma_val * attn_out

        # Level 3 也加一个简单 gate 作为补充 (标量 zero-init, 让 down3 有直接路径)
        cg_d3_simple = self.proj_down3_simple(cg_feats['down3'])
        cg_d3_simple = F.interpolate(
            cg_d3_simple, size=(H3, W3), mode='bilinear', align_corners=False
        )
        gate3_val = self.gate_down3
        fused[3] = fused[3] + gate3_val * cg_d3_simple

        # ============================================================
        # 诊断统计 — 全面对照设计预期
        # ============================================================
        with torch.no_grad():
            # --- 1. Gate/Gamma 值 (设计预期: 从 0 增长) ---
            stats['gate_L1'] = self.gate_down1.item()
            stats['gate_L2'] = self.gate_down2.item()
            stats['gate_L3'] = gate3_val.item()
            stats['gamma_L3'] = gamma_val.item()

            # --- 2. 特征范数 (谁在主导) ---
            stats['ld_L1_norm'] = ld_feats[1].norm().item() / B
            stats['ld_L2_norm'] = ld_feats[2].norm().item() / B
            stats['ld_L3_norm'] = ld_feats[3].norm().item() / B
            stats['cg_L1_norm'] = cg_d1.norm().item() / B
            stats['cg_L2_norm'] = cg_d2.norm().item() / B
            stats['cg_L3_simple_norm'] = cg_d3_simple.norm().item() / B
            stats['attn_out_norm'] = attn_out.norm().item() / B
            stats['fused_L1_norm'] = fused[1].norm().item() / B
            stats['fused_L2_norm'] = fused[2].norm().item() / B
            stats['fused_L3_norm'] = fused[3].norm().item() / B

            # --- 3. 贡献比例 (CG 实际影响 vs LD 基准) ---
            # cross-attn 贡献 = |gamma| * ||attn_out|| / ||ld_L3||
            ld3_norm = ld_feats[3].norm().item() / B + 1e-8
            stats['contrib_attn_ratio'] = (
                abs(gamma_val.item()) * attn_out.norm().item() / B
            ) / ld3_norm
            # gate 贡献 = |gate| * ||cg_proj|| / ||ld||
            stats['contrib_gate_L1_ratio'] = (
                abs(self.gate_down1.item()) * cg_d1.norm().item() / B
            ) / (ld_feats[1].norm().item() / B + 1e-8)
            stats['contrib_gate_L2_ratio'] = (
                abs(self.gate_down2.item()) * cg_d2.norm().item() / B
            ) / (ld_feats[2].norm().item() / B + 1e-8)
            stats['contrib_gate_L3_ratio'] = (
                abs(gate3_val.item()) * cg_d3_simple.norm().item() / B
            ) / ld3_norm

            # --- 4. Attention entropy (选择性融合的核心验证) ---
            # entropy 越低 → attention 越集中 (高选择性)
            # entropy 越高 → attention 越均匀 (无选择性)
            # uniform entropy = log(2*H3*W3) = log(1152) ≈ 7.05
            if attn_weights is not None:
                # attn_weights: B, L_q, L_kv
                attn_eps = attn_weights.clamp(min=1e-8)
                entropy = -(attn_eps * attn_eps.log()).sum(dim=-1)  # B, L_q
                stats['attn_entropy_mean'] = entropy.mean().item()
                stats['attn_entropy_std'] = entropy.std().item()
                stats['attn_entropy_uniform'] = float(np.log(attn_weights.shape[-1]))
                # 归一化 entropy: [0, 1], 1=完全均匀, 0=完全集中
                stats['attn_entropy_normalized'] = (
                    stats['attn_entropy_mean'] / stats['attn_entropy_uniform']
                )
                # attention max weight (峰值集中度)
                stats['attn_max_weight_mean'] = attn_weights.max(dim=-1)[0].mean().item()

            # --- 5. CG-LD 特征相似度 (信息增益验证) ---
            # cosine similarity: 若高度相似 → CG 无新信息
            # 若接近 0 → CG 提供正交信息 (理想)
            # 若为负 → CG 特征与 LD 对立 (可能有害)
            for level_name, ld_f, cg_f in [
                ('L1', ld_feats[1], cg_d1),
                ('L2', ld_feats[2], cg_d2),
                ('L3', ld_feats[3], cg_d3_simple),
            ]:
                ld_flat = ld_f.flatten(2)  # B, C, HW
                cg_flat = cg_f.flatten(2)  # B, C, HW
                cos_sim = F.cosine_similarity(ld_flat, cg_flat, dim=1)  # B, HW
                stats[f'cos_sim_{level_name}_mean'] = cos_sim.mean().item()
                stats[f'cos_sim_{level_name}_std'] = cos_sim.std().item()

            # --- 6. Fused-LD 差异 (CG 实际改变程度) ---
            for level_name, ld_f, fused_f in [
                ('L1', ld_feats[1], fused[1]),
                ('L2', ld_feats[2], fused[2]),
                ('L3', ld_feats[3], fused[3]),
            ]:
                diff = (fused_f - ld_f).norm().item() / B
                ld_n = ld_f.norm().item() / B + 1e-8
                stats[f'fused_minus_ld_{level_name}_ratio'] = diff / ld_n

        self._last_stats = stats

        return fused

    def extra_repr(self):
        return (
            f'cg_channels={self.cg_channels}, ld_channels={self.ld_channels}, '
            f'num_heads={self.num_heads}'
        )

    def get_gate_values(self):
        with torch.no_grad():
            return {
                'gate_down1': self.gate_down1.item(),
                'gate_down2': self.gate_down2.item(),
                'gate_down3': self.gate_down3.item(),
                'gamma_down3': self.gamma_down3.item(),
            }

    def get_diagnostics(self):
        """返回完整诊断信息"""
        diag = {}
        if hasattr(self, '_last_stats'):
            diag.update(self._last_stats)

        with torch.no_grad():
            for name, param in self.named_parameters():
                if param is None:
                    continue
                if name.startswith('gate_') or name == 'gamma_down3':
                    diag[f'param_{name}'] = param.item()
                    if param.grad is not None:
                        diag[f'grad_{name}_norm'] = param.grad.norm().item()
                elif 'weight' in name and 'proj' in name:
                    diag[f'param_{name}_mean'] = param.mean().item()
                    diag[f'param_{name}_std'] = param.std().item()

        return diag
