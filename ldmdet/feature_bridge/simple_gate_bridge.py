"""SimpleGateFeatureBridgeModule — 简单 zero-init gate 特征桥接

E6.4 分析发现 cross-attention gamma 未激活 (≈0), 但 simple gate 有效学习
(gate_down1=+0.11, gate_down2=-0.17, gate_down3=+0.29).

本模块移除 cross-attention, 只保留 zero-init 标量 gate:
- Level 1 (H/8): gate * proj(cg_down1)
- Level 2 (H/16): gate * proj(cg_down2)
- Level 3 (H/32): gate * proj(cg_down3) + gate * proj(cg_mid)

特点:
- 参数量极少 (仅 3 个标量 gate + proj 层)
- 梯度不消失 (无 sigmoid)
- 模型可选择性抑制或增强 CG 特征 (gate 可正可负)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class SimpleGateFeatureBridgeModule(nn.Module):
    """Simple zero-init gate 特征桥接模块"""

    def __init__(
        self,
        cg_channels=(320, 640, 1280, 1280),
        ld_channels=256,
        num_groups: int = 32,
    ):
        super().__init__()
        assert len(cg_channels) == 4
        assert ld_channels % num_groups == 0

        self.ld_channels = ld_channels
        self.cg_channels = cg_channels

        # Level 1 (H/8): proj + zero-init gate
        self.proj_down1 = self._build_proj(cg_channels[0], ld_channels, num_groups)
        self.gate_down1 = nn.Parameter(torch.zeros(1))

        # Level 2 (H/16): proj + zero-init gate
        self.proj_down2 = self._build_proj(cg_channels[1], ld_channels, num_groups)
        self.gate_down2 = nn.Parameter(torch.zeros(1))

        # Level 3 (H/32): proj down3 + proj mid + zero-init gate
        self.proj_down3 = self._build_proj(cg_channels[2], ld_channels, num_groups)
        self.proj_mid = self._build_proj(cg_channels[3], ld_channels, num_groups)
        self.gate_down3 = nn.Parameter(torch.zeros(1))
        self.gate_mid = nn.Parameter(torch.zeros(1))

    def _build_proj(self, in_channels: int, out_channels: int, num_groups: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU(),
        )

    def forward(self, ld_feats, cg_feats):
        assert len(ld_feats) == 4
        fused = list(ld_feats)

        # Level 1 (H/8)
        cg_d1 = self.proj_down1(cg_feats['down1'])
        cg_d1 = F.interpolate(
            cg_d1, size=ld_feats[1].shape[2:], mode='bilinear', align_corners=False
        )
        fused[1] = ld_feats[1] + self.gate_down1 * cg_d1

        # Level 2 (H/16)
        cg_d2 = self.proj_down2(cg_feats['down2'])
        cg_d2 = F.interpolate(
            cg_d2, size=ld_feats[2].shape[2:], mode='bilinear', align_corners=False
        )
        fused[2] = ld_feats[2] + self.gate_down2 * cg_d2

        # Level 3 (H/32): down3 + mid
        cg_d3 = self.proj_down3(cg_feats['down3'])
        cg_d3 = F.interpolate(
            cg_d3, size=ld_feats[3].shape[2:], mode='bilinear', align_corners=False
        )
        cg_mid = self.proj_mid(cg_feats['mid'])
        cg_mid = F.interpolate(
            cg_mid, size=ld_feats[3].shape[2:], mode='bilinear', align_corners=False
        )
        fused[3] = (
            ld_feats[3]
            + self.gate_down3 * cg_d3
            + self.gate_mid * cg_mid
        )

        # 诊断
        with torch.no_grad():
            self._last_stats = {
                'gate_L1': self.gate_down1.item(),
                'gate_L2': self.gate_down2.item(),
                'gate_L3': self.gate_down3.item(),
                'gate_mid': self.gate_mid.item(),
            }

        return fused

    def extra_repr(self):
        return f'cg_channels={self.cg_channels}, ld_channels={self.ld_channels}'

    def get_gate_values(self):
        with torch.no_grad():
            return {
                'gate_down1': self.gate_down1.item(),
                'gate_down2': self.gate_down2.item(),
                'gate_down3': self.gate_down3.item(),
                'gate_mid': self.gate_mid.item(),
            }

    def get_diagnostics(self):
        diag = {}
        if hasattr(self, '_last_stats'):
            diag.update(self._last_stats)
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name.startswith('gate_'):
                    diag[f'param_{name}'] = param.item()
                    if param.grad is not None:
                        diag[f'grad_{name}_norm'] = param.grad.norm().item()
        return diag
