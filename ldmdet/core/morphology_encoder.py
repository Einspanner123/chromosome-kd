"""形态感知 RoI 编码器 (Morphology-Aware RoI Encoder) — M1

零初始化残差分支, 在 7×7 RoI 特征上叠加染色体形态学先验。
D1 消融实验 (2026-07-22) 证实 7×7 空间结构至关重要 (抹平→mAP 0.863→0.009),
因此本模块作为**增强**而非**重建**接入, 不破坏已验证有效的空间通路。

ISCN 染色体形态学判据:
  - 臂长比 (p/q): 着丝粒位置 → 近中/亚中/近端着丝粒
  - 整体尺寸: A>G>Y, ~5× 跨度
  - 弯曲度: 部分类别有特征性弯曲

方向解耦设计:
  - h_conv (H,1): 沿臂长方向 (水平) 扫描, 捕获臂长比
  - v_conv (1,W): 沿着丝粒方向 (垂直) 扫描, 捕获着丝粒位置
  两个方向独立提取后融合为形态嵌入, 零初始化保证初始残差≡0。

详见 docs/research/STRUCTURAL_IMPROVEMENT_ANALYSIS.md §三 M1
"""

import torch
import torch.nn as nn


class MorphologyAwareRoIEncoder(nn.Module):
    """形态感知 RoI 编码器: 零初始化残差分支增强染色体形态特征.

    在 RoIAlign 输出的 [N, C, H, W] 特征上, 用方向解耦卷积分别提取
    水平 (臂长比) 和垂直 (着丝粒位置) 形态线索, 融合后以残差形式叠加。

    零初始化 fuse 层确保:
      - 初始 morph_emb ≡ 0 → 加载预训练权重时行为不变
      - 训练初期梯度通过残差路径流回, 逐步学到形态增强
      - 不破坏 D1 消融验证的 7×7 空间通路

    Args:
        channels: 输入通道数 (与 feat_channels 一致, 默认 256)
        reduction: 方向卷积通道缩减比 (默认 4, 即 256→64)
        pooler_resolution: RoI 空间分辨率 (默认 7, 与 RoIAlign output_size 一致)
    """

    def __init__(
        self,
        channels: int = 256,
        reduction: int = 4,
        pooler_resolution: int = 7,
    ):
        super().__init__()
        self.pooler_resolution = pooler_resolution
        reduced = channels // reduction

        # 水平方向 (臂长比): kernel 沿高度覆盖全图, 挤压高度维 → [N, C//r, 1, W]
        self.h_conv = nn.Conv2d(channels, reduced, (pooler_resolution, 1))
        # 垂直方向 (着丝粒): kernel 沿宽度覆盖全图, 挤压宽度维 → [N, C//r, H, 1]
        self.v_conv = nn.Conv2d(channels, reduced, (1, pooler_resolution))

        # 逐通道归一化 (GroupNorm num_groups=num_channels ≡ InstanceNorm)
        # + SiLU 激活, 处理融合前的方向特征
        self.norm = nn.GroupNorm(reduced * 2, reduced * 2)
        self.act = nn.SiLU()

        # 融合层: 1×1 卷积, 零初始化确保残差恒等 (morph_emb ≡ 0 initially)
        self.fuse = nn.Conv2d(reduced * 2, channels, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, roi_features: torch.Tensor) -> torch.Tensor:
        """前向传播: roi_features + morph_emb (残差).

        Args:
            roi_features: [N, C, H, W] RoIAlign 输出 (H=W=pooler_resolution)

        Returns:
            与输入同形状的增强 RoI 特征
        """
        # 方向解耦提取
        h_feat = self.h_conv(roi_features)  # [N, C//r, 1, W]
        v_feat = self.v_conv(roi_features)  # [N, C//r, H, 1]

        # 广播回 H×W (方向摘要沿正交维度复制)
        h_feat = h_feat.expand(
            -1, -1, self.pooler_resolution, -1
        )  # [N, C//r, H, W]
        v_feat = v_feat.expand(
            -1, -1, -1, self.pooler_resolution
        )  # [N, C//r, H, W]

        # 融合 → 归一化 → 激活 → 零初始化投影
        morph_emb = self.fuse(
            self.act(self.norm(torch.cat([h_feat, v_feat], dim=1)))
        )

        return roi_features + morph_emb  # 残差
