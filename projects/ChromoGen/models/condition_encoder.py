"""染色体类别/数量条件编码器

将24类染色体的类别分布编码为条件embedding，注入UNet的cross-attention层。
支持两种模式：
  1. 类别embedding：每类的可学习embedding
  2. 数量embedding：各类数量的正弦位置编码
"""

import math
from typing import Dict, List

import torch
import torch.nn as nn

from ..constants import NUM_CLASSES


class SinusoidalCountEncoding(nn.Module):
    """正弦数量编码：将各类的计数值编码为向量"""

    def __init__(self, num_classes: int, dim: int, max_count: int = 50):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.max_count = max_count

        # 预计算频率
        half_dim = dim // 2
        freq = torch.exp(
            -math.log(max_count)
            * torch.arange(half_dim, dtype=torch.float32)
            / half_dim
        )
        self.register_buffer('freq', freq)

        # 可学习的缩放和偏移
        self.scale = nn.Parameter(torch.ones(1) * 1.0)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            counts: [B, num_classes] 各类染色体数量
        Returns:
            encoding: [B, num_classes, dim]
        """
        # counts: [B, num_classes] -> [B, num_classes, 1]
        counts = counts.float().unsqueeze(-1)  # [B, C, 1]
        # freq: [half_dim] -> [1, 1, half_dim]
        angles = counts * self.freq.view(1, 1, -1) * self.scale + self.bias
        # [B, C, half_dim*2]
        encoding = torch.cat([angles.sin(), angles.cos()], dim=-1)

        # 如果dim是奇数，补零
        if encoding.shape[-1] < self.dim:
            padding = torch.zeros(
                *encoding.shape[:-1],
                self.dim - encoding.shape[-1],
                device=encoding.device,
            )
            encoding = torch.cat([encoding, padding], dim=-1)

        return encoding  # [B, num_classes, dim]


class ChromoConditionEncoder(nn.Module):
    """染色体条件编码器

    将24类染色体的类别+数量信息编码为条件序列，用于UNet的cross-attention。

    输出格式: [B, seq_len, embed_dim]
      - seq_len = num_classes + 1 (24个类别token + 1个全局token)
      - 每个类别token = class_embedding + count_encoding
      - 全局token = 可学习的[CLS] embedding
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = 768,
        max_count: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 类别embedding：每类一个可学习向量
        self.class_embedding = nn.Embedding(num_classes, embed_dim)

        # 数量编码
        self.count_encoding = SinusoidalCountEncoding(
            num_classes, embed_dim, max_count
        )

        # 数量投影层
        self.count_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 全局[CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # LayerNorm + Dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        class_labels: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            class_labels: [B, num_classes] 各类别的索引 (0-23)
            counts: [B, num_classes] 各类别的数量
        Returns:
            condition: [B, seq_len, embed_dim] 条件序列
        """
        B = class_labels.shape[0]

        # 类别embedding: [B, num_classes, embed_dim]
        cls_emb = self.class_embedding(class_labels)

        # 数量编码: [B, num_classes, embed_dim]
        count_enc = self.count_encoding(counts)
        count_enc = self.count_proj(count_enc)

        # 融合: 类别 + 数量
        tokens = cls_emb + count_enc  # [B, num_classes, embed_dim]

        # 拼接全局token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, 25, embed_dim]

        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)

        return tokens

    @staticmethod
    def build_class_labels(
        num_classes: int = NUM_CLASSES,
        device: torch.device = torch.device('cpu'),
    ) -> torch.Tensor:
        """构建标准类别标签 [0, 1, 2, ..., 23]"""
        return torch.arange(num_classes, device=device).unsqueeze(
            0
        )  # [1, num_classes]

    @staticmethod
    def parse_counts_from_coco(
        annotations: List[Dict],
        num_classes: int = NUM_CLASSES,
    ) -> torch.Tensor:
        """从COCO标注中统计各类别数量

        Args:
            annotations: COCO格式的annotation列表
            num_classes: 类别数
        Returns:
            counts: [num_classes] 各类数量
        """
        counts = torch.zeros(num_classes, dtype=torch.long)
        for ann in annotations:
            # COCO category_id 从1开始
            cat_idx = ann['category_id'] - 1
            if 0 <= cat_idx < num_classes:
                counts[cat_idx] += 1
        return counts
