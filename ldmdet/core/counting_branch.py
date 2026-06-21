"""计数分支 (路径 C) — 方向二: 计数先验约束的扩散生成

从 FPN 特征预测染色体总数 c, 为路径 B (拉格朗日约束推理) 提供计数输入.

结构: 多尺度池化 → 拼接 → MLP → softmax over {count_min,...,count_max}
训练时用交叉熵损失, 推理时用 argmax + count_min 得到预测计数.

开关: 通过 head.py 的 counting_branch 参数控制, 默认 None (不启用).
失败时直接删除本文件 + 还原 head.py 即可回滚.
"""

import torch
import torch.nn as nn
from torch import Tensor


class CountingBranch(nn.Module):
    """轻量计数分支.

    从 FPN 各层特征池化后预测染色体总数 c.
    输出范围 [count_min, count_max] (覆盖常见核型 44-48).

    Args:
        feat_channels: FPN 特征通道数 (默认 256)
        num_levels: FPN 层数 (默认 4, P2-P5)
        count_min: 最小计数 (默认 44)
        count_max: 最大计数 (默认 48)
    """

    def __init__(
        self,
        feat_channels: int = 256,
        num_levels: int = 4,
        count_min: int = 44,
        count_max: int = 48,
    ):
        super().__init__()
        self.count_min = count_min
        self.count_max = count_max
        self.num_classes = count_max - count_min + 1

        # 多尺度自适应池化 (每层独立池化到 1x1)
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in range(num_levels)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(feat_channels * num_levels, feat_channels * 2),
            nn.SiLU(),
            nn.Linear(feat_channels * 2, self.num_classes),
        )

    def forward(self, features: list[Tensor]) -> tuple[Tensor, Tensor]:
        """前向传播.

        Args:
            features: [P2, P3, P4, P5] FPN 特征, 每个 [bs, C, H, W]

        Returns:
            logits: [bs, num_count_classes] 计数分类 logits
            pred_count: [bs] 预测计数 (argmax + count_min)
        """
        if len(features) != len(self.pools):
            # 特征层数不匹配时, 取前 N 层或报错
            if len(features) < len(self.pools):
                raise ValueError(
                    f"CountingBranch 期望 {len(self.pools)} 层特征, "
                    f"得到 {len(features)} 层"
                )
            features = features[: len(self.pools)]

        pooled = [pool(f).flatten(1) for pool, f in zip(self.pools, features)]
        concat = torch.cat(pooled, dim=1)
        logits = self.mlp(concat)
        pred_count = logits.argmax(dim=-1) + self.count_min
        return logits, pred_count

    def compute_loss(self, logits: Tensor, gt_count: Tensor) -> Tensor:
        """计算计数分支的交叉熵损失.

        Args:
            logits: [bs, num_count_classes]
            gt_count: [bs] GT 染色体总数 (long)

        Returns:
            scalar loss
        """
        # 将 gt_count 映射到类别索引: c → c - count_min
        target = (gt_count - self.count_min).clamp(0, self.num_classes - 1)
        return nn.functional.cross_entropy(logits, target)
