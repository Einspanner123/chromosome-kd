"""分层分类头 — 方向五: 组条件化分层分类

将扁平 24 类分类分解为"先组后类"的分层分类:
1. 组分类: 8 组 (A-G + Sex)
2. 组内分类: 每组 K_g 类

概率分解: p(y|x) = p(g|x) * p(y|g,x)
信息论: H(Y) = H(G) + H(Y|G), 分层后组内分类熵降低 62%

若方向五废弃, 删除本文件 + tests/unit/test_hierarchical_classification.py 即可回滚.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ldmdet.utils.constants import CHROMO_GROUP_OF_CLASS, NUM_CHROMO_GROUPS


class HierarchicalClsHead(nn.Module):
    """分层分类头.

    将 24 类分类分解为:
    1. 组分类: 8 组 (A-G + Sex)
    2. 组内分类: 每组 K_g 类

    概率: p(y|x) = p(g|x) * p(y|g,x)
    组合输出 flat_logits 通过 log-sum-exp, 保持与现有匹配器/损失兼容.

    Args:
        feat_channels: 输入特征维度
        num_classes: 总类别数 (24)
        num_groups: 组数 (8)
        group_of_class: class_idx → group_idx 映射列表
        num_cls_convs: 分类头前的卷积层数 (与 SingleDiffusionDetHead 一致)
    """

    def __init__(
        self,
        feat_channels: int = 256,
        num_classes: int = 24,
        num_groups: int = NUM_CHROMO_GROUPS,
        group_of_class: Optional[List[int]] = None,
        num_cls_convs: int = 1,
    ):
        super().__init__()
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.num_groups = num_groups

        if group_of_class is None:
            group_of_class = CHROMO_GROUP_OF_CLASS
        assert len(group_of_class) == num_classes, (
            f"group_of_class 长度 ({len(group_of_class)}) 应等于 num_classes ({num_classes})"
        )

        # 注册组映射为 buffer (不进 state_dict, 避免迁移问题)
        self.register_buffer(
            'group_of_class',
            torch.tensor(group_of_class, dtype=torch.long),
            persistent=False,
        )

        # 预计算每组包含的 class_idx
        self.class_indices_per_group: List[List[int]] = []
        for g in range(num_groups):
            indices = [i for i, gc in enumerate(group_of_class) if gc == g]
            self.class_indices_per_group.append(indices)
        # 每组的类别数
        self.num_classes_per_group = [len(idx) for idx in self.class_indices_per_group]

        # 组分类头前的卷积 (与 SingleDiffusionDetHead 一致的结构)
        group_layers = []
        for _ in range(num_cls_convs):
            group_layers.append(nn.Sequential(
                nn.Linear(feat_channels, feat_channels, bias=False),
                nn.LayerNorm(feat_channels),
                nn.ReLU(inplace=True),
            ))
        group_layers.append(nn.Linear(feat_channels, num_groups))
        self.group_head = nn.Sequential(*group_layers)

        # 组内分类头: 每组一个独立的线性层
        self.class_heads = nn.ModuleList()
        for g in range(num_groups):
            K_g = self.num_classes_per_group[g]
            cls_layers = []
            for _ in range(num_cls_convs):
                cls_layers.append(nn.Sequential(
                    nn.Linear(feat_channels, feat_channels, bias=False),
                    nn.LayerNorm(feat_channels),
                    nn.ReLU(inplace=True),
                ))
            cls_layers.append(nn.Linear(feat_channels, K_g))
            self.class_heads.append(nn.Sequential(*cls_layers))

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """前向传播.

        Args:
            x: [bs, N, feat_channels] 或 [bs*N, feat_channels]

        Returns:
            dict with:
                group_logits: [bs, N, num_groups] 组分类 logits
                class_logits_per_group: list of [bs, N, K_g] 组内分类 logits
                flat_logits: [bs, N, num_classes] 组合后的扁平 logits (log-sum-exp)
        """
        original_shape = x.shape
        if x.dim() == 3:
            bs, N, C = x.shape
            x_flat = x.reshape(bs * N, C)
        else:
            x_flat = x

        # 组分类
        group_logits = self.group_head(x_flat)  # [bs*N, num_groups]

        # 组内分类
        class_logits_per_group = []
        for g, head in enumerate(self.class_heads):
            class_logits_per_group.append(head(x_flat))  # [bs*N, K_g]

        # 组合为扁平 logits: log p(y) = log p(g) + log p(y|g)
        flat_logits = torch.full(
            (x_flat.shape[0], self.num_classes),
            -10.0,  # 未选中组的类 logit 设为极小
            device=x_flat.device,
            dtype=x_flat.dtype,
        )

        log_p_group = F.log_softmax(group_logits, dim=-1)  # [bs*N, num_groups]
        for g, (cls_lg, indices) in enumerate(
            zip(class_logits_per_group, self.class_indices_per_group)
        ):
            log_p_class_given_group = F.log_softmax(cls_lg, dim=-1)  # [bs*N, K_g]
            log_p_y = log_p_group[:, g:g + 1] + log_p_class_given_group  # [bs*N, K_g]
            # scatter 到 flat_logits
            idx_tensor = torch.tensor(
                indices, device=x_flat.device, dtype=torch.long
            )
            flat_logits.scatter_(
                1, idx_tensor.unsqueeze(0).expand_as(log_p_y), log_p_y
            )

        # reshape 回原形状
        if len(original_shape) == 3:
            bs, N = original_shape[0], original_shape[1]
            group_logits = group_logits.view(bs, N, -1)
            class_logits_per_group = [
                cl.view(bs, N, -1) for cl in class_logits_per_group
            ]
            flat_logits = flat_logits.view(bs, N, -1)

        return {
            'group_logits': group_logits,
            'class_logits_per_group': class_logits_per_group,
            'flat_logits': flat_logits,
        }

    # ──────────────────────────────────────────
    # 损失计算
    # ──────────────────────────────────────────

    def loss(
        self,
        group_logits: Tensor,
        class_logits_per_group: List[Tensor],
        targets: Tensor,
        valid_mask: Optional[Tensor] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        lambda_class: float = 1.0,
    ) -> Dict[str, Tensor]:
        """计算分层分类损失.

        L = L_group + λ * L_class|group
        - L_group: 8 组分类的 Focal Loss
        - L_class|group: 组内分类的 Focal Loss (仅对正确组计算)

        Args:
            group_logits: [bs, N, num_groups]
            class_logits_per_group: list of [bs, N, K_g]
            targets: [bs, N] 全局类别标签 (-1 表示忽略)
            valid_mask: [bs, N] bool, 可选的有效样本掩码
            focal_alpha: Focal Loss alpha
            focal_gamma: Focal Loss gamma
            lambda_class: 组内损失权重

        Returns:
            dict with loss_group, loss_class, loss_total
        """
        bs, N = targets.shape[:2]
        device = targets.device

        # 组标签
        group_targets = self.group_of_class.to(device)[targets.clamp(min=0)]  # [bs, N]
        if valid_mask is None:
            valid_mask = targets >= 0
        else:
            valid_mask = valid_mask & (targets >= 0)

        # 组分类损失 (Focal Loss)
        loss_group = self._focal_loss(
            group_logits, group_targets, valid_mask,
            focal_alpha, focal_gamma,
        )

        # 组内分类损失 (仅对正确组计算)
        loss_class = torch.tensor(0.0, device=device)
        for g, (cls_lg, indices) in enumerate(
            zip(class_logits_per_group, self.class_indices_per_group)
        ):
            g_mask = (group_targets == g) & valid_mask  # [bs, N]
            if g_mask.sum() == 0:
                continue
            # 该组内的局部类别标签 (向量化构建)
            # indices: 该组包含的全局类别列表, local_idx = 在 indices 中的位置
            local_targets = torch.full_like(targets, -1)
            local_idx_map = torch.full(
                (self.num_classes,), -1, device=device, dtype=targets.dtype,
            )
            local_idx_map[indices] = torch.arange(
                len(indices), device=device, dtype=targets.dtype,
            )
            local_targets = local_idx_map[targets.clamp(min=0)]
            local_targets_g = local_targets[g_mask]  # [K]
            cls_lg_g = cls_lg[g_mask]  # [K, K_g]
            loss_class = loss_class + self._focal_loss_flat(
                cls_lg_g, local_targets_g,
                focal_alpha, focal_gamma,
            )

        return {
            'loss_group': loss_group,
            'loss_class': loss_class * lambda_class,
            'loss_total': loss_group + loss_class * lambda_class,
        }

    @staticmethod
    def _focal_loss(
        logits: Tensor,
        targets: Tensor,
        mask: Tensor,
        alpha: float,
        gamma: float,
    ) -> Tensor:
        """Focal Loss (支持 [bs, N, C] 输入).

        Args:
            logits: [bs, N, C]
            targets: [bs, N]
            mask: [bs, N] bool
            alpha, gamma: Focal Loss 参数

        Returns:
            scalar loss
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        logits_valid = logits[mask]  # [K, C]
        targets_valid = targets[mask]  # [K]
        return HierarchicalClsHead._focal_loss_flat(
            logits_valid, targets_valid, alpha, gamma
        )

    @staticmethod
    def _focal_loss_flat(
        logits: Tensor,
        targets: Tensor,
        alpha: float,
        gamma: float,
    ) -> Tensor:
        """Focal Loss (扁平输入).

        Args:
            logits: [K, C]
            targets: [K]
            alpha, gamma: Focal Loss 参数

        Returns:
            scalar loss
        """
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        ce = F.cross_entropy(logits, targets, reduction='none')
        p = torch.exp(-ce)
        loss = alpha * (1 - p) ** gamma * ce
        return loss.mean()

    # ──────────────────────────────────────────
    # 推理辅助
    # ──────────────────────────────────────────

    def predict(
        self,
        group_logits: Tensor,
        class_logits_per_group: List[Tensor],
        flat_logits: Optional[Tensor] = None,
    ) -> Tensor:
        """推理时预测全局类别.

        两种模式:
        1. 若提供 flat_logits: 用全局 MAP (flat_logits.argmax), 数学上等价于
           argmax_y [log p(g(y)) + log p(y|g(y))]
        2. 若不提供 flat_logits: 用贪心分层 (先选最佳组, 再选组内最佳类),
           这是近似, 不一定与全局 MAP 一致

        Args:
            group_logits: [bs, N, num_groups]
            class_logits_per_group: list of [bs, N, K_g]
            flat_logits: [bs, N, num_classes] 可选, 若提供则用全局 MAP

        Returns:
            pred_classes: [bs, N] 预测的全局类别索引
        """
        if flat_logits is not None:
            return flat_logits.argmax(dim=-1)

        # 贪心分层预测 (近似)
        bs, N = group_logits.shape[:2]
        device = group_logits.device

        # 1. 预测组
        pred_group = group_logits.argmax(dim=-1)  # [bs, N]

        # 2. 组内预测
        pred_classes = torch.zeros(bs, N, dtype=torch.long, device=device)
        for g, (cls_lg, indices) in enumerate(
            zip(class_logits_per_group, self.class_indices_per_group)
        ):
            g_mask = pred_group == g  # [bs, N]
            if g_mask.sum() == 0:
                continue
            # 该组的局部预测
            cls_lg_g = cls_lg[g_mask]  # [K, K_g]
            local_pred = cls_lg_g.argmax(dim=-1)  # [K]
            # 映射回全局类别
            idx_tensor = torch.tensor(
                indices, device=device, dtype=torch.long
            )
            global_pred = idx_tensor[local_pred]
            pred_classes[g_mask] = global_pred

        return pred_classes
