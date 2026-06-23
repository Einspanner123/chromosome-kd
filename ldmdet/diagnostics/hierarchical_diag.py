"""方向五分层分类诊断 — 验证分层分类和组条件化是否按预期工作.

若方向五废弃, 删除本文件 + tests/unit/test_diagnostics_hierarchical.py 即可清理.

诊断内容:
1. 组分类统计 (准确率、混淆矩阵、组间均衡)
2. 组内分类统计 (各组准确率、组内混淆)
3. 信息论验证 (H(Y), H(G), H(Y|G), 验证 H(Y) ≈ H(G) + H(Y|G))
4. 组条件化效果 (组嵌入利用率、组间特征分离度)

使用方式:
- 在 head.py 的 loss() 中调用 HierarchicalDiagnosticsCallback.update()
- TrainingDiagnosticsHook 通过 diagnostics_callback 自动采集
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────
# 组分类统计
# ──────────────────────────────────────────────

def compute_group_classification_stats(
    group_logits: Tensor,
    group_targets: Tensor,
    valid_mask: Optional[Tensor] = None,
) -> Dict[str, float]:
    """计算组分类统计.

    Args:
        group_logits: [bs, N, num_groups]
        group_targets: [bs, N] 真实组标签
        valid_mask: [bs, N] bool, 可选的有效样本掩码

    Returns:
        统计字典:
        - group_accuracy: 组分类准确率
        - group_confidence_mean: 平均置信度 (softmax 最大值)
        - group_confidence_std
        - group_entropy_mean: 平均预测熵 (越低越确信)
    """
    if valid_mask is None:
        valid_mask = torch.ones_like(group_targets, dtype=torch.bool)

    logits_valid = group_logits[valid_mask]  # [K, num_groups]
    targets_valid = group_targets[valid_mask]  # [K]

    if logits_valid.numel() == 0:
        return {
            'group_accuracy': 0.0,
            'group_confidence_mean': 0.0,
            'group_confidence_std': 0.0,
            'group_entropy_mean': 0.0,
        }

    pred = logits_valid.argmax(dim=-1)
    accuracy = float((pred == targets_valid).float().mean().item())

    probs = F.softmax(logits_valid, dim=-1)
    confidence = probs.max(dim=-1).values
    entropy = -(probs * (probs.clamp_min(1e-10).log())).sum(dim=-1)

    return {
        'group_accuracy': accuracy,
        'group_confidence_mean': float(confidence.mean().item()),
        'group_confidence_std': float(confidence.std().item()) if confidence.numel() > 1 else 0.0,
        'group_entropy_mean': float(entropy.mean().item()),
    }


def compute_group_confusion(
    group_logits: Tensor,
    group_targets: Tensor,
    num_groups: int,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """计算组分类混淆矩阵.

    Args:
        group_logits: [bs, N, num_groups]
        group_targets: [bs, N]
        num_groups: 组数
        valid_mask: [bs, N] bool

    Returns:
        confusion: [num_groups, num_groups] 混淆矩阵
            confusion[i, j] = 真实组 i 被预测为组 j 的次数
    """
    if valid_mask is None:
        valid_mask = torch.ones_like(group_targets, dtype=torch.bool)

    pred = group_logits.argmax(dim=-1)
    confusion = torch.zeros(num_groups, num_groups, dtype=torch.long)

    targets_valid = group_targets[valid_mask]
    pred_valid = pred[valid_mask]

    for t in range(num_groups):
        for p in range(num_groups):
            confusion[t, p] = ((targets_valid == t) & (pred_valid == p)).sum()

    return confusion


# ──────────────────────────────────────────────
# 组内分类统计
# ──────────────────────────────────────────────

def compute_intra_group_classification_stats(
    class_logits_per_group: List[Tensor],
    targets: Tensor,
    group_targets: Tensor,
    class_indices_per_group: List[List[int]],
    valid_mask: Optional[Tensor] = None,
) -> Dict[str, float]:
    """计算组内分类统计.

    Args:
        class_logits_per_group: list of [bs, N, K_g]
        targets: [bs, N] 全局类别标签
        group_targets: [bs, N] 组标签
        class_indices_per_group: 每组包含的全局类别索引
        valid_mask: [bs, N] bool

    Returns:
        统计字典:
        - intra_group_accuracy_mean: 组内分类平均准确率
        - intra_group_accuracy_per_group: 各组准确率 (g0, g1, ...)
        - intra_group_confidence_mean: 平均置信度
    """
    if valid_mask is None:
        valid_mask = torch.ones_like(targets, dtype=torch.bool)

    stats: Dict[str, float] = {}
    accuracies = []
    confidences = []

    for g, (cls_lg, indices) in enumerate(
        zip(class_logits_per_group, class_indices_per_group)
    ):
        g_mask = (group_targets == g) & valid_mask
        if g_mask.sum() == 0:
            continue

        # 局部类别标签
        idx_map = {
            global_idx: local_idx
            for local_idx, global_idx in enumerate(indices)
        }
        local_targets = torch.zeros_like(targets)
        for gi, global_idx in enumerate(indices):
            local_targets[targets == global_idx] = gi

        cls_lg_g = cls_lg[g_mask]
        local_targets_g = local_targets[g_mask]

        pred = cls_lg_g.argmax(dim=-1)
        acc = float((pred == local_targets_g).float().mean().item())
        accuracies.append(acc)
        stats[f'intra_group_accuracy_g{g}'] = acc

        probs = F.softmax(cls_lg_g, dim=-1)
        confidence = probs.max(dim=-1).values
        confidences.append(confidence.mean().item())

    if accuracies:
        stats['intra_group_accuracy_mean'] = sum(accuracies) / len(accuracies)
        stats['intra_group_confidence_mean'] = sum(confidences) / len(confidences)
    else:
        stats['intra_group_accuracy_mean'] = 0.0
        stats['intra_group_confidence_mean'] = 0.0

    return stats


# ──────────────────────────────────────────────
# 信息论验证
# ──────────────────────────────────────────────

def compute_information_stats(
    flat_logits: Tensor,
    group_logits: Tensor,
    class_logits_per_group: List[Tensor],
    valid_mask: Optional[Tensor] = None,
) -> Dict[str, float]:
    """计算信息论统计, 验证 H(Y) ≈ H(G) + H(Y|G).

    Args:
        flat_logits: [bs, N, num_classes] 扁平分类 logits
        group_logits: [bs, N, num_groups] 组分类 logits
        class_logits_per_group: list of [bs, N, K_g]
        valid_mask: [bs, N] bool

    Returns:
        统计字典:
        - H_Y: 扁平分类的预测熵
        - H_G: 组分类的预测熵
        - H_Y_given_G: 组内分类的条件熵 (加权平均)
        - H_decomposition_error: |H_Y - (H_G + H_Y_given_G)| 分解误差
    """
    if valid_mask is None:
        valid_mask = torch.ones(
            flat_logits.shape[:2], dtype=torch.bool, device=flat_logits.device
        )

    # H(Y): 扁平分类熵
    flat_probs = F.softmax(flat_logits[valid_mask], dim=-1)
    H_Y = -(flat_probs * flat_probs.clamp_min(1e-10).log()).sum(dim=-1).mean().item()

    # H(G): 组分类熵
    group_probs = F.softmax(group_logits[valid_mask], dim=-1)
    H_G = -(group_probs * group_probs.clamp_min(1e-10).log()).sum(dim=-1).mean().item()

    # H(Y|G): 组内分类熵的加权平均
    # p(g) * H(Y|G=g)
    H_Y_given_G = 0.0
    for g, cls_lg in enumerate(class_logits_per_group):
        p_g = group_probs[:, g].mean().item()  # 该组的平均概率
        cls_probs = F.softmax(cls_lg[valid_mask], dim=-1)
        H_g = -(cls_probs * cls_probs.clamp_min(1e-10).log()).sum(dim=-1).mean().item()
        H_Y_given_G += p_g * H_g

    # 分解误差
    decomp_error = abs(H_Y - (H_G + H_Y_given_G))

    return {
        'H_Y': float(H_Y),
        'H_G': float(H_G),
        'H_Y_given_G': float(H_Y_given_G),
        'H_decomposition_error': float(decomp_error),
    }


# ──────────────────────────────────────────────
# 组条件化效果统计
# ──────────────────────────────────────────────

def compute_group_conditioning_stats(
    group_embeddings: Tensor,
    group_labels: Tensor,
    num_groups: int = 8,
    valid_mask: Optional[Tensor] = None,
) -> Dict[str, float]:
    """计算组条件化效果统计.

    Args:
        group_embeddings: [bs, N, emb_dim] 组嵌入输出
        group_labels: [bs, N] 组标签
        num_groups: 总组数 (用于计算利用率, 默认 8)
        valid_mask: [bs, N] bool

    Returns:
        统计字典:
        - group_emb_norm_mean: 组嵌入平均 norm
        - group_emb_separation: 组间分离度 (组间距离 / 组内距离)
        - group_emb_utilization: 嵌入利用率 (不同组的嵌入数量 / 总组数)
    """
    if valid_mask is None:
        valid_mask = torch.ones_like(group_labels, dtype=torch.bool)

    emb_valid = group_embeddings[valid_mask]  # [K, emb_dim]
    labels_valid = group_labels[valid_mask]  # [K]

    if emb_valid.numel() == 0:
        return {
            'group_emb_norm_mean': 0.0,
            'group_emb_separation': 0.0,
            'group_emb_utilization': 0.0,
        }

    # 嵌入 norm
    norms = emb_valid.norm(2, dim=-1)
    norm_mean = float(norms.mean().item())

    # 组间分离度: 组间距离 / 组内距离
    unique_groups = labels_valid.unique()
    if unique_groups.numel() < 2:
        return {
            'group_emb_norm_mean': norm_mean,
            'group_emb_separation': 0.0,
            'group_emb_utilization': float(unique_groups.numel()) / max(num_groups, 1),
        }

    # 组中心
    centers = []
    for g in unique_groups:
        mask = labels_valid == g
        if mask.sum() > 0:
            centers.append(emb_valid[mask].mean(dim=0))
    centers = torch.stack(centers)  # [G, emb_dim]

    # 组间距离 (平均)
    inter_dists = torch.cdist(centers, centers)
    # 排除对角线
    mask = ~torch.eye(len(centers), dtype=torch.bool, device=centers.device)
    inter_dist = float(inter_dists[mask].mean().item())

    # 组内距离 (平均)
    intra_dists = []
    for g in unique_groups:
        mask = labels_valid == g
        if mask.sum() > 1:
            g_emb = emb_valid[mask]
            dists = torch.cdist(g_emb, g_emb)
            mask_inner = ~torch.eye(
                g_emb.shape[0], dtype=torch.bool, device=g_emb.device
            )
            intra_dists.append(dists[mask_inner].mean().item())
    intra_dist = sum(intra_dists) / max(len(intra_dists), 1)

    # 分离度 = 组间距离 / 组内距离
    separation = inter_dist / max(intra_dist, 1e-8)

    # 嵌入利用率
    utilization = float(unique_groups.numel()) / max(num_groups, 1)

    return {
        'group_emb_norm_mean': norm_mean,
        'group_emb_separation': float(separation),
        'group_emb_utilization': utilization,
    }


# ──────────────────────────────────────────────
# 诊断回调
# ──────────────────────────────────────────────

class HierarchicalDiagnosticsCallback:
    """分层分类诊断回调, 集成到 TrainingDiagnosticsHook.

    用法:
        1. 在 head.py 的 loss() 中调用:
           callback.update(group_logits, class_logits_per_group, flat_logits,
                          targets, group_targets, valid_mask)
        2. TrainingDiagnosticsHook 通过 diagnostics_callback 自动调用 collect()

    Args:
        interval: 采样间隔 (iter), 默认 100
        class_indices_per_group: 每组包含的全局类别索引 (用于组内分类统计)
    """

    def __init__(
        self,
        interval: int = 100,
        class_indices_per_group: Optional[List[List[int]]] = None,
    ):
        self.interval = interval
        self.class_indices_per_group = class_indices_per_group
        self.last_group_logits: Optional[Tensor] = None
        self.last_class_logits_per_group: Optional[List[Tensor]] = None
        self.last_flat_logits: Optional[Tensor] = None
        self.last_targets: Optional[Tensor] = None
        self.last_group_targets: Optional[Tensor] = None
        self.last_valid_mask: Optional[Tensor] = None
        self.last_group_embeddings: Optional[Tensor] = None
        self.last_group_labels: Optional[Tensor] = None

    def update(
        self,
        group_logits: Tensor,
        class_logits_per_group: List[Tensor],
        flat_logits: Tensor,
        targets: Tensor,
        group_targets: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> None:
        """更新分层分类数据."""
        self.last_group_logits = group_logits.detach().cpu()
        self.last_class_logits_per_group = [
            cl.detach().cpu() for cl in class_logits_per_group
        ]
        self.last_flat_logits = flat_logits.detach().cpu()
        self.last_targets = targets.detach().cpu()
        self.last_group_targets = group_targets.detach().cpu()
        if valid_mask is not None:
            self.last_valid_mask = valid_mask.detach().cpu()

    def update_group_embeddings(
        self,
        group_embeddings: Tensor,
        group_labels: Tensor,
    ) -> None:
        """更新组嵌入数据 (用于组条件化效果统计)."""
        self.last_group_embeddings = group_embeddings.detach().cpu()
        self.last_group_labels = group_labels.detach().cpu()

    def collect(self, step: int) -> Dict[str, float]:
        """采集诊断数据 (由 TrainingDiagnosticsHook 调用).

        Returns:
            诊断数据字典, 若非采样点则返回空字典
        """
        if step % self.interval != 0 or step == 0:
            return {}

        data: Dict[str, float] = {}

        # 组分类统计
        if (self.last_group_logits is not None
                and self.last_group_targets is not None):
            gstats = compute_group_classification_stats(
                self.last_group_logits,
                self.last_group_targets,
                self.last_valid_mask,
            )
            for k, v in gstats.items():
                data[f'hierarchical/{k}'] = v

        # 组内分类统计
        if (self.last_class_logits_per_group is not None
                and self.last_targets is not None
                and self.last_group_targets is not None
                and self.class_indices_per_group is not None):
            istats = compute_intra_group_classification_stats(
                self.last_class_logits_per_group,
                self.last_targets,
                self.last_group_targets,
                self.class_indices_per_group,
                self.last_valid_mask,
            )
            for k, v in istats.items():
                data[f'hierarchical/{k}'] = v

        # 信息论统计
        if (self.last_flat_logits is not None
                and self.last_group_logits is not None
                and self.last_class_logits_per_group is not None):
            istats = compute_information_stats(
                self.last_flat_logits,
                self.last_group_logits,
                self.last_class_logits_per_group,
                self.last_valid_mask,
            )
            for k, v in istats.items():
                data[f'hierarchical/{k}'] = v

        # 组条件化效果统计
        if (self.last_group_embeddings is not None
                and self.last_group_labels is not None):
            cstats = compute_group_conditioning_stats(
                self.last_group_embeddings,
                self.last_group_labels,
            )
            for k, v in cstats.items():
                data[f'hierarchical/{k}'] = v

        return data
