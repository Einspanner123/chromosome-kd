"""cls/reg 头输出分布分析 — 挖掘头内在问题

通过采集 cls_head 和 reg_head 的输入输出, 分析:
- cls_logits 分布: 类别坍塌? 易混淆类对?
- reg_delta 分布: 回归是否保守 (delta 太小)?
- 梯度比例: cls vs reg 哪个头主导?
- 困难样本: 同组混淆样本是否过度自信?

使用方法 (推理时, forward hook):
    collector = HeadOutputCollector()
    # hook single_head 的 cls_head 和 reg_head
    def hook_cls(module, input, output):
        collector.record_cls(output)
    def hook_reg(module, input, output):
        collector.record_reg(output)
    h1 = head.cls_head.register_forward_hook(hook_cls)
    h2 = head.reg_head.register_forward_hook(hook_reg)
    # ... 推理 ...
    collector.set_labels(current_labels)
    analyzer = HeadOutputAnalyzer(collector)
    report = analyzer.full_report()
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class HeadOutputCollector:
    """采集 cls_head 和 reg_head 的输入输出."""

    def __init__(self):
        self.fc_features: Optional[torch.Tensor] = None  # (N, C)
        self.cls_logits: Optional[torch.Tensor] = None   # (N, n_classes)
        self.reg_deltas: Optional[torch.Tensor] = None   # (N, 4)
        self.labels: Optional[torch.Tensor] = None       # (N,)
        self.cls_grad_norm: Optional[torch.Tensor] = None  # (N,) 每样本 cls 梯度范数
        self.reg_grad_norm: Optional[torch.Tensor] = None  # (N,) 每样本 reg 梯度范数

    def record(
        self,
        fc_feature: torch.Tensor,
        cls_logits: torch.Tensor,
        reg_deltas: torch.Tensor,
        labels: torch.Tensor,
    ):
        """记录一批头输出 (累积)."""
        fc = fc_feature.detach().cpu().float()
        cls = cls_logits.detach().cpu().float()
        reg = reg_deltas.detach().cpu().float()
        lab = labels.detach().cpu().long()

        if self.fc_features is None:
            self.fc_features = fc
            self.cls_logits = cls
            self.reg_deltas = reg
            self.labels = lab
        else:
            self.fc_features = torch.cat([self.fc_features, fc], dim=0)
            self.cls_logits = torch.cat([self.cls_logits, cls], dim=0)
            self.reg_deltas = torch.cat([self.reg_deltas, reg], dim=0)
            self.labels = torch.cat([self.labels, lab], dim=0)


class HeadOutputAnalyzer:
    """分析 cls/reg 头输出分布.

    Args:
        collector: HeadOutputCollector (已采集数据)
    """

    def __init__(self, collector: HeadOutputCollector):
        self.collector = collector
        self.cls_logits = collector.cls_logits
        self.reg_deltas = collector.reg_deltas
        self.labels = collector.labels
        self.n_classes = int(self.cls_logits.shape[1]) if self.cls_logits is not None else 0

    def analyze_cls_distribution(self) -> dict:
        """分析 cls_logits 的 per-class 分布.

        Returns:
            per_class_mean_logit: 每类平均 logit (n_classes,)
            per_class_std_logit: 每类 logit 标准差
            class_collapse: 是否有类别坍塌 (某类 logit 恒低)
            collapsed_classes: 坍塌的类别索引列表
            confusing_pairs: 易混淆类对 [(i, j, correlation), ...] (相关 >0.8)
        """
        # per-class 统计 (对每个类别列统计)
        per_class_mean = self.cls_logits.mean(dim=0)  # (C,)
        per_class_std = self.cls_logits.std(dim=0)    # (C,)

        # 类别坍塌: 某类 logit 均值显著低于全局均值 (>2 std)
        global_mean = per_class_mean.mean().item()
        global_std = per_class_mean.std().item()
        collapsed = (per_class_mean < global_mean - 2 * global_std).tolist()
        collapsed_classes = [i for i, c in enumerate(collapsed) if c]

        # 易混淆类对: logits 列间相关系数
        # 标准化后计算相关矩阵
        normed = (self.cls_logits - self.cls_logits.mean(dim=0)) / (self.cls_logits.std(dim=0) + 1e-8)
        corr_matrix = (normed.T @ normed) / normed.shape[0]  # (C, C)
        confusing_pairs = []
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                c = corr_matrix[i, j].item()
                if abs(c) > 0.8:
                    confusing_pairs.append((i, j, c))
        # 按相关度排序
        confusing_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return {
            'per_class_mean_logit': per_class_mean.tolist(),
            'per_class_std_logit': per_class_std.tolist(),
            'class_collapse': len(collapsed_classes) > 0,
            'collapsed_classes': collapsed_classes,
            'confusing_pairs': confusing_pairs[:10],  # top 10
        }

    def analyze_reg_distribution(self) -> dict:
        """分析 reg_delta 的幅度分布.

        Returns:
            delta_mean: delta 整体均值
            delta_std: delta 整体标准差
            per_dim_mean: 每维 delta 均值 (4 维)
            per_dim_std: 每维 delta 标准差
            is_conservative: 回归是否保守 (|delta_mean| < 0.1)
        """
        delta = self.reg_deltas  # (N, 4)
        delta_mean = delta.mean().item()
        delta_std = delta.std().item()
        per_dim_mean = delta.mean(dim=0).tolist()
        per_dim_std = delta.std(dim=0).tolist()

        is_conservative = abs(delta_mean) < 0.1

        return {
            'delta_mean': delta_mean,
            'delta_std': delta_std,
            'per_dim_mean': per_dim_mean,
            'per_dim_std': per_dim_std,
            'is_conservative': is_conservative,
        }

    def analyze_gradient_ratio(self) -> dict:
        """分析 cls vs reg 头的梯度比例.

        Returns:
            mean_cls_grad: cls 平均梯度范数
            mean_reg_grad: reg 平均梯度范数
            cls_reg_ratio: cls/reg 梯度比
            dominant_head: 主导头 ('cls' / 'reg' / 'balanced')
        """
        if self.collector.cls_grad_norm is None or self.collector.reg_grad_norm is None:
            return {
                'mean_cls_grad': 0.0,
                'mean_reg_grad': 0.0,
                'cls_reg_ratio': 0.0,
                'dominant_head': 'unknown',
            }

        mean_cls = self.collector.cls_grad_norm.mean().item()
        mean_reg = self.collector.reg_grad_norm.mean().item()
        ratio = mean_cls / (mean_reg + 1e-8)

        if ratio > 2.0:
            dominant = 'cls'
        elif ratio < 0.5:
            dominant = 'reg'
        else:
            dominant = 'balanced'

        return {
            'mean_cls_grad': mean_cls,
            'mean_reg_grad': mean_reg,
            'cls_reg_ratio': ratio,
            'dominant_head': dominant,
        }

    def analyze_hard_samples(self, group_mapping: dict) -> dict:
        """分析困难样本 (同组混淆) 的 logits 特征.

        Args:
            group_mapping: {group_id: [class_idx, ...]}

        Returns:
            hard_sample_count: 困难样本数 (同组内 top1≠GT 且 top2∈同组)
            hard_sample_confidence: 困难样本 top1 置信度均值
            hard_sample_top2_gap: 困难样本 top1-top2 间距均值
            overconfident: 困难样本是否过度自信 (置信度 > 0.7)
        """
        # 构建 class -> group 映射
        class_to_group = {}
        for gid, cs in group_mapping.items():
            for c in cs:
                class_to_group[c] = gid

        probs = F.softmax(self.cls_logits, dim=-1)  # (N, C)
        top2_probs, top2_idx = probs.topk(2, dim=-1)  # (N, 2)

        hard_mask = torch.zeros(self.labels.shape[0], dtype=torch.bool)
        for i in range(self.labels.shape[0]):
            gt = int(self.labels[i].item())
            top1 = int(top2_idx[i, 0].item())
            top2 = int(top2_idx[i, 1].item())
            # 困难: top1 错且 top1, top2 同组 (或 top2 是 GT)
            if gt in class_to_group and top1 != gt:
                if top1 in class_to_group and class_to_group.get(top1) == class_to_group[gt]:
                    hard_mask[i] = True
                elif top2 == gt:
                    hard_mask[i] = True

        n_hard = int(hard_mask.sum().item())
        if n_hard == 0:
            return {
                'hard_sample_count': 0,
                'hard_sample_confidence': 0.0,
                'hard_sample_top2_gap': 0.0,
                'overconfident': False,
            }

        hard_conf = top2_probs[hard_mask, 0].mean().item()
        hard_gap = (top2_probs[hard_mask, 0] - top2_probs[hard_mask, 1]).mean().item()
        overconfident = hard_conf > 0.7

        return {
            'hard_sample_count': n_hard,
            'hard_sample_confidence': hard_conf,
            'hard_sample_top2_gap': hard_gap,
            'overconfident': overconfident,
        }

    def full_report(self, group_mapping: Optional[dict] = None) -> dict:
        """生成完整头输出分析报告."""
        result = {
            'n_samples': int(self.cls_logits.shape[0]) if self.cls_logits is not None else 0,
            'n_classes': self.n_classes,
        }

        result['cls_distribution'] = self.analyze_cls_distribution()
        result['reg_distribution'] = self.analyze_reg_distribution()

        if self.collector.cls_grad_norm is not None:
            result['gradient_ratio'] = self.analyze_gradient_ratio()

        if group_mapping is not None:
            result['hard_samples'] = self.analyze_hard_samples(group_mapping)

        # 文字分析
        analyses = []
        cls_dist = result['cls_distribution']
        if cls_dist['class_collapse']:
            analyses.append(
                f"检测到类别坍塌: {cls_dist['collapsed_classes']}, "
                f"这些类几乎不被预测"
            )
        if cls_dist['confusing_pairs']:
            top_pair = cls_dist['confusing_pairs'][0]
            analyses.append(
                f"最易混淆类对: ({top_pair[0]}, {top_pair[1]}) 相关={top_pair[2]:.3f}"
            )
        reg_dist = result['reg_distribution']
        if reg_dist['is_conservative']:
            analyses.append(
                f"回归保守 (delta_mean={reg_dist['delta_mean']:.4f}), "
                f"可能导致高 IoU 定位不足"
            )
        if 'hard_samples' in result:
            hs = result['hard_samples']
            if hs['hard_sample_count'] > 0 and hs['overconfident']:
                analyses.append(
                    f"困难样本过度自信 (置信度={hs['hard_sample_confidence']:.3f}), "
                    f"应增强困难样本损失 (方向 C ContrastiveLoss)"
                )
        result['analysis'] = ' | '.join(analyses) if analyses else '无明显问题'

        return result
