"""RoI 特征区分度分析 — 挖掘分类瓶颈根因

通过采集 roi_extractor 输出的 RoI 特征, 分析:
- 同组类别相似度: 分类误差来自特征不区分还是 cls_head 不够强?
- 尺度特征范数: 小目标特征是否太弱?
- 误差类型可分性: TP/Cls/Loc 在特征空间是否可分?

使用方法 (推理时, forward hook):
    collector = RoIFeatureCollector()
    # 注册 hook 到 roi_extractor 或 single_head 的 roi_features
    def hook(module, input, output):
        collector.record(roi_features=output, labels=current_labels, scales=current_scales)
    handle = model.bbox_head.roi_extractor.register_forward_hook(hook)
    # ... 推理 ...
    handle.remove()
    analyzer = RoIFeatureAnalyzer(collector)
    report = analyzer.full_report(group_mapping={0: [0,1,2], 1: [3,4,5]})
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class RoIFeatureCollector:
    """采集 RoI 特征及对应标签."""

    def __init__(self):
        self.features: Optional[torch.Tensor] = None  # (N, C)
        self.labels: Optional[torch.Tensor] = None    # (N,)
        self.scales: Optional[torch.Tensor] = None    # (N,) 目标面积
        self.error_types: Optional[list[str]] = None  # (N,) 'TP'/'Cls'/'Loc'/'Both'/'Bkg'/'Miss'

    def record(
        self,
        roi_features: torch.Tensor,  # (N, C) 或 (N, C, H, W)
        labels: torch.Tensor,         # (N,)
        scales: torch.Tensor,         # (N,) 目标面积 (bbox 面积)
        error_types: Optional[list[str]] = None,
    ):
        """记录一批 RoI 特征.

        多次调用会累积 (用于跨 batch 采集).
        """
        # 若是 (N, C, H, W) 做 global avg pool
        if roi_features.dim() == 4:
            roi_features = roi_features.mean(dim=[2, 3])
        feats = roi_features.detach().cpu().float()
        labels = labels.detach().cpu().long()
        scales = scales.detach().cpu().float()

        if self.features is None:
            self.features = feats
            self.labels = labels
            self.scales = scales
            if error_types is not None:
                self.error_types = list(error_types)
        else:
            self.features = torch.cat([self.features, feats], dim=0)
            self.labels = torch.cat([self.labels, labels], dim=0)
            self.scales = torch.cat([self.scales, scales], dim=0)
            if error_types is not None:
                self.error_types.extend(error_types)


class RoIFeatureAnalyzer:
    """分析 RoI 特征区分度.

    Args:
        collector: RoIFeatureCollector (已采集数据)
    """

    def __init__(self, collector: RoIFeatureCollector):
        self.collector = collector
        self.features = collector.features
        self.labels = collector.labels
        self.scales = collector.scales
        self.error_types = collector.error_types

    def _cosine_sim_matrix(self, feats: torch.Tensor) -> torch.Tensor:
        """计算特征矩阵的余弦相似度 (N, N)."""
        normed = F.normalize(feats, dim=-1)
        return normed @ normed.T

    def _class_centroids(self) -> torch.Tensor:
        """计算每个类别的特征中心 (均值)."""
        unique_labels = torch.unique(self.labels)
        n_classes = int(unique_labels.max().item()) + 1
        centroids = torch.zeros(n_classes, self.features.shape[1])
        counts = torch.zeros(n_classes)
        for c in unique_labels:
            mask = self.labels == c
            centroids[c] = self.features[mask].mean(dim=0)
            counts[c] = mask.sum()
        return centroids

    def analyze_same_group_similarity(self, group_mapping: dict) -> dict:
        """分析同组类别的特征相似度.

        Args:
            group_mapping: {group_id: [class_idx, ...]}

        Returns:
            per_group_intra_similarity: 每组的组内平均余弦相似度
            cross_group_similarity: 跨组的平均余弦相似度 (对照)
            same_group_too_similar: 同组是否过于相似 (>0.9 且 > cross_group + 0.1)
        """
        centroids = self._class_centroids()
        sim_matrix = self._cosine_sim_matrix(centroids)

        intra_sims = []
        for gid, class_ids in group_mapping.items():
            if len(class_ids) < 2:
                continue
            # 组内两两相似度
            pair_sims = []
            for i in range(len(class_ids)):
                for j in range(i + 1, len(class_ids)):
                    pair_sims.append(sim_matrix[class_ids[i], class_ids[j]].item())
            if pair_sims:
                intra_sims.append(sum(pair_sims) / len(pair_sims))

        # 跨组相似度 (所有不同组的类对)
        all_classes = [c for cs in group_mapping.values() for c in cs]
        cross_sims = []
        class_to_group = {}
        for gid, cs in group_mapping.items():
            for c in cs:
                class_to_group[c] = gid
        for i in range(len(all_classes)):
            for j in range(i + 1, len(all_classes)):
                if class_to_group[all_classes[i]] != class_to_group[all_classes[j]]:
                    cross_sims.append(sim_matrix[all_classes[i], all_classes[j]].item())
        cross_sim = sum(cross_sims) / len(cross_sims) if cross_sims else 0.0

        avg_intra = sum(intra_sims) / len(intra_sims) if intra_sims else 0.0
        too_similar = avg_intra > 0.9 and avg_intra > cross_sim + 0.1

        return {
            'per_group_intra_similarity': intra_sims,
            'cross_group_similarity': cross_sim,
            'same_group_too_similar': too_similar,
        }

    def analyze_scale_feature_norm(self) -> dict:
        """分析不同尺度目标的特征范数.

        Returns:
            per_scale_norm: 各尺度区间的特征 L2 范数均值
                {'small': float, 'medium': float, 'large': float}
            small_vs_large_norm_ratio: 小/大目标范数比
            scale_affects_feature: 尺度是否显著影响特征 (比 < 0.7)
        """
        norms = self.features.norm(dim=-1)  # (N,)

        # 按面积分三档 (COCO 标准: small<32^2, medium 32^2~96^2, large>96^2)
        # 这里用面积 1024 和 9216 作为阈值
        small_mask = self.scales < 1024
        medium_mask = (self.scales >= 1024) & (self.scales < 9216)
        large_mask = self.scales >= 9216

        per_scale = {}
        for name, mask in [('small', small_mask), ('medium', medium_mask), ('large', large_mask)]:
            if mask.any():
                per_scale[name] = norms[mask].mean().item()
            else:
                per_scale[name] = 0.0

        small_norm = per_scale.get('small', 0.0) or 1e-8
        large_norm = per_scale.get('large', 0.0) or 1e-8
        ratio = small_norm / large_norm if large_norm > 1e-8 else 0.0
        scale_affects = ratio < 0.7

        return {
            'per_scale_norm': per_scale,
            'small_vs_large_norm_ratio': ratio,
            'scale_affects_feature': scale_affects,
        }

    def analyze_error_type_separability(self) -> dict:
        """分析 TP/Cls/Loc 误差类型在特征空间的可分性.

        Returns:
            cls_error_cluster_center: Cls 误差样本的特征中心
            tp_cluster_center: TP 样本的特征中心
            cls_tp_separation: Cls-TP 中心的余弦距离
            loc_tp_separation: Loc-TP 中心的余弦距离
            is_separable: 误差类型是否可分 (分离度 > 0.1)
        """
        if self.error_types is None:
            return {
                'cls_error_cluster_center': None,
                'tp_cluster_center': None,
                'cls_tp_separation': 0.0,
                'loc_tp_separation': 0.0,
                'is_separable': False,
            }

        feats_by_type = {}
        for et in set(self.error_types):
            mask = [e == et for e in self.error_types]
            mask_idx = torch.tensor(mask)
            if mask_idx.any():
                feats_by_type[et] = self.features[mask_idx].mean(dim=0)

        tp_center = feats_by_type.get('TP')
        cls_center = feats_by_type.get('Cls')
        loc_center = feats_by_type.get('Loc')

        def cos_dist(a, b):
            if a is None or b is None:
                return 0.0
            return (1 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))).item()

        cls_tp_sep = cos_dist(cls_center, tp_center)
        loc_tp_sep = cos_dist(loc_center, tp_center)
        is_separable = max(cls_tp_sep, loc_tp_sep) > 0.1

        return {
            'cls_error_cluster_center': cls_center.tolist() if cls_center is not None else None,
            'tp_cluster_center': tp_center.tolist() if tp_center is not None else None,
            'cls_tp_separation': cls_tp_sep,
            'loc_tp_separation': loc_tp_sep,
            'is_separable': is_separable,
        }

    def full_report(self, group_mapping: Optional[dict] = None) -> dict:
        """生成完整 RoI 特征分析报告."""
        result = {
            'n_samples': int(self.features.shape[0]) if self.features is not None else 0,
            'feature_dim': int(self.features.shape[1]) if self.features is not None else 0,
        }

        if group_mapping is not None:
            result['same_group_similarity'] = self.analyze_same_group_similarity(group_mapping)

        result['scale_feature_norm'] = self.analyze_scale_feature_norm()

        if self.error_types is not None:
            result['error_type_separability'] = self.analyze_error_type_separability()

        # 文字分析
        analyses = []
        if 'same_group_similarity' in result:
            sg = result['same_group_similarity']
            if sg['same_group_too_similar']:
                analyses.append(
                    f"同组类别特征过于相似 (组内={sg['per_group_intra_similarity']}, "
                    f"跨组={sg['cross_group_similarity']:.3f}), 分类瓶颈在特征提取阶段, "
                    f"方向 C (ShapeAttention) 应优先"
                )
            else:
                analyses.append(
                    f"同组类别特征可区分 (组内={sg['per_group_intra_similarity']}, "
                    f"跨组={sg['cross_group_similarity']:.3f}), 分类瓶颈在 cls_head"
                )
        if 'scale_feature_norm' in result:
            sn = result['scale_feature_norm']
            if sn['scale_affects_feature']:
                analyses.append(
                    f"小目标特征范数偏弱 (small/large={sn['small_vs_large_norm_ratio']:.3f}), "
                    f"尺度影响特征质量"
                )
        result['analysis'] = ' | '.join(analyses) if analyses else '数据不足'

        return result
