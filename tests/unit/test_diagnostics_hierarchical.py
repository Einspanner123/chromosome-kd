"""方向五分层分类诊断测试

验证 hierarchical_diag.py 中的统计计算正确性.

测试覆盖:
1. 组分类统计 (准确率、置信度、熵)
2. 组混淆矩阵
3. 组内分类统计
4. 信息论验证 (H(Y) ≈ H(G) + H(Y|G))
5. 组条件化效果统计
6. 诊断回调行为
"""

import pytest
import torch
import torch.nn.functional as F

from ldmdet.diagnostics.hierarchical_diag import (
    HierarchicalDiagnosticsCallback,
    compute_group_classification_stats,
    compute_group_confusion,
    compute_intra_group_classification_stats,
    compute_information_stats,
    compute_group_conditioning_stats,
)


# ================================================================
# 组分类统计
# ================================================================

class TestGroupClassificationStats:
    """测试组分类统计."""

    def test_perfect_predictions(self):
        """完美预测: 准确率=1, 熵=0."""
        bs, N, num_groups = 2, 10, 8
        group_logits = torch.full((bs, N, num_groups), -10.0)
        group_targets = torch.randint(0, num_groups, (bs, N))
        # 完美预测: 真实组 logit = 10, 其他 = -10
        for b in range(bs):
            for i in range(N):
                group_logits[b, i, group_targets[b, i]] = 10.0

        stats = compute_group_classification_stats(group_logits, group_targets)
        assert stats['group_accuracy'] == 1.0
        assert stats['group_entropy_mean'] < 0.1

    def test_wrong_predictions(self):
        """全错预测: 准确率=0."""
        bs, N, num_groups = 2, 10, 8
        group_logits = torch.full((bs, N, num_groups), -10.0)
        group_targets = torch.randint(0, num_groups, (bs, N))
        # 全错预测: 真实组 logit = -10, 其他组 = 10
        for b in range(bs):
            for i in range(N):
                wrong_g = (group_targets[b, i] + 1) % num_groups
                group_logits[b, i, wrong_g] = 10.0

        stats = compute_group_classification_stats(group_logits, group_targets)
        assert stats['group_accuracy'] == 0.0

    def test_uniform_logits_high_entropy(self):
        """均匀 logits: 熵应接近 log(num_groups)."""
        bs, N, num_groups = 2, 10, 8
        group_logits = torch.zeros(bs, N, num_groups)
        group_targets = torch.randint(0, num_groups, (bs, N))
        stats = compute_group_classification_stats(group_logits, group_targets)
        import math
        expected_entropy = math.log(num_groups)
        assert abs(stats['group_entropy_mean'] - expected_entropy) < 0.1

    def test_confidence_range(self):
        """置信度应在 [1/num_groups, 1] 范围内."""
        bs, N, num_groups = 2, 10, 8
        group_logits = torch.randn(bs, N, num_groups)
        group_targets = torch.randint(0, num_groups, (bs, N))
        stats = compute_group_classification_stats(group_logits, group_targets)
        assert 1.0 / num_groups - 1e-5 <= stats['group_confidence_mean'] <= 1.0 + 1e-5

    def test_with_valid_mask(self):
        """带 valid_mask 的统计."""
        bs, N, num_groups = 2, 10, 8
        group_logits = torch.randn(bs, N, num_groups)
        group_targets = torch.randint(0, num_groups, (bs, N))
        valid_mask = torch.zeros(bs, N, dtype=torch.bool)
        valid_mask[0, :5] = True
        stats = compute_group_classification_stats(
            group_logits, group_targets, valid_mask
        )
        assert 'group_accuracy' in stats

    def test_all_invalid(self):
        """全部无效: 返回 0."""
        bs, N, num_groups = 2, 10, 8
        group_logits = torch.randn(bs, N, num_groups)
        group_targets = torch.randint(0, num_groups, (bs, N))
        valid_mask = torch.zeros(bs, N, dtype=torch.bool)
        stats = compute_group_classification_stats(
            group_logits, group_targets, valid_mask
        )
        assert stats['group_accuracy'] == 0.0


# ================================================================
# 组混淆矩阵
# ================================================================

class TestGroupConfusion:
    """测试组混淆矩阵."""

    def test_perfect_predictions_diagonal(self):
        """完美预测: 混淆矩阵应为对角阵."""
        bs, N, num_groups = 2, 100, 8
        group_targets = torch.randint(0, num_groups, (bs, N))
        group_logits = torch.full((bs, N, num_groups), -10.0)
        for b in range(bs):
            for i in range(N):
                group_logits[b, i, group_targets[b, i]] = 10.0

        confusion = compute_group_confusion(
            group_logits, group_targets, num_groups
        )
        # 对角线应非零, 非对角线应为 0
        for i in range(num_groups):
            for j in range(num_groups):
                if i == j:
                    pass  # 对角线可能非零
                else:
                    assert confusion[i, j] == 0

    def test_confusion_shape(self):
        """混淆矩阵形状应为 [num_groups, num_groups]."""
        bs, N, num_groups = 2, 10, 8
        group_logits = torch.randn(bs, N, num_groups)
        group_targets = torch.randint(0, num_groups, (bs, N))
        confusion = compute_group_confusion(
            group_logits, group_targets, num_groups
        )
        assert confusion.shape == (num_groups, num_groups)

    def test_confusion_sum_equals_valid_count(self):
        """混淆矩阵总和应等于有效样本数."""
        bs, N, num_groups = 2, 100, 8
        group_logits = torch.randn(bs, N, num_groups)
        group_targets = torch.randint(0, num_groups, (bs, N))
        confusion = compute_group_confusion(
            group_logits, group_targets, num_groups
        )
        assert confusion.sum().item() == bs * N


# ================================================================
# 组内分类统计
# ================================================================

class TestIntraGroupClassificationStats:
    """测试组内分类统计."""

    def test_basic_stats(self):
        """基本统计应返回."""
        bs, N = 2, 100
        num_groups = 8
        # 模拟 class_logits_per_group
        class_indices_per_group = [
            [0, 1, 2], [3, 4], [5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14], [15, 16, 17], [18, 19], [20, 21], [22, 23],
        ]
        class_logits_per_group = []
        for indices in class_indices_per_group:
            class_logits_per_group.append(torch.randn(bs, N, len(indices)))

        targets = torch.randint(0, 24, (bs, N))
        group_targets = torch.randint(0, num_groups, (bs, N))

        stats = compute_intra_group_classification_stats(
            class_logits_per_group, targets, group_targets,
            class_indices_per_group,
        )
        assert 'intra_group_accuracy_mean' in stats
        assert 'intra_group_confidence_mean' in stats
        # 每组准确率
        for g in range(num_groups):
            assert f'intra_group_accuracy_g{g}' in stats

    def test_perfect_intra_group(self):
        """完美组内分类: 准确率=1."""
        bs, N = 1, 100
        class_indices_per_group = [[0, 1, 2], [3, 4]]
        # 所有样本属于组 0, 类别 0
        targets = torch.zeros(bs, N, dtype=torch.long)
        group_targets = torch.zeros(bs, N, dtype=torch.long)

        class_logits_per_group = [
            torch.full((bs, N, 3), -10.0),
            torch.randn(bs, N, 2),
        ]
        # 组 0 完美预测类别 0
        class_logits_per_group[0][:, :, 0] = 10.0

        stats = compute_intra_group_classification_stats(
            class_logits_per_group, targets, group_targets,
            class_indices_per_group,
        )
        assert stats['intra_group_accuracy_g0'] == 1.0

    def test_empty_group(self):
        """空组应跳过."""
        bs, N = 1, 10
        class_indices_per_group = [[0, 1], [2, 3]]
        class_logits_per_group = [
            torch.randn(bs, N, 2),
            torch.randn(bs, N, 2),
        ]
        targets = torch.zeros(bs, N, dtype=torch.long)  # 全部组 0
        group_targets = torch.zeros(bs, N, dtype=torch.long)

        stats = compute_intra_group_classification_stats(
            class_logits_per_group, targets, group_targets,
            class_indices_per_group,
        )
        # 组 0 有样本, 组 1 无样本
        assert 'intra_group_accuracy_g0' in stats
        assert 'intra_group_accuracy_g1' not in stats


# ================================================================
# 信息论验证
# ================================================================

class TestInformationStats:
    """测试信息论统计."""

    def test_basic_stats(self):
        """基本统计应返回."""
        bs, N = 2, 100
        num_groups = 8
        num_classes = 24
        flat_logits = torch.randn(bs, N, num_classes)
        group_logits = torch.randn(bs, N, num_groups)
        class_logits_per_group = [
            torch.randn(bs, N, 3),
            torch.randn(bs, N, 2),
            torch.randn(bs, N, 7),
            torch.randn(bs, N, 3),
            torch.randn(bs, N, 3),
            torch.randn(bs, N, 2),
            torch.randn(bs, N, 2),
            torch.randn(bs, N, 2),
        ]
        stats = compute_information_stats(
            flat_logits, group_logits, class_logits_per_group
        )
        assert 'H_Y' in stats
        assert 'H_G' in stats
        assert 'H_Y_given_G' in stats
        assert 'H_decomposition_error' in stats

    def test_decomposition_error_small(self):
        """当 flat_logits 由 group + class 精确分解时, 分解误差应小."""
        bs, N = 2, 100
        num_groups = 8
        # 构造精确分解的 flat_logits
        group_logits = torch.randn(bs, N, num_groups)
        class_indices_per_group = [
            [0, 1, 2], [3, 4], [5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14], [15, 16, 17], [18, 19], [20, 21], [22, 23],
        ]
        class_logits_per_group = []
        for indices in class_indices_per_group:
            class_logits_per_group.append(torch.randn(bs, N, len(indices)))

        # 手动构造 flat_logits = log p(g) + log p(y|g)
        log_p_group = F.log_softmax(group_logits, dim=-1)
        flat_logits = torch.full((bs, N, 24), -1e10)
        for g, (cls_lg, indices) in enumerate(
            zip(class_logits_per_group, class_indices_per_group)
        ):
            log_p_class = F.log_softmax(cls_lg, dim=-1)
            log_p_y = log_p_group[:, :, g:g + 1] + log_p_class
            idx_tensor = torch.tensor(indices)
            flat_logits.scatter_(
                2, idx_tensor.unsqueeze(0).unsqueeze(0).expand(bs, N, -1), log_p_y
            )

        stats = compute_information_stats(
            flat_logits, group_logits, class_logits_per_group
        )
        # 分解误差应较小
        assert stats['H_decomposition_error'] < 0.5

    def test_H_Y_nonneg(self):
        """H(Y) 应非负."""
        bs, N = 2, 10
        flat_logits = torch.randn(bs, N, 24)
        group_logits = torch.randn(bs, N, 8)
        class_logits_per_group = [torch.randn(bs, N, 3) for _ in range(8)]
        stats = compute_information_stats(
            flat_logits, group_logits, class_logits_per_group
        )
        assert stats['H_Y'] >= 0
        assert stats['H_G'] >= 0
        assert stats['H_Y_given_G'] >= 0


# ================================================================
# 组条件化效果统计
# ================================================================

class TestGroupConditioningStats:
    """测试组条件化效果统计."""

    def test_basic_stats(self):
        """基本统计应返回."""
        bs, N = 2, 100
        emb_dim = 32
        group_embeddings = torch.randn(bs, N, emb_dim)
        group_labels = torch.randint(0, 8, (bs, N))
        stats = compute_group_conditioning_stats(group_embeddings, group_labels)
        assert 'group_emb_norm_mean' in stats
        assert 'group_emb_separation' in stats
        assert 'group_emb_utilization' in stats

    def test_separated_groups_high_separation(self):
        """分离的组嵌入: 分离度应较高."""
        emb_dim = 32
        num_groups = 4
        samples_per_group = 50
        # 每组用不同的中心
        centers = torch.randn(num_groups, emb_dim) * 10  # 远距离中心
        group_embeddings = []
        group_labels = []
        for g in range(num_groups):
            emb = centers[g] + torch.randn(samples_per_group, emb_dim) * 0.1
            group_embeddings.append(emb)
            group_labels.append(torch.full((samples_per_group,), g))
        group_embeddings = torch.cat(group_embeddings).unsqueeze(0)
        group_labels = torch.cat(group_labels).unsqueeze(0)

        stats = compute_group_conditioning_stats(group_embeddings, group_labels)
        # 组间距离应远大于组内距离
        assert stats['group_emb_separation'] > 1.0

    def test_overlapping_groups_low_separation(self):
        """重叠的组嵌入: 分离度应较低."""
        emb_dim = 32
        num_groups = 4
        samples_per_group = 50
        # 所有组用相同中心
        center = torch.randn(emb_dim)
        group_embeddings = []
        group_labels = []
        for g in range(num_groups):
            emb = center + torch.randn(samples_per_group, emb_dim) * 1.0
            group_embeddings.append(emb)
            group_labels.append(torch.full((samples_per_group,), g))
        group_embeddings = torch.cat(group_embeddings).unsqueeze(0)
        group_labels = torch.cat(group_labels).unsqueeze(0)

        stats = compute_group_conditioning_stats(group_embeddings, group_labels)
        # 分离度应接近 1 (组间距离 ≈ 组内距离)
        assert stats['group_emb_separation'] < 2.0

    def test_single_group(self):
        """单组: 分离度应为 0 (无组间距离)."""
        bs, N = 1, 50
        emb_dim = 32
        group_embeddings = torch.randn(bs, N, emb_dim)
        group_labels = torch.zeros(bs, N, dtype=torch.long)
        stats = compute_group_conditioning_stats(group_embeddings, group_labels)
        assert stats['group_emb_separation'] == 0.0
        assert stats['group_emb_utilization'] == 1.0 / 8.0

    def test_empty_input(self):
        """空输入: 返回 0."""
        bs, N = 2, 10
        emb_dim = 32
        group_embeddings = torch.randn(bs, N, emb_dim)
        group_labels = torch.randint(0, 8, (bs, N))
        valid_mask = torch.zeros(bs, N, dtype=torch.bool)
        stats = compute_group_conditioning_stats(
            group_embeddings, group_labels, num_groups=8, valid_mask=valid_mask,
        )
        assert stats['group_emb_norm_mean'] == 0.0


# ================================================================
# 诊断回调
# ================================================================

class TestHierarchicalDiagnosticsCallback:
    """测试分层分类诊断回调."""

    def test_init_defaults(self):
        """默认初始化."""
        callback = HierarchicalDiagnosticsCallback()
        assert callback.interval == 100
        assert callback.last_group_logits is None

    def test_update_stores_data(self):
        """update 应存储数据."""
        callback = HierarchicalDiagnosticsCallback()
        bs, N = 2, 10
        group_logits = torch.randn(bs, N, 8)
        class_logits_per_group = [torch.randn(bs, N, 3) for _ in range(8)]
        flat_logits = torch.randn(bs, N, 24)
        targets = torch.randint(0, 24, (bs, N))
        group_targets = torch.randint(0, 8, (bs, N))

        callback.update(
            group_logits, class_logits_per_group, flat_logits,
            targets, group_targets,
        )
        assert callback.last_group_logits is not None
        assert callback.last_class_logits_per_group is not None
        assert callback.last_flat_logits is not None
        assert callback.last_targets is not None
        assert callback.last_group_targets is not None

    def test_collect_non_sampling_step(self):
        """非采样点应返回空字典."""
        callback = HierarchicalDiagnosticsCallback(interval=100)
        # 没有调用 update, collect 应返回空
        data = callback.collect(step=50)
        assert data == {}

    def test_collect_sampling_step(self):
        """采样点应返回诊断数据."""
        callback = HierarchicalDiagnosticsCallback(interval=100)
        bs, N = 2, 10
        group_logits = torch.randn(bs, N, 8)
        class_logits_per_group = [torch.randn(bs, N, 3) for _ in range(8)]
        flat_logits = torch.randn(bs, N, 24)
        targets = torch.randint(0, 24, (bs, N))
        group_targets = torch.randint(0, 8, (bs, N))
        callback.update(
            group_logits, class_logits_per_group, flat_logits,
            targets, group_targets,
        )

        data = callback.collect(step=100)
        assert 'hierarchical/group_accuracy' in data
        assert 'hierarchical/H_Y' in data

    def test_collect_zero_step(self):
        """step=0 应返回空."""
        callback = HierarchicalDiagnosticsCallback(interval=100)
        data = callback.collect(step=0)
        assert data == {}

    def test_collect_with_class_indices(self):
        """带 class_indices_per_group 应返回组内分类统计."""
        class_indices_per_group = [
            [0, 1, 2], [3, 4], [5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14], [15, 16, 17], [18, 19], [20, 21], [22, 23],
        ]
        callback = HierarchicalDiagnosticsCallback(
            interval=100,
            class_indices_per_group=class_indices_per_group,
        )
        bs, N = 2, 10
        group_logits = torch.randn(bs, N, 8)
        class_logits_per_group = [
            torch.randn(bs, N, len(indices))
            for indices in class_indices_per_group
        ]
        flat_logits = torch.randn(bs, N, 24)
        targets = torch.randint(0, 24, (bs, N))
        group_targets = torch.randint(0, 8, (bs, N))
        callback.update(
            group_logits, class_logits_per_group, flat_logits,
            targets, group_targets,
        )

        data = callback.collect(step=100)
        assert 'hierarchical/intra_group_accuracy_mean' in data
        assert 'hierarchical/intra_group_accuracy_g0' in data

    def test_update_group_embeddings(self):
        """update_group_embeddings 应存储组嵌入数据."""
        callback = HierarchicalDiagnosticsCallback()
        bs, N = 2, 10
        group_embeddings = torch.randn(bs, N, 32)
        group_labels = torch.randint(0, 8, (bs, N))
        callback.update_group_embeddings(group_embeddings, group_labels)
        assert callback.last_group_embeddings is not None
        assert callback.last_group_labels is not None

        data = callback.collect(step=100)
        assert 'hierarchical/group_emb_norm_mean' in data
        assert 'hierarchical/group_emb_separation' in data
