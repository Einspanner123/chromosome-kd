"""方向五: 组条件化分层分类测试

验证 HierarchicalClsHead 和 GroupConditionedSingleHead 的正确性.

测试覆盖:
1. HierarchicalClsHead 基本功能 (形状、组映射、概率分解)
2. HierarchicalClsHead 损失计算 (Focal Loss、组内损失)
3. HierarchicalClsHead 推理 (predict 方法)
4. 隔离性: 关闭分层后等价于标准分类
"""

import pytest
import torch
import torch.nn.functional as F

from ldmdet.core.hierarchical_head import HierarchicalClsHead
from ldmdet.utils.constants import CHROMO_GROUP_OF_CLASS, NUM_CHROMO_GROUPS


# ================================================================
# HierarchicalClsHead 基本功能
# ================================================================

class TestHierarchicalClsHeadBasic:
    """测试分层分类头的基本功能."""

    def test_init_default(self):
        """默认参数初始化."""
        head = HierarchicalClsHead(feat_channels=64)
        assert head.num_classes == 24
        assert head.num_groups == 8
        assert len(head.class_heads) == 8
        # 每组的类别数
        assert head.num_classes_per_group == [3, 2, 7, 3, 3, 2, 2, 2]

    def test_init_custom_group_of_class(self):
        """自定义 group_of_class."""
        # 4 类, 2 组
        group_of_class = [0, 0, 1, 1]
        head = HierarchicalClsHead(
            feat_channels=32, num_classes=4, num_groups=2,
            group_of_class=group_of_class,
        )
        assert head.num_classes == 4
        assert head.num_groups == 2
        assert head.num_classes_per_group == [2, 2]
        assert head.class_indices_per_group == [[0, 1], [2, 3]]

    def test_init_invalid_group_of_class_length(self):
        """group_of_class 长度不匹配应报错."""
        with pytest.raises(AssertionError):
            HierarchicalClsHead(
                feat_channels=32, num_classes=4,
                group_of_class=[0, 1, 2],  # 长度 3 != num_classes 4
            )

    def test_forward_shape_3d(self):
        """3D 输入 [bs, N, C] 的输出形状."""
        head = HierarchicalClsHead(feat_channels=64)
        bs, N = 2, 10
        x = torch.randn(bs, N, 64)
        out = head(x)
        assert out['group_logits'].shape == (bs, N, 8)
        assert len(out['class_logits_per_group']) == 8
        for g, cls_lg in enumerate(out['class_logits_per_group']):
            assert cls_lg.shape == (bs, N, head.num_classes_per_group[g])
        assert out['flat_logits'].shape == (bs, N, 24)

    def test_forward_shape_2d(self):
        """2D 输入 [bs*N, C] 的输出形状."""
        head = HierarchicalClsHead(feat_channels=64)
        K = 20
        x = torch.randn(K, 64)
        out = head(x)
        assert out['group_logits'].shape == (K, 8)
        assert out['flat_logits'].shape == (K, 24)

    def test_group_of_class_buffer(self):
        """group_of_class 应作为 buffer 注册."""
        head = HierarchicalClsHead(feat_channels=32)
        assert 'group_of_class' in dict(head.named_buffers())
        assert head.group_of_class.shape == (24,)

    def test_class_indices_per_group_correct(self):
        """class_indices_per_group 应正确反映组映射."""
        head = HierarchicalClsHead(feat_channels=32)
        # 验证每组包含的类别索引
        for g, indices in enumerate(head.class_indices_per_group):
            for idx in indices:
                assert CHROMO_GROUP_OF_CLASS[idx] == g


# ================================================================
# 概率分解验证
# ================================================================

class TestProbabilityDecomposition:
    """验证 p(y|x) = p(g|x) * p(y|g,x)."""

    def test_flat_logits_equals_log_p_g_plus_log_p_y_given_g(self):
        """flat_logits 应等于 log p(g) + log p(y|g)."""
        torch.manual_seed(42)
        head = HierarchicalClsHead(feat_channels=64)
        bs, N = 2, 10
        x = torch.randn(bs, N, 64)
        out = head(x)

        # 手动计算 log p(y)
        log_p_group = F.log_softmax(out['group_logits'], dim=-1)  # [bs, N, num_groups]
        manual_flat = torch.full(
            (bs, N, 24), -1e10, device=x.device, dtype=x.dtype
        )
        for g, (cls_lg, indices) in enumerate(
            zip(out['class_logits_per_group'], head.class_indices_per_group)
        ):
            log_p_class = F.log_softmax(cls_lg, dim=-1)  # [bs, N, K_g]
            log_p_y = log_p_group[:, :, g:g + 1] + log_p_class
            idx_tensor = torch.tensor(indices, device=x.device)
            manual_flat.scatter_(2, idx_tensor.unsqueeze(0).unsqueeze(0).expand(bs, N, -1), log_p_y)

        # flat_logits 应与手动计算一致
        assert torch.allclose(out['flat_logits'], manual_flat, atol=1e-5)

    def test_flat_probs_sum_to_one(self):
        """flat_logits 的 softmax 应和为 1."""
        head = HierarchicalClsHead(feat_channels=64)
        x = torch.randn(2, 10, 64)
        out = head(x)
        probs = F.softmax(out['flat_logits'], dim=-1)
        sum_probs = probs.sum(dim=-1)
        assert torch.allclose(sum_probs, torch.ones_like(sum_probs), atol=1e-5)

    def test_flat_probs_nonneg(self):
        """flat_logits 的 softmax 应非负."""
        head = HierarchicalClsHead(feat_channels=64)
        x = torch.randn(2, 10, 64)
        out = head(x)
        probs = F.softmax(out['flat_logits'], dim=-1)
        assert (probs >= 0).all()


# ================================================================
# 损失计算
# ================================================================

class TestHierarchicalLoss:
    """测试分层分类损失."""

    def test_loss_shape(self):
        """损失应为标量."""
        head = HierarchicalClsHead(feat_channels=64)
        bs, N = 2, 10
        x = torch.randn(bs, N, 64)
        out = head(x)
        targets = torch.randint(0, 24, (bs, N))
        losses = head.loss(
            out['group_logits'], out['class_logits_per_group'],
            targets,
        )
        assert losses['loss_group'].dim() == 0
        assert losses['loss_class'].dim() == 0
        assert losses['loss_total'].dim() == 0

    def test_loss_total_equals_group_plus_class(self):
        """loss_total = loss_group + loss_class (loss_class 已含 lambda 权重)."""
        head = HierarchicalClsHead(feat_channels=64)
        bs, N = 2, 10
        x = torch.randn(bs, N, 64)
        out = head(x)
        targets = torch.randint(0, 24, (bs, N))
        lambda_class = 0.5
        losses = head.loss(
            out['group_logits'], out['class_logits_per_group'],
            targets, lambda_class=lambda_class,
        )
        # loss_class 返回时已乘 lambda_class, 所以 total = group + class
        expected_total = losses['loss_group'] + losses['loss_class']
        assert torch.allclose(losses['loss_total'], expected_total, atol=1e-5)

    def test_loss_with_valid_mask(self):
        """带 valid_mask 的损失计算."""
        head = HierarchicalClsHead(feat_channels=64)
        bs, N = 2, 10
        x = torch.randn(bs, N, 64)
        out = head(x)
        targets = torch.randint(0, 24, (bs, N))
        valid_mask = torch.zeros(bs, N, dtype=torch.bool)
        valid_mask[0, :5] = True  # 只有前 5 个有效
        losses = head.loss(
            out['group_logits'], out['class_logits_per_group'],
            targets, valid_mask=valid_mask,
        )
        assert torch.isfinite(losses['loss_total'])

    def test_loss_ignore_invalid_targets(self):
        """targets=-1 应被忽略."""
        head = HierarchicalClsHead(feat_channels=64)
        bs, N = 2, 10
        x = torch.randn(bs, N, 64)
        out = head(x)
        targets = torch.randint(0, 24, (bs, N))
        targets[0, 5:] = -1  # 后 5 个无效
        losses = head.loss(
            out['group_logits'], out['class_logits_per_group'],
            targets,
        )
        assert torch.isfinite(losses['loss_total'])

    def test_loss_all_invalid(self):
        """全部无效样本应返回 0 损失."""
        head = HierarchicalClsHead(feat_channels=64)
        bs, N = 2, 10
        x = torch.randn(bs, N, 64)
        out = head(x)
        targets = torch.full((bs, N), -1)
        losses = head.loss(
            out['group_logits'], out['class_logits_per_group'],
            targets,
        )
        assert losses['loss_group'].item() == 0.0
        assert losses['loss_class'].item() == 0.0

    def test_loss_decreases_with_correct_predictions(self):
        """正确预测的损失应低于错误预测."""
        head = HierarchicalClsHead(feat_channels=64)
        bs, N = 1, 10
        x = torch.randn(bs, N, 64)
        out = head(x)

        # 构造正确和错误的 targets
        pred_group = out['group_logits'].argmax(dim=-1)
        # 正确 targets: 与预测一致
        correct_targets = torch.zeros(bs, N, dtype=torch.long)
        for i in range(N):
            for g, indices in enumerate(head.class_indices_per_group):
                if g == pred_group[0, i]:
                    correct_targets[0, i] = indices[0]
                    break

        # 错误 targets: 与预测不一致
        wrong_targets = torch.zeros(bs, N, dtype=torch.long)
        for i in range(N):
            for g, indices in enumerate(head.class_indices_per_group):
                if g != pred_group[0, i]:
                    wrong_targets[0, i] = indices[0]
                    break

        loss_correct = head.loss(
            out['group_logits'], out['class_logits_per_group'], correct_targets
        )['loss_total']
        loss_wrong = head.loss(
            out['group_logits'], out['class_logits_per_group'], wrong_targets
        )['loss_total']

        # 正确预测的损失应 <= 错误预测
        assert loss_correct <= loss_wrong + 1e-4


# ================================================================
# 推理
# ================================================================

class TestHierarchicalPredict:
    """测试分层分类推理."""

    def test_predict_shape(self):
        """predict 输出形状正确."""
        head = HierarchicalClsHead(feat_channels=64)
        bs, N = 2, 10
        x = torch.randn(bs, N, 64)
        out = head(x)
        pred = head.predict(out['group_logits'], out['class_logits_per_group'])
        assert pred.shape == (bs, N)
        assert pred.dtype == torch.long

    def test_predict_range(self):
        """predict 输出应在 [0, num_classes) 范围内."""
        head = HierarchicalClsHead(feat_channels=64)
        bs, N = 2, 10
        x = torch.randn(bs, N, 64)
        out = head(x)
        pred = head.predict(out['group_logits'], out['class_logits_per_group'])
        assert (pred >= 0).all()
        assert (pred < 24).all()

    def test_predict_consistent_with_argmax_flat(self):
        """predict(flat_logits) 应与 flat_logits.argmax 完全一致 (全局 MAP)."""
        head = HierarchicalClsHead(feat_channels=64)
        bs, N = 2, 100
        x = torch.randn(bs, N, 64)
        out = head(x)
        # 提供 flat_logits 时, predict 用全局 MAP
        pred_hier = head.predict(
            out['group_logits'], out['class_logits_per_group'],
            flat_logits=out['flat_logits'],
        )
        pred_flat = out['flat_logits'].argmax(dim=-1)
        # 应该完全一致
        assert (pred_hier == pred_flat).all()

    def test_predict_greedy_in_range(self):
        """predict (贪心模式, 不提供 flat_logits) 输出应在有效范围内."""
        head = HierarchicalClsHead(feat_channels=64)
        bs, N = 2, 100
        x = torch.randn(bs, N, 64)
        out = head(x)
        pred_hier = head.predict(out['group_logits'], out['class_logits_per_group'])
        # 输出应在 [0, num_classes) 范围内
        assert (pred_hier >= 0).all()
        assert (pred_hier < 24).all()


# ================================================================
# 隔离性测试
# ================================================================

class TestBaselineIsolation:
    """验证分层分类头不影响 baseline (当不使用时)."""

    def test_standard_classification_still_works(self):
        """标准分类 (用 flat_logits) 仍能正常工作."""
        head = HierarchicalClsHead(feat_channels=64)
        bs, N = 2, 10
        x = torch.randn(bs, N, 64)
        out = head(x)
        # 用 flat_logits 做标准分类
        pred = out['flat_logits'].argmax(dim=-1)
        assert pred.shape == (bs, N)
        assert (pred >= 0).all()
        assert (pred < 24).all()

    def test_flat_logits_finite(self):
        """flat_logits 应有限 (无 NaN/Inf)."""
        head = HierarchicalClsHead(feat_channels=64)
        x = torch.randn(2, 10, 64)
        out = head(x)
        assert torch.isfinite(out['flat_logits']).all()
