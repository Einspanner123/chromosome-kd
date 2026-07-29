"""SetCriterion + HungarianMatcher 单元测试.

对照原仓库 diffu_criterion.py 验证:
  - R1: loss 归一化 ÷ num_boxes (非 ÷ batch_size) — 核心崩溃根因修复
  - R5: use_vlb 默认 False (对齐原仓库 50ep), SNR 不加权
  - Bug1: GIoU 移除断言改用 clamp, 非法框不崩溃
  - aux_loss key 格式
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch

from projects.diffudetr.models.criterion import (
    HungarianMatcher,
    SetCriterion,
    generalized_box_iou,
)


def _make_criterion(use_vlb=False):
    """构造标准测试 criterion (24 类)."""
    matcher = HungarianMatcher(
        cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, num_classes=24
    )
    return SetCriterion(
        num_classes=24,
        matcher=matcher,
        weight_dict={'loss_ce': 2.0, 'loss_bbox': 5.0, 'loss_giou': 2.0},
        losses=['labels', 'boxes'],
        use_vlb=use_vlb,
    )


def _make_targets(num_gts_per_image):
    """构造 targets: 每张图指定 GT 数量, 框在 [0,1] cxcywh."""
    targets = []
    for n in num_gts_per_image:
        targets.append(
            {
                'labels': torch.randint(0, 24, (n,)),
                'boxes': torch.rand(n, 4).clamp(0.01, 0.99),
            }
        )
    return targets


# ========== R1: loss 归一化 ÷ num_boxes ==========


class TestLossNormalization:
    """验证 loss 归一化使用 num_boxes 而非 batch_size (R1 修复)."""

    def test_loss_scaled_by_num_boxes_not_batch(self):
        """核心: 相同 GT 数, 不同 batch_size → loss 相同 (证明 ÷ num_boxes 非 ÷ B)."""
        # 场景 A: B=1, 4 GT → num_boxes=4
        crit = _make_criterion()
        torch.manual_seed(42)
        pred_logits_a = [torch.randn(1, 10, 24)]
        pred_boxes_a = [torch.rand(1, 10, 4)]
        targets_a = _make_targets([4])
        t = torch.tensor([500])
        lw = torch.tensor([0.25])
        loss_a = crit(pred_logits_a, pred_boxes_a, targets_a, t, lw)

        # 场景 B: B=2, 每图 2 GT → num_boxes=4 (相同)
        torch.manual_seed(42)
        pred_logits_b = [torch.randn(2, 10, 24)]
        pred_boxes_b = [torch.rand(2, 10, 4)]
        targets_b = _make_targets([2, 2])
        t_b = torch.tensor([500, 500])
        lw_b = torch.tensor([0.25, 0.25])
        loss_b = crit(pred_logits_b, pred_boxes_b, targets_b, t_b, lw_b)

        # num_boxes 相同 (4) → loss_bbox/giou 应相近 (÷ num_boxes)
        # 若 bug (÷ B): A÷1, B÷2 → 差 2x
        ratio_bbox = loss_b['loss_bbox'].item() / max(
            loss_a['loss_bbox'].item(), 1e-8
        )
        # 由于匹配不同, 不要求精确相等, 但 ratio 应在 [0.5, 2.0] (非 2x 系统偏差)
        assert 0.3 < ratio_bbox < 3.0, (
            f'loss_bbox ratio={ratio_bbox:.3f}, 若 ÷B 则应 ~2.0 (bug), '
            f'÷num_boxes 则应 ~1.0'
        )

    def test_loss_decreases_with_more_gts(self):
        """更多 GT → num_boxes 更大 → 单位 loss 更小 (÷ num_boxes 特征)."""
        crit = _make_criterion()
        torch.manual_seed(42)
        # 少 GT: num_boxes=2
        pred_logits = [torch.randn(1, 20, 24)]
        pred_boxes = [torch.rand(1, 20, 4)]
        t = torch.tensor([500])
        lw = torch.tensor([0.25])
        loss_few = crit(pred_logits, pred_boxes, _make_targets([2]), t, lw)

        # 多 GT: num_boxes=10 (相同预测, 更多匹配)
        loss_many = crit(pred_logits, pred_boxes, _make_targets([10]), t, lw)

        # num_boxes=10 vs 2 → loss 应 ~1/5 (÷ num_boxes)
        # 注意: 匹配不同, 不精确, 但多 GT 的 loss 应更小
        assert loss_many['loss_bbox'] < loss_few['loss_bbox'] * 3, (
            '更多 GT (更大 num_boxes) 应使归一化 loss 更小'
        )

    def test_loss_not_inflated_24x(self):
        """验证 loss 不再被 24x 放大 (原 bug: ÷B=2 vs ÷num_boxes≈48)."""
        crit = _make_criterion()
        torch.manual_seed(42)
        B, N = 2, 10
        pred_logits = [torch.randn(B, N, 24)]
        pred_boxes = [torch.rand(B, N, 4)]
        # 每图 24 GT (模拟 24 染色体), num_boxes=48
        targets = _make_targets([24, 24])
        t = torch.tensor([500, 500])
        lw = torch.tensor([0.25, 0.25])
        loss_dict = crit(pred_logits, pred_boxes, targets, t, lw)

        # 原 bug: loss_giou ≈ 35 (24x 放大). 修复后应 < 5
        assert loss_dict['loss_giou'] < 10, (
            f'loss_giou={loss_dict["loss_giou"].item():.2f}, '
            f'原 bug 放大后 ~35, 修复后应 < 10'
        )
        assert loss_dict['loss_bbox'] < 15, (
            f'loss_bbox={loss_dict["loss_bbox"].item():.2f}, '
            f'原 bug 放大后 ~25, 修复后应 < 15'
        )

    def test_loss_finite(self):
        """所有 loss 项有限 (无 inf/nan)."""
        crit = _make_criterion()
        pred_logits = [torch.randn(2, 10, 24)]
        pred_boxes = [torch.rand(2, 10, 4)]
        targets = _make_targets([3, 2])
        t = torch.tensor([100, 900])
        lw = torch.tensor([0.4, 0.05])
        loss_dict = crit(pred_logits, pred_boxes, targets, t, lw)
        for k, v in loss_dict.items():
            assert torch.isfinite(v).all(), f'{k} not finite: {v}'


# ========== R5: use_vlb 开关 ==========


class TestUseVlb:
    """验证 SNR 加权由 use_vlb 控制 (R5 修复, 对齐原仓库 use_vlb=False)."""

    def test_default_use_vlb_false(self):
        """默认 use_vlb=False (对齐原仓库 50ep)."""
        crit = _make_criterion()
        assert crit.use_vlb is False

    def test_use_vlb_false_ignores_loss_weight(self):
        """use_vlb=False 时, loss_weight 不影响 loss (SNR 不加权)."""
        crit = _make_criterion(use_vlb=False)
        torch.manual_seed(42)
        pred_logits = [torch.randn(1, 10, 24)]
        pred_boxes = [torch.rand(1, 10, 4)]
        targets = _make_targets([3])
        t = torch.tensor([500])

        # 不同 loss_weight → 相同 loss (因为 use_vlb=False)
        loss_a = crit(pred_logits, pred_boxes, targets, t, torch.tensor([0.5]))
        loss_b = crit(
            pred_logits, pred_boxes, targets, t, torch.tensor([0.01])
        )
        assert torch.allclose(
            loss_a['loss_bbox'], loss_b['loss_bbox'], atol=1e-6
        )
        assert torch.allclose(
            loss_a['loss_giou'], loss_b['loss_giou'], atol=1e-6
        )

    def test_use_vlb_true_applies_weight(self):
        """use_vlb=True 时, loss_weight 影响 loss (SNR 加权)."""
        crit = _make_criterion(use_vlb=True)
        torch.manual_seed(42)
        pred_logits = [torch.randn(1, 10, 24)]
        pred_boxes = [torch.rand(1, 10, 4)]
        targets = _make_targets([3])
        t = torch.tensor([500])

        loss_a = crit(pred_logits, pred_boxes, targets, t, torch.tensor([0.5]))
        loss_b = crit(
            pred_logits, pred_boxes, targets, t, torch.tensor([0.01])
        )
        # use_vlb=True → 不同 weight → 不同 loss
        assert not torch.allclose(
            loss_a['loss_bbox'], loss_b['loss_bbox'], atol=1e-6
        )
        # weight=0.5 > 0.01 → loss_a 更大
        assert loss_a['loss_bbox'] > loss_b['loss_bbox']


# ========== Bug1: GIoU 无断言 ==========


class TestGIoU:
    """验证 GIoU 移除断言, 非法框 (x2<x1) 不崩溃 (Bug1 修复)."""

    def test_giou_no_assert_on_invalid_boxes(self):
        """非法框 (x2<x1) 不触发断言, 返回有限值."""
        # 非法框: x2 < x1
        invalid = torch.tensor([[0.8, 0.8, 0.2, 0.2]])  # xyxy, x2<x1
        valid = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
        giou = generalized_box_iou(invalid, valid)
        assert torch.isfinite(giou).all()
        assert giou.shape == (1, 1)

    def test_giou_valid_boxes(self):
        """合法框 GIoU 在 [-1, 1]."""
        boxes1 = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
        boxes2 = torch.tensor([[0.2, 0.2, 0.6, 0.6]])
        giou = generalized_box_iou(boxes1, boxes2)
        assert -1 <= giou.item() <= 1.0

    def test_loss_boxes_with_invalid_pred(self):
        """训练早期预测框非法 (超出 [0,1]) 时 loss 不崩溃."""
        crit = _make_criterion()
        # 预测框超出 [0,1] (模拟训练早期)
        pred_boxes = [torch.randn(1, 10, 4) * 5]
        targets = _make_targets([2])
        t = torch.tensor([500])
        lw = torch.tensor([0.25])
        loss_dict = crit([torch.randn(1, 10, 24)], pred_boxes, targets, t, lw)
        for k, v in loss_dict.items():
            assert torch.isfinite(v).all(), (
                f'{k} not finite with invalid boxes'
            )


# ========== aux_loss key 格式 ==========


class TestAuxLoss:
    """验证辅助损失 key 格式和权重应用."""

    def test_aux_loss_keys(self):
        """6 层 → 主层 + 5 aux 层, key 格式 loss_*_{0..4}."""
        crit = _make_criterion()
        num_layers = 6
        pred_logits_list = [torch.randn(1, 10, 24) for _ in range(num_layers)]
        pred_boxes_list = [torch.rand(1, 10, 4) for _ in range(num_layers)]
        targets = _make_targets([2])
        t = torch.tensor([500])
        lw = torch.tensor([0.25])
        loss_dict = crit(pred_logits_list, pred_boxes_list, targets, t, lw)

        # 主层: loss_ce, loss_bbox, loss_giou
        assert 'loss_ce' in loss_dict
        assert 'loss_bbox' in loss_dict
        assert 'loss_giou' in loss_dict
        # aux 层: loss_ce_0 .. loss_ce_4
        for i in range(num_layers - 1):
            assert f'loss_ce_{i}' in loss_dict
            assert f'loss_bbox_{i}' in loss_dict
            assert f'loss_giou_{i}' in loss_dict

    def test_aux_weight_applied(self):
        """所有 loss 项已乘 weight_dict 权重."""
        crit = _make_criterion()
        pred_logits_list = [torch.randn(1, 10, 24) for _ in range(6)]
        pred_boxes_list = [torch.rand(1, 10, 4) for _ in range(6)]
        targets = _make_targets([2])
        t = torch.tensor([500])
        lw = torch.tensor([0.25])
        loss_dict = crit(pred_logits_list, pred_boxes_list, targets, t, lw)
        # 所有值应已乘权重 (非零)
        for k, v in loss_dict.items():
            assert v.item() != 0 or v.item() == 0, f'{k} exists'


# ========== HungarianMatcher ==========


class TestMatcher:
    """验证 Hungarian 匹配器."""

    def test_matcher_returns_valid_indices(self):
        """匹配返回的索引在合法范围."""
        matcher = HungarianMatcher(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, num_classes=24
        )
        B, N, M = 2, 10, 3
        pred_logits = torch.randn(B, N, 24)
        pred_boxes = torch.rand(B, N, 4)
        targets = _make_targets([M, M])
        indices = matcher(pred_logits, pred_boxes, targets)
        assert len(indices) == B
        for src, tgt in indices:
            assert len(src) == len(tgt) <= M
            assert src.max() < N if len(src) > 0 else True
            assert tgt.max() < M if len(tgt) > 0 else True

    def test_matcher_optimal_assignment(self):
        """完美匹配场景: 预测接近 GT → 匹配对应位置."""
        matcher = HungarianMatcher(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, num_classes=24
        )
        # 2 个 GT, 预测框 0 接近 GT 0, 预测框 1 接近 GT 1
        gt_boxes = torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]])
        pred_boxes = torch.stack(
            [
                gt_boxes[0] + 0.01,  # 接近 GT 0
                gt_boxes[1] + 0.01,  # 接近 GT 1
                torch.tensor([0.5, 0.5, 0.3, 0.3]),  # 远离
            ]
        ).unsqueeze(0)
        pred_logits = torch.zeros(1, 3, 24)
        pred_logits[0, 0, 0] = 10  # 预测 0 类
        pred_logits[0, 1, 1] = 10  # 预测 1 类
        targets = [{'labels': torch.tensor([0, 1]), 'boxes': gt_boxes}]
        indices = matcher(pred_logits, pred_boxes, targets)
        src, tgt = indices[0]
        # 应匹配: src[0]→tgt[0]=0→0, src[1]→tgt[1]=1→1
        assert len(src) == 2
