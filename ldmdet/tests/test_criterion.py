"""测试 ldmdet.criterion — 代价函数、损失函数、匹配器、完整准则"""

import torch
import pytest
from ldmdet.criterion.costs import FocalLossCost, BBoxL1Cost, IoUCost, RelativeL1Cost
from ldmdet.criterion.losses import FocalLoss, GIoULoss, L1Loss, sigmoid_focal_loss
from ldmdet.criterion.matcher import DiffusionDetMatcher
from ldmdet.criterion.criterion import DiffusionDetCriterion
from ldmdet.data.structures import InstanceData, ModelOutput


# ============================================================
# 代价函数 — 注意签名: (pred_logits, pred_bboxes, gt_labels, gt_bboxes)
# ============================================================

class TestFocalLossCost:
    def test_output_shape(self):
        cost_fn = FocalLossCost(weight=2.0)
        pred_logits = torch.randn(100, 24)
        pred_bboxes = torch.rand(100, 4)
        gt_labels = torch.randint(0, 24, (30,))
        gt_bboxes = torch.rand(30, 4)
        cost = cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
        assert cost.shape == (100, 30)

    def test_deterministic(self):
        cost_fn = FocalLossCost(weight=2.0)
        pred_logits = torch.randn(100, 24)
        pred_bboxes = torch.rand(100, 4)
        gt_labels = torch.randint(0, 24, (30,))
        gt_bboxes = torch.rand(30, 4)
        c1 = cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
        c2 = cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
        assert torch.allclose(c1, c2, atol=1e-6)


class TestBBoxL1Cost:
    def test_output_shape(self):
        cost_fn = BBoxL1Cost(weight=5.0)
        pred_logits = torch.randn(100, 24)
        pred_bboxes = torch.rand(100, 4)  # [0,1] 归一化坐标
        gt_labels = torch.randint(0, 24, (30,))
        gt_bboxes = torch.rand(30, 4)
        cost = cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
        assert cost.shape == (100, 30)

    def test_identical_boxes_low_cost(self):
        cost_fn = BBoxL1Cost(weight=5.0)
        # 归一化坐标 [0,1] xyxy 格式
        tl = torch.rand(30, 2) * 0.3
        boxes = torch.cat([tl, tl + 0.2], dim=-1)
        pred_logits = torch.randn(30, 24)
        gt_labels = torch.randint(0, 24, (30,))
        cost = cost_fn(pred_logits, boxes, gt_labels, boxes)
        # 对角线 (自身匹配) 应为 0
        assert torch.allclose(cost.diag(), torch.zeros(30), atol=1e-5)

    def test_non_negative(self):
        cost_fn = BBoxL1Cost(weight=5.0)
        pred_logits = torch.randn(100, 24)
        pred_bboxes = torch.rand(100, 4)
        gt_labels = torch.randint(0, 24, (30,))
        gt_bboxes = torch.rand(30, 4)
        cost = cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
        assert (cost >= -1e-6).all()


class TestIoUCost:
    def test_output_shape(self):
        cost_fn = IoUCost(iou_mode='giou', weight=2.0)
        pred_logits = torch.randn(100, 24)
        # 归一化坐标 [0,1] xyxy 格式
        pred_bboxes = torch.rand(100, 2) * 0.5
        pred_bboxes = torch.cat([pred_bboxes, pred_bboxes + torch.rand(100, 2) * 0.3], dim=-1)
        gt_labels = torch.randint(0, 24, (30,))
        gt_bboxes = torch.rand(30, 2) * 0.5
        gt_bboxes = torch.cat([gt_bboxes, gt_bboxes + torch.rand(30, 2) * 0.3], dim=-1)
        cost = cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
        assert cost.shape == (100, 30)

    def test_identical_boxes_low_cost(self):
        cost_fn = IoUCost(iou_mode='giou', weight=2.0)
        # 归一化坐标 [0,1] xyxy 格式, 确保 x2>x1, y2>y1
        tl = torch.rand(30, 2) * 0.3
        boxes = torch.cat([tl, tl + 0.2], dim=-1)
        pred_logits = torch.randn(30, 24)
        gt_labels = torch.randint(0, 24, (30,))
        cost = cost_fn(pred_logits, boxes, gt_labels, boxes)
        # 对角线 IoU=1 -> cost=0
        assert torch.allclose(cost.diag(), torch.zeros(30), atol=1e-3)


class TestRelativeL1Cost:
    def test_output_shape(self):
        cost_fn = RelativeL1Cost(weight=1.0)
        pred_logits = torch.randn(100, 24)
        pred_bboxes = torch.rand(100, 4)
        pred_bboxes[:, 2:] += pred_bboxes[:, :2]
        gt_labels = torch.randint(0, 24, (30,))
        gt_bboxes = torch.rand(30, 4)
        gt_bboxes[:, 2:] += gt_bboxes[:, :2]
        cost = cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
        assert cost.shape == (100, 30)

    def test_non_negative(self):
        cost_fn = RelativeL1Cost(weight=1.0)
        pred_logits = torch.randn(100, 24)
        pred_bboxes = torch.rand(100, 4)
        pred_bboxes[:, 2:] += pred_bboxes[:, :2]
        gt_labels = torch.randint(0, 24, (30,))
        gt_bboxes = torch.rand(30, 4)
        gt_bboxes[:, 2:] += gt_bboxes[:, :2]
        cost = cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
        assert (cost >= -1e-6).all()


# ============================================================
# 损失函数
# ============================================================

class TestSigmoidFocalLoss:
    def test_output_shape(self):
        pred = torch.randn(100, 24)
        target = torch.randint(0, 2, (100, 24)).float()
        loss = sigmoid_focal_loss(pred, target, alpha=0.25, gamma=2.0)
        assert loss.shape == (100, 24)

    def test_non_negative(self):
        pred = torch.randn(100, 24)
        target = torch.randint(0, 2, (100, 24)).float()
        loss = sigmoid_focal_loss(pred, target, alpha=0.25, gamma=2.0)
        assert (loss >= -1e-6).all()

    def test_deterministic(self):
        pred = torch.randn(100, 24)
        target = torch.randint(0, 2, (100, 24)).float()
        l1 = sigmoid_focal_loss(pred, target, alpha=0.25, gamma=2.0)
        l2 = sigmoid_focal_loss(pred, target, alpha=0.25, gamma=2.0)
        assert torch.allclose(l1, l2, atol=1e-7)

    def test_reduction_mean(self):
        pred = torch.randn(100, 24)
        target = torch.randint(0, 2, (100, 24)).float()
        loss = sigmoid_focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean')
        assert loss.ndim == 0

    def test_reduction_sum(self):
        pred = torch.randn(100, 24)
        target = torch.randint(0, 2, (100, 24)).float()
        loss = sigmoid_focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='sum')
        assert loss.ndim == 0


class TestFocalLoss:
    def test_forward_with_class_indices(self):
        """输入 class index (非 one-hot)"""
        loss_fn = FocalLoss(loss_weight=2.0)
        pred = torch.randn(100, 24)
        target = torch.randint(0, 24, (100,))
        loss = loss_fn(pred, target)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_forward_with_onehot(self):
        """输入 one-hot"""
        loss_fn = FocalLoss(loss_weight=2.0)
        pred = torch.randn(100, 24)
        target = torch.randint(0, 2, (100, 24)).float()
        loss = loss_fn(pred, target)
        assert loss.ndim == 0

    def test_gradient_flow(self):
        loss_fn = FocalLoss(loss_weight=2.0)
        pred = torch.randn(100, 24, requires_grad=True)
        target = torch.randint(0, 24, (100,))
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestGIoULoss:
    def test_forward(self):
        loss_fn = GIoULoss(loss_weight=2.0)
        pred = torch.rand(100, 4)
        pred[:, 2:] += pred[:, :2]
        target = torch.rand(100, 4)
        target[:, 2:] += target[:, :2]
        loss = loss_fn(pred, target)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_identical_boxes_low_loss(self):
        loss_fn = GIoULoss(loss_weight=2.0)
        boxes = torch.rand(50, 4)
        boxes[:, 2:] += boxes[:, :2]
        loss = loss_fn(boxes, boxes)
        assert loss.item() < 0.01


class TestL1Loss:
    def test_forward(self):
        loss_fn = L1Loss(loss_weight=5.0)
        pred = torch.rand(100, 4)
        target = torch.rand(100, 4)
        loss = loss_fn(pred, target)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_identical_zero_loss(self):
        loss_fn = L1Loss(loss_weight=5.0)
        boxes = torch.rand(50, 4)
        loss = loss_fn(boxes, boxes)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


# ============================================================
# 匹配器
# ============================================================

class TestDiffusionDetMatcher:
    @pytest.fixture
    def matcher(self):
        return DiffusionDetMatcher(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            candidate_topk=5,
        )

    def _make_output(self, num_queries=100, num_classes=24, bs=2):
        """创建 [bs, N, C] 格式的 ModelOutput"""
        return ModelOutput(
            pred_logits=torch.randn(bs, num_queries, num_classes),
            pred_boxes=torch.rand(bs, num_queries, 4),
        )

    def _make_targets(self, num_gts=30, num_classes=24, bs=2):
        targets = []
        for _ in range(bs):
            # 归一化坐标 [0,1] xyxy 格式
            gt_bboxes = torch.rand(num_gts, 2) * 0.5
            gt_bboxes = torch.cat([gt_bboxes, gt_bboxes + torch.rand(num_gts, 2) * 0.3], dim=-1)
            gt_labels = torch.randint(0, num_classes, (num_gts,))
            targets.append(InstanceData(bboxes=gt_bboxes, labels=gt_labels, img_shape=(512, 512)))
        return targets

    def test_output_structure(self, matcher):
        output = self._make_output()
        targets = self._make_targets()
        indices = matcher(output, targets)
        assert len(indices) == 2  # batch size
        for fg_mask, matched_gt_inds in indices:
            assert fg_mask.dtype == torch.bool
            assert matched_gt_inds.dtype == torch.long
            assert fg_mask.shape[0] == matched_gt_inds.shape[0]
            # 正样本的 GT 索引应非负
            if fg_mask.any():
                assert (matched_gt_inds[fg_mask] >= 0).all()

    def test_deterministic(self, matcher):
        output = self._make_output()
        targets = self._make_targets()
        idx1 = matcher(output, targets)
        idx2 = matcher(output, targets)
        for (m1, g1), (m2, g2) in zip(idx1, idx2):
            assert torch.equal(m1, m2)
            assert torch.equal(g1, g2)

    def test_few_gt(self, matcher):
        output = self._make_output(num_queries=500)
        targets = self._make_targets(num_gts=5)
        indices = matcher(output, targets)
        for fg_mask, matched_gt_inds in indices:
            assert fg_mask.sum() > 0

    def test_empty_gt(self, matcher):
        output = self._make_output(bs=1)
        targets = [InstanceData(bboxes=torch.zeros(0, 4), labels=torch.zeros(0, dtype=torch.long), img_shape=(512, 512))]
        indices = matcher(output, targets)
        assert len(indices) == 1
        fg_mask, matched_gt_inds = indices[0]
        assert fg_mask.sum() == 0  # 无正样本


# ============================================================
# 完整准则
# ============================================================

class TestDiffusionDetCriterion:
    @pytest.fixture
    def criterion(self):
        matcher = DiffusionDetMatcher(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, candidate_topk=5,
        )
        return DiffusionDetCriterion(
            num_classes=24,
            matcher=matcher,
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            deep_supervision=True,
        )

    def _make_outputs(self, num_heads=6, num_queries=100, num_classes=24, bs=2):
        main_logits = torch.randn(bs, num_queries, num_classes)
        main_bboxes = torch.rand(bs, num_queries, 4)
        main_bboxes[:, :, 2:] += main_bboxes[:, :, :2]
        aux_outputs = [
            ModelOutput(
                pred_logits=torch.randn(bs, num_queries, num_classes),
                pred_boxes=torch.rand(bs, num_queries, 4),
            )
            for _ in range(num_heads - 1)
        ]
        # 也让 aux bboxes 有效
        for aux in aux_outputs:
            aux.pred_boxes[:, :, 2:] += aux.pred_boxes[:, :, :2]
        return ModelOutput(
            pred_logits=main_logits,
            pred_boxes=main_bboxes,
            aux_outputs=aux_outputs,
        )

    def _make_targets(self, num_gts=30, num_classes=24, bs=2):
        targets = []
        for _ in range(bs):
            gt_bboxes = torch.rand(num_gts, 4)
            gt_bboxes[:, 2:] += gt_bboxes[:, :2]
            gt_labels = torch.randint(0, num_classes, (num_gts,))
            targets.append(InstanceData(bboxes=gt_bboxes, labels=gt_labels, img_shape=(512, 512)))
        return targets

    def test_loss_keys(self, criterion):
        outputs = self._make_outputs()
        targets = self._make_targets()
        losses = criterion(outputs, targets)
        assert 'loss_cls' in losses
        assert 'loss_bbox' in losses
        assert 'loss_giou' in losses
        # aux losses
        assert any(k.startswith('aux_') for k in losses)

    def test_losses_finite(self, criterion):
        outputs = self._make_outputs()
        targets = self._make_targets()
        losses = criterion(outputs, targets)
        for key, val in losses.items():
            assert torch.isfinite(val), f"{key} is not finite: {val.item()}"

    def test_deterministic(self, criterion):
        outputs = self._make_outputs()
        targets = self._make_targets()
        losses1 = criterion(outputs, targets)
        losses2 = criterion(outputs, targets)
        for key in losses1:
            assert torch.allclose(losses1[key], losses2[key], atol=1e-5), f"{key} mismatch"

    def test_gradient_flow(self, criterion):
        outputs = self._make_outputs()
        targets = self._make_targets()
        outputs.pred_logits.requires_grad_(True)
        outputs.pred_boxes.requires_grad_(True)
        losses = criterion(outputs, targets)
        total = sum(losses.values())
        total.backward()
        assert outputs.pred_logits.grad is not None
        assert outputs.pred_boxes.grad is not None

    def test_no_deep_supervision(self):
        matcher = DiffusionDetMatcher(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, candidate_topk=5,
        )
        criterion = DiffusionDetCriterion(
            num_classes=24,
            matcher=matcher,
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            deep_supervision=False,
        )
        outputs = self._make_outputs()
        targets = self._make_targets()
        losses = criterion(outputs, targets)
        assert all(not k.startswith('aux_') for k in losses)

    def test_empty_gt(self, criterion):
        outputs = self._make_outputs()
        targets = [InstanceData(bboxes=torch.zeros(0, 4), labels=torch.zeros(0, dtype=torch.long), img_shape=(512, 512))
                   for _ in range(2)]
        losses = criterion(outputs, targets)
        for key, val in losses.items():
            assert torch.isfinite(val), f"{key} is not finite"

    def test_localization_utility_is_terminal_only_and_metric_ordered(self):
        matcher = DiffusionDetMatcher(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, candidate_topk=1,
        )
        criterion = DiffusionDetCriterion(
            num_classes=24,
            matcher=matcher,
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            deep_supervision=True,
            localization_utility_loss_weight=0.5,
            localization_utility_temperature=0.025,
        )
        targets = [InstanceData(
            bboxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            labels=torch.tensor([0]),
            img_shape=(32, 32),
        )]
        indices = [(torch.tensor([True]), torch.tensor([0]))]
        high_iou = ModelOutput(
            pred_logits=torch.zeros(1, 1, 24),
            pred_boxes=torch.tensor(
                [[[0.0, 0.0, 0.9, 1.0]]], requires_grad=True),
        )
        low_iou = ModelOutput(
            pred_logits=torch.zeros(1, 1, 24),
            pred_boxes=torch.tensor([[[0.0, 0.0, 0.6, 1.0]]]),
        )
        high_loss = criterion._loss_localization_utility(
            high_iou, targets, indices)
        low_loss = criterion._loss_localization_utility(
            low_iou, targets, indices)
        assert high_loss < low_loss
        high_loss.backward()
        assert high_iou.pred_boxes.grad is not None
        assert torch.isfinite(high_iou.pred_boxes.grad).all()
        assert high_iou.pred_boxes.grad.abs().sum() > 0

        high_iou.pred_boxes.grad = None
        high_iou.aux_outputs = [low_iou]
        losses = criterion(high_iou, targets)
        assert 'loss_localization_utility' in losses
        assert not any(
            key.startswith('aux_') and 'localization_utility' in key
            for key in losses)

    def test_iou_survival_loss_supervises_all_thresholds(self, criterion):
        criterion.quality_thresholds = (0.50, 0.75, 0.95)
        targets = [InstanceData(
            bboxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            labels=torch.tensor([0]),
            img_shape=(32, 32),
        )]
        indices = [(torch.tensor([True]), torch.tensor([0]))]
        pred_boxes = torch.tensor([[[0.0, 0.0, 0.6, 1.0]]])
        common = dict(
            pred_logits=torch.zeros(1, 1, 24),
            pred_boxes=pred_boxes,
        )
        good = ModelOutput(
            **common, pred_quality=torch.tensor([[[10.0, -10.0, -10.0]]])
        )
        bad = ModelOutput(
            **common, pred_quality=torch.tensor([[[-10.0, 10.0, 10.0]]])
        )
        assert criterion._loss_quality(good, targets, indices) < (
            criterion._loss_quality(bad, targets, indices)
        )

    def test_set_mass_target_conserves_one_unit_per_gt(self, criterion):
        indices = [(
            torch.tensor([True, True, False]),
            torch.tensor([0, 0, 0]),
        )]
        targets = [InstanceData(
            bboxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            labels=torch.tensor([0]),
            img_shape=(32, 32),
        )]
        common = dict(
            pred_logits=torch.zeros(1, 3, 24),
            pred_boxes=torch.zeros(1, 3, 4),
        )
        conserved = ModelOutput(
            **common,
            pred_mass=torch.tensor([[[0.0], [0.0], [-10.0]]]),
        )
        excessive = ModelOutput(
            **common,
            pred_mass=torch.tensor([[[4.0], [4.0], [-10.0]]]),
        )
        conserved_losses = criterion._loss_set_mass(
            conserved, targets, indices
        )
        excessive_losses = criterion._loss_set_mass(
            excessive, targets, indices
        )
        assert conserved_losses[1] < excessive_losses[1]

    def test_quality_weighted_mass_prefers_best_duplicate(self, criterion):
        criterion.mass_target_mode = 'coco_utility_softmax'
        criterion.mass_target_temperature = 0.1
        indices = [(
            torch.tensor([True, True, False]),
            torch.tensor([0, 0, 0]),
        )]
        targets = [InstanceData(
            bboxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            labels=torch.tensor([0]),
            img_shape=(32, 32),
        )]
        outputs = ModelOutput(
            pred_logits=torch.zeros(1, 3, 24),
            pred_boxes=torch.tensor([[[0.0, 0.0, 1.0, 1.0],
                                      [0.0, 0.0, 0.4, 1.0],
                                      [0.0, 0.0, 0.0, 0.0]]]),
            pred_mass=torch.zeros(1, 3, 1),
        )
        mass_targets, _ = criterion._set_mass_targets(
            outputs, targets, indices
        )
        assert mass_targets[0, 0] > mass_targets[0, 1]
        assert torch.allclose(
            mass_targets[0, :2].sum(), torch.tensor(1.0)
        )
        assert mass_targets[0, 2] == 0


class TestIoUCostPlain:
    """测试 IoUCost 的 iou_mode='iou' (非 giou)"""

    def test_output_shape(self):
        cost_fn = IoUCost(iou_mode='iou', weight=2.0)
        pred_logits = torch.randn(100, 24)
        pred_bboxes = torch.rand(100, 2) * 0.5
        pred_bboxes = torch.cat([pred_bboxes, pred_bboxes + torch.rand(100, 2) * 0.3], dim=-1)
        gt_labels = torch.randint(0, 24, (30,))
        gt_bboxes = torch.rand(30, 2) * 0.5
        gt_bboxes = torch.cat([gt_bboxes, gt_bboxes + torch.rand(30, 2) * 0.3], dim=-1)
        cost = cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
        assert cost.shape == (100, 30)

    def test_identical_boxes_low_cost(self):
        cost_fn = IoUCost(iou_mode='iou', weight=2.0)
        tl = torch.rand(30, 2) * 0.3
        boxes = torch.cat([tl, tl + 0.2], dim=-1)
        pred_logits = torch.randn(30, 24)
        gt_labels = torch.randint(0, 24, (30,))
        cost = cost_fn(pred_logits, boxes, gt_labels, boxes)
        # IoU=1 → cost=0
        assert torch.allclose(cost.diag(), torch.zeros(30), atol=1e-3)
