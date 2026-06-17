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
        for pred_idx, gt_idx in indices:
            assert pred_idx.shape == gt_idx.shape
            assert (pred_idx >= 0).all()
            assert (gt_idx >= 0).all()

    def test_deterministic(self, matcher):
        output = self._make_output()
        targets = self._make_targets()
        idx1 = matcher(output, targets)
        idx2 = matcher(output, targets)
        for (p1, g1), (p2, g2) in zip(idx1, idx2):
            assert torch.equal(p1, p2)
            assert torch.equal(g1, g2)

    def test_few_gt(self, matcher):
        output = self._make_output(num_queries=500)
        targets = self._make_targets(num_gts=5)
        indices = matcher(output, targets)
        for pred_idx, gt_idx in indices:
            assert gt_idx.shape[0] > 0

    def test_empty_gt(self, matcher):
        output = self._make_output(bs=1)
        targets = [InstanceData(bboxes=torch.zeros(0, 4), labels=torch.zeros(0, dtype=torch.long), img_shape=(512, 512))]
        indices = matcher(output, targets)
        assert len(indices) == 1
        pred_idx, gt_idx = indices[0]
        assert pred_idx.shape[0] == 0


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

    def test_scale_aware(self):
        matcher = DiffusionDetMatcher(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, candidate_topk=5,
        )
        criterion = DiffusionDetCriterion(
            num_classes=24,
            matcher=matcher,
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            deep_supervision=True,
            scale_aware=True,
            scale_aware_mode='inverse',
        )
        outputs = self._make_outputs()
        targets = self._make_targets()
        losses = criterion(outputs, targets)
        for key, val in losses.items():
            assert torch.isfinite(val), f"{key} not finite with scale_aware"


class TestDiffusionDetCriterionScaleAwareModes:
    """测试 scale_aware 的三种模式"""

    def _make_criterion(self, mode):
        matcher = DiffusionDetMatcher(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, candidate_topk=5,
        )
        return DiffusionDetCriterion(
            num_classes=24,
            matcher=matcher,
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            deep_supervision=False,
            scale_aware=True,
            scale_aware_mode=mode,
        )

    def _make_data(self):
        outputs = ModelOutput(
            pred_logits=torch.randn(2, 100, 24),
            pred_boxes=torch.rand(2, 100, 4),
        )
        outputs.pred_boxes[:, :, 2:] += outputs.pred_boxes[:, :, :2]
        targets = []
        for _ in range(2):
            gt_bboxes = torch.rand(30, 4)
            gt_bboxes[:, 2:] += gt_bboxes[:, :2]
            gt_labels = torch.randint(0, 24, (30,))
            targets.append(InstanceData(bboxes=gt_bboxes, labels=gt_labels, img_shape=(512, 512)))
        return outputs, targets

    def test_log_linear_mode(self):
        criterion = self._make_criterion('log_linear')
        outputs, targets = self._make_data()
        losses = criterion(outputs, targets)
        for key, val in losses.items():
            assert torch.isfinite(val), f"{key} not finite with log_linear"

    def test_sqrt_inverse_mode(self):
        criterion = self._make_criterion('sqrt_inverse')
        outputs, targets = self._make_data()
        losses = criterion(outputs, targets)
        for key, val in losses.items():
            assert torch.isfinite(val), f"{key} not finite with sqrt_inverse"

    def test_inverse_mode(self):
        criterion = self._make_criterion('inverse')
        outputs, targets = self._make_data()
        losses = criterion(outputs, targets)
        for key, val in losses.items():
            assert torch.isfinite(val), f"{key} not finite with inverse"

    def test_scale_aware_giou(self):
        """scale_aware + scale_aware_giou 组合"""
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
            scale_aware=True,
            scale_aware_mode='inverse',
            scale_aware_giou=True,
        )
        outputs, targets = self._make_data()
        losses = criterion(outputs, targets)
        for key, val in losses.items():
            assert torch.isfinite(val), f"{key} not finite with scale_aware_giou"

    def test_scale_aware_weights_differ_from_uniform(self):
        """scale_aware=True 时，inverse 模式使小框相对权重更大"""
        matcher = DiffusionDetMatcher(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, candidate_topk=5,
        )

        # 构造多 GT 场景: 大框 + 小框混合
        big_boxes = torch.tensor([
            [0.1, 0.1, 0.65, 0.65],   # area=0.3025
            [0.2, 0.2, 0.7, 0.7],     # area=0.25
        ])
        small_boxes = torch.tensor([
            [0.1, 0.1, 0.2, 0.2],     # area=0.01
            [0.3, 0.3, 0.4, 0.4],     # area=0.01
        ])

        # 同样的预测偏差
        pred_big = big_boxes + 0.05
        pred_small = small_boxes + 0.05

        # scale_aware=True (inverse 模式)
        criterion_sa = DiffusionDetCriterion(
            num_classes=24,
            matcher=matcher,
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            deep_supervision=False,
            scale_aware=True,
            scale_aware_mode='inverse',
        )
        # scale_aware=False
        criterion_no = DiffusionDetCriterion(
            num_classes=24,
            matcher=matcher,
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            deep_supervision=False,
            scale_aware=False,
        )

        def _make_output(pred_boxes):
            return ModelOutput(
                pred_logits=torch.randn(1, 2, 24),
                pred_boxes=pred_boxes.unsqueeze(0),
            )

        def _make_targets(gt_boxes):
            return [InstanceData(
                bboxes=gt_boxes,
                labels=torch.tensor([0, 1]),
                img_shape=(512, 512),
            )]

        # 分别计算 loss_bbox
        losses_big_sa = criterion_sa(_make_output(pred_big), _make_targets(big_boxes))
        losses_small_sa = criterion_sa(_make_output(pred_small), _make_targets(small_boxes))
        losses_big_no = criterion_no(_make_output(pred_big), _make_targets(big_boxes))
        losses_small_no = criterion_no(_make_output(pred_small), _make_targets(small_boxes))

        l1_big_sa = losses_big_sa['loss_bbox'].item()
        l1_small_sa = losses_small_sa['loss_bbox'].item()
        l1_big_no = losses_big_no['loss_bbox'].item()
        l1_small_no = losses_small_no['loss_bbox'].item()

        # inverse 模式: scale_w = 1/area / mean(1/area)
        # 小框 area 小 → 1/area 大 → scale_w 大
        # 验证: scale_aware 下小框与大框的 loss 比值应不同于无 scale_aware
        # (scale_aware 改变了权重分配)
        ratio_no = l1_small_no / max(l1_big_no, 1e-10)
        ratio_sa = l1_small_sa / max(l1_big_sa, 1e-10)
        # 两个比值应不同，说明 scale_aware 确实改变了权重
        assert abs(ratio_no - ratio_sa) > 0.01, (
            f"scale_aware 未改变权重分配: ratio_no={ratio_no:.4f}, ratio_sa={ratio_sa:.4f}"
        )


class TestDiffusionDetCriterionRelativeL1:
    """测试 bbox_loss_mode='relative_l1'"""

    def test_relative_l1_finite(self):
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
            bbox_loss_mode='relative_l1',
        )
        outputs = ModelOutput(
            pred_logits=torch.randn(2, 100, 24),
            pred_boxes=torch.rand(2, 100, 4),
        )
        outputs.pred_boxes[:, :, 2:] += outputs.pred_boxes[:, :, :2]
        targets = []
        for _ in range(2):
            gt_bboxes = torch.rand(30, 4)
            gt_bboxes[:, 2:] += gt_bboxes[:, :2]
            gt_labels = torch.randint(0, 24, (30,))
            targets.append(InstanceData(bboxes=gt_bboxes, labels=gt_labels, img_shape=(512, 512)))
        losses = criterion(outputs, targets)
        for key, val in losses.items():
            assert torch.isfinite(val), f"{key} not finite with relative_l1"

    def test_relative_l1_gradient(self):
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
            bbox_loss_mode='relative_l1',
        )
        outputs = ModelOutput(
            pred_logits=torch.randn(2, 100, 24, requires_grad=True),
            pred_boxes=torch.rand(2, 100, 4, requires_grad=True),
        )
        outputs.pred_boxes.data[:, :, 2:] += outputs.pred_boxes.data[:, :, :2]
        targets = []
        for _ in range(2):
            gt_bboxes = torch.rand(30, 4)
            gt_bboxes[:, 2:] += gt_bboxes[:, :2]
            gt_labels = torch.randint(0, 24, (30,))
            targets.append(InstanceData(bboxes=gt_bboxes, labels=gt_labels, img_shape=(512, 512)))
        losses = criterion(outputs, targets)
        sum(losses.values()).backward()
        assert outputs.pred_logits.grad is not None
        assert outputs.pred_boxes.grad is not None


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
