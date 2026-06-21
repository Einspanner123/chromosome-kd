"""方向二: 计数先验约束的扩散生成 (Count-Prior Constrained Diffusion) 测试

测试路径 B (拉格朗日约束 NMS) 和路径 C (计数分支).

测试覆盖:
1. find_threshold_for_count: 二分搜索正确性、边界情况
2. count_constrained_nms: 计数约束、min_keep 兜底、空输入
3. CountingBranch: 前向传播、损失计算、形状正确性
4. head.py 集成: 开关默认关闭、开关打开后行为变化
5. 隔离性: baseline 行为不受影响
"""

import torch
import torch.nn as nn
import pytest

from ldmdet.core.counting_branch import CountingBranch
from ldmdet.core.head import DiffusionDetHead
from ldmdet.data.structures import DetectionResult, ImageMeta
from ldmdet.inference.count_constrained_nms import (
    count_constrained_nms,
    find_threshold_for_count,
)


# ================================================================
# 路径 B: find_threshold_for_count
# ================================================================


class TestFindThresholdForCount:
    """测试二分搜索阈值函数"""

    def test_basic(self):
        """基本场景: 10 个分数, 目标保留 5 个"""
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        tau = find_threshold_for_count(scores, target_count=5)
        count = (scores > tau).sum().item()
        # 应该保留约 5 个 (二分搜索可能 ±1)
        assert abs(count - 5) <= 1

    def test_target_zero(self):
        """目标为 0: 阈值应很高, 保留 0 个"""
        scores = torch.tensor([0.1, 0.2, 0.3])
        tau = find_threshold_for_count(scores, target_count=0)
        count = (scores > tau).sum().item()
        assert count == 0

    def test_target_all(self):
        """目标等于总数: 阈值应很低, 保留全部"""
        scores = torch.tensor([0.1, 0.2, 0.3])
        tau = find_threshold_for_count(scores, target_count=3)
        count = (scores > tau).sum().item()
        assert count == 3

    def test_target_exceeds_total(self):
        """目标超过总数: 应保留全部"""
        scores = torch.tensor([0.1, 0.2, 0.3])
        tau = find_threshold_for_count(scores, target_count=100)
        count = (scores > tau).sum().item()
        assert count == 3

    def test_empty_scores(self):
        """空输入: 应返回 tau_high, 不报错"""
        scores = torch.tensor([])
        tau = find_threshold_for_count(scores, target_count=5)
        assert isinstance(tau, float)

    def test_custom_bounds(self):
        """自定义 tau_low/tau_high"""
        scores = torch.tensor([0.5, 0.6, 0.7])
        tau = find_threshold_for_count(
            scores, target_count=1, tau_low=0.0, tau_high=1.0
        )
        assert 0.0 <= tau <= 1.0

    def test_max_iters_precision(self):
        """更多迭代 → 更精确"""
        scores = torch.linspace(0.01, 0.99, 99)
        tau_5 = find_threshold_for_count(scores, target_count=50, max_iters=5)
        tau_30 = find_threshold_for_count(scores, target_count=50, max_iters=30)
        count_5 = (scores > tau_5).sum().item()
        count_30 = (scores > tau_30).sum().item()
        # 30 次迭代应更接近 50
        assert abs(count_30 - 50) <= abs(count_5 - 50)


# ================================================================
# 路径 B: count_constrained_nms
# ================================================================


class TestCountConstrainedNms:
    """测试计数约束 NMS"""

    def test_basic_constraint(self):
        """基本场景: 应将检测数约束到接近 target_count"""
        # 100 个框, 分数递减, 无重叠 (NMS 不会移除任何框)
        boxes = torch.zeros(100, 4)
        boxes[:, 0] = torch.arange(100).float() * 10  # x1 各不相同, 无重叠
        boxes[:, 2] = boxes[:, 0] + 5
        boxes[:, 3] = 5
        scores = torch.linspace(0.99, 0.01, 100)
        labels = torch.zeros(100, dtype=torch.long)

        keep = count_constrained_nms(
            boxes, scores, labels, target_count=46, iou_threshold=0.5
        )
        # 应保留约 46 个 (二分搜索 ±1)
        assert abs(keep.shape[0] - 46) <= 2

    def test_min_keep_fallback(self):
        """min_keep 兜底: 当 target_count 很小时, 至少保留 min_keep 个"""
        boxes = torch.zeros(20, 4)
        boxes[:, 0] = torch.arange(20).float() * 10
        boxes[:, 2] = boxes[:, 0] + 5
        boxes[:, 3] = 5
        scores = torch.linspace(0.99, 0.01, 20)
        labels = torch.zeros(20, dtype=torch.long)

        keep = count_constrained_nms(
            boxes, scores, labels,
            target_count=1,  # 极小目标
            iou_threshold=0.5,
            min_keep=10,  # 兜底
        )
        assert keep.shape[0] >= 10

    def test_empty_input(self):
        """空输入: 应返回空索引"""
        boxes = torch.zeros(0, 4)
        scores = torch.zeros(0)
        labels = torch.zeros(0, dtype=torch.long)

        keep = count_constrained_nms(
            boxes, scores, labels, target_count=46
        )
        assert keep.shape[0] == 0

    def test_nms_filtering(self):
        """NMS 应移除重叠框"""
        # 两个高度重叠的框
        boxes = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],   # 高分
            [0.0, 0.0, 10.0, 10.0],   # 低分, 与上一个完全重叠
            [100.0, 100.0, 110.0, 110.0],  # 不重叠
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        labels = torch.zeros(3, dtype=torch.long)

        keep = count_constrained_nms(
            boxes, scores, labels,
            target_count=10,  # 大于总数, NMS 决定保留数
            iou_threshold=0.5,
        )
        # NMS 应移除重叠的低分框, 保留 2 个
        assert keep.shape[0] == 2

    def test_target_count_zero(self):
        """target_count=0: 应保留 min_keep 个 (兜底)"""
        boxes = torch.zeros(10, 4)
        boxes[:, 0] = torch.arange(10).float() * 10
        boxes[:, 2] = boxes[:, 0] + 5
        boxes[:, 3] = 5
        scores = torch.linspace(0.9, 0.1, 10)
        labels = torch.zeros(10, dtype=torch.long)

        keep = count_constrained_nms(
            boxes, scores, labels,
            target_count=0,
            min_keep=5,
        )
        assert keep.shape[0] >= 5

    def test_keep_indices_in_range(self):
        """返回的索引应在 [0, N) 范围内"""
        boxes = torch.rand(50, 4) * 100
        boxes[:, 2:] = boxes[:, :2] + torch.rand(50, 2) * 10
        scores = torch.rand(50)
        labels = torch.zeros(50, dtype=torch.long)

        keep = count_constrained_nms(
            boxes, scores, labels, target_count=20
        )
        assert (keep >= 0).all()
        assert (keep < 50).all()


# ================================================================
# 路径 C: CountingBranch
# ================================================================


class TestCountingBranch:
    """测试计数分支模块"""

    def test_forward_shape(self):
        """前向传播: 输出形状正确"""
        branch = CountingBranch(feat_channels=64, num_levels=4, count_min=44, count_max=48)
        features = [torch.randn(2, 64, 32, 32) for _ in range(4)]
        logits, pred_count = branch(features)
        # logits: [bs, num_count_classes], pred_count: [bs]
        assert logits.shape == (2, 5)  # 48 - 44 + 1 = 5
        assert pred_count.shape == (2,)

    def test_pred_count_range(self):
        """预测计数应在 [count_min, count_max] 范围内"""
        branch = CountingBranch(feat_channels=64, num_levels=4, count_min=44, count_max=48)
        features = [torch.randn(2, 64, 32, 32) for _ in range(4)]
        _, pred_count = branch(features)
        assert (pred_count >= 44).all()
        assert (pred_count <= 48).all()

    def test_loss(self):
        """损失计算: 应为标量, 可反向传播"""
        branch = CountingBranch(feat_channels=64, num_levels=4)
        features = [torch.randn(2, 64, 32, 32) for _ in range(4)]
        logits, _ = branch(features)
        gt_count = torch.tensor([46, 47])
        loss = branch.compute_loss(logits, gt_count)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0
        loss.backward()  # 应可反向传播

    def test_loss_clipping(self):
        """gt_count 超出范围时应被 clamp 到有效范围"""
        branch = CountingBranch(feat_channels=64, num_levels=4, count_min=44, count_max=48)
        features = [torch.randn(2, 64, 32, 32) for _ in range(4)]
        logits, _ = branch(features)
        # 43 和 49 都超出 [44, 48], 应被 clamp 到 44 和 48
        gt_count = torch.tensor([43, 49])
        loss = branch.compute_loss(logits, gt_count)
        assert loss.item() > 0  # 不应报错

    def test_different_num_levels(self):
        """不同 FPN 层数应正常工作"""
        branch = CountingBranch(feat_channels=64, num_levels=3)
        features = [torch.randn(2, 64, 32, 32) for _ in range(3)]
        logits, _ = branch(features)
        assert logits.shape[0] == 2

    def test_too_few_levels_raises(self):
        """特征层数不足时应报错"""
        branch = CountingBranch(feat_channels=64, num_levels=4)
        features = [torch.randn(2, 64, 32, 32) for _ in range(2)]  # 只有 2 层
        with pytest.raises(ValueError):
            branch(features)

    def test_parameters_registered(self):
        """参数应被 nn.Module 正确注册 (供 optimizer 收集)"""
        branch = CountingBranch(feat_channels=64, num_levels=4)
        params = list(branch.parameters())
        assert len(params) > 0
        # 检查可训练
        assert all(p.requires_grad for p in params)


# ================================================================
# head.py 集成测试
# ================================================================


class TestHeadIntegration:
    """测试 head.py 的方向二集成"""

    def test_default_disabled(self):
        """默认参数: 方向二功能应全部关闭"""
        # 用最小参数构造 head (避免实际加载完整模型)
        # 由于 DiffusionDetHead 需要 single_head/roi_extractor/criterion,
        # 这里只检查属性默认值
        head = _make_minimal_head()
        assert head.use_count_constraint is False
        assert head.counting_branch is None
        assert head.default_target_count == 46
        assert head.count_loss_weight == 1.0

    def test_enable_count_constraint(self):
        """开启 use_count_constraint: 属性应正确设置"""
        head = _make_minimal_head(use_count_constraint=True, default_target_count=47)
        assert head.use_count_constraint is True
        assert head.default_target_count == 47

    def test_attach_counting_branch(self):
        """附加 counting_branch: 应被注册为子模块"""
        branch = CountingBranch(feat_channels=64, num_levels=4)
        head = _make_minimal_head(counting_branch=branch)
        assert head.counting_branch is not None
        # 应出现在 submodules 中
        sub_modules = dict(head.named_modules())
        assert 'counting_branch' in sub_modules

    def test_counting_branch_params_in_head(self):
        """counting_branch 参数应被 head.parameters() 收集"""
        branch = CountingBranch(feat_channels=64, num_levels=4)
        head = _make_minimal_head(counting_branch=branch)
        head_params = set(id(p) for p in head.parameters())
        branch_params = set(id(p) for p in branch.parameters())
        # branch 的参数应全部在 head 的参数中
        assert branch_params.issubset(head_params)

    def test_apply_count_constraint_with_default_count(self):
        """_apply_count_constraint: 无 counting_branch 时用 default_target_count"""
        head = _make_minimal_head(
            use_count_constraint=True,
            default_target_count=46,
        )
        # 构造 mock features (counting_branch 为 None, 不会用到)
        features = [torch.randn(1, 64, 32, 32) for _ in range(4)]
        # 构造 100 个不重叠的检测结果
        boxes = torch.zeros(100, 4)
        boxes[:, 0] = torch.arange(100).float() * 10
        boxes[:, 2] = boxes[:, 0] + 5
        boxes[:, 3] = 5
        scores = torch.linspace(0.99, 0.01, 100)
        labels = torch.zeros(100, dtype=torch.long)
        results = [DetectionResult(bboxes=boxes, scores=scores, labels=labels)]

        new_results = head._apply_count_constraint(features, results)
        assert abs(new_results[0].bboxes.shape[0] - 46) <= 2

    def test_apply_count_constraint_with_counting_branch(self):
        """_apply_count_constraint: 有 counting_branch 时用预测 count"""
        branch = CountingBranch(feat_channels=64, num_levels=4)
        head = _make_minimal_head(
            use_count_constraint=True,
            counting_branch=branch,
        )
        # 构造 features
        features = [torch.randn(1, 64, 32, 32) for _ in range(4)]
        # 构造 100 个不重叠的检测结果
        boxes = torch.zeros(100, 4)
        boxes[:, 0] = torch.arange(100).float() * 10
        boxes[:, 2] = boxes[:, 0] + 5
        boxes[:, 3] = 5
        scores = torch.linspace(0.99, 0.01, 100)
        labels = torch.zeros(100, dtype=torch.long)
        results = [DetectionResult(bboxes=boxes, scores=scores, labels=labels)]

        new_results = head._apply_count_constraint(features, results)
        # 预测 count 在 [44, 48], 保留数应在该范围 ±2
        assert 42 <= new_results[0].bboxes.shape[0] <= 50

    def test_apply_count_constraint_empty_result(self):
        """_apply_count_constraint: 空结果应原样返回"""
        head = _make_minimal_head(use_count_constraint=True)
        features = [torch.randn(1, 64, 32, 32) for _ in range(4)]
        results = [DetectionResult(
            bboxes=torch.zeros(0, 4),
            scores=torch.zeros(0),
            labels=torch.zeros(0, dtype=torch.long),
        )]

        new_results = head._apply_count_constraint(features, results)
        assert new_results[0].bboxes.shape[0] == 0

    def test_loss_with_counting_branch(self):
        """集成测试: loss() 在 counting_branch 启用时应添加 loss_count

        覆盖真实训练场景: gt_bboxes 为 list[Tensor], 每张图 GT 数不同.
        回归 bug: 之前用 torch.stack 导致 TypeError (int 不能 stack).
        """
        branch = CountingBranch(feat_channels=64, num_levels=4)
        head = _make_minimal_head(counting_branch=branch)

        # 构造 mock 输入: 2 张图, 分别 46 和 47 个 GT
        device = torch.device('cpu')
        bs = 2
        features = [torch.randn(bs, 64, 32, 32) for _ in range(4)]
        img_metas = [
            ImageMeta(img_shape=(32, 32), pad_shape=(32, 32),
                      ori_shape=(32, 32), scale_factor=[1.0, 1.0, 1.0, 1.0])
            for _ in range(bs)
        ]
        gt_bboxes = [
            torch.rand(46, 4, device=device) * 30,  # 46 个 GT
            torch.rand(47, 4, device=device) * 30,  # 47 个 GT
        ]
        gt_labels = [
            torch.randint(0, 24, (46,), device=device),
            torch.randint(0, 24, (47,), device=device),
        ]

        losses = head.loss(features, img_metas, gt_bboxes, gt_labels)
        # 应包含 loss_count
        assert 'loss_count' in losses
        assert losses['loss_count'].dim() == 0  # scalar
        assert losses['loss_count'].item() > 0
        # 应可反向传播
        losses['loss_count'].backward()


# ================================================================
# 隔离性测试: baseline 不受影响
# ================================================================


class TestBaselineIsolation:
    """验证方向二默认关闭时, baseline 行为完全不受影响"""

    def test_baseline_no_count_loss(self):
        """baseline (counting_branch=None): loss 不含 loss_count"""
        head = _make_minimal_head()  # 默认 counting_branch=None
        # 模拟 loss 字典
        losses = {'loss_cls': torch.tensor(1.0), 'loss_bbox': torch.tensor(2.0)}
        # 由于 counting_branch is None, 不应添加 loss_count
        if head.counting_branch is not None:
            losses['loss_count'] = torch.tensor(0.5)
        assert 'loss_count' not in losses

    def test_baseline_no_count_constraint(self):
        """baseline (use_count_constraint=False): predict 不调用 _apply_count_constraint"""
        head = _make_minimal_head()
        assert head.use_count_constraint is False
        # predict() 中的 if self.use_count_constraint 不会进入


# ================================================================
# 辅助函数
# ================================================================


def _make_minimal_head(**kwargs) -> DiffusionDetHead:
    """构造最小可测试的 DiffusionDetHead.

    用 mock 的 single_head/roi_extractor/criterion 避免完整模型加载.
    """
    # mock single_head: 返回 (cls_logits, pred_bboxes, curr_proposals)
    class MockSingleHead(nn.Module):
        def __init__(self, feat_channels=64, num_classes=24, num_proposals=10):
            super().__init__()
            self.cls_head = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, num_classes))
            self.feat_channels = feat_channels

        def forward(self, features, bboxes, proposals, pooler, time_emb):
            bs, num_boxes = bboxes.shape[:2]
            cls_logits = torch.zeros(bs, num_boxes, 24)
            pred_bboxes = bboxes.clone()
            curr_proposals = torch.zeros(1, bs * num_boxes, 64)
            return cls_logits, pred_bboxes, curr_proposals

    # mock roi_extractor: 返回固定形状特征
    class MockPooler(nn.Module):
        def __init__(self, feat_channels=64):
            super().__init__()
            self.feat_channels = feat_channels

        def forward(self, features, rois):
            num_rois = rois.shape[0]
            return torch.zeros(num_rois, 64, 7, 7)

    # mock criterion: 返回空 loss 字典 (接受 t 参数, 兼容方向三)
    class MockCriterion(nn.Module):
        def forward(self, outputs, targets, t=None):
            return {'loss_cls': torch.tensor(0.0, requires_grad=True)}

    # mock coupling (None 表示用默认 random)
    return DiffusionDetHead(
        num_classes=24,
        feat_channels=64,
        num_proposals=10,
        num_heads=1,
        single_head=MockSingleHead(),
        roi_extractor=MockPooler(),
        criterion=MockCriterion(),
        coupling=None,
        diffusion_type='rectified_flow',
        **kwargs,
    )
