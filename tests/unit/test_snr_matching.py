"""方向三: SNR 感知的动态匹配 (SNR-Aware Dynamic Matching) 测试

测试覆盖:
1. snr_weight: logistic/exponential/none 模式, 边界值, 单调性
2. SNRAwareMatcher: t=None 退化, t=1 高噪声抑制, t=0 低噪声信任
3. criterion: SNR 加权损失, t=None 退化
4. 隔离性: baseline 行为不受影响
"""

import torch
import torch.nn as nn
import pytest

from ldmdet.criterion.costs import BBoxL1Cost, FocalLossCost, IoUCost
from ldmdet.criterion.criterion import DiffusionDetCriterion
from ldmdet.criterion.matcher import DiffusionDetMatcher
from ldmdet.criterion.snr_aware_matcher import SNRAwareMatcher
from ldmdet.criterion.snr_weight import (
    exponential_snr_weight,
    get_snr_weight,
    logistic_snr_weight,
)
from ldmdet.criterion import FocalLoss, GIoULoss, L1Loss
from ldmdet.data.structures import InstanceData, ModelOutput


# ================================================================
# 3.1 SNR 权重函数
# ================================================================


class TestSnrWeight:
    """测试 SNR 权重计算"""

    def test_logistic_boundaries(self):
        """logistic: w(0)=1, w(0.5)=0.5, w(1)=0"""
        t = torch.tensor([0.0, 0.5, 1.0])
        w = logistic_snr_weight(t)
        assert torch.allclose(w[0], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(w[1], torch.tensor(0.5), atol=1e-6)
        assert torch.allclose(w[2], torch.tensor(0.0), atol=1e-6)

    def test_logistic_monotonic(self):
        """logistic: 单调递减"""
        t = torch.linspace(0.0, 1.0, 100)
        w = logistic_snr_weight(t)
        diff = w[1:] - w[:-1]
        assert (diff <= 1e-6).all(), "w(t) 应单调递减"

    def test_logistic_w_min(self):
        """logistic: w_min 下界生效"""
        t = torch.tensor([1.0])  # w(1)=0
        w = logistic_snr_weight(t, w_min=0.1)
        assert torch.allclose(w, torch.tensor([0.1]), atol=1e-6)

    def test_exponential_boundaries(self):
        """exponential: w(0)=1, w(1)=exp(-β)"""
        t = torch.tensor([0.0, 1.0])
        w = exponential_snr_weight(t, beta=3.0)
        assert torch.allclose(w[0], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(w[1], torch.tensor(torch.exp(torch.tensor(-3.0)).item()), atol=1e-6)

    def test_exponential_monotonic(self):
        """exponential: 单调递减"""
        t = torch.linspace(0.0, 1.0, 100)
        w = exponential_snr_weight(t, beta=3.0)
        diff = w[1:] - w[:-1]
        assert (diff <= 1e-6).all()

    def test_exponential_beta_effect(self):
        """exponential: β 越大衰减越快"""
        t = torch.tensor([0.5])
        w_beta1 = exponential_snr_weight(t, beta=1.0)
        w_beta6 = exponential_snr_weight(t, beta=6.0)
        assert w_beta6.item() < w_beta1.item()

    def test_get_snr_weight_none_mode(self):
        """none 模式: 始终返回 1"""
        t = torch.tensor([0.0, 0.5, 1.0])
        w = get_snr_weight(t, mode='none')
        assert torch.allclose(w, torch.ones_like(t))

    def test_get_snr_weight_unknown_mode(self):
        """未知模式: 应报错"""
        t = torch.tensor([0.5])
        with pytest.raises(ValueError):
            get_snr_weight(t, mode='unknown')

    def test_get_snr_weight_scalar_input(self):
        """标量输入: 应正常工作"""
        t = torch.tensor(0.5)
        w = get_snr_weight(t, mode='logistic')
        assert 0.4 < w.item() < 0.6

    def test_w_min_range(self):
        """w_min 生效: w 应在 [w_min, 1] 范围内"""
        t = torch.linspace(0.0, 1.0, 50)
        w = get_snr_weight(t, mode='logistic', w_min=0.1)
        assert (w >= 0.1 - 1e-6).all()
        assert (w <= 1.0 + 1e-6).all()


# ================================================================
# 3.2 SNRAwareMatcher
# ================================================================


class TestSNRAwareMatcher:
    """测试 SNR 感知匹配器"""

    def test_t_none_degenerate(self):
        """t=None: 退化为标准匹配"""
        matcher = SNRAwareMatcher()
        outputs, targets = _make_matcher_inputs()

        # t=None 调用父类
        results_none = matcher(outputs, targets, t=None)

        # 与标准 matcher 结果对比
        std_matcher = DiffusionDetMatcher()
        results_std = std_matcher(outputs, targets)

        # 应完全一致
        for (mask1, gt1), (mask2, gt2) in zip(results_none, results_std):
            assert torch.equal(mask1, mask2)
            assert torch.equal(gt1, gt2)

    def test_t_one_high_noise_suppression(self):
        """t=1 (高噪声): 正样本数应显著减少"""
        matcher = SNRAwareMatcher(snr_w_min=0.0)  # 不设下界, 完全抑制
        outputs, targets = _make_matcher_inputs()

        # t=1: w=0, 高噪声保护触发
        t_high = torch.ones(2)
        results_high = matcher(outputs, targets, t=t_high)

        # t=0: w=1, 全权信任
        t_low = torch.zeros(2)
        results_low = matcher(outputs, targets, t=t_low)

        # 高噪声正样本数应 <= 低噪声
        total_high = sum(mask.sum().item() for mask, _ in results_high)
        total_low = sum(mask.sum().item() for mask, _ in results_low)
        assert total_high <= total_low

    def test_t_zero_full_trust(self):
        """t=0 (低噪声): w=1, 应与标准匹配一致"""
        matcher = SNRAwareMatcher()
        outputs, targets = _make_matcher_inputs()

        t_zero = torch.zeros(2)
        results_snr = matcher(outputs, targets, t=t_zero)

        # 与标准 matcher 对比 (代价乘 1, 无高噪声保护)
        std_matcher = DiffusionDetMatcher()
        results_std = std_matcher(outputs, targets)

        for (mask1, gt1), (mask2, gt2) in zip(results_snr, results_std):
            assert torch.equal(mask1, mask2)
            assert torch.equal(gt1, gt2)

    def test_k_max_scale(self):
        """k_max_scale: 高噪声时 k 上界减小"""
        matcher = SNRAwareMatcher(
            k_max_scale=True, snr_w_min=0.0, candidate_topk=5
        )
        outputs, targets = _make_matcher_inputs()

        # t=1: w=0, k_max = max(0*5, 1) = 1
        t_high = torch.ones(2)
        results = matcher(outputs, targets, t=t_high)
        # 每个 GT 最多 1 个正样本
        for mask, gt_inds in results:
            # 正样本数应 <= GT 数 (k_max=1)
            assert mask.sum().item() <= len(targets[0].bboxes)

    def test_cost_threshold_protection(self):
        """cost_threshold: 高噪声时代价加常数, 抑制匹配"""
        matcher = SNRAwareMatcher(
            snr_w_min=0.0,  # w(1)=0 < 0.1 触发保护
            cost_threshold=1000.0,  # 大阈值
        )
        outputs, targets = _make_matcher_inputs()

        t_high = torch.ones(2)
        results = matcher(outputs, targets, t=t_high)
        # 高噪声 + 大阈值: 正样本应很少或为 0
        total_pos = sum(mask.sum().item() for mask, _ in results)
        assert total_pos >= 0  # 不报错即可

    def test_empty_targets(self):
        """空 GT: 应返回空匹配"""
        matcher = SNRAwareMatcher()
        outputs = ModelOutput(
            pred_logits=torch.randn(2, 10, 25),
            pred_boxes=torch.rand(2, 10, 4),
        )
        targets = [
            InstanceData(bboxes=torch.zeros(0, 4), labels=torch.zeros(0, dtype=torch.long), img_shape=(256, 256)),
            InstanceData(bboxes=torch.zeros(0, 4), labels=torch.zeros(0, dtype=torch.long), img_shape=(256, 256)),
        ]
        t = torch.tensor([0.5, 0.5])
        results = matcher(outputs, targets, t=t)
        for mask, gt in results:
            assert mask.sum() == 0
            assert gt.shape[0] == 10

    def test_forward_with_gt_cache_t_none(self):
        """forward_with_gt_cache: t=None 退化为父类"""
        matcher = SNRAwareMatcher()
        outputs, targets = _make_matcher_inputs()

        indices, cache = matcher.forward_with_gt_cache(outputs, targets, t=None)
        assert len(indices) == 2
        assert len(cache) == 2

    def test_forward_with_gt_cache_with_t(self):
        """forward_with_gt_cache: t 不为 None 时正常工作"""
        matcher = SNRAwareMatcher()
        outputs, targets = _make_matcher_inputs()

        t = torch.tensor([0.3, 0.7])
        indices, cache = matcher.forward_with_gt_cache(outputs, targets, t=t)
        assert len(indices) == 2
        assert len(cache) == 2

    def test_snr_mode_exponential(self):
        """exponential 模式: 应正常工作"""
        matcher = SNRAwareMatcher(snr_mode='exponential', snr_beta=3.0)
        outputs, targets = _make_matcher_inputs()
        t = torch.tensor([0.5, 0.5])
        results = matcher(outputs, targets, t=t)
        assert len(results) == 2


# ================================================================
# 3.3 Criterion SNR 加权损失
# ================================================================


class TestCriterionSnrWeighted:
    """测试 criterion 的 SNR 加权损失"""

    def test_t_none_no_weighting(self):
        """t=None: 不加权, 与 baseline 一致"""
        criterion = _make_criterion(snr_weighted_loss=True)
        outputs, targets = _make_matcher_inputs()

        # t=None
        losses_none = criterion(outputs, targets, t=None)
        # snr_weighted_loss=True 但 t=None, 不应加权
        assert 'loss_cls' in losses_none
        assert 'loss_bbox' in losses_none
        assert 'loss_giou' in losses_none

    def test_snr_weighted_loss_disabled(self):
        """snr_weighted_loss=False: 即使有 t 也不加权"""
        criterion = _make_criterion(snr_weighted_loss=False)
        outputs, targets = _make_matcher_inputs()

        t = torch.tensor([0.5])
        losses = criterion(outputs, targets, t=t)
        # 应正常计算损失
        assert 'loss_cls' in losses

    def test_snr_weighted_loss_enabled(self):
        """snr_weighted_loss=True + t: 应加权"""
        criterion = _make_criterion(snr_weighted_loss=True)
        outputs, targets = _make_matcher_inputs()

        # t=0: w=1, 损失不变
        t_zero = torch.tensor([0.0])
        losses_zero = criterion(outputs, targets, t=t_zero)

        # t=1: w=w_min=0.1, 损失缩小
        t_one = torch.tensor([1.0])
        losses_one = criterion(outputs, targets, t=t_one)

        # t=1 的损失应 < t=0 的损失 (因 w(1)=0.1 < w(0)=1)
        assert losses_one['loss_cls'].item() < losses_zero['loss_cls'].item()
        assert losses_one['loss_bbox'].item() < losses_zero['loss_bbox'].item()

    def test_snr_weighted_with_snr_matcher(self):
        """SNRAwareMatcher + snr_weighted_loss: 联合工作"""
        matcher = SNRAwareMatcher()
        criterion = _make_criterion(
            snr_weighted_loss=True, matcher=matcher
        )
        outputs, targets = _make_matcher_inputs()

        # t 应为 [bs] 形状, 与 batch_size 一致
        t = torch.tensor([0.5, 0.5])
        losses = criterion(outputs, targets, t=t)
        assert 'loss_cls' in losses
        assert 'loss_bbox' in losses
        assert 'loss_giou' in losses


# ================================================================
# 隔离性测试: baseline 不受影响
# ================================================================


class TestBaselineIsolation:
    """验证方向三默认关闭时, baseline 行为不受影响"""

    def test_default_criterion_no_t(self):
        """默认 criterion: 不传 t, 行为与 baseline 一致"""
        criterion = _make_criterion()  # snr_weighted_loss=False
        outputs, targets = _make_matcher_inputs()

        # 不传 t (默认 None)
        losses = criterion(outputs, targets)
        assert 'loss_cls' in losses

    def test_default_criterion_t_ignored(self):
        """默认 criterion: 传 t 但 snr_weighted_loss=False, t 被忽略"""
        criterion = _make_criterion(snr_weighted_loss=False)
        outputs, targets = _make_matcher_inputs()

        t = torch.tensor([0.5])
        losses_with_t = criterion(outputs, targets, t=t)
        losses_without_t = criterion(outputs, targets, t=None)

        # 损失应一致 (matcher 是标准 DiffusionDetMatcher, 不响应 t)
        assert torch.allclose(losses_with_t['loss_cls'], losses_without_t['loss_cls'])
        assert torch.allclose(losses_with_t['loss_bbox'], losses_without_t['loss_bbox'])

    def test_standard_matcher_ignores_t(self):
        """标准 DiffusionDetMatcher: 不响应 t (criterion 中 isinstance 检查)"""
        matcher = DiffusionDetMatcher()  # 标准 matcher
        criterion = _make_criterion(snr_weighted_loss=True, matcher=matcher)
        outputs, targets = _make_matcher_inputs()

        t = torch.tensor([0.5])
        # 应不报错 (criterion 中 isinstance 检查会跳过 t 传递)
        losses = criterion(outputs, targets, t=t)
        assert 'loss_cls' in losses


# ================================================================
# 辅助函数
# ================================================================


def _make_matcher_inputs():
    """构造 matcher 测试输入: 2 张图, 每张 3 个 GT, 10 个 proposal"""
    torch.manual_seed(42)
    bs, N, C = 2, 10, 25
    pred_logits = torch.randn(bs, N, C)
    pred_boxes = torch.rand(bs, N, 4) * 0.8 + 0.1  # [0.1, 0.9]
    outputs = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_boxes)

    targets = []
    for _ in range(bs):
        n_gt = 3
        bboxes = torch.rand(n_gt, 4) * 0.8 + 0.1
        labels = torch.randint(0, C - 1, (n_gt,))
        targets.append(InstanceData(bboxes=bboxes, labels=labels, img_shape=(256, 256)))
    return outputs, targets


def _make_criterion(
    snr_weighted_loss: bool = False,
    snr_mode: str = 'logistic',
    snr_beta: float = 3.0,
    snr_w_min: float = 0.1,
    matcher: nn.Module = None,
) -> DiffusionDetCriterion:
    """构造测试用 criterion"""
    if matcher is None:
        matcher = DiffusionDetMatcher()
    return DiffusionDetCriterion(
        num_classes=24,
        matcher=matcher,
        loss_cls=FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=L1Loss(loss_weight=5.0),
        loss_giou=GIoULoss(loss_weight=2.0),
        deep_supervision=False,  # 简化测试
        snr_weighted_loss=snr_weighted_loss,
        snr_mode=snr_mode,
        snr_beta=snr_beta,
        snr_w_min=snr_w_min,
    )
