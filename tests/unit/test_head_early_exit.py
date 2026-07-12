"""IO4 级联头提前退出单元测试

测试方案核心数学理论:
1. 收敛判据:
   - 框位置: 框对角线归一化的 L2 变化 < box_threshold
   - 分类: 高置信度框 (sigmoid > 0.3) 的 argmax 一致率 > cls_threshold
2. 时间步感知: 不同 t 使用不同阈值
   - 早期 (t_norm > 0.5): 严格 (box=0.002, cls=0.99) — 框变化大, 少退出
   - 中期 (0.2 < t_norm ≤ 0.5): 适中 (box=0.005, cls=0.98)
   - 后期 (t_norm ≤ 0.2): 宽松 (box=0.01, cls=0.95) — 框已稳定, 多退出
3. min_heads 保证: 至少执行 min_heads 个头后才检测收敛
4. 输出形状不变: 提前退出后用最后一头结果填充, stack 形状始终 [num_heads, ...]
5. 训练时不退出: training=True 时始终跑完所有头 (保持 deep supervision)
6. 推理时才退出: training=False 时才启用提前退出
"""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ldmdet.core.head import DiffusionDetHead


# ================================================================
# Mock 组件: 构建可测试的 DiffusionDetHead
# ================================================================


class MockSingleHead(nn.Module):
    """Mock single head, 返回可控的框精化和分类输出。

    Args:
        box_delta: 框变化幅度 (像素). 0.0 = 完全收敛, 大值 = 发散
        cls_consistent: True=分类固定一致, False=分类随机变化
        num_classes: 类别数
    """

    def __init__(self, box_delta=0.0, cls_consistent=True, num_classes=24):
        super().__init__()
        self.box_delta = box_delta
        self.cls_consistent = cls_consistent
        self.num_classes = num_classes

    def forward(self, features, bboxes, curr_proposals, roi_extractor, time_emb):
        bs, n = bboxes.shape[:2]
        noise = torch.randn_like(bboxes) * self.box_delta
        pred_bboxes = bboxes + noise
        if self.cls_consistent:
            cls_logits = torch.zeros(bs, n, self.num_classes)
            cls_logits[..., 0] = 10.0
        else:
            cls_logits = torch.randn(bs, n, self.num_classes) * 5.0
        return cls_logits, pred_bboxes, bboxes


def _make_lightweight_head(**kwargs):
    """创建轻量 head 对象, 只设置 IO4 相关属性, 绑定方法。

    用于纯收敛检测方法测试, 避免构建完整 DiffusionDetHead。
    """
    defaults = dict(
        head_early_exit_enabled=True,
        head_exit_box_threshold=0.005,
        head_exit_cls_threshold=0.98,
        head_exit_min_heads=3,
        head_exit_max_heads=6,
        head_exit_time_aware=True,
        timesteps=1000,
        num_heads=6,
        training=False,
        _exit_stats={},
    )
    defaults.update(kwargs)
    head = SimpleNamespace(**defaults)
    head._check_head_convergence = (
        DiffusionDetHead._check_head_convergence.__get__(head)
    )
    head._get_exit_threshold = (
        DiffusionDetHead._get_exit_threshold.__get__(head)
    )
    return head


def _make_full_head(
    box_delta=0.0,
    cls_consistent=True,
    num_heads=6,
    min_heads=3,
    deep_supervision=True,
    **io4_kwargs,
):
    """创建完整 DiffusionDetHead, 用 MockSingleHead。

    Args:
        box_delta: 传给 MockSingleHead 的框变化幅度
        cls_consistent: 传给 MockSingleHead 的分类一致性
        num_heads: 级联头数
        min_heads: 最少执行头数
        deep_supervision: 是否启用深度监督
        **io4_kwargs: IO4 参数覆盖
    """
    mock_single = MockSingleHead(
        box_delta=box_delta, cls_consistent=cls_consistent, num_classes=24
    )
    io4_defaults = dict(
        head_early_exit_enabled=True,
        head_exit_box_threshold=0.005,
        head_exit_cls_threshold=0.98,
        head_exit_min_heads=min_heads,
        head_exit_max_heads=num_heads,
        head_exit_time_aware=True,
    )
    io4_defaults.update(io4_kwargs)
    head = DiffusionDetHead(
        num_classes=24,
        feat_channels=256,
        num_proposals=100,
        num_heads=num_heads,
        single_head=mock_single,
        roi_extractor=None,
        criterion=None,
        coupling=None,
        diffusion_type='rectified_flow',
        deep_supervision=deep_supervision,
        **io4_defaults,
    )
    return head


# ================================================================
# 测试 1: 收敛检测数学正确性
# ================================================================


class TestCheckHeadConvergence:
    """测试 _check_head_convergence 方法的数学正确性。"""

    def setup_method(self):
        self.head = _make_lightweight_head()
        self.bs, self.n, self.num_classes = 2, 100, 24
        # 框: xyxy 格式, 中心 (50,50), 大小 40x40, 对角线 ≈ 56.57
        self.base_bboxes = (
            torch.tensor([30.0, 30.0, 70.0, 70.0])
            .expand(self.bs, self.n, 4)
            .clone()
        )

    def test_converged_boxes_and_cls(self):
        """框变化极小 + 分类一致 → converged=True"""
        curr_bboxes = self.base_bboxes.clone()
        prev_bboxes = self.base_bboxes.clone() + 0.001  # 极小变化
        curr_logits = torch.zeros(self.bs, self.n, self.num_classes)
        curr_logits[..., 0] = 10.0  # 高置信度 class 0
        prev_logits = curr_logits.clone()
        t = torch.full((self.bs,), 100.0)  # t_norm=0.1, 后期步

        converged, stats = self.head._check_head_convergence(
            curr_bboxes, prev_bboxes, curr_logits, prev_logits, t
        )
        assert converged is True
        assert stats['mean_relative_delta'] < 0.01

    def test_diverged_boxes(self):
        """框变化大 → converged=False"""
        curr_bboxes = self.base_bboxes.clone()
        prev_bboxes = self.base_bboxes.clone() + 10.0  # 大变化
        curr_logits = torch.zeros(self.bs, self.n, self.num_classes)
        curr_logits[..., 0] = 10.0
        prev_logits = curr_logits.clone()
        t = torch.full((self.bs,), 100.0)

        converged, stats = self.head._check_head_convergence(
            curr_bboxes, prev_bboxes, curr_logits, prev_logits, t
        )
        assert converged is False
        assert stats['mean_relative_delta'] > 0.01

    def test_diverged_cls(self):
        """分类不一致 → converged=False"""
        curr_bboxes = self.base_bboxes.clone()
        prev_bboxes = self.base_bboxes.clone()
        curr_logits = torch.zeros(self.bs, self.n, self.num_classes)
        curr_logits[..., 0] = 10.0  # class 0
        prev_logits = torch.zeros(self.bs, self.n, self.num_classes)
        prev_logits[..., 1] = 10.0  # class 1 (不同)
        t = torch.full((self.bs,), 100.0)

        converged, stats = self.head._check_head_convergence(
            curr_bboxes, prev_bboxes, curr_logits, prev_logits, t
        )
        assert converged is False
        assert stats['mean_consistency'] < 0.98

    def test_low_confidence_ignored(self):
        """低置信度框 (sigmoid < 0.3) 不参与分类收敛判定。"""
        curr_bboxes = self.base_bboxes.clone()
        prev_bboxes = self.base_bboxes.clone()
        # 所有框低置信度: sigmoid(-2.0) ≈ 0.12 < 0.3
        curr_logits = torch.full(
            (self.bs, self.n, self.num_classes), -2.0
        )
        curr_logits[:, :, 0] = -1.9  # 略高, argmax=0
        prev_logits = torch.full(
            (self.bs, self.n, self.num_classes), -2.0
        )
        prev_logits[:, :, 1] = -1.9  # 略高, argmax=1
        t = torch.full((self.bs,), 100.0)

        converged, stats = self.head._check_head_convergence(
            curr_bboxes, prev_bboxes, curr_logits, prev_logits, t
        )
        # 低置信度框被忽略, mean_consistency 默认 1.0
        # 框无变化 → converged=True
        assert converged is True
        assert stats['mean_consistency'] == pytest.approx(1.0, abs=0.01)

    def test_time_aware_early_strict(self):
        """早期步 (t_norm > 0.5) 用严格阈值, 不易退出。"""
        # 框变化 0.1 像素:
        # delta = ||[0.1,0.1,0.1,0.1]|| = 0.2
        # box_scale = sqrt(40²+40²) ≈ 56.57
        # relative_delta = 0.2 / 56.57 ≈ 0.00354
        curr_bboxes = self.base_bboxes.clone()
        prev_bboxes = self.base_bboxes.clone() + 0.1
        curr_logits = torch.zeros(self.bs, self.n, self.num_classes)
        curr_logits[..., 0] = 10.0
        prev_logits = curr_logits.clone()

        # 早期步: t=800, t_norm=0.8, 阈值 0.002 (严格)
        t_early = torch.full((self.bs,), 800.0)
        converged_early, stats_early = self.head._check_head_convergence(
            curr_bboxes, prev_bboxes, curr_logits, prev_logits, t_early
        )
        # 0.00354 > 0.002 → 框不收敛
        assert converged_early is False
        assert stats_early['box_thr'] == pytest.approx(0.002, abs=1e-6)

    def test_time_aware_late_lenient(self):
        """后期步 (t_norm ≤ 0.2) 用宽松阈值, 易退出。"""
        curr_bboxes = self.base_bboxes.clone()
        prev_bboxes = self.base_bboxes.clone() + 0.1
        curr_logits = torch.zeros(self.bs, self.n, self.num_classes)
        curr_logits[..., 0] = 10.0
        prev_logits = curr_logits.clone()

        # 后期步: t=100, t_norm=0.1, 阈值 0.01 (宽松)
        t_late = torch.full((self.bs,), 100.0)
        converged_late, stats_late = self.head._check_head_convergence(
            curr_bboxes, prev_bboxes, curr_logits, prev_logits, t_late
        )
        # 0.00354 < 0.01 → 框收敛
        assert converged_late is True
        assert stats_late['box_thr'] == pytest.approx(0.01, abs=1e-6)

    def test_time_aware_mid_moderate(self):
        """中期步 (0.2 < t_norm ≤ 0.5) 用适中阈值。"""
        curr_bboxes = self.base_bboxes.clone()
        prev_bboxes = self.base_bboxes.clone() + 0.1
        curr_logits = torch.zeros(self.bs, self.n, self.num_classes)
        curr_logits[..., 0] = 10.0
        prev_logits = curr_logits.clone()

        # 中期步: t=400, t_norm=0.4, 阈值 0.005
        t_mid = torch.full((self.bs,), 400.0)
        converged_mid, stats_mid = self.head._check_head_convergence(
            curr_bboxes, prev_bboxes, curr_logits, prev_logits, t_mid
        )
        # 0.00354 < 0.005 → 框收敛
        assert converged_mid is True
        assert stats_mid['box_thr'] == pytest.approx(0.005, abs=1e-6)

    def test_time_aware_disabled(self):
        """关闭时间步感知时用固定阈值。"""
        head = _make_lightweight_head(head_exit_time_aware=False)
        curr_bboxes = self.base_bboxes.clone()
        prev_bboxes = self.base_bboxes.clone() + 0.1
        curr_logits = torch.zeros(self.bs, self.n, self.num_classes)
        curr_logits[..., 0] = 10.0
        prev_logits = curr_logits.clone()

        t = torch.full((self.bs,), 800.0)  # 早期步, 但关闭感知
        converged, stats = head._check_head_convergence(
            curr_bboxes, prev_bboxes, curr_logits, prev_logits, t
        )
        # 固定阈值 0.005, 0.00354 < 0.005 → 收敛
        assert converged is True
        assert stats['box_thr'] == pytest.approx(0.005, abs=1e-6)

    def test_stats_returned(self):
        """收敛检测返回完整统计信息。"""
        curr_bboxes = self.base_bboxes.clone()
        prev_bboxes = self.base_bboxes.clone()
        curr_logits = torch.zeros(self.bs, self.n, self.num_classes)
        curr_logits[..., 0] = 10.0
        prev_logits = curr_logits.clone()
        t = torch.full((self.bs,), 100.0)

        _, stats = self.head._check_head_convergence(
            curr_bboxes, prev_bboxes, curr_logits, prev_logits, t
        )
        assert 'mean_relative_delta' in stats
        assert 'mean_consistency' in stats
        assert 'box_thr' in stats
        assert 'cls_thr' in stats
        assert 't_norm' in stats
        assert 'converged' in stats


# ================================================================
# 测试 2: get_exit_threshold
# ================================================================


class TestGetExitThreshold:
    """测试 _get_exit_threshold 方法。"""

    def setup_method(self):
        self.head = _make_lightweight_head()

    def test_early_step(self):
        """t_norm > 0.5 → 严格阈值 (0.002, 0.99)。"""
        box_thr, cls_thr = self.head._get_exit_threshold(0.8)
        assert box_thr == pytest.approx(0.002, abs=1e-6)
        assert cls_thr == pytest.approx(0.99, abs=1e-6)

    def test_mid_step(self):
        """0.2 < t_norm ≤ 0.5 → 适中阈值 (0.005, 0.98)。"""
        box_thr, cls_thr = self.head._get_exit_threshold(0.4)
        assert box_thr == pytest.approx(0.005, abs=1e-6)
        assert cls_thr == pytest.approx(0.98, abs=1e-6)

    def test_late_step(self):
        """t_norm ≤ 0.2 → 宽松阈值 (0.01, 0.95)。"""
        box_thr, cls_thr = self.head._get_exit_threshold(0.1)
        assert box_thr == pytest.approx(0.01, abs=1e-6)
        assert cls_thr == pytest.approx(0.95, abs=1e-6)

    def test_boundary_0_5(self):
        """t_norm = 0.5 → 中期阈值。"""
        box_thr, _ = self.head._get_exit_threshold(0.5)
        assert box_thr == pytest.approx(0.005, abs=1e-6)

    def test_boundary_0_2(self):
        """t_norm = 0.2 → 后期阈值。"""
        box_thr, _ = self.head._get_exit_threshold(0.2)
        assert box_thr == pytest.approx(0.01, abs=1e-6)


# ================================================================
# 测试 3: forward 集成 — 提前退出逻辑
# ================================================================


class TestForwardWithEarlyExit:
    """测试 forward 方法中的提前退出逻辑。"""

    def test_disabled_no_exit(self):
        """禁用时跑完所有头, 无退出统计。"""
        head = _make_full_head(
            box_delta=0.0,  # 框无变化 (会收敛)
            cls_consistent=True,
            head_early_exit_enabled=False,  # 禁用
        )
        head.eval()

        features = [torch.randn(1, 256, 16, 16)]
        bboxes = (
            torch.tensor([30.0, 30.0, 70.0, 70.0])
            .expand(1, 100, 4)
            .clone()
        )
        t = torch.full((1,), 100.0)  # 后期步

        cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        assert cls_logits.shape[0] == 6  # 跑完 6 头
        assert head._exit_stats == {}  # 无退出统计

    def test_enabled_converged_early_exit(self):
        """启用且收敛时提前退出。"""
        head = _make_full_head(
            box_delta=0.0,  # 框无变化 → 收敛
            cls_consistent=True,  # 分类一致 → 收敛
            min_heads=3,
        )
        head.eval()

        features = [torch.randn(1, 256, 16, 16)]
        bboxes = (
            torch.tensor([30.0, 30.0, 70.0, 70.0])
            .expand(1, 100, 4)
            .clone()
        )
        t = torch.full((1,), 100.0)  # 后期步

        cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        # 提前退出, exit_head_idx < 6
        assert head._exit_stats.get('exit_head_idx', 6) < 6
        # 至少 min_heads=3 个头 (idx >= 2)
        assert head._exit_stats['exit_head_idx'] >= 2

    def test_enabled_diverged_no_exit(self):
        """启用但框发散时不提前退出, 跑完所有头。"""
        head = _make_full_head(
            box_delta=20.0,  # 框变化大 → 不收敛
            cls_consistent=True,
            min_heads=3,
        )
        head.eval()

        features = [torch.randn(1, 256, 16, 16)]
        bboxes = (
            torch.tensor([30.0, 30.0, 70.0, 70.0])
            .expand(1, 100, 4)
            .clone()
        )
        t = torch.full((1,), 100.0)

        cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        # 不收敛, 跑完 6 头
        assert head._exit_stats.get('exit_head_idx', 6) == 6 or not head._exit_stats

    def test_min_heads_respected(self):
        """至少执行 min_heads 个头后才检测收敛。"""
        head = _make_full_head(
            box_delta=0.0,
            cls_consistent=True,
            min_heads=5,  # 至少 5 头
        )
        head.eval()

        features = [torch.randn(1, 256, 16, 16)]
        bboxes = (
            torch.tensor([30.0, 30.0, 70.0, 70.0])
            .expand(1, 100, 4)
            .clone()
        )
        t = torch.full((1,), 100.0)

        head(features, bboxes, t)
        # 退出头 idx >= min_heads - 1 = 4
        exit_idx = head._exit_stats.get('exit_head_idx', 6)
        if exit_idx < 6:
            assert exit_idx >= 4  # 至少第 5 头才退出

    def test_training_always_full(self):
        """训练模式始终跑完所有头, 不启用提前退出。"""
        head = _make_full_head(
            box_delta=0.0,  # 框无变化 (会收敛)
            cls_consistent=True,
        )
        head.train()  # 训练模式

        features = [torch.randn(1, 256, 16, 16)]
        bboxes = (
            torch.tensor([30.0, 30.0, 70.0, 70.0])
            .expand(1, 100, 4)
            .clone()
        )
        t = torch.full((1,), 100.0)

        cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        assert cls_logits.shape[0] == 6  # 跑完 6 头
        assert head._exit_stats == {}  # 无退出统计

    def test_output_shape_preserved(self):
        """提前退出后输出形状始终 [num_heads, bs, N, ...]。"""
        head = _make_full_head(
            box_delta=0.0,
            cls_consistent=True,
            min_heads=3,
        )
        head.eval()

        features = [torch.randn(1, 256, 16, 16)]
        bboxes = (
            torch.tensor([30.0, 30.0, 70.0, 70.0])
            .expand(1, 100, 4)
            .clone()
        )
        t = torch.full((1,), 100.0)

        cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        # 形状始终 [6, 1, 100, 24] 和 [6, 1, 100, 4]
        assert cls_logits.shape == (6, 1, 100, 24)
        assert pred_bboxes.shape == (6, 1, 100, 4)

    def test_filled_heads_equal_exit_head(self):
        """退出后填充的头输出 == 退出头输出。"""
        head = _make_full_head(
            box_delta=0.0,
            cls_consistent=True,
            min_heads=3,
        )
        head.eval()

        features = [torch.randn(1, 256, 16, 16)]
        bboxes = (
            torch.tensor([30.0, 30.0, 70.0, 70.0])
            .expand(1, 100, 4)
            .clone()
        )
        t = torch.full((1,), 100.0)

        cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        exit_idx = head._exit_stats.get('exit_head_idx', -1)

        if exit_idx >= 0 and exit_idx < 5:
            # 填充头 (exit_idx+1 到 5) 应等于 exit_idx 的输出
            for j in range(exit_idx + 1, 6):
                assert torch.allclose(
                    cls_logits[j], cls_logits[exit_idx], atol=1e-6
                )
                assert torch.allclose(
                    pred_bboxes[j], pred_bboxes[exit_idx], atol=1e-6
                )

    def test_exit_stats_recorded(self):
        """退出时记录完整统计信息。"""
        head = _make_full_head(
            box_delta=0.0,
            cls_consistent=True,
            min_heads=3,
        )
        head.eval()

        features = [torch.randn(1, 256, 16, 16)]
        bboxes = (
            torch.tensor([30.0, 30.0, 70.0, 70.0])
            .expand(1, 100, 4)
            .clone()
        )
        t = torch.full((1,), 100.0)

        head(features, bboxes, t)
        if head._exit_stats:
            assert 'exit_head_idx' in head._exit_stats
            assert 'total_heads' in head._exit_stats
            assert 'mean_relative_delta' in head._exit_stats
            assert 'mean_consistency' in head._exit_stats
            assert 't_norm' in head._exit_stats

    def test_early_step_less_likely_exit(self):
        """早期步 (t大) 比后期步更不容易退出。"""
        # 用相同框变化, 比较早期和后期的退出行为
        features = [torch.randn(1, 256, 16, 16)]
        bboxes = (
            torch.tensor([30.0, 30.0, 70.0, 70.0])
            .expand(1, 100, 4)
            .clone()
        )

        # 框变化 0.1 像素: relative_delta ≈ 0.00354
        # 早期阈值 0.002 → 不收敛; 后期阈值 0.01 → 收敛
        head_early = _make_full_head(
            box_delta=0.1, cls_consistent=True, min_heads=3
        )
        head_early.eval()
        head_early(features, bboxes, torch.full((1,), 800.0))  # 早期
        early_exit = head_early._exit_stats.get('exit_head_idx', 6)

        head_late = _make_full_head(
            box_delta=0.1, cls_consistent=True, min_heads=3
        )
        head_late.eval()
        head_late(features, bboxes, torch.full((1,), 100.0))  # 后期
        late_exit = head_late._exit_stats.get('exit_head_idx', 6)

        # 后期更容易退出 (exit_idx 更小或相等)
        assert late_exit <= early_exit


# ================================================================
# 测试 4: 与 deep_supervision 的兼容性
# ================================================================


class TestDeepSupervisionCompat:
    """测试提前退出与 deep_supervision 的兼容性。"""

    def test_deep_supervision_shape(self):
        """deep_supervision=True 时返回所有头 (含填充)。"""
        head = _make_full_head(
            box_delta=0.0,
            cls_consistent=True,
            min_heads=3,
            deep_supervision=True,
        )
        head.eval()

        features = [torch.randn(1, 256, 16, 16)]
        bboxes = (
            torch.tensor([30.0, 30.0, 70.0, 70.0])
            .expand(1, 100, 4)
            .clone()
        )
        t = torch.full((1,), 100.0)

        cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        # deep_supervision=True 返回所有 6 头
        assert cls_logits.shape[0] == 6
        assert pred_bboxes.shape[0] == 6

    def test_no_deep_supervision_shape(self):
        """deep_supervision=False 时只返回最后一级。"""
        head = _make_full_head(
            box_delta=0.0,
            cls_consistent=True,
            min_heads=3,
            deep_supervision=False,
        )
        head.eval()

        features = [torch.randn(1, 256, 16, 16)]
        bboxes = (
            torch.tensor([30.0, 30.0, 70.0, 70.0])
            .expand(1, 100, 4)
            .clone()
        )
        t = torch.full((1,), 100.0)

        cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        # deep_supervision=False 只返回最后 1 头
        assert cls_logits.shape[0] == 1
        assert pred_bboxes.shape[0] == 1
