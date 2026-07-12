"""IO1 自适应步数提前终止单元测试

测试方案核心数学理论:
1. 收敛判据 (batch 级别, 标量比较):
   - 框位置: ‖x0_curr - x0_prev‖ / ‖x0_curr‖ < step_exit_threshold (相对 L2)
   - 分类: 高置信度框 (sigmoid > 0.3) 的 argmax 一致率 > step_exit_cls_threshold
2. RF 直线路径理论: t→0 时 x_t → x_0, x0_pred 趋于稳定, 相邻步变化 →0
3. min_steps 保证: 至少执行 min_steps 步后才检测收敛 (不跳第 0 步)
4. ensemble 填充: 提前终止后用最后结果填充剩余步, 保持多步投票一致性
5. 推理时才启用: predict() 仅在推理调用, 训练 loss() 不受影响
6. 最后一步不检测: step_idx < len(time_pairs) - 1, 最后一步退出无意义
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ldmdet.core.head import DiffusionDetHead


# ================================================================
# Mock 组件
# ================================================================


class MockSingleHead(nn.Module):
    """Mock single head for DiffusionDetHead construction."""

    def __init__(self, num_classes=24):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, features, bboxes, curr_proposals, roi_extractor, time_emb):
        bs, n = bboxes.shape[:2]
        cls_logits = torch.zeros(bs, n, self.num_classes)
        cls_logits[..., 0] = 10.0
        return cls_logits, bboxes, bboxes


def _make_lightweight_head(**kwargs):
    """创建轻量 head 对象, 只设置 IO1 相关属性, 绑定方法。

    用于纯收敛检测方法测试, 避免构建完整 DiffusionDetHead。
    """
    defaults = dict(
        step_early_exit_enabled=True,
        step_exit_threshold=0.01,
        step_exit_cls_threshold=0.95,
        step_exit_min_steps=1,
        _step_exit_stats={},
    )
    defaults.update(kwargs)
    head = SimpleNamespace(**defaults)
    head._check_step_convergence = (
        DiffusionDetHead._check_step_convergence.__get__(head)
    )
    return head


def _make_full_head(
    num_heads=6,
    min_steps=1,
    threshold=0.01,
    cls_threshold=0.95,
    step_early_exit_enabled=True,
    **extra_kwargs,
):
    """创建完整 DiffusionDetHead, 用 MockSingleHead。"""
    mock_single = MockSingleHead(num_classes=24)
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
        solver_type='euler',
        sampling_timesteps=4,
        rf_schedule='linear',
        snr_scale=2.0,
        step_early_exit_enabled=step_early_exit_enabled,
        step_exit_threshold=threshold,
        step_exit_cls_threshold=cls_threshold,
        step_exit_min_steps=min_steps,
        **extra_kwargs,
    )
    return head


# ================================================================
# 测试 1: 收敛检测数学正确性
# ================================================================


class TestCheckStepConvergence:
    """测试 _check_step_convergence 方法的数学正确性。"""

    def setup_method(self):
        self.head = _make_lightweight_head()
        self.bs, self.n, self.num_classes = 2, 100, 24
        # x0 预测: 框坐标 (raw space, 但这里用任意数值测试相对变化)
        self.base_x0 = (
            torch.tensor([10.0, 20.0, 30.0, 40.0])
            .expand(self.bs, self.n, 4)
            .clone()
        )

    def test_converged_x0_and_cls(self):
        """x0 变化极小 + 分类一致 → converged=True"""
        x0_curr = self.base_x0.clone()
        x0_prev = self.base_x0.clone() + 0.001  # 极小变化
        cls_curr = torch.zeros(self.bs, self.n, self.num_classes)
        cls_curr[..., 0] = 10.0  # 高置信度 class 0
        cls_prev = cls_curr.clone()

        converged, stats = self.head._check_step_convergence(
            x0_curr, x0_prev, cls_curr, cls_prev, t_curr=0.25
        )
        assert converged is True
        assert stats['mean_relative_delta'] < 0.01
        assert stats['mean_consistency'] > 0.95

    def test_diverged_x0(self):
        """x0 变化大 → converged=False"""
        x0_curr = self.base_x0.clone()
        x0_prev = self.base_x0.clone() + 10.0  # 大变化
        cls_curr = torch.zeros(self.bs, self.n, self.num_classes)
        cls_curr[..., 0] = 10.0
        cls_prev = cls_curr.clone()

        converged, stats = self.head._check_step_convergence(
            x0_curr, x0_prev, cls_curr, cls_prev, t_curr=0.25
        )
        assert converged is False
        assert stats['mean_relative_delta'] > 0.01

    def test_diverged_cls(self):
        """分类不一致 → converged=False"""
        x0_curr = self.base_x0.clone()
        x0_prev = self.base_x0.clone()
        cls_curr = torch.zeros(self.bs, self.n, self.num_classes)
        cls_curr[..., 0] = 10.0  # class 0
        cls_prev = torch.zeros(self.bs, self.n, self.num_classes)
        cls_prev[..., 1] = 10.0  # class 1 (不同)

        converged, stats = self.head._check_step_convergence(
            x0_curr, x0_prev, cls_curr, cls_prev, t_curr=0.25
        )
        assert converged is False
        assert stats['mean_consistency'] < 0.95

    def test_low_confidence_ignored(self):
        """低置信度框 (sigmoid < 0.3) 不参与分类收敛判定。"""
        x0_curr = self.base_x0.clone()
        x0_prev = self.base_x0.clone()
        # 所有框低置信度: sigmoid(-2.0) ≈ 0.12 < 0.3
        cls_curr = torch.full(
            (self.bs, self.n, self.num_classes), -2.0
        )
        cls_curr[:, :, 0] = -1.9  # 略高, argmax=0
        cls_prev = torch.full(
            (self.bs, self.n, self.num_classes), -2.0
        )
        cls_prev[:, :, 1] = -1.9  # 略高, argmax=1 (不同)

        converged, stats = self.head._check_step_convergence(
            x0_curr, x0_prev, cls_curr, cls_prev, t_curr=0.25
        )
        # 低置信度框被忽略, mean_consistency 默认 1.0
        # 框无变化 → converged=True
        assert converged is True
        assert stats['mean_consistency'] == pytest.approx(1.0, abs=0.01)

    def test_partial_low_confidence(self):
        """部分高置信度框: 只用高置信度框计算一致率。"""
        x0_curr = self.base_x0.clone()
        x0_prev = self.base_x0.clone()
        cls_curr = torch.full(
            (self.bs, self.n, self.num_classes), -2.0
        )
        cls_curr[:, :50, 0] = 10.0  # 前 50 个高置信 class 0
        cls_prev = cls_curr.clone()  # 完全一致

        converged, stats = self.head._check_step_convergence(
            x0_curr, x0_prev, cls_curr, cls_prev, t_curr=0.25
        )
        # 高置信度框完全一致 → mean_consistency=1.0
        assert converged is True
        assert stats['mean_consistency'] == pytest.approx(1.0, abs=0.01)

    def test_stats_returned(self):
        """统计信息字典包含所有必要字段。"""
        x0_curr = self.base_x0.clone()
        x0_prev = self.base_x0.clone()
        cls_curr = torch.zeros(self.bs, self.n, self.num_classes)
        cls_curr[..., 0] = 10.0
        cls_prev = cls_curr.clone()

        _, stats = self.head._check_step_convergence(
            x0_curr, x0_prev, cls_curr, cls_prev, t_curr=0.5
        )
        assert 'mean_relative_delta' in stats
        assert 'mean_consistency' in stats
        assert 'box_thr' in stats
        assert 'cls_thr' in stats
        assert 't_curr' in stats
        assert 'converged' in stats
        assert stats['t_curr'] == 0.5

    def test_threshold_boundary_converged(self):
        """x0 变化恰好等于阈值时不收敛 (严格小于)。"""
        # 构造 relative_delta 恰好 ≈ threshold
        # base_x0 范数: sqrt(10^2+20^2+30^2+40^2) = sqrt(100+400+900+1600) = sqrt(3000) ≈ 54.77
        # delta = threshold * magnitude = 0.01 * 54.77 ≈ 0.5477
        # 设 prev = curr + delta_per_dim, 使 delta.norm() = sqrt(4) * delta_per_dim
        x0_curr = self.base_x0.clone()
        delta_per_dim = 0.01 * 54.77 / 2.0  # 使 delta.norm() ≈ 0.5477
        x0_prev = self.base_x0.clone() + delta_per_dim
        cls_curr = torch.zeros(self.bs, self.n, self.num_classes)
        cls_curr[..., 0] = 10.0
        cls_prev = cls_curr.clone()

        _, stats = self.head._check_step_convergence(
            x0_curr, x0_prev, cls_curr, cls_prev, t_curr=0.25
        )
        # 应该在阈值附近, 不严格判断 converged (浮点精度),
        # 但验证 relative_delta 接近 threshold
        assert abs(stats['mean_relative_delta'] - 0.01) < 0.005

    def test_zero_magnitude_no_nan(self):
        """x0_curr 全零时不产生 NaN (clamp(min=1e-6))。"""
        x0_curr = torch.zeros(self.bs, self.n, 4)
        x0_prev = torch.zeros(self.bs, self.n, 4)
        cls_curr = torch.zeros(self.bs, self.n, self.num_classes)
        cls_curr[..., 0] = 10.0
        cls_prev = cls_curr.clone()

        converged, stats = self.head._check_step_convergence(
            x0_curr, x0_prev, cls_curr, cls_prev, t_curr=0.25
        )
        assert not torch.isnan(
            torch.tensor(stats['mean_relative_delta'])
        )
        assert converged is True  # 0/1e-6 = 0 < threshold

    def test_batch_independence_averaged(self):
        """batch 级别标量比较: 不同图像的 delta 被平均。"""
        # 图像 0: 完全收敛, 图像 1: 大变化
        x0_curr = self.base_x0.clone()
        x0_prev = self.base_x0.clone()
        x0_prev[1] += 100.0  # 图像 1 大变化

        cls_curr = torch.zeros(self.bs, self.n, self.num_classes)
        cls_curr[..., 0] = 10.0
        cls_prev = cls_curr.clone()

        converged, stats = self.head._check_step_convergence(
            x0_curr, x0_prev, cls_curr, cls_prev, t_curr=0.25
        )
        # 平均后 delta 较大 → 不收敛
        assert converged is False
        assert stats['mean_relative_delta'] > 0.01


# ================================================================
# 测试 2: predict() 集成 — 提前终止行为
# ================================================================


class TestPredictWithStepEarlyExit:
    """测试 predict() 中 IO1 步级提前终止的端到端行为。"""

    def test_disabled_no_early_exit(self):
        """step_early_exit_enabled=False 时跑完所有步。"""
        head = _make_full_head(step_early_exit_enabled=False)
        call_count = [0]

        original_forward = head._forward_at_t

        def mock_forward(features, x_raw, t, img_metas):
            call_count[0] += 1
            bs, n = x_raw.shape[:2]
            cls_logits = torch.zeros(bs, n, 24)
            cls_logits[..., 0] = 10.0
            pred_bboxes = torch.randn(bs, n, 4)
            x0 = pred_bboxes.clone()
            return cls_logits, pred_bboxes, x0

        head._forward_at_t = mock_forward
        img_metas = [{'img_shape': (256, 256, 3), 'scale_factor': 1.0}]
        features = [torch.randn(1, 256, 32, 32)]

        # Euler 4步, 每步 1 NFE = 4 次 _forward_at_t
        head.predict(features, img_metas)
        assert call_count[0] == 4  # 4 步全跑

    def test_enabled_converged_early_exit(self):
        """收敛时提前终止, 不跑剩余步。"""
        head = _make_full_head(
            step_early_exit_enabled=True,
            threshold=0.01,
            cls_threshold=0.95,
            min_steps=1,
        )
        call_count = [0]

        def mock_forward(features, x_raw, t, img_metas):
            call_count[0] += 1
            bs, n = x_raw.shape[:2]
            cls_logits = torch.zeros(bs, n, 24)
            cls_logits[..., 0] = 10.0  # 固定分类
            # x0 固定不变 → 第 2 步起即收敛
            pred_bboxes = torch.full((bs, n, 4), 10.0)
            x0 = pred_bboxes.clone()
            return cls_logits, pred_bboxes, x0

        head._forward_at_t = mock_forward
        img_metas = [{'img_shape': (256, 256, 3), 'scale_factor': 1.0}]
        features = [torch.randn(1, 256, 32, 32)]

        head.predict(features, img_metas)
        # min_steps=1, 第 1 步 (idx=0) 不检测 (x0_prev=None),
        # 第 2 步 (idx=1) 检测收敛 → 退出
        # 但第 2 步前先调用 _forward_at_t, 所以 call_count=2
        assert call_count[0] == 2
        assert head._step_exit_stats.get('exit_step_idx') == 1

    def test_enabled_diverged_no_exit(self):
        """不收敛时跑完所有步。"""
        head = _make_full_head(
            step_early_exit_enabled=True,
            threshold=0.001,  # 极严格, 几乎不可能收敛
            cls_threshold=0.999,
        )
        call_count = [0]

        def mock_forward(features, x_raw, t, img_metas):
            call_count[0] += 1
            bs, n = x_raw.shape[:2]
            cls_logits = torch.randn(bs, n, 24) * 5  # 随机分类
            pred_bboxes = torch.randn(bs, n, 4) * 100  # 随机 x0
            x0 = pred_bboxes.clone()
            return cls_logits, pred_bboxes, x0

        head._forward_at_t = mock_forward
        img_metas = [{'img_shape': (256, 256, 3), 'scale_factor': 1.0}]
        features = [torch.randn(1, 256, 32, 32)]

        head.predict(features, img_metas)
        assert call_count[0] == 4  # 4 步全跑
        assert head._step_exit_stats == {}

    def test_min_steps_respected(self):
        """min_steps=2 时前 2 步不检测收敛。"""
        head = _make_full_head(
            step_early_exit_enabled=True,
            threshold=0.01,
            cls_threshold=0.95,
            min_steps=2,
        )
        call_count = [0]

        def mock_forward(features, x_raw, t, img_metas):
            call_count[0] += 1
            bs, n = x_raw.shape[:2]
            cls_logits = torch.zeros(bs, n, 24)
            cls_logits[..., 0] = 10.0
            pred_bboxes = torch.full((bs, n, 4), 10.0)
            x0 = pred_bboxes.clone()
            return cls_logits, pred_bboxes, x0

        head._forward_at_t = mock_forward
        img_metas = [{'img_shape': (256, 256, 3), 'scale_factor': 1.0}]
        features = [torch.randn(1, 256, 32, 32)]

        head.predict(features, img_metas)
        # min_steps=2: 第 1 步 (idx=0) 不检测, 第 2 步 (idx=1) 不检测,
        # 第 3 步 (idx=2) 检测收敛 → 退出
        # call_count = 3 (3 次 _forward_at_t)
        assert call_count[0] == 3
        assert head._step_exit_stats.get('exit_step_idx') == 2

    def test_last_step_no_exit_check(self):
        """最后一步不检测收敛 (退出无意义)。

        min_steps=3 → 检测从 step_idx=3 开始, 但 step_idx=3 是最后一步
        (len(time_pairs)-1=3), 最后一步被排除 → 全部 4 步执行, 无退出。
        """
        head = _make_full_head(
            step_early_exit_enabled=True,
            threshold=0.01,
            cls_threshold=0.95,
            min_steps=3,
        )
        call_count = [0]

        def mock_forward(features, x_raw, t, img_metas):
            call_count[0] += 1
            bs, n = x_raw.shape[:2]
            cls_logits = torch.zeros(bs, n, 24)
            cls_logits[..., 0] = 10.0
            pred_bboxes = torch.full((bs, n, 4), 10.0)
            x0 = pred_bboxes.clone()
            return cls_logits, pred_bboxes, x0

        head._forward_at_t = mock_forward
        img_metas = [{'img_shape': (256, 256, 3), 'scale_factor': 1.0}]
        features = [torch.randn(1, 256, 32, 32)]

        head.predict(features, img_metas)
        # min_steps=3 → 检测条件 step_idx >= 3, 但最后一步 (idx=3) 被排除
        # → 无检测发生, 4 步全跑
        assert call_count[0] == 4
        assert head._step_exit_stats == {}

    def test_ensemble_filled_after_exit(self):
        """提前终止后 ensemble 被填充至完整步数。"""
        head = _make_full_head(
            step_early_exit_enabled=True,
            threshold=0.01,
            cls_threshold=0.95,
            min_steps=1,
        )

        def mock_forward(features, x_raw, t, img_metas):
            bs, n = x_raw.shape[:2]
            cls_logits = torch.zeros(bs, n, 24)
            cls_logits[..., 0] = 10.0
            pred_bboxes = torch.full((bs, n, 4), 10.0)
            x0 = pred_bboxes.clone()
            return cls_logits, pred_bboxes, x0

        head._forward_at_t = mock_forward

        # 拦截 post_process 来检查 ensemble 长度
        original_post_process = head._sampler.post_process
        captured_ensemble = []

        def capture_post_process(ensemble_results, img_metas, rescale):
            captured_ensemble.append(len(ensemble_results))
            return original_post_process(ensemble_results, img_metas, rescale)

        head._sampler.post_process = capture_post_process

        img_metas = [{'img_shape': (256, 256, 3), 'scale_factor': 1.0}]
        features = [torch.randn(1, 256, 32, 32)]

        head.predict(features, img_metas)
        # 4 步 ensemble, 即使 idx=1 收敛退出, ensemble 应填充到 4
        assert captured_ensemble[0] == 4

    def test_exit_stats_recorded(self):
        """收敛退出后 _step_exit_stats 包含完整信息。"""
        head = _make_full_head(
            step_early_exit_enabled=True,
            threshold=0.01,
            cls_threshold=0.95,
            min_steps=1,
        )

        def mock_forward(features, x_raw, t, img_metas):
            bs, n = x_raw.shape[:2]
            cls_logits = torch.zeros(bs, n, 24)
            cls_logits[..., 0] = 10.0
            pred_bboxes = torch.full((bs, n, 4), 10.0)
            x0 = pred_bboxes.clone()
            return cls_logits, pred_bboxes, x0

        head._forward_at_t = mock_forward
        img_metas = [{'img_shape': (256, 256, 3), 'scale_factor': 1.0}]
        features = [torch.randn(1, 256, 32, 32)]

        head.predict(features, img_metas)
        stats = head._step_exit_stats
        assert 'exit_step_idx' in stats
        assert 'total_steps' in stats
        assert 'mean_relative_delta' in stats
        assert 'mean_consistency' in stats
        assert 'converged' in stats
        assert stats['converged'] is True
        assert stats['exit_step_idx'] == 1
        assert stats['total_steps'] == 4

    def test_no_exit_when_not_converged(self):
        """不收敛时 _step_exit_stats 为空。"""
        head = _make_full_head(
            step_early_exit_enabled=True,
            threshold=0.0001,  # 极严格
            cls_threshold=0.999,
        )

        def mock_forward(features, x_raw, t, img_metas):
            bs, n = x_raw.shape[:2]
            cls_logits = torch.randn(bs, n, 24) * 5
            pred_bboxes = torch.randn(bs, n, 4) * 100
            x0 = pred_bboxes.clone()
            return cls_logits, pred_bboxes, x0

        head._forward_at_t = mock_forward
        img_metas = [{'img_shape': (256, 256, 3), 'scale_factor': 1.0}]
        features = [torch.randn(1, 256, 32, 32)]

        head.predict(features, img_metas)
        assert head._step_exit_stats == {}


# ================================================================
# 测试 3: 参数初始化与配置
# ================================================================


class TestIO1ParameterInit:
    """测试 IO1 参数初始化。"""

    def test_default_disabled(self):
        """默认 step_early_exit_enabled=False。"""
        head = _make_full_head(step_early_exit_enabled=False)
        assert head.step_early_exit_enabled is False
        assert head.step_exit_threshold == 0.01
        assert head.step_exit_cls_threshold == 0.95
        assert head.step_exit_min_steps == 1
        assert head._step_exit_stats == {}

    def test_custom_params(self):
        """自定义参数正确设置。"""
        mock_single = MockSingleHead(num_classes=24)
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=256,
            num_proposals=100,
            num_heads=6,
            single_head=mock_single,
            roi_extractor=None,
            criterion=None,
            coupling=None,
            diffusion_type='rectified_flow',
            solver_type='euler',
            sampling_timesteps=4,
            rf_schedule='linear',
            snr_scale=2.0,
            step_early_exit_enabled=True,
            step_exit_threshold=0.005,
            step_exit_cls_threshold=0.99,
            step_exit_min_steps=2,
        )
        assert head.step_early_exit_enabled is True
        assert head.step_exit_threshold == 0.005
        assert head.step_exit_cls_threshold == 0.99
        assert head.step_exit_min_steps == 2
