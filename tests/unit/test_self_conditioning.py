"""SC-RF (Self-Conditioned Rectified Flow) 单元测试

测试自条件化机制的核心理论性质:
1. 零初始化保证: x0_prev_proj 初始化为零, 初始时 SC-RF 等价于标准 RF
2. 信息注入: 非零 x0_prev 改变模型输出 (自条件化提供额外信息)
3. 梯度流: x0_prev_proj 梯度正常回传 (残差学习可优化)
4. 训练切换: 50% 概率启用自条件化, 50% 使用零输入 (fallback 路径)
5. 推理传递: 多步推理中 x0_prev 正确传递给下一步 (跨时间步信息)
6. 求解器兼容: SC-RF 与 DPM-Solver++/Heun 兼容

对应 docs/paper/proposals/SC-RF_Self-Conditioned_Rectified_Flow.md
理论依据: 2.2 条件化不增加熵, 2.4 残差学习视角, 2.5 训练-推理失配缓解
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ldmdet.core.head import DiffusionDetHead
from ldmdet.core.single_head import SingleDiffusionDetHead
from ldmdet.core.roi_extractor import SingleRoIExtractor


def _make_single_head(num_classes=24, feat_channels=64, use_self_conditioning=True):
    """构建支持自条件化的 SingleDiffusionDetHead"""
    return SingleDiffusionDetHead(
        num_classes=num_classes,
        feat_channels=feat_channels,
        num_cls_convs=1,
        num_reg_convs=2,
        use_focal_loss=True,
        use_normalized_classifier=False,
        use_self_conditioning=use_self_conditioning,
    )


def _make_single_head_no_sc(num_classes=24, feat_channels=64):
    """构建不支持自条件化的标准 SingleDiffusionDetHead (用于对比)"""
    return SingleDiffusionDetHead(
        num_classes=num_classes,
        feat_channels=feat_channels,
        num_cls_convs=1,
        num_reg_convs=2,
        use_focal_loss=True,
        use_normalized_classifier=False,
        use_self_conditioning=False,
    )


def _make_roi_extractor(out_channels=64):
    """构建最小 RoIExtractor"""
    return SingleRoIExtractor(
        featmap_strides=[16], out_channels=out_channels,
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0, aligned=True),
    )


def _make_head(use_self_conditioning=True, num_heads=3, feat_channels=64):
    """构建 DiffusionDetHead"""
    return DiffusionDetHead(
        num_classes=24,
        feat_channels=feat_channels,
        num_heads=num_heads,
        single_head=_make_single_head(use_self_conditioning=use_self_conditioning),
        roi_extractor=_make_roi_extractor(feat_channels),
        criterion=None,
        diffusion_type='rectified_flow',
        solver_type='euler',
        sampling_timesteps=1,
        use_self_conditioning=use_self_conditioning,
    )


def _make_dummy_input(bs=2, num_boxes=10, feat_channels=64):
    """构造 dummy 输入"""
    features = [torch.randn(bs, feat_channels, 16, 16)]
    bboxes = torch.rand(bs, num_boxes, 4)
    # 确保有效框: x2 > x1, y2 > y1
    bboxes[:, :, 2] = bboxes[:, :, 0] + 0.2
    bboxes[:, :, 3] = bboxes[:, :, 1] + 0.2
    t = torch.rand(bs)
    return features, bboxes, t


# ============================================================
# 1. 零初始化验证 (理论 2.2: 条件化不损害, 零初始化保证平滑过渡)
# ============================================================

class TestZeroInitialization:
    """测试 x0_prev_proj 的零初始化"""

    def test_x0_prev_proj_exists(self):
        """use_self_conditioning=True 时应创建 x0_prev_proj"""
        head = _make_head(use_self_conditioning=True)
        assert hasattr(head.head_series[0], 'x0_prev_proj'), \
            'SingleDiffusionDetHead 应有 x0_prev_proj 属性'

    def test_x0_prev_proj_not_exists_without_sc(self):
        """use_self_conditioning=False 时不应创建 x0_prev_proj"""
        head = _make_head(use_self_conditioning=False)
        assert not hasattr(head.head_series[0], 'x0_prev_proj'), \
            'use_self_conditioning=False 时不应有 x0_prev_proj'

    def test_x0_prev_proj_weight_zero(self):
        """x0_prev_proj 的 weight 应零初始化"""
        head = _make_head(use_self_conditioning=True)
        for h in head.head_series:
            w = h.x0_prev_proj.weight
            assert torch.allclose(w, torch.zeros_like(w)), \
                'x0_prev_proj weight 应零初始化'

    def test_x0_prev_proj_bias_zero(self):
        """x0_prev_proj 的 bias 应零初始化"""
        head = _make_head(use_self_conditioning=True)
        for h in head.head_series:
            b = h.x0_prev_proj.bias
            assert b is not None
            assert torch.allclose(b, torch.zeros_like(b)), \
                'x0_prev_proj bias 应零初始化'

    def test_x0_prev_proj_is_linear_4_to_feat(self):
        """x0_prev_proj 应是 Linear(4, feat_channels)"""
        head = _make_head(use_self_conditioning=True, feat_channels=64)
        proj = head.head_series[0].x0_prev_proj
        assert isinstance(proj, nn.Linear), 'x0_prev_proj 应是 nn.Linear'
        assert proj.in_features == 4, 'x0_prev_proj 输入维度应为 4 (bbox)'
        assert proj.out_features == 64, 'x0_prev_proj 输出维度应为 feat_channels'


# ============================================================
# 2. 零输入等价性 (理论 2.2: 零初始化时 SC-RF 等价于标准 RF)
# ============================================================

class TestZeroInputEquivalence:
    """测试 x0_prev=0 时 SC-RF 输出与标准 RF 一致"""

    def test_zero_x0_prev_forward_works(self):
        """x0_prev=0 时 forward 应正常工作"""
        head = _make_head(use_self_conditioning=True)
        head.eval()
        features, bboxes, t = _make_dummy_input()
        x0_prev = torch.zeros(2, 10, 4)

        with torch.no_grad():
            cls_logits, pred_bboxes, _ = head(features, bboxes, t, x0_prev)
        assert cls_logits is not None
        assert pred_bboxes is not None

    def test_zero_x0_prev_equals_no_x0_prev(self):
        """x0_prev=0 应与不传 x0_prev (默认零) 输出一致"""
        head = _make_head(use_self_conditioning=True)
        head.eval()
        features, bboxes, t = _make_dummy_input()
        x0_prev_zero = torch.zeros(2, 10, 4)

        with torch.no_grad():
            out_with_zero = head(features, bboxes, t, x0_prev_zero)
            out_without = head(features, bboxes, t)

        # 两者应完全一致 (零初始化 + 零输入 = 无自条件化)
        torch.testing.assert_close(out_with_zero[0], out_without[0])
        torch.testing.assert_close(out_with_zero[1], out_without[1])


# ============================================================
# 3. 信息注入验证 (理论 2.3: 非零 x0_prev 提供额外信息)
# ============================================================

class TestInformationInjection:
    """测试非零 x0_prev 改变模型输出"""

    def test_nonzero_x0_prev_changes_output(self):
        """非零 x0_prev 应改变输出 (自条件化提供信息)"""
        head = _make_head(use_self_conditioning=True)
        # 破坏零初始化, 使 x0_prev_proj 有非零权重
        for h in head.head_series:
            nn.init.normal_(h.x0_prev_proj.weight, std=0.02)
            nn.init.normal_(h.x0_prev_proj.bias, std=0.02)
        head.eval()
        features, bboxes, t = _make_dummy_input()
        x0_prev_zero = torch.zeros(2, 10, 4)
        x0_prev_nonzero = torch.randn(2, 10, 4) * 0.5

        with torch.no_grad():
            out_zero = head(features, bboxes, t, x0_prev_zero)
            out_nonzero = head(features, bboxes, t, x0_prev_nonzero)

        # 非零 x0_prev 应改变输出
        diff_cls = (out_zero[0] - out_nonzero[0]).abs().sum()
        diff_box = (out_zero[1] - out_nonzero[1]).abs().sum()
        assert diff_cls > 1e-6 or diff_box > 1e-6, \
            '非零 x0_prev 应改变模型输出'

    def test_different_x0_prev_different_output(self):
        """不同 x0_prev 应产生不同输出"""
        head = _make_head(use_self_conditioning=True)
        for h in head.head_series:
            nn.init.normal_(h.x0_prev_proj.weight, std=0.02)
            nn.init.normal_(h.x0_prev_proj.bias, std=0.02)
        head.eval()
        features, bboxes, t = _make_dummy_input()
        x0_prev_a = torch.randn(2, 10, 4) * 0.5
        x0_prev_b = torch.randn(2, 10, 4) * 0.5

        with torch.no_grad():
            out_a = head(features, bboxes, t, x0_prev_a)
            out_b = head(features, bboxes, t, x0_prev_b)

        diff = (out_a[0] - out_b[0]).abs().sum() + (out_a[1] - out_b[1]).abs().sum()
        assert diff > 1e-6, '不同 x0_prev 应产生不同输出'

    def test_zero_init_nonzero_prev_no_change(self):
        """零初始化时, 非零 x0_prev 不改变输出 (零初始化保证)"""
        head = _make_head(use_self_conditioning=True)
        head.eval()  # 保持零初始化
        features, bboxes, t = _make_dummy_input()
        x0_prev_zero = torch.zeros(2, 10, 4)
        x0_prev_nonzero = torch.randn(2, 10, 4)

        with torch.no_grad():
            out_zero = head(features, bboxes, t, x0_prev_zero)
            out_nonzero = head(features, bboxes, t, x0_prev_nonzero)

        # 零初始化: x0_prev_proj(weight=0) * x0_prev = 0, 故输出不变
        torch.testing.assert_close(out_zero[0], out_nonzero[0])
        torch.testing.assert_close(out_zero[1], out_nonzero[1])


# ============================================================
# 4. 梯度流验证 (理论 2.4: 残差学习可优化)
# ============================================================

class TestGradientFlow:
    """测试 x0_prev_proj 的梯度流"""

    def test_gradient_flows_to_x0_prev_proj(self):
        """损失应能回传梯度到 x0_prev_proj"""
        head = _make_head(use_self_conditioning=True)
        # 破坏零初始化以使 x0_prev 有实际影响
        for h in head.head_series:
            nn.init.normal_(h.x0_prev_proj.weight, std=0.02)
            nn.init.normal_(h.x0_prev_proj.bias, std=0.02)
        head.train()
        features, bboxes, t = _make_dummy_input()
        x0_prev = torch.randn(2, 10, 4) * 0.5

        cls_logits, pred_bboxes, _ = head(features, bboxes, t, x0_prev)
        loss = cls_logits.sum() + pred_bboxes.sum()
        loss.backward()

        for i, h in enumerate(head.head_series):
            assert h.x0_prev_proj.weight.grad is not None, \
                f'head[{i}] x0_prev_proj.weight 应有梯度'
            assert h.x0_prev_proj.weight.grad.abs().sum() > 0, \
                f'head[{i}] x0_prev_proj.weight 梯度应非零'

    def test_x0_prev_receives_gradient(self):
        """x0_prev 本身应接收梯度 (当 requires_grad=True 时)"""
        head = _make_head(use_self_conditioning=True)
        for h in head.head_series:
            nn.init.normal_(h.x0_prev_proj.weight, std=0.02)
            nn.init.normal_(h.x0_prev_proj.bias, std=0.02)
        head.train()
        features, bboxes, t = _make_dummy_input()
        # 先创建张量再设置 requires_grad, 避免 * 0.5 产生非叶子张量
        x0_prev = (torch.randn(2, 10, 4) * 0.5).requires_grad_(True)

        cls_logits, pred_bboxes, _ = head(features, bboxes, t, x0_prev)
        loss = pred_bboxes.sum()
        loss.backward()

        assert x0_prev.grad is not None, 'x0_prev 应接收梯度'
        assert x0_prev.grad.abs().sum() > 0, 'x0_prev 梯度应非零'


# ============================================================
# 5. 训练自条件化切换 (理论 2.5: 50% 切换提供 fallback 路径)
# ============================================================

class TestTrainingToggle:
    """测试训练时 50% 自条件化切换"""

    def test_use_self_conditioning_flag(self):
        """DiffusionDetHead 应有 use_self_conditioning 标志"""
        head = _make_head(use_self_conditioning=True)
        assert hasattr(head, 'use_self_conditioning')
        assert head.use_self_conditioning is True

    def test_self_conditioning_prob_default(self):
        """self_conditioning_prob 默认应为 0.5"""
        head = _make_head(use_self_conditioning=True)
        assert hasattr(head, 'self_conditioning_prob')
        assert head.self_conditioning_prob == 0.5

    def test_self_conditioning_prob_customizable(self):
        """self_conditioning_prob 应可自定义"""
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_heads=3,
            single_head=_make_single_head(),
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            diffusion_type='rectified_flow',
            solver_type='euler',
            sampling_timesteps=1,
            use_self_conditioning=True,
            self_conditioning_prob=0.3,
        )
        assert head.self_conditioning_prob == 0.3


# ============================================================
# 6. 推理多步传递 (理论 2.3: 跨时间步信息传递)
# ============================================================

class TestInferenceMultiStep:
    """测试推理时多步 x0_prev 传递"""

    def test_forward_at_t_accepts_x0_prev(self):
        """_forward_at_t 应接受 x0_prev 参数"""
        head = _make_head(use_self_conditioning=True)
        head.eval()
        features, _, _ = _make_dummy_input()
        x_raw = torch.randn(2, 10, 4)
        x0_prev = torch.zeros(2, 10, 4)

        # _forward_at_t 应接受 x0_prev 参数
        import inspect
        sig = inspect.signature(head._forward_at_t)
        assert 'x0_prev' in sig.parameters, \
            '_forward_at_t 应有 x0_prev 参数'

    def test_predict_passes_x0_prev_across_steps(self):
        """predict 应在多步间传递 x0_prev"""
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_heads=3,
            num_proposals=10,
            single_head=_make_single_head(),
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            diffusion_type='rectified_flow',
            solver_type='euler',
            sampling_timesteps=4,  # 多步
            use_self_conditioning=True,
        )
        head.eval()
        features = [torch.randn(2, 64, 16, 16)]
        img_metas = [
            {'img_shape': (16, 16, 3), 'scale_factor': 1.0, 'ori_shape': (16, 16, 3)}
            for _ in range(2)
        ]

        # predict 应正常工作 (多步推理 + 自条件化)
        results = head.predict(features, img_metas)
        assert results is not None
        assert len(results) == 2  # batch size


# ============================================================
# 7. 求解器兼容性 (理论 2.7: 与 DPM-Solver++ 兼容)
# ============================================================

class TestSolverCompatibility:
    """测试 SC-RF 与不同求解器的兼容性"""

    def test_dpm_solver_pp_compatibility(self):
        """SC-RF + DPM-Solver++ 应正常工作"""
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_heads=3,
            num_proposals=10,
            single_head=_make_single_head(),
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            diffusion_type='rectified_flow',
            solver_type='dpm_solver_pp',
            sampling_timesteps=4,
            use_self_conditioning=True,
        )
        head.eval()
        features = [torch.randn(2, 64, 16, 16)]
        img_metas = [
            {'img_shape': (16, 16, 3), 'scale_factor': 1.0, 'ori_shape': (16, 16, 3)}
            for _ in range(2)
        ]

        results = head.predict(features, img_metas)
        assert results is not None

    def test_heun_compatibility(self):
        """SC-RF + Heun 应正常工作"""
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_heads=3,
            num_proposals=10,
            single_head=_make_single_head(),
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            diffusion_type='rectified_flow',
            solver_type='heun',
            sampling_timesteps=4,
            use_self_conditioning=True,
        )
        head.eval()
        features = [torch.randn(2, 64, 16, 16)]
        img_metas = [
            {'img_shape': (16, 16, 3), 'scale_factor': 1.0, 'ori_shape': (16, 16, 3)}
            for _ in range(2)
        ]

        results = head.predict(features, img_metas)
        assert results is not None


# ============================================================
# 8. 残差学习性质 (理论 2.4: 校正项从零开始增长)
# ============================================================

class TestResidualLearning:
    """测试残差学习性质"""

    def test_initial_correction_is_zero(self):
        """零初始化时, 校正项 Δ = f(x_t, x0_prev) - x0_prev ≈ 0 (无影响)"""
        head = _make_head(use_self_conditioning=True)
        head.eval()
        features, bboxes, t = _make_dummy_input()
        x0_prev_zero = torch.zeros(2, 10, 4)
        x0_prev_nonzero = torch.randn(2, 10, 4) * 0.5

        with torch.no_grad():
            out_zero = head(features, bboxes, t, x0_prev_zero)
            out_nonzero = head(features, bboxes, t, x0_prev_nonzero)

        # 零初始化: x0_prev_proj 输出为零, 校正项为零
        # 故两个输出应完全一致 (x0_prev 不影响)
        correction = (out_zero[1] - out_nonzero[1]).abs().sum()
        assert correction < 1e-6, \
            f'零初始化时校正项应为零, 实际: {correction}'

    def test_correction_grows_after_training(self):
        """训练后 (非零权重), 校正项应非零"""
        head = _make_head(use_self_conditioning=True)
        # 模拟训练后的状态: 非零权重
        for h in head.head_series:
            nn.init.normal_(h.x0_prev_proj.weight, std=0.02)
            nn.init.normal_(h.x0_prev_proj.bias, std=0.02)
        head.eval()
        features, bboxes, t = _make_dummy_input()
        x0_prev_zero = torch.zeros(2, 10, 4)
        x0_prev_nonzero = torch.randn(2, 10, 4) * 0.5

        with torch.no_grad():
            out_zero = head(features, bboxes, t, x0_prev_zero)
            out_nonzero = head(features, bboxes, t, x0_prev_nonzero)

        # 训练后: 非零 x0_prev 应改变输出, 校正项非零
        correction = (out_zero[1] - out_nonzero[1]).abs().sum()
        assert correction > 1e-6, \
            f'训练后校正项应非零, 实际: {correction}'


# ============================================================
# 9. 向后兼容性 (不影响标准 RF 的使用)
# ============================================================

class TestBackwardCompatibility:
    """测试 use_self_conditioning=False 时的向后兼容性"""

    def test_standard_rf_forward_without_x0_prev(self):
        """use_self_conditioning=False 时, forward 不需要 x0_prev"""
        head = _make_head(use_self_conditioning=False)
        head.eval()
        features, bboxes, t = _make_dummy_input()

        # 不传 x0_prev 应正常工作
        with torch.no_grad():
            cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        assert cls_logits is not None
        assert pred_bboxes is not None

    def test_standard_rf_forward_ignores_x0_prev(self):
        """use_self_conditioning=False 时, 传 x0_prev 应被忽略"""
        head = _make_head(use_self_conditioning=False)
        head.eval()
        features, bboxes, t = _make_dummy_input()
        x0_prev = torch.randn(2, 10, 4)

        with torch.no_grad():
            out_without = head(features, bboxes, t)
            out_with = head(features, bboxes, t, x0_prev)

        # use_self_conditioning=False 时 x0_prev 应被忽略
        torch.testing.assert_close(out_without[0], out_with[0])
        torch.testing.assert_close(out_without[1], out_with[1])

    def test_standard_rf_no_x0_prev_proj(self):
        """use_self_conditioning=False 时, single_head 不应有 x0_prev_proj"""
        head = _make_head(use_self_conditioning=False)
        for h in head.head_series:
            assert not hasattr(h, 'x0_prev_proj'), \
                'use_self_conditioning=False 时不应有 x0_prev_proj'
