"""测试 ldmdet.coupling — 所有耦合策略

当前可用策略:
  - random: 随机匹配
  - hard_ot: 硬 OT (最近邻, 确定性)
  - ot_flow: Sinkhorn OT (支持 coupling_mode='argmax' 确定性 / 'multinomial' 随机)

已移除策略 (代码清理 2026-07-23):
  - sinkhorn_argmax → 合并入 ot_flow (coupling_mode='argmax')
  - sinkhorn_stochastic → 合并入 ot_flow (coupling_mode='multinomial')
  - ghss → 已证伪移除 (GHSS coupling strategy 未能正确集成)
"""

import torch
import pytest
from ldmdet.coupling import build_coupling


# 所有耦合策略名
COUPLING_NAMES = ['random', 'hard_ot', 'ot_flow']

# 需要 epsilon/num_iters 参数的策略
_OT_NAMES = {'ot_flow'}


def _build(name, **kwargs):
    """构建耦合策略，仅对 OT 类策略传递额外参数"""
    if name in _OT_NAMES:
        return build_coupling(name, **kwargs)
    return build_coupling(name)


class TestBuildCoupling:
    def test_all_strategies(self):
        for name in COUPLING_NAMES:
            c = build_coupling(name)
            assert c is not None

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown coupling"):
            build_coupling('nonexistent')

    def test_with_kwargs(self):
        c = build_coupling('ot_flow', epsilon=5.0, num_iters=20)
        assert c.epsilon == 5.0
        assert c.num_iters == 20


class TestCouplingOutputShape:
    """所有策略输出形状一致"""

    @pytest.mark.parametrize("name", COUPLING_NAMES)
    def test_output_shape(self, name):
        coupling = _build(name, epsilon=1.0, num_iters=10)
        noise = torch.randn(50, 4)
        gt = torch.randn(10, 4)
        gt_labels = torch.randint(0, 24, (10,))
        x_start, matched_idx = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        assert x_start.shape == (50, 4)
        assert matched_idx.shape == (50,)

    @pytest.mark.parametrize("name", COUPLING_NAMES)
    def test_matched_from_gt(self, name):
        """x_start 应来自 gt_diffusion (x_start == gt[matched_idx])"""
        coupling = _build(name, epsilon=1.0, num_iters=10)
        noise = torch.randn(50, 4)
        gt = torch.randn(10, 4)
        gt_labels = torch.randint(0, 24, (10,))
        x_start, matched_idx = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        assert torch.allclose(x_start, gt[matched_idx], atol=1e-5)

    @pytest.mark.parametrize("name", COUPLING_NAMES)
    def test_matched_idx_valid(self, name):
        """matched_idx 应在 [0, M) 范围内"""
        coupling = _build(name, epsilon=1.0, num_iters=10)
        noise = torch.randn(50, 4)
        gt = torch.randn(10, 4)
        gt_labels = torch.randint(0, 24, (10,))
        _, matched_idx = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        assert (matched_idx >= 0).all()
        assert (matched_idx < 10).all()


class TestCouplingEmptyGT:
    """空 GT 边界情况"""

    @pytest.mark.parametrize("name", COUPLING_NAMES)
    def test_empty_gt(self, name):
        coupling = _build(name, epsilon=1.0, num_iters=10)
        noise = torch.randn(50, 4)
        gt = torch.zeros(0, 4)
        gt_labels = torch.zeros(0, dtype=torch.long)
        x_start, matched_idx = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        assert x_start.shape == (50, 4)
        assert matched_idx.shape == (50,)


class TestRandomCoupling:
    def test_randomness(self):
        """两次调用结果不同 (概率极高)"""
        coupling = build_coupling('random')
        noise = torch.randn(100, 4)
        gt = torch.randn(10, 4)
        gt_labels = torch.randint(0, 24, (10,))
        _, idx1 = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        _, idx2 = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        # 两次随机匹配几乎不可能完全相同
        assert not torch.equal(idx1, idx2)


class TestHardOTCoupling:
    def test_deterministic(self):
        """Hard OT 是确定性的"""
        coupling = build_coupling('hard_ot')
        noise = torch.randn(50, 4)
        gt = torch.randn(10, 4)
        gt_labels = torch.randint(0, 24, (10,))
        x1, idx1 = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        x2, idx2 = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        assert torch.equal(idx1, idx2)
        assert torch.allclose(x1, x2, atol=1e-7)

    def test_nearest_neighbor(self):
        """每个噪声匹配最近的 GT"""
        coupling = build_coupling('hard_ot')
        gt = torch.tensor([[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]])
        gt_labels = torch.tensor([0, 1])
        noise = torch.tensor([[0.1, 0.1, 0.1, 0.1], [9.9, 9.9, 9.9, 9.9]])
        x_start, idx = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        assert idx[0].item() == 0
        assert idx[1].item() == 1


class TestOTFlowArgmaxCoupling:
    """ot_flow with coupling_mode='argmax' (确定性)"""

    def test_deterministic(self):
        """argmax 解码是确定性的"""
        coupling = build_coupling(
            'ot_flow', epsilon=1.0, num_iters=10, coupling_mode='argmax'
        )
        noise = torch.randn(50, 4)
        gt = torch.randn(10, 4)
        gt_labels = torch.randint(0, 24, (10,))
        _, idx1 = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        _, idx2 = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        assert torch.equal(idx1, idx2)


class TestOTFlowMultinomialCoupling:
    """ot_flow with coupling_mode='multinomial' (随机采样)"""

    def test_stochastic(self):
        """随机采样应产生不同结果 (概率极高)"""
        coupling = build_coupling(
            'ot_flow', epsilon=5.0, num_iters=10, coupling_mode='multinomial'
        )
        noise = torch.randn(100, 4)
        gt = torch.randn(10, 4)
        gt_labels = torch.randint(0, 24, (10,))
        _, idx1 = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        _, idx2 = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        assert not torch.equal(idx1, idx2)
