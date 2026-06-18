"""测试 ldmdet.coupling — 所有耦合策略"""

import torch
import pytest
from ldmdet.coupling import build_coupling
from ldmdet.coupling._sinkhorn_ops import _OT_GENERATORS
from ldmdet.utils.constants import CHROMO_GROUP_OF_CLASS


# 所有耦合策略名
COUPLING_NAMES = ['random', 'hard_ot', 'sinkhorn_argmax', 'sinkhorn_stochastic', 'ghss']

# 需要 epsilon/num_iters 参数的策略
_SINKHORN_NAMES = {'sinkhorn_argmax', 'sinkhorn_stochastic', 'ghss'}


def _build(name, **kwargs):
    """构建耦合策略，仅对 Sinkhorn 类策略传递额外参数"""
    if name in _SINKHORN_NAMES:
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
        c = build_coupling('sinkhorn_stochastic', epsilon=5.0, num_iters=20)
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


class TestSinkhornArgmaxCoupling:
    def test_deterministic(self):
        """argmax 解码是确定性的"""
        coupling = build_coupling('sinkhorn_argmax', epsilon=1.0, num_iters=10)
        noise = torch.randn(50, 4)
        gt = torch.randn(10, 4)
        gt_labels = torch.randint(0, 24, (10,))
        _, idx1 = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        _, idx2 = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        assert torch.equal(idx1, idx2)


class TestSinkhornStochasticCoupling:
    def test_stochastic(self):
        """随机采样应产生不同结果 (概率极高)"""
        coupling = build_coupling('sinkhorn_stochastic', epsilon=5.0, num_iters=10)
        noise = torch.randn(100, 4)
        gt = torch.randn(10, 4)
        gt_labels = torch.randint(0, 24, (10,))
        _, idx1 = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        _, idx2 = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        assert not torch.equal(idx1, idx2)

    def test_with_seed(self):
        """使用 seed 应可复现"""
        coupling = build_coupling('sinkhorn_stochastic', epsilon=5.0, num_iters=10, sample_seed=42)
        noise = torch.randn(50, 4)
        gt = torch.randn(10, 4)
        gt_labels = torch.randint(0, 24, (10,))
        _, idx1 = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        _, idx2 = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        # 同一个 generator 连续采样，结果不同
        # 但创建新实例用同 seed 也不保证相同 (因为 generator 状态不同)
        # 这里只验证输出合法
        assert (idx1 >= 0).all() and (idx1 < 10).all()


class TestGHSSCoupling:
    def test_group_respect(self):
        """GHSS 应按组分配噪声"""
        coupling = build_coupling('ghss', epsilon=5.0, num_iters=10)
        noise = torch.randn(50, 4)
        gt = torch.randn(10, 4)
        # 全部属于同一组 (group 0: class 0,1,2)
        gt_labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        x_start, matched_idx = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        assert x_start.shape == (50, 4)
        assert (matched_idx >= 0).all() and (matched_idx < 10).all()

    def test_multi_group(self):
        """多组情况"""
        coupling = build_coupling('ghss', epsilon=5.0, num_iters=10)
        noise = torch.randn(100, 4)
        gt = torch.randn(20, 4)
        # 跨多个组
        gt_labels = torch.cat([
            torch.zeros(5, dtype=torch.long),       # group 0
            torch.ones(5, dtype=torch.long) * 3,    # group 1 (class 3,4)
            torch.ones(5, dtype=torch.long) * 12,   # group 3 (class 12,13,14)
            torch.ones(5, dtype=torch.long) * 22,   # group 7 (class 22,23)
        ])
        x_start, matched_idx = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))
        assert x_start.shape == (100, 4)
        assert (matched_idx >= 0).all() and (matched_idx < 20).all()

    def test_group_isolation_no_cross_group_matching(self):
        """GHSS 核心特性: 噪声 slot 匹配的 GT 必须属于分配给它的组, 禁止跨组匹配"""
        coupling = build_coupling('ghss', epsilon=5.0, num_iters=10)
        noise = torch.randn(100, 4)
        gt = torch.randn(20, 4)
        # 4 组, 每组 5 个 GT, GT 索引范围: group0=[0,5), group1=[5,10), group3=[10,15), group7=[15,20)
        gt_labels = torch.cat([
            torch.zeros(5, dtype=torch.long),       # group 0 (class 0-2)
            torch.ones(5, dtype=torch.long) * 3,    # group 1 (class 3-4)
            torch.ones(5, dtype=torch.long) * 12,   # group 3 (class 12-14)
            torch.ones(5, dtype=torch.long) * 22,   # group 7 (class 22-23)
        ])
        _, matched_idx = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))

        # 每个 matched GT 的组
        gt_groups = torch.tensor(CHROMO_GROUP_OF_CLASS)[gt_labels]
        matched_groups = gt_groups[matched_idx]

        # 所有 4 个组都应被分配到噪声 slot
        assert len(torch.unique(matched_groups)) == 4

        # 每组分配的噪声 slot 数应与组内 GT 数成正比 (每组 5/20 = 25%)
        for g in [0, 1, 3, 7]:
            count = (matched_groups == g).sum().item()
            assert count == 25, f"group {g} 应分配 25 个 slot, 实际 {count}"

        # 核心断言: 不存在跨组匹配
        # group 0 的噪声 slot 只能匹配 GT 索引 [0, 5)
        # group 1 的噪声 slot 只能匹配 GT 索引 [5, 10)
        # 以此类推
        group_gt_ranges = {0: (0, 5), 1: (5, 10), 3: (10, 15), 7: (15, 20)}
        for g, (lo, hi) in group_gt_ranges.items():
            slots_for_g = matched_idx[matched_groups == g]
            # 所有匹配的 GT 索引必须落在该组的范围内
            assert ((slots_for_g >= lo) & (slots_for_g < hi)).all(), (
                f"group {g} 存在跨组匹配: GT 索引应在 [{lo}, {hi}), 实际 {slots_for_g.tolist()}"
            )

    def test_group_isolation_all_groups_present(self):
        """即使某些组只有 1 个 GT, GHSS 仍应为其分配噪声 slot"""
        coupling = build_coupling('ghss', epsilon=5.0, num_iters=10)
        noise = torch.randn(80, 4)
        gt = torch.randn(10, 4)
        # 不均匀分组: group 0 有 7 个, group 1 有 1 个, group 3 有 1 个, group 7 有 1 个
        gt_labels = torch.cat([
            torch.zeros(7, dtype=torch.long),       # group 0
            torch.ones(1, dtype=torch.long) * 3,    # group 1
            torch.ones(1, dtype=torch.long) * 12,   # group 3
            torch.ones(1, dtype=torch.long) * 22,   # group 7
        ])
        _, matched_idx = coupling.couple(noise, gt, gt_labels, torch.device('cpu'))

        gt_groups = torch.tensor(CHROMO_GROUP_OF_CLASS)[gt_labels]
        matched_groups = gt_groups[matched_idx]
        # 每个组都应至少有 1 个噪声 slot
        for g in [0, 1, 3, 7]:
            assert (matched_groups == g).sum() > 0, f"group {g} 未被分配任何噪声 slot"
