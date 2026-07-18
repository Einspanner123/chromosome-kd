"""测试 Per-Head Time Reparameterization (PHTR)

方向2: 时间条件×级联头层级耦合
通过给每个级联头可学习的time_scale和time_shift，
使不同头可以关注不同的时间区间。

根因: DiffusionDetHead.forward 中 time_emb = self.time_mlp(t) 只计算一次,
所有6个级联头接收完全相同的 time_emb, 没有头层级特化机制。
PHTR 给每个头一对 (scale_i, shift_i) 做仿射变换, 初始化为 identity 保持兼容。

核心验证:
1. 启用 PHTR 时参数存在且形状正确 [num_heads, feat_channels*4]
2. 默认不启用 PHTR (向后兼容)
3. PHTR 初始化为 identity: scale=1, shift=0 (不破坏预训练)
4. PHTR 启用时 forward 正常, 且不同头接收不同的 time_emb (hook 验证)
5. PHTR 禁用时不创建参数, 行为与原始一致
6. identity PHTR 与禁用 PHTR 输出完全一致
7. 非 identity PHTR 在 adaln_mlp 非零时改变输出 (验证端到端耦合)
"""

import os
import sys

import torch

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ldmdet.core.head import DiffusionDetHead
from ldmdet.core.roi_extractor import SingleRoIExtractor
from ldmdet.core.single_head import SingleDiffusionDetHead


def _make_single_head(num_classes=24, feat_channels=64):
    """构建最小 SingleDiffusionDetHead (AdaLN-Zero 时间条件化)"""
    return SingleDiffusionDetHead(
        num_classes=num_classes,
        feat_channels=feat_channels,
        num_cls_convs=1,
        num_reg_convs=2,
        use_focal_loss=True,
        use_normalized_classifier=False,
        time_conditioning='adaln_zero',
    )


def _make_roi_extractor(out_channels=64):
    """构建最小 RoIExtractor (与 test_cascade_detach.py 一致)"""
    return SingleRoIExtractor(
        featmap_strides=[16],
        out_channels=out_channels,
        roi_layer=dict(
            type='RoIAlign', output_size=7, sampling_ratio=0, aligned=True
        ),
    )


def _enable_adaln_effect(head):
    """将各 head 的 adaln_mlp 末层权重置为非零。

    AdaLN-Zero 默认零初始化 adaln_mlp 末层, 导致 time_emb 对输出无影响
    (这正是 A1=A2 的根因)。本辅助函数打破零初始化, 使 time_emb 真正参与
    计算, 用于验证 PHTR 的端到端耦合效果 (梯度回传 / 输出变化)。
    """
    for h in head.head_series:
        last = h.adaln_mlp[-1]
        torch.nn.init.normal_(last.weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(last.bias, mean=0.0, std=0.1)


def _make_dummy_input(bs=2, num_boxes=10, feat_channels=64):
    """构造 dummy 输入 (有效 xyxy 框)"""
    torch.manual_seed(42)
    features = [torch.randn(bs, feat_channels, 16, 16)]
    bboxes = torch.rand(bs, num_boxes, 4) * 100
    bboxes[..., 2:] = bboxes[..., :2] + torch.rand(bs, num_boxes, 2) * 50 + 10
    t = torch.tensor([500.0, 500.0])
    return features, bboxes, t


class TestPerHeadTimeReparam:
    """PHTR: 每个头的time_emb做独立的仿射变换"""

    def test_phtr_parameters_exist_when_enabled(self):
        """启用PHTR时，DiffusionDetHead应包含head_time_scale和head_time_shift参数"""
        single_head = _make_single_head()
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_proposals=10,
            num_heads=6,
            single_head=single_head,
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_time_reparam=True,  # 新参数
        )
        assert hasattr(head, 'head_time_scale')
        assert hasattr(head, 'head_time_shift')
        # [num_heads, feat_channels*4]
        assert head.head_time_scale.shape == (6, 64 * 4)
        assert head.head_time_shift.shape == (6, 64 * 4)
        # 应为 nn.Parameter (可学习)
        assert isinstance(head.head_time_scale, torch.nn.Parameter)
        assert isinstance(head.head_time_shift, torch.nn.Parameter)

    def test_phtr_disabled_by_default(self):
        """默认不启用PHTR，保持向后兼容"""
        single_head = _make_single_head()
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_proposals=10,
            num_heads=6,
            single_head=single_head,
            roi_extractor=_make_roi_extractor(),
            criterion=None,
        )
        assert not hasattr(head, 'head_time_scale')
        assert not hasattr(head, 'head_time_shift')
        assert head.use_time_reparam is False

    def test_phtr_identity_initialization(self):
        """PHTR初始化为identity: scale=1, shift=0，确保不改变原始行为"""
        single_head = _make_single_head()
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_proposals=10,
            num_heads=6,
            single_head=single_head,
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_time_reparam=True,
        )
        assert torch.allclose(
            head.head_time_scale, torch.ones_like(head.head_time_scale)
        )
        assert torch.allclose(
            head.head_time_shift, torch.zeros_like(head.head_time_shift)
        )

    def test_phtr_passes_different_time_emb_per_head(self):
        """PHTR启用时, 不同头应接收不同的 time_emb (通过 forward pre-hook 验证)

        设置 head 0 的 scale=2.0/shift=0.5, head 2 保持 identity,
        则 head 0 收到的 time_emb 应 != head 2 收到的 time_emb。
        """
        single_head = _make_single_head()
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_proposals=10,
            num_heads=6,
            single_head=single_head,
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_time_reparam=True,
        )
        # head 0: 非identity; head 2: 保持 identity
        with torch.no_grad():
            head.head_time_scale[0] = 2.0
            head.head_time_shift[0] = 0.5

        # 用 forward pre-hook 捕获每个 head 收到的 time_emb (第5个位置参数)
        captured = {}

        def make_hook(idx):
            def hook(module, args):
                # forward(features, bboxes, proposals, pooler, time_emb)
                captured[idx] = args[4].clone()

            return hook

        for i, h in enumerate(head.head_series):
            h.register_forward_pre_hook(make_hook(i))

        features, bboxes, t = _make_dummy_input()
        head.eval()
        with torch.no_grad():
            cls_logits, pred_bboxes, _ = head(features, bboxes, t)

        # 形状检查 (deep_supervision=True 默认返回所有头)
        assert cls_logits.shape == (6, 2, 10, 24)
        assert pred_bboxes.shape == (6, 2, 10, 4)

        # 核心验证: head 0 收到的 time_emb != head 2 收到的 time_emb
        assert 0 in captured and 2 in captured, 'hook 应捕获 head 0 和 head 2'
        assert not torch.allclose(captured[0], captured[2], atol=1e-6), (
            'PHTR 应使 head 0 (非identity) 和 head 2 (identity) 收到不同 time_emb'
        )
        # head 2 保持 identity: time_emb_2 = time_emb * 1 + 0 = 原始 time_emb
        # head 0: time_emb_0 = time_emb * 2.0 + 0.5
        # 验证仿射关系: time_emb_0 ≈ time_emb_2 * 2.0 + 0.5
        assert torch.allclose(
            captured[0], captured[2] * 2.0 + 0.5, atol=1e-5
        ), 'head 0 的 time_emb 应满足 time_emb * 2.0 + 0.5'

    def test_phtr_backward_compatible_when_disabled(self):
        """PHTR禁用时，forward行为与原始完全一致 (无PHTR参数)"""
        single_head = _make_single_head()
        head_disabled = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_proposals=10,
            num_heads=6,
            single_head=single_head,
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_time_reparam=False,
        )
        # 禁用时不应该有 phtr 参数
        assert not hasattr(head_disabled, 'head_time_scale')
        assert not hasattr(head_disabled, 'head_time_shift')
        assert head_disabled.use_time_reparam is False

        # forward 应正常工作
        features, bboxes, t = _make_dummy_input()
        cls_logits, pred_bboxes, _ = head_disabled(features, bboxes, t)
        assert cls_logits.shape == (6, 2, 10, 24)
        assert pred_bboxes.shape == (6, 2, 10, 4)

    def test_phtr_gradient_flows_when_adaln_active(self):
        """PHTR 参数在 adaln_mlp 非零时应接收梯度 (端到端耦合验证)

        AdaLN-Zero 默认零初始化 adaln_mlp, 使 time_emb 对输出无影响,
        因此 PHTR 参数梯度为零。本测试先打破零初始化, 验证梯度能回传。
        """
        single_head = _make_single_head()
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_proposals=10,
            num_heads=6,
            single_head=single_head,
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_time_reparam=True,
        )
        _enable_adaln_effect(head)  # 打破 adaln_mlp 零初始化

        features, bboxes, t = _make_dummy_input()
        head.train()
        cls_logits, pred_bboxes, _ = head(features, bboxes, t)
        loss = cls_logits.sum() + pred_bboxes.sum()
        loss.backward()

        assert head.head_time_scale.grad is not None, (
            'PHTR scale 应接收梯度 (adaln 非零时)'
        )
        assert head.head_time_shift.grad is not None, (
            'PHTR shift 应接收梯度 (adaln 非零时)'
        )
        # 至少部分梯度非零
        assert head.head_time_scale.grad.abs().sum() > 0


class TestPerHeadTimeReparamNumerics:
    """PHTR 数值正确性: 验证仿射变换的数学性质"""

    def test_phtr_identity_equals_disabled(self):
        """PHTR启用但保持identity时, 输出应与禁用PHTR时一致"""
        features, bboxes, t = _make_dummy_input()

        # 禁用 PHTR
        single_head_a = _make_single_head()
        head_disabled = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_proposals=10,
            num_heads=6,
            single_head=single_head_a,
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_time_reparam=False,
        )
        head_disabled.eval()
        with torch.no_grad():
            cls_a, bboxes_a, _ = head_disabled(features, bboxes, t)

        # 启用 PHTR 但保持 identity (scale=1, shift=0)
        single_head_b = _make_single_head()
        # 复制权重使两者初始参数一致
        single_head_b.load_state_dict(single_head_a.state_dict())
        head_identity = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_proposals=10,
            num_heads=6,
            single_head=single_head_b,
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_time_reparam=True,  # identity 初始化
        )
        # 复制 time_mlp 和 head_series 权重 (deepcopy 后参数独立)
        head_identity.time_mlp.load_state_dict(
            head_disabled.time_mlp.state_dict()
        )
        for i in range(6):
            head_identity.head_series[i].load_state_dict(
                head_disabled.head_series[i].state_dict()
            )
        head_identity.eval()
        with torch.no_grad():
            cls_b, bboxes_b, _ = head_identity(features, bboxes, t)

        # identity 变换: time_emb * 1 + 0 = time_emb, 输出应完全一致
        assert torch.allclose(cls_a, cls_b, atol=1e-6), (
            'PHTR identity 应与禁用 PHTR 输出一致'
        )
        assert torch.allclose(bboxes_a, bboxes_b, atol=1e-6)

    def test_phtr_different_scales_change_output_when_adaln_active(self):
        """非identity的scale/shift在adaln_mlp非零时应改变输出

        AdaLN-Zero 默认零初始化使 time_emb 无效, 因此需先打破零初始化,
        才能验证 PHTR 的非 identity 变换对输出的影响。
        """
        features, bboxes, t = _make_dummy_input()

        single_head = _make_single_head()
        head = DiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_proposals=10,
            num_heads=6,
            single_head=single_head,
            roi_extractor=_make_roi_extractor(),
            criterion=None,
            use_time_reparam=True,
        )
        _enable_adaln_effect(head)  # 打破 adaln_mlp 零初始化
        head.eval()

        # identity 时的输出
        with torch.no_grad():
            cls_identity, bboxes_identity, _ = head(features, bboxes, t)

        # 设置非 identity 的 scale/shift
        with torch.no_grad():
            head.head_time_scale.fill_(1.5)
            head.head_time_shift.fill_(0.2)

        with torch.no_grad():
            cls_modified, bboxes_modified, _ = head(features, bboxes, t)

        # 非identity变换应改变输出 (adaln_mlp 非零时 time_emb 影响输出)
        assert not torch.allclose(cls_identity, cls_modified, atol=1e-4), (
            '非 identity 的 scale/shift 应改变分类输出 (adaln 非零时)'
        )
        assert not torch.allclose(
            bboxes_identity, bboxes_modified, atol=1e-4
        ), '非 identity 的 scale/shift 应改变框回归输出 (adaln 非零时)'
