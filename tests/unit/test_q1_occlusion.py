"""方向Q Phase 1 (Q1): 训练时遮挡模拟单元测试

测试 SingleDiffusionDetHead 的 occlusion_prob / occlusion_ratio_range 参数:
- occlusion_prob=0.0 (默认): 不施加遮挡 (向后兼容)
- occlusion_prob>0: 训练时对 RoI 特征施加随机矩形零填充遮挡
- eval 模式: 不施加遮挡 (推理零开销)

对应 docs/research/frontier_directions/方向Q_ATD_Amodal轨迹扩散.md §3.1 Q1 设计。

核心验证:
1. 参数存在且默认值正确 (向后兼容)
2. _apply_occlusion 方法在 prob=0 时不遮挡, prob=1 时全部遮挡
3. 遮挡区域为零, 非遮挡区域保持不变
4. training 模式施加遮挡, eval 模式不施加
5. 遮挡比例在 occlusion_ratio_range 范围内
"""

import os
import sys

import pytest
import torch

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ldmdet.core.single_head import SingleDiffusionDetHead


def _make_single_head(
    num_classes=24,
    feat_channels=64,
    occlusion_prob=0.0,
    occlusion_ratio_range=(0.2, 0.6),
):
    """构建最小 SingleDiffusionDetHead, 支持遮挡参数"""
    return SingleDiffusionDetHead(
        num_classes=num_classes,
        feat_channels=feat_channels,
        num_cls_convs=1,
        num_reg_convs=2,
        use_focal_loss=True,
        use_normalized_classifier=False,
        occlusion_prob=occlusion_prob,
        occlusion_ratio_range=occlusion_ratio_range,
    )


class TestOcclusionParameters:
    """测试遮挡参数存在性和默认值"""

    def test_occlusion_prob_exists_default_zero(self):
        """SingleDiffusionDetHead 应有 occlusion_prob 参数, 默认 0.0"""
        head = _make_single_head()
        assert hasattr(head, 'occlusion_prob')
        assert head.occlusion_prob == 0.0

    def test_occlusion_ratio_range_exists_default(self):
        """SingleDiffusionDetHead 应有 occlusion_ratio_range, 默认 (0.2, 0.6)"""
        head = _make_single_head()
        assert hasattr(head, 'occlusion_ratio_range')
        assert head.occlusion_ratio_range == (0.2, 0.6)

    def test_occlusion_prob_can_be_set(self):
        """occlusion_prob 应可设置为非零值"""
        head = _make_single_head(occlusion_prob=0.3)
        assert head.occlusion_prob == 0.3

    def test_occlusion_ratio_range_can_be_set(self):
        """occlusion_ratio_range 应可自定义"""
        head = _make_single_head(occlusion_ratio_range=(0.3, 0.5))
        assert head.occlusion_ratio_range == (0.3, 0.5)

    def test_backward_compatible_no_occlusion_args(self):
        """不传遮挡参数时应正常构造 (向后兼容)"""
        head = SingleDiffusionDetHead(
            num_classes=24,
            feat_channels=64,
            num_cls_convs=1,
            num_reg_convs=2,
            use_focal_loss=True,
        )
        assert head.occlusion_prob == 0.0
        assert head.occlusion_ratio_range == (0.2, 0.6)


class TestApplyOcclusionBehavior:
    """测试 _apply_occlusion 方法的核心行为"""

    def test_prob_zero_no_occlusion(self):
        """occlusion_prob=0.0 时, 输出应与输入完全一致"""
        head = _make_single_head(occlusion_prob=0.0)
        head.train()
        roi_features = torch.randn(20, 64, 7, 7)
        out = head._apply_occlusion(roi_features)
        assert torch.equal(out, roi_features)

    def test_prob_one_all_occluded(self):
        """occlusion_prob=1.0 时, 所有 RoI 应被遮挡 (至少有一个零区域)"""
        torch.manual_seed(42)
        head = _make_single_head(occlusion_prob=1.0)
        head.train()
        # 用非零特征确保零填充可检测
        roi_features = torch.ones(20, 64, 7, 7)
        out = head._apply_occlusion(roi_features)
        # 每个 RoI 应有至少一个零元素
        for i in range(20):
            assert (out[i] == 0).any(), f'RoI {i} 应有遮挡区域 (零值)'

    def test_occluded_region_is_zero(self):
        """遮挡区域应为零"""
        torch.manual_seed(42)
        head = _make_single_head(occlusion_prob=1.0)
        head.train()
        roi_features = torch.ones(10, 64, 7, 7) * 5.0
        out = head._apply_occlusion(roi_features)
        # 零值区域就是遮挡区域
        assert (out == 0).any(), '应有零值遮挡区域'
        # 非零区域应保持原值 5.0
        non_zero_mask = out != 0
        assert torch.all(
            out[non_zero_mask] == 5.0
        ), '非遮挡区域应保持原值'

    def test_non_occluded_region_unchanged(self):
        """非遮挡区域应保持原值不变"""
        torch.manual_seed(123)
        head = _make_single_head(occlusion_prob=1.0)
        head.train()
        roi_features = torch.randn(10, 64, 7, 7)
        out = head._apply_occlusion(roi_features)
        # 非零区域应等于原始值
        non_zero_mask = out != 0
        assert torch.equal(
            out[non_zero_mask], roi_features[non_zero_mask]
        ), '非遮挡区域应保持原值'

    def test_occlusion_ratio_in_range(self):
        """遮挡面积比例应在 occlusion_ratio_range 范围内 (允许 int 截断误差)"""
        torch.manual_seed(42)
        ratio_range = (0.2, 0.6)
        head = _make_single_head(
            occlusion_prob=1.0, occlusion_ratio_range=ratio_range
        )
        head.train()
        H, W = 7, 7
        total = H * W  # 49
        roi_features = torch.ones(100, 64, H, W)
        out = head._apply_occlusion(roi_features)
        for i in range(100):
            # 遮挡掩码在所有通道上一致, 只需统计单通道的空间零像素数
            zero_count = (out[i, 0] == 0).sum().item()
            actual_ratio = zero_count / total
            # int 截断可能导致比例略低于下限, 允许 1 像素容差
            min_allowed = ratio_range[0] - 1.0 / total
            max_allowed = ratio_range[1] + 1.0 / total
            assert (
                min_allowed <= actual_ratio <= max_allowed
            ), f'RoI {i}: 遮挡比例 {actual_ratio:.3f} 超出范围 [{min_allowed:.3f}, {max_allowed:.3f}]'

    def test_eval_mode_no_occlusion(self):
        """eval 模式下 _apply_occlusion 应不施加遮挡"""
        head = _make_single_head(occlusion_prob=1.0)
        head.eval()
        roi_features = torch.randn(10, 64, 7, 7)
        out = head._apply_occlusion(roi_features)
        assert torch.equal(out, roi_features), 'eval 模式不应施加遮挡'

    def test_different_rois_independent(self):
        """不同 RoI 的遮挡应独立 (遮挡区域/比例不同)"""
        torch.manual_seed(42)
        head = _make_single_head(occlusion_prob=1.0)
        head.train()
        roi_features = torch.ones(50, 64, 7, 7)
        out = head._apply_occlusion(roi_features)
        # 至少有两个 RoI 的遮挡模式不同
        patterns = [(out[i] == 0).sum().item() for i in range(50)]
        assert len(set(patterns)) > 1, '不同 RoI 的遮挡应有差异'

    def test_preserves_shape_and_dtype(self):
        """遮挡后形状和数据类型应保持不变"""
        head = _make_single_head(occlusion_prob=1.0)
        head.train()
        roi_features = torch.randn(10, 64, 7, 7, dtype=torch.float32)
        out = head._apply_occlusion(roi_features)
        assert out.shape == roi_features.shape
        assert out.dtype == roi_features.dtype


class TestOcclusionInForward:
    """测试 forward 中遮挡的集成行为"""

    def _make_dummy_input(self, bs=2, num_boxes=10, feat_channels=64):
        """构造 forward 所需的 dummy 输入"""
        features = [torch.randn(bs, feat_channels, 16, 16)]
        bboxes = torch.rand(bs, num_boxes, 4)
        # 确保有效框: x2 > x1, y2 > y1
        bboxes[:, :, 2] = bboxes[:, :, 0] + 0.3
        bboxes[:, :, 3] = bboxes[:, :, 1] + 0.3
        return features, bboxes

    def _make_pooler(self, out_channels=64):
        """构建最小 RoIExtractor"""
        from ldmdet.core.roi_extractor import SingleRoIExtractor
        return SingleRoIExtractor(
            featmap_strides=[16], out_channels=out_channels,
            roi_layer=dict(
                type='RoIAlign', output_size=7, sampling_ratio=0, aligned=True
            ),
        )

    def test_forward_training_with_occlusion(self):
        """training + occlusion_prob=0.5 时 forward 应正常工作"""
        head = _make_single_head(occlusion_prob=0.5)
        head.train()
        features, bboxes = self._make_dummy_input()
        pooler = self._make_pooler()
        time_emb = torch.randn(2, 64 * 4)
        proposals = None
        # forward 应不报错
        cls_logits, pred_bboxes, fc = head(
            features, bboxes, proposals, pooler, time_emb
        )
        assert cls_logits.shape[0] == 2
        assert pred_bboxes.shape[0] == 2

    def test_forward_eval_no_occlusion(self):
        """eval 模式 forward 应正常工作, 不施加遮挡"""
        head = _make_single_head(occlusion_prob=1.0)
        head.eval()
        features, bboxes = self._make_dummy_input()
        pooler = self._make_pooler()
        time_emb = torch.randn(2, 64 * 4)
        cls_logits, pred_bboxes, fc = head(
            features, bboxes, None, pooler, time_emb
        )
        assert cls_logits.shape[0] == 2

    def test_forward_default_no_occlusion(self):
        """默认参数 (occlusion_prob=0.0) forward 应正常工作"""
        head = _make_single_head(occlusion_prob=0.0)
        head.train()
        features, bboxes = self._make_dummy_input()
        pooler = self._make_pooler()
        time_emb = torch.randn(2, 64 * 4)
        cls_logits, pred_bboxes, fc = head(
            features, bboxes, None, pooler, time_emb
        )
        assert cls_logits.shape[0] == 2

    def test_forward_gradient_flows_with_occlusion(self):
        """遮挡后梯度仍能正常回传"""
        head = _make_single_head(occlusion_prob=0.5)
        head.train()
        features, bboxes = self._make_dummy_input()
        pooler = self._make_pooler()
        time_emb = torch.randn(2, 64 * 4)
        cls_logits, pred_bboxes, _ = head(
            features, bboxes, None, pooler, time_emb
        )
        loss = cls_logits.sum() + pred_bboxes.sum()
        loss.backward()
        # head 参数应有梯度
        has_grad = sum(
            1 for p in head.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert has_grad > 0, '遮挡后参数应有梯度'


class TestBackwardCompatibility:
    """测试向后兼容性: 默认参数行为与无遮挡一致"""

    def test_default_prob_zero_identical_to_no_occlusion(self):
        """occlusion_prob=0.0 的 _apply_occlusion 应返回原始张量"""
        head = _make_single_head(occlusion_prob=0.0)
        head.train()
        x = torch.randn(10, 64, 7, 7)
        out = head._apply_occlusion(x)
        assert out is x or torch.equal(out, x)

    def test_existing_configs_unaffected(self):
        """不传遮挡参数的 SingleDiffusionDetHead 应正常构造"""
        head = SingleDiffusionDetHead(
            num_classes=24,
            feat_channels=256,
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            use_focal_loss=True,
            time_conditioning='adaln_zero',
        )
        assert head.occlusion_prob == 0.0
        assert head.occlusion_ratio_range == (0.2, 0.6)
