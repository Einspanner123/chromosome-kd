"""方向 A 测试: 多尺度特征增强 (P1 FPN + DeformableRoIExtractor)

- A1: FPNWithP1 — 在标准 FPN 基础上增加 stride 2 的 P1 层
- A2: DeformableRoIExtractor — RoIAlign + DeformConv2d (可学习空间偏移)
"""

import pytest
import torch
import torch.nn as nn

from ldmdet.core.roi_extractor import SingleRoIExtractor
from ldmdet.core.deformable_roi_extractor import DeformableRoIExtractor


# ──────────────────────────────────────────────────────────────
# A2: DeformableRoIExtractor (纯 PyTorch, 不依赖 mmdet)
# ──────────────────────────────────────────────────────────────

class TestDeformableRoIExtractor:
    """A2: DeformableRoIExtractor = RoIAlign + DeformConv2d (residual)"""

    @pytest.fixture
    def extractor(self):
        return DeformableRoIExtractor(
            roi_layer={
                'type': 'RoIAlign',
                'output_size': 7,
                'sampling_ratio': 2,
                'aligned': True,
            },
            out_channels=64,
            featmap_strides=[4, 8, 16, 32],
            deform_groups=1,
        )

    def _make_feats(self, bs=1, h=64, w=64, channels=64):
        return tuple([
            torch.randn(bs, channels, h // s, w // s)
            for s in [4, 8, 16, 32]
        ])

    def test_output_shape(self, extractor):
        """输出形状应与 SingleRoIExtractor 一致: (N, C, 7, 7)"""
        feats = self._make_feats()
        rois = torch.tensor([[0, 10.0, 20.0, 50.0, 80.0],
                             [0, 30.0, 40.0, 100.0, 120.0]])
        out = extractor(feats, rois)
        assert out.shape == (2, 64, 7, 7)

    def test_is_subclass_of_single_extractor(self, extractor):
        """应继承 SingleRoIExtractor, 复用层级映射逻辑"""
        assert isinstance(extractor, SingleRoIExtractor)

    def test_has_deform_conv(self, extractor):
        """必须包含 DeformConv2d (可学习偏移)"""
        from torchvision.ops import DeformConv2d
        assert any(isinstance(m, DeformConv2d) for m in extractor.modules())

    def test_has_offset_predictor(self, extractor):
        """必须有偏移预测卷积"""
        assert hasattr(extractor, 'offset_conv')
        assert isinstance(extractor.offset_conv, nn.Conv2d)
        # offset 输出通道 = 2 * kh * kw * deform_groups
        # 3x3 kernel, deform_groups=1 => 2*3*3*1 = 18
        assert extractor.offset_conv.out_channels == 18

    def test_residual_connection(self, extractor):
        """deform conv 应使用残差连接, 不改变通道数"""
        extractor.eval()
        feats = self._make_feats()
        rois = torch.tensor([[0, 10.0, 20.0, 50.0, 80.0]])
        # 不应抛出异常, 输出通道 = 输入通道
        out = extractor(feats, rois)
        assert out.shape[1] == 64  # out_channels

    def test_gradient_flow(self, extractor):
        """梯度应能流过 DeformConv2d 和 offset 预测器"""
        feats = self._make_feats()
        rois = torch.tensor([[0, 10.0, 20.0, 50.0, 80.0]])
        for f in feats:
            f.requires_grad_(True)
        out = extractor(feats, rois)
        out.sum().backward()
        # DeformConv2d 权重应有梯度
        from torchvision.ops import DeformConv2d
        deform = next(m for m in extractor.modules() if isinstance(m, DeformConv2d))
        assert deform.weight.grad is not None
        # offset 预测器权重应有梯度
        assert extractor.offset_conv.weight.grad is not None

    def test_different_deform_groups(self):
        """支持不同 deform_groups"""
        extractor = DeformableRoIExtractor(
            roi_layer={'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 2},
            out_channels=64,
            featmap_strides=[4, 8, 16, 32],
            deform_groups=2,
        )
        assert extractor.offset_conv.out_channels == 2 * 3 * 3 * 2  # 36

    def test_deterministic_eval(self, extractor):
        """eval 模式下应确定性输出"""
        extractor.eval()
        feats = self._make_feats()
        rois = torch.tensor([[0, 10.0, 20.0, 50.0, 80.0]])
        with torch.no_grad():
            out1 = extractor(feats, rois)
            out2 = extractor(feats, rois)
        assert torch.allclose(out1, out2, atol=1e-6)


# ──────────────────────────────────────────────────────────────
# A1: FPNWithP1 (依赖 mmdet, 单独测试)
# ──────────────────────────────────────────────────────────────

class TestFPNWithP1:
    """A1: FPNWithP1 — 在标准 FPN 基础上增加 P1 (stride 2) 层"""

    @pytest.fixture
    def neck(self):
        from experiments.mmdet_bridge.necks.fpn_with_p1 import FPNWithP1
        return FPNWithP1(
            in_channels=[256, 512, 1024, 2048],
            out_channels=64,
            num_outs=4,  # FPN 原有输出 P2-P5
            start_level=0,
        )

    def _make_backbone_inputs(self, bs=2, h=128, w=128):
        """ResNet50 backbone 输出 (strides 4,8,16,32)"""
        return tuple([
            torch.randn(bs, 256, h // 4, w // 4),    # C2 stride 4
            torch.randn(bs, 512, h // 8, w // 8),    # C3 stride 8
            torch.randn(bs, 1024, h // 16, w // 16), # C4 stride 16
            torch.randn(bs, 2048, h // 32, w // 32), # C5 stride 32
        ])

    def test_output_5_levels(self, neck):
        """输出应为 5 层 (P1 + P2-P5)"""
        inputs = self._make_backbone_inputs()
        outs = neck(inputs)
        assert len(outs) == 5

    def test_p1_is_stride_2(self, neck):
        """P1 (outs[0]) 应为 stride 2 = 输入尺寸的 1/2"""
        inputs = self._make_backbone_inputs(h=128, w=128)
        outs = neck(inputs)
        # P1 应为 64x64 (输入 128x128, stride 2)
        assert outs[0].shape[-2:] == (64, 64)

    def test_p2_is_stride_4(self, neck):
        """P2 (outs[1]) 应为 stride 4 = 32x32"""
        inputs = self._make_backbone_inputs(h=128, w=128)
        outs = neck(inputs)
        assert outs[1].shape[-2:] == (32, 32)

    def test_all_outs_channels(self, neck):
        """所有输出应有相同的 out_channels"""
        inputs = self._make_backbone_inputs()
        outs = neck(inputs)
        for o in outs:
            assert o.shape[1] == 64

    def test_gradient_flow(self, neck):
        """梯度应流过 P1 分支"""
        inputs = self._make_backbone_inputs()
        for i in inputs:
            i.requires_grad_(True)
        outs = neck(inputs)
        outs[0].sum().backward()  # 只对 P1 反向传播
        # P1 来自 inputs[0] (C2), 应有梯度
        assert inputs[0].grad is not None

    def test_p1_lateral_conv_exists(self, neck):
        """必须有 p1_lateral_conv 投影 C2 → out_channels"""
        assert hasattr(neck, 'p1_lateral_conv')
        assert isinstance(neck.p1_lateral_conv, nn.Conv2d)
        assert neck.p1_lateral_conv.in_channels == 256  # C2 通道
        assert neck.p1_lateral_conv.out_channels == 64  # out_channels

    def test_p1_conv_exists(self, neck):
        """必须有 p1_conv 平滑卷积"""
        assert hasattr(neck, 'p1_conv')
        assert isinstance(neck.p1_conv, nn.Conv2d)
