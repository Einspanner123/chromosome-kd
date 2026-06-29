"""方向 G 测试: LAMFPN 局部注意力特征金字塔

G1: LAMModule — softmax 注意力融合 (代替 FPN 简单相加)
G1: DualAttention — 通道注意力 + 空间注意力
G1: LAMFPN — 继承 FPN, 在 top-down 路径应用 LAM + DualAttention
G3: CrossLayerAttention — 跨层注意力 (可选附加)
"""

import pytest
import torch

from ldmdet.necks.lam_fpn import LAMFPN, LAMModule, DualAttention
from ldmdet.necks.lam_fpn import CrossLayerAttention


class TestLAMModule:
    """G1: 局部注意力融合模块"""

    @pytest.fixture
    def lam(self):
        return LAMModule(channels=256, channel_reduction=4)

    def test_output_shape(self, lam):
        """输出 shape 应与输入一致"""
        target = torch.randn(2, 256, 16, 16)
        source = torch.randn(2, 256, 16, 16)
        out = lam(target, source)
        assert out.shape == (2, 256, 16, 16)

    def test_attention_weights_sum_to_one(self, lam):
        """softmax 后权重和应为 1"""
        target = torch.randn(2, 256, 8, 8)
        source = torch.randn(2, 256, 8, 8)
        concat = torch.cat([target, source], dim=1)
        attn_feat = lam.attention_conv(concat)
        weights = lam.softmax(lam.attention_weights(attn_feat))
        assert weights.shape == (2, 2, 8, 8)
        assert torch.allclose(weights.sum(dim=1), torch.ones(2, 8, 8), atol=1e-5)

    def test_gradient_flow(self, lam):
        """梯度应能通过 attention_conv 和 fusion_conv 回传"""
        target = torch.randn(2, 256, 8, 8, requires_grad=True)
        source = torch.randn(2, 256, 8, 8, requires_grad=True)
        out = lam(target, source)
        out.sum().backward()
        # 检查 attention_conv 和 fusion_conv 的参数有梯度
        assert any(p.grad is not None for p in lam.attention_conv.parameters())
        assert any(p.grad is not None for p in lam.fusion_conv.parameters())

    def test_not_simple_addition(self, lam):
        """输出不应等于简单相加 (证明注意力生效)"""
        target = torch.randn(2, 256, 8, 8)
        source = torch.randn(2, 256, 8, 8)
        out = lam(target, source)
        simple_sum = target + source
        assert not torch.allclose(out, simple_sum, atol=1e-3)


class TestDualAttention:
    """G1: 通道 + 空间双重注意力"""

    @pytest.fixture
    def da(self):
        return DualAttention(channels=256, reduction=16)

    def test_output_shape(self, da):
        """输出 shape 应与输入一致"""
        x = torch.randn(2, 256, 16, 16)
        out = da(x)
        assert out.shape == (2, 256, 16, 16)

    def test_channel_attention_range(self, da):
        """通道注意力权重应在 [0, 1]"""
        x = torch.randn(2, 256, 8, 8)
        avg_pool = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        max_pool = torch.nn.functional.adaptive_max_pool2d(x, 1)
        avg_out = da.shared_mlp(avg_pool)
        max_out = da.shared_mlp(max_pool)
        channel_att = torch.sigmoid(avg_out + max_out)
        assert channel_att.min() >= 0.0
        assert channel_att.max() <= 1.0

    def test_gradient_flow(self, da):
        """梯度应能通过 shared_mlp 和 spatial_attention 回传"""
        x = torch.randn(2, 256, 8, 8, requires_grad=True)
        out = da(x)
        out.sum().backward()
        assert da.shared_mlp[0].weight.grad is not None
        # spatial_attention: [Conv2d(2,1,1), Conv2d(1,1,7), Sigmoid] — 取第 0 个 Conv
        assert da.spatial_attention[0].weight.grad is not None


class TestLAMFPN:
    """G1: LAMFPN 整体 (继承 FPN, 接口兼容)"""

    @pytest.fixture
    def fpn(self):
        return LAMFPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4,
            apply_lam_levels=[1, 2],
            attention_type='dual',
            use_cross_layer_attention=False,
            use_dw_conv=True,
            channel_reduction=4,
            use_modern_norm=True,
            use_modern_act='silu',
            use_checkpoint=False,
        )

    @pytest.fixture
    def inputs(self):
        # 模拟 ResNet-50 backbone 输出 (stride 4/8/16/32)
        return [
            torch.randn(2, 256, 64, 64),
            torch.randn(2, 512, 32, 32),
            torch.randn(2, 1024, 16, 16),
            torch.randn(2, 2048, 8, 8),
        ]

    def test_forward_shape(self, fpn, inputs):
        """前向应输出 num_outs 个特征图, 通道均为 out_channels"""
        outs = fpn(inputs)
        assert len(outs) == 4
        for i, o in enumerate(outs):
            assert o.shape[1] == 256
            # 分辨率应递减: 64, 32, 16, 8
            expected_h = [64, 32, 16, 8][i]
            assert o.shape[2] == expected_h

    def test_backward_compatible_with_fpn(self, fpn, inputs):
        """应能接受与标准 FPN 相同的输入 (backbone 4 stage 输出)"""
        outs = fpn(inputs)
        assert len(outs) == 4
        # 通道数应统一为 out_channels
        for o in outs:
            assert o.shape[1] == 256

    def test_lam_modules_at_correct_levels(self, fpn):
        """LAM 模块应只在 apply_lam_levels 指定层级应用"""
        # lam_modules 长度 = num_outs - 1 = 3 (索引 0,1,2 对应 P2-P3, P3-P4, P4-P5 融合)
        assert len(fpn.lam_modules) == 3
        # apply_lam_levels=[1,2] → 索引 1, 2 应有 LAM, 索引 0 应为 None
        assert fpn.lam_modules[0] is None
        assert fpn.lam_modules[1] is not None
        assert fpn.lam_modules[2] is not None

    def test_feature_attention_at_correct_levels(self, fpn):
        """DualAttention 应只在指定层级 [1, 2] 应用"""
        assert len(fpn.feature_attention) == 4
        assert fpn.feature_attention[0] is None
        assert fpn.feature_attention[1] is not None
        assert fpn.feature_attention[2] is not None
        assert fpn.feature_attention[3] is None

    def test_no_lam_when_empty_levels(self, inputs):
        """apply_lam_levels=[] 时, 应退化为标准 FPN (无 LAM 模块)"""
        fpn = LAMFPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4,
            apply_lam_levels=[],
            attention_type=None,
            use_cross_layer_attention=False,
            use_checkpoint=False,
        )
        outs = fpn(inputs)
        assert len(outs) == 4
        # 所有的 LAM 模块应为 None
        for m in fpn.lam_modules:
            assert m is None

    def test_gradient_flow(self, fpn, inputs):
        """梯度应能通过 LAM 模块回传"""
        for x in inputs:
            x.requires_grad_(True)
        outs = fpn(inputs)
        loss = sum(o.sum() for o in outs)
        loss.backward()
        # LAM 模块的参数应有梯度
        assert any(p.grad is not None for p in fpn.lam_modules[1].parameters())
        assert any(p.grad is not None for p in fpn.lam_modules[2].parameters())


class TestCrossLayerAttention:
    """G3: 跨层注意力"""

    @pytest.fixture
    def cla(self):
        return CrossLayerAttention(channels=256, num_levels=4)

    @pytest.fixture
    def features(self):
        return [
            torch.randn(2, 256, 64, 64),
            torch.randn(2, 256, 32, 32),
            torch.randn(2, 256, 16, 16),
            torch.randn(2, 256, 8, 8),
        ]

    def test_output_shapes_preserved(self, cla, features):
        """输出数量和 shape 应与输入一致"""
        outs = cla(features)
        assert len(outs) == 4
        for i, o in enumerate(outs):
            assert o.shape == features[i].shape

    def test_all_levels_fused(self, cla, features):
        """每个输出层级都应融合了其他层级 (输出 != 输入)"""
        outs = cla(features)
        for i in range(4):
            assert not torch.allclose(outs[i], features[i], atol=1e-3)

    def test_gradient_flow(self, cla, features):
        """梯度应能通过 shared_transform 和 fuse_conv 回传"""
        for f in features:
            f.requires_grad_(True)
        outs = cla(features)
        sum(out.sum() for out in outs).backward()
        assert any(p.grad is not None for p in cla.shared_transform.parameters())
        assert any(p.grad is not None for p in cla.fuse_conv.parameters())


class TestLAMFPNWithCrossAttention:
    """G3: LAMFPN + CrossLayerAttention 集成"""

    def test_forward_with_cross_attention(self):
        """启用 cross_layer_attention 时应正常前向"""
        fpn = LAMFPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4,
            apply_lam_levels=[1, 2],
            attention_type='dual',
            use_cross_layer_attention=True,
            use_checkpoint=False,
        )
        inputs = [
            torch.randn(1, 256, 32, 32),
            torch.randn(1, 512, 16, 16),
            torch.randn(1, 1024, 8, 8),
            torch.randn(1, 2048, 4, 4),
        ]
        outs = fpn(inputs)
        assert len(outs) == 4
        assert fpn.cross_attention is not None
