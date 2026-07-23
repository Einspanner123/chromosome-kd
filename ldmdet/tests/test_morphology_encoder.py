"""测试 MorphologyAwareRoIEncoder (M1 形态感知 RoI 编码器)

核心测试:
  1. 零初始化恒等性: 初始 morph_emb≡0, 输出≡输入 (保证加载 A4 预训练时行为不变)
  2. 方向解耦: h_conv 和 v_conv 提取不同方向的特征
  3. 形状正确性: 输出形状与输入一致
  4. 梯度流: 残差路径梯度可流回
  5. 不同参数配置: pooler_resolution / reduction / channels
"""

import torch
import pytest

from ldmdet.core.morphology_encoder import MorphologyAwareRoIEncoder


class TestMorphologyAwareRoIEncoder:
    """MorphologyAwareRoIEncoder 核心功能测试"""

    @pytest.fixture
    def encoder(self):
        """标准配置: channels=256, reduction=4, pooler_resolution=7"""
        return MorphologyAwareRoIEncoder(
            channels=256, reduction=4, pooler_resolution=7
        )

    @pytest.fixture
    def roi_features(self):
        """模拟 RoIAlign 输出: [N, C, 7, 7]"""
        torch.manual_seed(42)
        return torch.randn(50, 256, 7, 7)

    # ================================================================
    # 零初始化恒等性 (最关键测试)
    # ================================================================

    def test_zero_init_identity(self, encoder, roi_features):
        """零初始化: 初始输出必须严格等于输入 (morph_emb≡0)

        这是 M1 的核心保证 — 加载 A4 预训练权重时模型行为不变,
        D1 消融验证的空间通路不被破坏。
        """
        with torch.no_grad():
            out = encoder(roi_features)
        assert torch.allclose(out, roi_features, atol=1e-6), (
            "零初始化失败: 初始输出应恒等于输入 (morph_emb 应为 0)"
        )

    def test_fuse_zero_init(self, encoder):
        """fuse 层权重和偏置应全为零"""
        assert torch.all(encoder.fuse.weight == 0), "fuse.weight 应零初始化"
        assert torch.all(encoder.fuse.bias == 0), "fuse.bias 应零初始化"

    def test_zero_init_after_multiple_calls(self, encoder, roi_features):
        """多次前向传播后零初始化仍保持 (无状态污染)"""
        with torch.no_grad():
            out1 = encoder(roi_features)
            out2 = encoder(roi_features)
        assert torch.allclose(out1, roi_features, atol=1e-6)
        assert torch.allclose(out2, roi_features, atol=1e-6)

    # ================================================================
    # 形状正确性
    # ================================================================

    def test_output_shape(self, encoder, roi_features):
        """输出形状必须与输入一致"""
        out = encoder(roi_features)
        assert out.shape == roi_features.shape

    def test_different_batch_sizes(self, encoder):
        """不同 N (RoI 数量) 均可处理"""
        for n in [1, 10, 100, 500]:
            x = torch.randn(n, 256, 7, 7)
            out = encoder(x)
            assert out.shape == (n, 256, 7, 7)

    def test_different_channels(self):
        """不同通道数"""
        for ch in [64, 128, 256, 512]:
            enc = MorphologyAwareRoIEncoder(channels=ch, reduction=4)
            x = torch.randn(10, ch, 7, 7)
            out = enc(x)
            assert out.shape == (10, ch, 7, 7)

    # ================================================================
    # 方向解耦
    # ================================================================

    def test_direction_decoupling(self, encoder):
        """h_conv 和 v_conv 提取不同方向的特征

        h_conv (7,1): 沿高度卷积, 输出 [N, C, 1, W]。水平翻转只是列重排,
          值不变 (torch.flip(h_orig, dim=3) ≈ h_flip)。
        v_conv (1,7): 沿宽度卷积, 输出 [N, C, H, 1]。水平翻转改变加权求和,
          值变 (v_orig ≠ v_flip)。
        """
        torch.manual_seed(123)
        x = torch.randn(10, 256, 7, 7)
        x_flip_h = torch.flip(x, dims=[3])  # 水平翻转 (沿宽度翻转)

        with torch.no_grad():
            h_orig = encoder.h_conv(x)      # [N, C//r, 1, 7]
            h_flip = encoder.h_conv(x_flip_h)
            v_orig = encoder.v_conv(x)      # [N, C//r, 7, 1]
            v_flip = encoder.v_conv(x_flip_h)

        # h_conv (kernel 沿高度): 水平翻转只重排列, 值不变
        assert torch.allclose(
            h_orig, torch.flip(h_flip, dims=[3]), atol=1e-5
        ), "h_conv 水平翻转后应只是列重排 (kernel 宽度=1, 各列独立)"
        # v_conv (kernel 沿宽度): 水平翻转改变加权求和, 值变
        assert not torch.allclose(v_orig, v_flip, atol=1e-5), (
            "v_conv 应受水平翻转影响 (kernel 沿宽度方向, 求和顺序改变)"
        )

    def test_vertical_flip_decoupling(self, encoder):
        """垂直翻转: h_conv 值变 (kernel 沿高度), v_conv 只行重排 (kernel 高度=1)"""
        torch.manual_seed(456)
        x = torch.randn(10, 256, 7, 7)
        x_flip_v = torch.flip(x, dims=[2])  # 垂直翻转 (沿高度翻转)

        with torch.no_grad():
            h_orig = encoder.h_conv(x)
            h_flip = encoder.h_conv(x_flip_v)
            v_orig = encoder.v_conv(x)
            v_flip = encoder.v_conv(x_flip_v)

        # v_conv (kernel 沿宽度, 高度=1): 垂直翻转只重排行, 值不变
        assert torch.allclose(
            v_orig, torch.flip(v_flip, dims=[2]), atol=1e-5
        ), "v_conv 垂直翻转后应只是行重排 (kernel 高度=1, 各行独立)"
        # h_conv (kernel 沿高度): 垂直翻转改变加权求和, 值变
        assert not torch.allclose(h_orig, h_flip, atol=1e-5), (
            "h_conv 应受垂直翻转影响 (kernel 沿高度方向, 求和顺序改变)"
        )

    # ================================================================
    # 梯度流
    # ================================================================

    def test_gradient_flow(self, encoder, roi_features):
        """梯度通过残差路径流回输入"""
        x = roi_features.clone().requires_grad_(True)
        out = encoder(x)
        out.sum().backward()
        assert x.grad is not None, "梯度应流回输入"
        # 残差路径: grad 至少为 1 (来自 out = x + morph_emb 的 x 项)
        assert torch.allclose(x.grad, torch.ones_like(x.grad), atol=1e-6), (
            "零初始化时梯度应全为 1 (残差直通)"
        )

    def test_fuse_gradient_flow(self, encoder, roi_features):
        """梯度流到 fuse 层权重 (训练将逐步激活 morph_emb)"""
        x = roi_features.clone().requires_grad_(True)
        out = encoder(x)
        out.sum().backward()
        # fuse.weight 初始为 0, 但梯度应非零 (来自上游 loss 对输出 的梯度)
        assert encoder.fuse.weight.grad is not None
        # 零初始化时 fuse 输出为 0, 但 fuse.weight 的梯度 = upstream_grad @ input.T
        # input (cat 后) 非零, upstream_grad 非零 (全 1), 所以 fuse.weight.grad 非零
        assert encoder.fuse.weight.grad.abs().sum() > 0, (
            "fuse.weight 应有非零梯度 (训练将逐步激活形态增强)"
        )

    # ================================================================
    # 不同参数配置
    # ================================================================

    def test_different_pooler_resolution(self):
        """不同 pooler_resolution"""
        for res in [7, 14]:
            enc = MorphologyAwareRoIEncoder(
                channels=64, reduction=4, pooler_resolution=res
            )
            x = torch.randn(10, 64, res, res)
            out = enc(x)
            assert out.shape == (10, 64, res, res)

    def test_different_reduction(self):
        """不同 reduction 比"""
        for r in [2, 4, 8]:
            enc = MorphologyAwareRoIEncoder(
                channels=256, reduction=r, pooler_resolution=7
            )
            x = torch.randn(10, 256, 7, 7)
            out = enc(x)
            assert out.shape == (10, 256, 7, 7)
            # 零初始化恒等性
            assert torch.allclose(out, x, atol=1e-6)

    def test_zero_init_with_all_configs(self):
        """所有参数配置下零初始化恒等性都成立"""
        configs = [
            dict(channels=64, reduction=4, pooler_resolution=7),
            dict(channels=128, reduction=2, pooler_resolution=7),
            dict(channels=256, reduction=8, pooler_resolution=7),
            dict(channels=512, reduction=4, pooler_resolution=14),
        ]
        for cfg in configs:
            enc = MorphologyAwareRoIEncoder(**cfg)
            x = torch.randn(10, cfg['channels'], cfg['pooler_resolution'],
                            cfg['pooler_resolution'])
            with torch.no_grad():
                out = enc(x)
            assert torch.allclose(out, x, atol=1e-6), (
                f"零初始化失败: config={cfg}"
            )

    # ================================================================
    # 训练后行为 (模拟微调)
    # ================================================================

    def test_nonzero_after_training(self, encoder, roi_features):
        """微调几步后 fuse 权重非零, morph_emb ≠ 0, 输出 ≠ 输入"""
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
        x = roi_features.clone()

        # 模拟微调: 最小化一个非平凡目标 (让输出接近特定 target)
        target = torch.randn_like(x)
        for _ in range(10):
            optimizer.zero_grad()
            out = encoder(x)
            loss = ((out - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        # 训练后 fuse 权重应非零
        assert encoder.fuse.weight.abs().sum() > 0, (
            "微调后 fuse 权重应非零"
        )
        # 输出应不再恒等于输入
        with torch.no_grad():
            out = encoder(x)
        assert not torch.allclose(out, x, atol=1e-6), (
            "微调后输出应不同于输入 (morph_emb 被激活)"
        )

    def test_parameter_count(self, encoder):
        """参数量在合理范围 (~265K for channels=256, reduction=4)"""
        total = sum(p.numel() for p in encoder.parameters())
        # h_conv: 256*64*(7*1) + 64 = 114944
        # v_conv: 256*64*(1*7) + 64 = 114944
        # norm: 128*2 = 256
        # fuse: 128*256*(1*1) + 256 = 33024
        # total ≈ 263168
        assert 200000 < total < 300000, f"参数量 {total} 不在预期范围 200K-300K"
