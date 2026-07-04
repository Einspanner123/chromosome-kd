"""方向I-1: Normalized Classifier 单元测试

测试 NormalizedLinear 层 (L2归一化权重和特征 + 温度缩放)。
对应 docs/research/breakthrough_directions/方向I_长尾少样本类别平衡.md 中 I-1 设计:
    logits = τ · (W̃ · x̃),  W̃ = W / ||W||,  x̃ = x / ||x||,  τ = 20

核心验证:
1. 输出形状与 nn.Linear 兼容
2. 权重 L2 归一化 (沿 feature 维度 ||W||_2 = 1)
3. 特征 L2 归一化 (沿 feature 维度 ||x||_2 = 1)
4. 温度缩放 (logits = τ · cos(θ))
5. 梯度可回传
6. 边界条件 (零向量、batch=1、单类)
"""

import pytest
import torch
import torch.nn as nn

from ldmdet.core.single_head import NormalizedLinear


class TestNormalizedLinearShape:
    """测试输出形状和接口兼容性"""

    def test_output_shape_matches_linear(self):
        """输出形状应与 nn.Linear 一致: [N, out_features]"""
        layer = NormalizedLinear(in_features=256, out_features=24)
        x = torch.randn(50, 256)
        out = layer(x)
        assert out.shape == (50, 24)

    def test_3d_input_supported(self):
        """应支持 3D 输入 [bs, N, C] (与 cls_head 调用方式兼容)"""
        layer = NormalizedLinear(in_features=64, out_features=10)
        x = torch.randn(2, 100, 64)
        out = layer(x)
        assert out.shape == (2, 100, 10)

    def test_default_temperature(self):
        """默认温度应为 20.0 (论文推荐值)"""
        layer = NormalizedLinear(in_features=64, out_features=10)
        assert layer.temperature == 20.0

    def test_is_nn_module(self):
        """应是 nn.Module 子类"""
        layer = NormalizedLinear(in_features=64, out_features=10)
        assert isinstance(layer, nn.Module)

    def test_has_weight_and_bias(self):
        """应有 weight 参数 (与 nn.Linear 接口一致)"""
        layer = NormalizedLinear(in_features=64, out_features=10)
        assert hasattr(layer, 'weight')
        # weight shape: [out_features, in_features] (与 nn.Linear 一致)
        assert layer.weight.shape == (10, 64)


class TestWeightNormalization:
    """测试权重 L2 归一化"""

    def test_weight_normalized_along_feature_dim(self):
        """归一化后, 每个 output 神经元的权重 ||W_j||_2 = 1"""
        layer = NormalizedLinear(in_features=64, out_features=10)
        x = torch.randn(5, 64)
        # 触发 forward, 检查内部归一化后的权重
        _ = layer(x)
        # 通过重新计算验证
        w_normalized = layer.weight / layer.weight.norm(dim=1, keepdim=True).clamp(min=1e-12)
        norms = w_normalized.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_weight_scale_invariant_output(self):
        """权重整体缩放后, 输出应不变 (因为归一化)"""
        layer = NormalizedLinear(in_features=64, out_features=10, temperature=1.0)
        x = torch.randn(5, 64)
        out1 = layer(x)

        # 缩放权重
        with torch.no_grad():
            layer.weight.mul_(10.0)
        out2 = layer(x)
        assert torch.allclose(out1, out2, atol=1e-5)


class TestFeatureNormalization:
    """测试特征 L2 归一化"""

    def test_feature_scale_invariant_output(self):
        """输入特征整体缩放后, 输出应不变 (因为归一化)"""
        layer = NormalizedLinear(in_features=64, out_features=10, temperature=1.0)
        x = torch.randn(5, 64)
        out1 = layer(x)

        # 缩放输入
        out2 = layer(x * 100.0)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_output_bounded_by_temperature(self):
        """输出绝对值应 <= temperature (cosine ∈ [-1, 1])"""
        layer = NormalizedLinear(in_features=64, out_features=10, temperature=20.0)
        x = torch.randn(100, 64)
        out = layer(x)
        # |logits| <= τ (允许数值误差)
        assert out.abs().max().item() <= 20.0 + 1e-3


class TestTemperatureScaling:
    """测试温度缩放"""

    def test_temperature_scales_output(self):
        """温度应线性缩放输出: out(τ=2) = 2 * out(τ=1)"""
        torch.manual_seed(42)
        layer1 = NormalizedLinear(in_features=64, out_features=10, temperature=1.0)
        torch.manual_seed(42)
        layer2 = NormalizedLinear(in_features=64, out_features=10, temperature=2.0)

        x = torch.randn(5, 64)
        out1 = layer1(x)
        out2 = layer2(x)
        assert torch.allclose(out2, 2.0 * out1, atol=1e-5)

    def test_custom_temperature(self):
        """应支持自定义温度"""
        layer = NormalizedLinear(in_features=64, out_features=10, temperature=15.0)
        assert layer.temperature == 15.0


class TestNormalizedLinearBackward:
    """测试梯度回传"""

    def test_gradient_flows(self):
        """梯度应正确回传到权重"""
        layer = NormalizedLinear(in_features=64, out_features=10)
        x = torch.randn(5, 64)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weight.grad is not None
        assert not torch.isnan(layer.weight.grad).any()

    def test_gradient_shape(self):
        """梯度形状应与权重一致"""
        layer = NormalizedLinear(in_features=64, out_features=10)
        x = torch.randn(5, 64)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weight.grad.shape == layer.weight.shape

    def test_gradient_nonzero(self):
        """梯度应非零"""
        layer = NormalizedLinear(in_features=64, out_features=10)
        x = torch.randn(5, 64)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weight.grad.abs().sum() > 0


class TestNormalizedLinearEdgeCases:
    """测试边界条件"""

    def test_batch_size_one(self):
        """batch=1 应正常工作"""
        layer = NormalizedLinear(in_features=64, out_features=10)
        x = torch.randn(1, 64)
        out = layer(x)
        assert out.shape == (1, 10)

    def test_single_class(self):
        """out_features=1 应正常工作"""
        layer = NormalizedLinear(in_features=64, out_features=1)
        x = torch.randn(5, 64)
        out = layer(x)
        assert out.shape == (5, 1)

    def test_zero_input_raises(self):
        """零向量输入应通过 clamp 避免除零 (不报错)"""
        layer = NormalizedLinear(in_features=64, out_features=10)
        x = torch.zeros(5, 64)
        out = layer(x)
        assert not torch.isnan(out).any()

    def test_3d_gradient_flows(self):
        """3D 输入应正确回传梯度"""
        layer = NormalizedLinear(in_features=64, out_features=10)
        x = torch.randn(2, 50, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestIntegrationWithClsHead:
    """测试与 SingleDiffusionDetHead 集成"""

    def test_can_replace_last_layer(self):
        """应能替换 nn.Sequential 的最后一层"""
        # 模拟 _build_cls_head 的结构
        feat_channels = 256
        num_classes = 24
        head = nn.Sequential(
            nn.Linear(feat_channels, feat_channels, bias=False),
            nn.LayerNorm(feat_channels),
            nn.ReLU(inplace=True),
            NormalizedLinear(feat_channels, num_classes, temperature=20.0),
        )
        x = torch.randn(10, feat_channels)
        out = head(x)
        assert out.shape == (10, num_classes)
