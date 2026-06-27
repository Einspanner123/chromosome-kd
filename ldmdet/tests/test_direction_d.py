"""方向 D 测试: 扩散过程改进 (Diffusion Process Improvements)

D1: BoxRefineNet — 扩散采样后的残差框精化网络
    在 reg_head 输出后加可选的细粒度回归头, 优化高 IoU 精度
"""

import pytest
import torch
import torch.nn as nn

from ldmdet.core.box_refine import BoxRefineNet


class TestBoxRefineNet:
    """D1: 框细化网络单元测试"""

    @pytest.fixture
    def refine(self):
        return BoxRefineNet(feat_channels=256)

    def test_output_shape(self, refine):
        """输出 shape 应为 (N, 4) — 4 维残差偏移"""
        x = torch.randn(10, 256)
        out = refine(x)
        assert out.shape == (10, 4)

    def test_zero_init_last_layer(self, refine):
        """D1: 最后一层应零初始化 (初始时残差为 0, 不破坏 baseline)"""
        last_layer = refine.mlp[-1]
        assert torch.allclose(last_layer.weight, torch.zeros_like(last_layer.weight))
        assert torch.allclose(last_layer.bias, torch.zeros_like(last_layer.bias))

    def test_residual_zero_at_init(self, refine):
        """D1: 初始化时输出应为 0 (残差结构, 不改变 baseline 预测)"""
        x = torch.randn(5, 256)
        out = refine(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_gradient_flow(self, refine):
        """梯度应能流经 MLP"""
        x = torch.randn(4, 256)
        out = refine(x)
        out.sum().backward()
        for layer in refine.mlp:
            if hasattr(layer, 'weight') and layer.weight is not None:
                assert layer.weight.grad is not None

    def test_structure(self, refine):
        """D1: 应有两层 MLP (Linear → ReLU → Linear)"""
        layers = list(refine.mlp)
        assert isinstance(layers[0], nn.Linear)
        assert layers[0].in_features == 256
        assert layers[0].out_features == 256
        assert isinstance(layers[1], nn.ReLU)
        assert isinstance(layers[2], nn.Linear)
        assert layers[2].in_features == 256
        assert layers[2].out_features == 4
