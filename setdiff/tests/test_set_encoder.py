"""测试 setdiff.core.set_encoder — SetEncoder + MLP"""

import torch
import pytest
from setdiff.core.set_encoder import MLP, SetEncoder


class TestMLP:
    def test_output_shape(self):
        mlp = MLP(input_dim=4, hidden_dim=64, output_dim=32, num_layers=3)
        x = torch.randn(8, 4)
        out = mlp(x)
        assert out.shape == (8, 32)

    def test_two_layers(self):
        """num_layers=2: 一个隐藏层 + 一个输出层"""
        mlp = MLP(input_dim=4, hidden_dim=64, output_dim=4, num_layers=2)
        x = torch.randn(2, 4)
        out = mlp(x)
        assert out.shape == (2, 4)

    def test_gradient_flow(self):
        mlp = MLP(input_dim=4, hidden_dim=64, output_dim=4, num_layers=3)
        x = torch.randn(4, 4, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None

    def test_deterministic(self):
        mlp = MLP(input_dim=4, hidden_dim=32, output_dim=4, num_layers=2)
        mlp.eval()
        x = torch.randn(2, 4)
        with torch.no_grad():
            out1 = mlp(x)
            out2 = mlp(x)
        assert torch.allclose(out1, out2)


class TestSetEncoder:
    @pytest.fixture
    def encoder(self):
        return SetEncoder(
            num_queries=10,
            feat_channels=64,
            num_heads=4,
            num_layers=2,
            dim_feedforward=128,
            num_classes=20,
        )

    def _make_inputs(self, B=2, N=10, HW=64, C=64):
        x_t = torch.randn(B, N, 4)
        t_emb = torch.randn(B, C)
        image_features = torch.randn(B, HW, C)
        return x_t, t_emb, image_features

    def test_output_shapes(self, encoder):
        x_t, t_emb, image_features = self._make_inputs()
        cls_logits, pred_boxes = encoder(x_t, t_emb, image_features)
        assert cls_logits.shape == (2, 10, 20)
        assert pred_boxes.shape == (2, 10, 4)

    def test_different_batch_size(self, encoder):
        x_t, t_emb, image_features = self._make_inputs(B=4)
        cls_logits, pred_boxes = encoder(x_t, t_emb, image_features)
        assert cls_logits.shape[0] == 4
        assert pred_boxes.shape[0] == 4

    def test_differentiability(self, encoder):
        """反向传播应能到达输入 x_t 和 image_features"""
        x_t = torch.randn(2, 10, 4, requires_grad=True)
        t_emb = torch.randn(2, 64)
        image_features = torch.randn(
            2, 64, 64, requires_grad=True
        )
        cls_logits, pred_boxes = encoder(x_t, t_emb, image_features)
        loss = cls_logits.sum() + pred_boxes.sum()
        loss.backward()
        assert x_t.grad is not None
        assert image_features.grad is not None

    def test_gradient_reaches_parameters(self, encoder):
        x_t, t_emb, image_features = self._make_inputs()
        cls_logits, pred_boxes = encoder(x_t, t_emb, image_features)
        loss = cls_logits.sum() + pred_boxes.sum()
        loss.backward()
        for name, p in encoder.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f'{name} has no gradient'

    def test_deterministic_eval(self, encoder):
        """eval 模式下相同输入应得到相同输出"""
        encoder.eval()
        x_t, t_emb, image_features = self._make_inputs()
        with torch.no_grad():
            out1 = encoder(x_t, t_emb, image_features)
            out2 = encoder(x_t, t_emb, image_features)
        assert torch.allclose(out1[0], out2[0], atol=1e-6)
        assert torch.allclose(out1[1], out2[1], atol=1e-6)

    def test_different_num_queries(self):
        encoder = SetEncoder(
            num_queries=20,
            feat_channels=32,
            num_heads=4,
            num_layers=1,
            dim_feedforward=64,
            num_classes=5,
        )
        x_t = torch.randn(2, 20, 4)
        t_emb = torch.randn(2, 32)
        image_features = torch.randn(2, 16, 32)
        cls_logits, pred_boxes = encoder(x_t, t_emb, image_features)
        assert cls_logits.shape == (2, 20, 5)
        assert pred_boxes.shape == (2, 20, 4)

    def test_pred_boxes_finite(self, encoder):
        """输出应有限"""
        x_t, t_emb, image_features = self._make_inputs()
        cls_logits, pred_boxes = encoder(x_t, t_emb, image_features)
        assert torch.isfinite(cls_logits).all()
        assert torch.isfinite(pred_boxes).all()
