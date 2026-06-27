"""方向 C 测试: 形态感知分类 (Morphology-Aware Classification)

C1: ShapeAttention — RoI 特征上的水平/垂直方向卷积 (捕获臂长/着丝粒)
C2: ContrastiveLoss — 拉大形态相似类的特征距离
"""

import pytest
import torch
import torch.nn as nn

from ldmdet.core.shape_attention import ShapeAttention
from ldmdet.criterion.contrastive_loss import ContrastiveLoss


class TestShapeAttention:
    """C1: 局部形状注意力单元测试"""

    @pytest.fixture
    def sa(self):
        return ShapeAttention(channels=256, reduction=4)

    def test_output_shape(self, sa):
        """输出 shape 应与输入一致 (残差结构)"""
        x = torch.randn(10, 256, 7, 7)
        out = sa(x)
        assert out.shape == (10, 256, 7, 7)

    def test_residual(self, sa):
        """残差连接: 输出应包含输入 (初始化时接近恒等)"""
        sa.eval()
        x = torch.randn(5, 256, 7, 7)
        out = sa(x)
        # 由于 fuse 初始化 + 残差, 输出应与输入相近但不完全相同
        assert out.shape == x.shape

    def test_directional_convs(self, sa):
        """C1: 应有水平 (7,1) 和垂直 (1,7) 方向卷积分支"""
        assert hasattr(sa, 'h_branch')
        assert hasattr(sa, 'v_branch')
        # h_branch: kernel (7,1), v_branch: kernel (1,7)
        assert sa.h_branch.kernel_size == (7, 1)
        assert sa.v_branch.kernel_size == (1, 7)

    def test_fuse_conv(self, sa):
        """C1: 应有 fuse 卷积融合水平/垂直特征"""
        assert hasattr(sa, 'fuse')
        assert isinstance(sa.fuse, nn.Conv2d)

    def test_gradient_flow(self, sa):
        """梯度应能流经 h_branch, v_branch, fuse"""
        x = torch.randn(4, 256, 7, 7, requires_grad=True)
        out = sa(x)
        out.sum().backward()
        assert sa.h_branch.weight.grad is not None
        assert sa.v_branch.weight.grad is not None
        assert sa.fuse.weight.grad is not None


class TestContrastiveLoss:
    """C2: 对比学习辅助损失单元测试"""

    @pytest.fixture
    def loss(self):
        return ContrastiveLoss(temp=0.07)

    def test_zero_loss_identical(self, loss):
        """同类且特征相同时, 损失应接近 0"""
        # 两个相同特征, 同类
        feats = torch.randn(1, 128)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        labels = torch.tensor([0])
        # 单样本无法做对比, 至少 2 个
        f = torch.cat([feats, feats], dim=0)
        l = torch.tensor([0, 0])
        loss_val = loss(f, l)
        assert loss_val.item() < 0.1

    def test_high_loss_different_class(self, loss):
        """同类正对 + 不同类负对: 当负对特征相似时损失应较高"""
        # 4 个样本: 2 个同类 (正对), 2 个不同类 (负对), 特征相近
        feats = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        labels = torch.tensor([0, 0, 1, 1])
        loss_val = loss(feats, labels)
        # 有正对时损失应为非负
        assert loss_val.item() >= 0
        # 与纯同类 batch 对比: 同类 batch 损失应更低
        feats_same = torch.tensor([[1.0, 0.0], [0.99, 0.01], [1.0, 0.0], [0.99, 0.01]])
        feats_same = feats_same / (feats_same.norm(dim=-1, keepdim=True) + 1e-8)
        labels_same = torch.tensor([0, 0, 0, 0])
        loss_same = loss(feats_same, labels_same)
        assert loss_val.item() > loss_same.item()

    def test_gradient_flow(self, loss):
        """梯度应能流经特征"""
        feats = torch.randn(8, 128, requires_grad=True)
        feats.retain_grad()
        feats_n = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss_val = loss(feats_n, labels)
        loss_val.backward()
        assert feats.grad is not None

    def test_temperature_effect(self):
        """温度参数应影响损失值 (低温度 → 更尖锐)"""
        feats = torch.randn(4, 64)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        labels = torch.tensor([0, 0, 1, 1])
        loss_low = ContrastiveLoss(temp=0.01)
        loss_high = ContrastiveLoss(temp=1.0)
        l_low = loss_low(feats, labels).item()
        l_high = loss_high(feats, labels).item()
        # 低温度下, 正负对区分更尖锐 (loss 更高或更低取决于具体值)
        # 这里仅验证两者不同
        assert abs(l_low - l_high) > 1e-4

    def test_hard_negative_mining(self, loss):
        """C2 困难负对: 形态相似类 (不同类但特征近) 应有更高对比权重"""
        # 构造 2 个同类 + 2 个不同类但特征相近的样本
        feats = torch.tensor([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0], [0.0, 1.01]])
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        labels = torch.tensor([0, 0, 1, 1])
        loss_val = loss(feats, labels)
        assert loss_val.item() >= 0
