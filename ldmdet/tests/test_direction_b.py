"""方向 B 测试: 解耦分类定位 (Decoupled Classification/Localization)

B1: 双分支头 — cls 和 reg 各有独立的 self_attn + FFN
B2: 每分支独立 LayerNorm
B3: 阶段化 cls loss 权重 (criterion 层面, 集成测试验证)
"""

import math

import pytest
import torch
import torch.nn as nn

from ldmdet.core.decoupled_head import DecoupledSingleHead


class TestDecoupledSingleHead:
    """B1+B2: 解耦双分支头单元测试"""

    @pytest.fixture
    def head(self):
        return DecoupledSingleHead(
            num_classes=24,
            feat_channels=256,
            dim_feedforward=2048,
            num_cls_convs=1,
            num_reg_convs=3,
            num_heads=8,
            pooler_resolution=7,
            time_conditioning='scale_shift',
        )

    def test_branches_are_separate(self, head):
        """B1: cls 和 reg 分支应当是独立的参数对象"""
        # cls 分支应有自己的 self_attn
        assert hasattr(head, 'cls_self_attn')
        assert hasattr(head, 'reg_self_attn')
        # cls 和 reg 的 self_attn 不应是同一个对象
        assert head.cls_self_attn is not head.reg_self_attn
        # cls 和 reg 不应共享 FFN
        assert head.cls_linear1 is not head.reg_linear1
        assert head.cls_linear2 is not head.reg_linear2

    def test_no_shared_self_attn(self, head):
        """B1: 不应存在共享的 self_attn (父类的 self_attn 应被覆盖)"""
        # 父类的 self_attn 仍存在但不应在 forward 中使用
        # 关键: cls/reg 分支各有自己的 attention
        assert isinstance(head.cls_self_attn, nn.MultiheadAttention)
        assert isinstance(head.reg_self_attn, nn.MultiheadAttention)

    def test_branch_norms_independent(self, head):
        """B2: 每分支应有独立的 LayerNorm"""
        # cls 分支 norm
        assert hasattr(head, 'cls_norm1')
        assert hasattr(head, 'cls_norm2')
        assert hasattr(head, 'cls_norm3')
        # reg 分支 norm
        assert hasattr(head, 'reg_norm1')
        assert hasattr(head, 'reg_norm2')
        assert hasattr(head, 'reg_norm3')
        # cls 和 reg 的 norm 独立
        assert head.cls_norm1 is not head.reg_norm1

    def test_forward_output_shapes(self, head):
        """前向: 输出 shapes 应与基类一致"""
        bs, num_boxes = 2, 10
        feat_channels = head.feat_channels
        features = [torch.randn(1, feat_channels, 32, 32)]
        bboxes = torch.rand(bs, num_boxes, 4) * 100
        bboxes[..., 2:] += bboxes[..., :2]  # 确保 xyxy
        proposals = torch.randn(bs, num_boxes, feat_channels)
        time_emb = torch.randn(bs, feat_channels * 4)

        # pooler mock
        class MockPooler:
            def __call__(self, feats, rois):
                n = rois.shape[0]
                return torch.randn(n, feat_channels, 7, 7)

        cls_logits, pred_bboxes, fc_feat = head(
            features, bboxes, proposals, MockPooler(), time_emb
        )
        assert cls_logits.shape == (bs, num_boxes, 24)
        assert pred_bboxes.shape == (bs, num_boxes, 4)
        assert fc_feat.shape == (1, bs * num_boxes, feat_channels)

    def test_forward_backward(self, head):
        """前向+反向: 梯度应能流经 cls 和 reg 两个分支"""
        bs, num_boxes = 1, 5
        feat_channels = head.feat_channels
        features = [torch.randn(1, feat_channels, 16, 16)]
        bboxes = torch.rand(bs, num_boxes, 4) * 50
        bboxes[..., 2:] += bboxes[..., :2]
        proposals = torch.randn(bs, num_boxes, feat_channels)
        time_emb = torch.randn(bs, feat_channels * 4)

        class MockPooler:
            def __call__(self, feats, rois):
                n = rois.shape[0]
                return torch.randn(n, feat_channels, 7, 7)

        cls_logits, pred_bboxes, _ = head(
            features, bboxes, proposals, MockPooler(), time_emb
        )
        loss = cls_logits.sum() + pred_bboxes.sum()
        loss.backward()

        # 验证 cls 分支有梯度
        assert head.cls_self_attn.in_proj_weight.grad is not None
        assert head.cls_linear1.weight.grad is not None
        # 验证 reg 分支有梯度
        assert head.reg_self_attn.in_proj_weight.grad is not None
        assert head.reg_linear1.weight.grad is not None

    def test_param_count_larger_than_single(self):
        """B1: 解耦头参数量应大于单分支头 (因为分支翻倍)"""
        from ldmdet.core.single_head import SingleDiffusionDetHead

        kwargs = dict(
            num_classes=24, feat_channels=256, dim_feedforward=2048,
            num_cls_convs=1, num_reg_convs=3, num_heads=8,
            pooler_resolution=7, time_conditioning='scale_shift',
        )
        single = SingleDiffusionDetHead(**kwargs)
        decoupled = DecoupledSingleHead(**kwargs)
        n_single = sum(p.numel() for p in single.parameters())
        n_decoupled = sum(p.numel() for p in decoupled.parameters())
        assert n_decoupled > n_single
