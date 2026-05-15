import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from mods.single_head import SingleDiffusionDetHead


class TestSingleDiffusionDetHead(unittest.TestCase):

    def setUp(self):
        """测试前的初始化工作"""
        # 基础配置
        self.batch_size = 2
        self.num_boxes = 4
        self.feat_channels = 256
        self.num_classes = 80
        self.pooler_resolution = 7

        # 初始化模型
        self.model = SingleDiffusionDetHead(
            num_classes=self.num_classes,
            feat_channels=self.feat_channels,
            dim_feedforward=1024,
            num_cls_convs=1,
            num_reg_convs=1,
            num_heads=4,
            pooler_resolution=self.pooler_resolution,
            use_focal_loss=True,
        )
        self.model.eval()  # 设置为评估模式

    def _create_dummy_inputs(self):
        """创建虚拟输入数据"""
        # Features: 模拟 FPN 层级 (P2-P5)
        features = [
            torch.randn(self.batch_size, self.feat_channels, 64, 64),
            torch.randn(self.batch_size, self.feat_channels, 32, 32),
            torch.randn(self.batch_size, self.feat_channels, 16, 16),
            torch.randn(self.batch_size, self.feat_channels, 8, 8),
        ]

        # BBoxes: (Batch, Num_Boxes, 4)
        bboxes = torch.randn(self.batch_size, self.num_boxes, 4)
        # 确保 x2 > x1, y2 > y1
        bboxes[..., 2:] = bboxes[..., :2] + torch.abs(bboxes[..., 2:]) + 1.0

        # Proposals: (Batch, Num_Boxes, C)
        proposals = torch.randn(self.batch_size, self.num_boxes,
                                self.feat_channels)

        # Time Embedding
        time_emb = torch.randn(self.batch_size, self.feat_channels * 4)

        return features, bboxes, proposals, time_emb

    def _get_mock_pooler(self):
        """创建 Mock ROI Pooler"""
        feat_channels = self.feat_channels

        class MockPooler(nn.Module):

            def forward(self, feats, rois):
                # rois shape: (N, 5)
                num_rois = rois.shape[0]
                return torch.randn(num_rois, feat_channels, 7, 7)

        return MockPooler()

    def test_inference_shape(self):
        """测试推理输出形状是否正确"""
        features, bboxes, proposals, time_emb = self._create_dummy_inputs()
        pooler = self._get_mock_pooler()

        with torch.no_grad():
            class_logits, pred_bboxes, obj_features = self.model(
                features, bboxes, proposals, pooler, time_emb)

        # 验证 Class Logits 形状
        expected_cls_dim = self.num_classes  # use_focal_loss=True
        self.assertEqual(
            class_logits.shape,
            (self.batch_size, self.num_boxes, expected_cls_dim),
            f'Class logits shape mismatch: got {class_logits.shape}',
        )

        # 验证 BBox 形状
        self.assertEqual(
            pred_bboxes.shape,
            (self.batch_size, self.num_boxes, 4),
            f'Pred bboxes shape mismatch: got {pred_bboxes.shape}',
        )

        # 验证 Object Features 形状
        self.assertEqual(
            obj_features.shape,
            (1, self.batch_size * self.num_boxes, self.feat_channels),
            f'Object features shape mismatch: got {obj_features.shape}',
        )

    def test_forward_with_different_batch_size(self):
        """测试不同 Batch Size"""
        self.batch_size = 4
        self.setUp()  # 重新初始化
        self.test_inference_shape()

    def test_forward_without_proposals(self):
        """测试 proposals=None 的情况"""
        features, bboxes, _, time_emb = self._create_dummy_inputs()
        pooler = self._get_mock_pooler()

        # proposals 设为 None
        with torch.no_grad():
            class_logits, pred_bboxes, _ = self.model(features, bboxes, None,
                                                      pooler, time_emb)

        self.assertEqual(class_logits.shape[0], self.batch_size)


if __name__ == '__main__':
    unittest.main()
