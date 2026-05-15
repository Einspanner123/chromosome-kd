import sys
import unittest
from pathlib import Path

import torch

# 将项目根目录添加到 python 路径
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from mods.loss import (DiffusionDetCriterion, DiffusionDetMatcher, FocalLoss,
                       GIoULoss, L1Loss)
from mods.roi_extractor import SingleRoIExtractor
from mods.single_head import SingleDiffusionDetHead

from projects.LDMDet.mods.diffusiondet_head import DiffusionDetHead


class TestDiffusionDetHead(unittest.TestCase):

    def setUp(self):
        """初始化测试环境"""
        self.num_classes = 80
        self.feat_channels = 256
        self.num_proposals = 100
        self.num_heads = 3
        self.batch_size = 2
        self.img_shape = (800, 1216)

        # 1. 初始化 SingleHead
        self.single_head = SingleDiffusionDetHead(
            num_classes=self.num_classes,
            feat_channels=self.feat_channels,
            num_heads=4,
            use_focal_loss=True,
        )

        # 2. 初始化 RoIExtractor
        self.roi_extractor = SingleRoIExtractor(
            roi_layer=dict(
                type='RoIAlign', output_size=7, sampling_ratio=2,
                aligned=True),
            out_channels=self.feat_channels,
            featmap_strides=[4, 8, 16, 32],
        )

        # 3. 初始化 DiffusionDetHead
        self.model = DiffusionDetHead(
            num_classes=self.num_classes,
            feat_channels=self.feat_channels,
            num_proposals=self.num_proposals,
            num_heads=self.num_heads,
            single_head=self.single_head,
            roi_extractor=self.roi_extractor,
            sampling_timesteps=1,  # 为了测试速度，设置采样步数为 1
            use_nms=True,
            nms_thr=0.5,
            score_thr=0.05,
            min_keep=10,
        )
        self.model.eval()

    def _create_dummy_features(self):
        """创建虚拟 FPN 特征"""
        return (
            torch.randn(self.batch_size, self.feat_channels, 200, 304),
            torch.randn(self.batch_size, self.feat_channels, 100, 152),
            torch.randn(self.batch_size, self.feat_channels, 50, 76),
            torch.randn(self.batch_size, self.feat_channels, 25, 38),
        )

    def test_forward_shape(self):
        """测试前向传播的输出形状"""
        features = self._create_dummy_features()
        bboxes = torch.randn(self.batch_size, self.num_proposals, 4)
        # 简单处理成 xyxy 格式且在图像范围内
        bboxes = bboxes.sigmoid() * 800
        t = torch.randint(0, 1000, (self.batch_size, ))

        with torch.no_grad():
            all_cls_logits, all_pred_bboxes = self.model(features, bboxes, t)

        # 验证输出形状
        # [num_heads, bs, num_proposals, num_classes]
        self.assertEqual(
            all_cls_logits.shape,
            (self.num_heads, self.batch_size, self.num_proposals,
             self.num_classes),
        )
        # [num_heads, bs, num_proposals, 4]
        self.assertEqual(
            all_pred_bboxes.shape,
            (self.num_heads, self.batch_size, self.num_proposals, 4),
        )

    def test_predict(self):
        """测试推理接口"""
        features = self._create_dummy_features()
        img_metas = [
            dict(img_shape=self.img_shape, scale_factor=[1.0, 1.0, 1.0, 1.0]),
            dict(img_shape=self.img_shape, scale_factor=[1.0, 1.0, 1.0, 1.0]),
        ]

        with torch.no_grad():
            results = self.model.predict(features, img_metas, rescale=True)

        # 验证结果结构
        self.assertEqual(len(results), self.batch_size)
        for res in results:
            self.assertIn('bboxes', res)
            self.assertIn('scores', res)
            self.assertIn('labels', res)

            # 验证张量形状
            num_dets = res['bboxes'].shape[0]
            self.assertEqual(res['bboxes'].shape, (num_dets, 4))
            self.assertEqual(res['scores'].shape, (num_dets, ))
            self.assertEqual(res['labels'].shape, (num_dets, ))

    def test_q_sample(self):
        """测试扩散采样函数"""
        x_start = torch.randn(2, 100, 4)
        t = torch.tensor([10, 500])

        x_t = self.model.q_sample(x_start, t)

        self.assertEqual(x_t.shape, x_start.shape)
        self.assertFalse(torch.allclose(x_t, x_start))

    def test_box_renewal_logic(self):
        """测试 Box Renewal 逻辑"""
        # 构造一些极低分数的 cls_logits 来触发 renewal
        features = self._create_dummy_features()
        img_metas = [dict(img_shape=self.img_shape)] * self.batch_size

        # 初始噪声
        x_raw = torch.randn(self.batch_size, self.num_proposals, 4)
        # 模拟全负样本的 logits
        cls_logits = torch.full(
            (self.batch_size, self.num_proposals, self.num_classes), -10.0)
        # 模拟全 0 的框
        pred_bboxes = torch.zeros(self.batch_size, self.num_proposals, 4)

        # 运行一步 ddim_step
        next_bboxes, next_x_raw = self.model._ddim_step(
            t_curr=999,
            t_next=900,
            x_raw=x_raw,
            cls_logits=cls_logits,
            pred_bboxes=pred_bboxes,
            img_metas=img_metas,
        )

        # 验证 renewal 后的形状
        self.assertEqual(next_x_raw.shape, x_raw.shape)
        # 由于 scores 很低，应该触发了 renewal，x_raw 会发生变化
        self.assertFalse(torch.allclose(next_x_raw, x_raw))

    def test_loss(self):
        """测试 loss 计算接口"""
        # 初始化 Criterion
        matcher = DiffusionDetMatcher(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
        loss_cls = FocalLoss(
            use_sigmoid=True, alpha=0.25, gamma=2.0, loss_weight=2.0)
        loss_bbox = L1Loss(loss_weight=5.0)
        loss_giou = GIoULoss(loss_weight=2.0)

        criterion = DiffusionDetCriterion(
            num_classes=self.num_classes,
            matcher=matcher,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_giou=loss_giou,
            deep_supervision=True,
        )
        self.model.criterion = criterion

        # 模拟输入
        img_metas = [{'img_shape': (800, 800)}, {'img_shape': (800, 1000)}]
        gt_bboxes = [
            torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]],
                         dtype=torch.float32),
            torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
        ]
        gt_labels = [
            torch.tensor([1, 5], dtype=torch.long),
            torch.tensor([10], dtype=torch.long),
        ]

        features = self._create_dummy_features()
        losses = self.model.loss(features, img_metas, gt_bboxes, gt_labels)

        # 检查损失
        self.assertIn('loss_cls', losses)
        self.assertIn('loss_bbox', losses)
        self.assertIn('loss_giou', losses)
        for k, v in losses.items():
            self.assertEqual(v.shape, ())
            self.assertFalse(torch.isnan(v))


if __name__ == '__main__':
    unittest.main()
