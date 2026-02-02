import sys
import unittest
from pathlib import Path

import torch

# 将项目根目录添加到 python 路径
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from mods.loss import (
    DiffusionDetCriterion,
    DiffusionDetMatcher,
    FocalLoss,
    GIoULoss,
    L1Loss,
)
from mods.roi_extractor import SingleRoIExtractor
from mods.single_head import SingleDiffusionDetHead

from projects.LDMDet.mods.diffusiondet_head import DiffusionDetHead
from projects.LDMDet.mods.structures import ImageMeta
from projects.LDMDet.mods.utils import raw_to_xyxy, xyxy_to_raw


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
                type="RoIAlign", output_size=7, sampling_ratio=2, aligned=True
            ),
            out_channels=self.feat_channels,
            featmap_strides=[4, 8, 16, 32],
        )

        # 3. 初始化 Criterion
        matcher = DiffusionDetMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
        loss_cls = FocalLoss(use_sigmoid=True, alpha=0.25, gamma=2.0, loss_weight=2.0)
        loss_bbox = L1Loss(loss_weight=5.0)
        loss_giou = GIoULoss(loss_weight=2.0)
        self.criterion = DiffusionDetCriterion(
            num_classes=self.num_classes,
            matcher=matcher,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_giou=loss_giou,
            deep_supervision=True,
        )

        # 4. 初始化 DiffusionDetHead
        self.model = DiffusionDetHead(
            num_classes=self.num_classes,
            feat_channels=self.feat_channels,
            num_proposals=self.num_proposals,
            num_heads=self.num_heads,
            single_head=self.single_head,
            roi_extractor=self.roi_extractor,
            criterion=self.criterion,
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
        t = torch.randint(0, 1000, (self.batch_size,))

        with torch.no_grad():
            all_cls_logits, all_pred_bboxes = self.model(features, bboxes, t)

        # 验证输出形状
        # [num_heads, bs, num_proposals, num_classes]
        self.assertEqual(
            all_cls_logits.shape,
            (self.num_heads, self.batch_size, self.num_proposals, self.num_classes),
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
            ImageMeta(img_shape=self.img_shape, scale_factor=[1.0, 1.0, 1.0, 1.0]),
            ImageMeta(img_shape=self.img_shape, scale_factor=[1.0, 1.0, 1.0, 1.0]),
        ]

        with torch.no_grad():
            results = self.model.predict(features, img_metas, rescale=True)

        # 验证结果结构
        self.assertEqual(len(results), self.batch_size)
        for res in results:
            self.assertTrue(hasattr(res, "bboxes"))
            self.assertTrue(hasattr(res, "scores"))
            self.assertTrue(hasattr(res, "labels"))

            # 验证张量形状
            num_dets = res.bboxes.shape[0]
            self.assertEqual(res.bboxes.shape, (num_dets, 4))
            self.assertEqual(res.scores.shape, (num_dets,))
            self.assertEqual(res.labels.shape, (num_dets,))

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
        img_metas = [ImageMeta(img_shape=self.img_shape)] * self.batch_size

        # 初始噪声
        x_raw = torch.randn(self.batch_size, self.num_proposals, 4)
        # 模拟全负样本的 logits
        cls_logits = torch.full(
            (self.batch_size, self.num_proposals, self.num_classes), -10.0
        )
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
        # 模拟输入
        img_metas = [
            ImageMeta(img_shape=(800, 800)),
            ImageMeta(img_shape=(800, 1000)),
        ]
        gt_bboxes = [
            torch.tensor(
                [[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32
            ),
            torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
        ]
        gt_labels = [
            torch.tensor([1, 5], dtype=torch.long),
            torch.tensor([10], dtype=torch.long),
        ]

        features = self._create_dummy_features()
        losses = self.model.loss(features, img_metas, gt_bboxes, gt_labels)

        # 检查损失
        self.assertIn("loss_cls", losses)
        self.assertIn("loss_bbox", losses)
        self.assertIn("loss_giou", losses)
        for k, v in losses.items():
            self.assertEqual(v.shape, ())
            self.assertFalse(torch.isnan(v))

    def test_rf_mode(self):
        """测试 Rectified Flow 模式"""
        # 创建 RF 模式的模型
        rf_model = DiffusionDetHead(
            num_classes=self.num_classes,
            feat_channels=self.feat_channels,
            num_proposals=self.num_proposals,
            num_heads=self.num_heads,
            single_head=self.single_head,
            roi_extractor=self.roi_extractor,
            criterion=self.criterion,
            diffusion_type="rectified_flow",
            rf_schedule="shifted",
            rf_shift=3.0,
            sampling_timesteps=2,
        )
        rf_model.eval()

        features = self._create_dummy_features()
        img_metas = [ImageMeta(img_shape=self.img_shape)] * self.batch_size

        # 1. 测试推理
        with torch.no_grad():
            results = rf_model.predict(features, img_metas)
        self.assertEqual(len(results), self.batch_size)

        # 2. 测试 loss
        gt_bboxes = [
            torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        ] * self.batch_size
        gt_labels = [torch.tensor([1], dtype=torch.long)] * self.batch_size

        losses = rf_model.loss(features, img_metas, gt_bboxes, gt_labels)
        self.assertIn("loss_cls", losses)

    def test_predict_trajectory(self):
        """测试 return_trajectory=True"""
        features = self._create_dummy_features()
        img_metas = [ImageMeta(img_shape=self.img_shape)] * self.batch_size
        sampling_steps = 2
        self.model.sampling_timesteps = sampling_steps

        with torch.no_grad():
            results, trajectory = self.model.predict(
                features, img_metas, return_trajectory=True
            )

        self.assertEqual(len(results), self.batch_size)
        # trajectory 长度应该是 sampling_steps
        self.assertEqual(len(trajectory), sampling_steps)
        for step_logits, step_bboxes in trajectory:
            self.assertEqual(
                step_logits.shape,
                (self.batch_size, self.num_proposals, self.num_classes),
            )
            self.assertEqual(
                step_bboxes.shape, (self.batch_size, self.num_proposals, 4)
            )

    def test_loss_empty_gt(self):
        """测试空 GT 情况下的 loss 计算"""
        features = self._create_dummy_features()
        img_metas = [ImageMeta(img_shape=self.img_shape)] * self.batch_size
        # 一个图像有 GT，一个没有
        gt_bboxes = [
            torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
            torch.zeros((0, 4), dtype=torch.float32),
        ]
        gt_labels = [
            torch.tensor([1], dtype=torch.long),
            torch.zeros((0,), dtype=torch.long),
        ]

        losses = self.model.loss(features, img_metas, gt_bboxes, gt_labels)
        self.assertIn("loss_cls", losses)
        self.assertFalse(torch.isnan(losses["loss_cls"]))

    def test_data_flow_consistency(self):
        """验证 forward 和 loss 之间的数据流一致性"""
        # 1. 验证 _raw_to_xyxy 和 _xyxy_to_raw 对于图像内框的互逆性
        img_metas = [ImageMeta(img_shape=(800, 1200))]
        # 构造一个完全在图像内的 cxcywh 框 (normalized to [0, 1])
        # cx, cy in [0.2, 0.8], wh in [0.1, 0.3] -> 保证 xyxy 在 [0, 1]
        inner_boxes_norm = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]], dtype=torch.float32)
        # 映射到 raw 空间 [-snr, snr]
        raw_boxes = (inner_boxes_norm * 2 - 1) * self.model.snr_scale

        xyxy_boxes = raw_to_xyxy(raw_boxes, img_metas, self.model.snr_scale)
        raw_boxes_back = xyxy_to_raw(xyxy_boxes, img_metas, self.model.snr_scale)

        # 允许微小的浮点误差
        self.assertTrue(torch.allclose(raw_boxes, raw_boxes_back, atol=1e-5))

        # 2. 验证 loss 中生成的 curr_bboxes 范围是否正确 (应该被裁剪到图像内)
        features = self._create_dummy_features()
        gt_bboxes = [
            torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
        ] * self.batch_size
        gt_labels = [torch.tensor([1], dtype=torch.long)] * self.batch_size

        # 临时 hook 捕获 self(features, curr_bboxes, t_input) 的输入
        captured_inputs = []
        original_forward = self.model.forward

        def mock_forward(features, bboxes, t, proposals=None):
            captured_inputs.append(bboxes)
            return original_forward(features, bboxes, t, proposals)

        self.model.forward = mock_forward
        try:
            self.model.loss(features, img_metas * self.batch_size, gt_bboxes, gt_labels)
        finally:
            self.model.forward = original_forward

        # 检查传给 forward 的 bboxes 是否在图像范围内且是 xyxy 格式
        input_bboxes = captured_inputs[0]
        self.assertTrue((input_bboxes >= 0).all())
        h, w = self.img_shape
        # 注意：这里我们使用了 img_metas * self.batch_size，所以每个 batch 的 w, h 可能不同
        # 但在 setUp 中我们定义了 self.img_shape = (800, 1216)
        # test_loss 覆盖了不同形状的情况，这里我们保持简单
        self.assertTrue((input_bboxes[..., 0] <= w).all())
        self.assertTrue((input_bboxes[..., 1] <= h).all())
        self.assertTrue((input_bboxes[..., 2] <= w).all())
        self.assertTrue((input_bboxes[..., 3] <= h).all())


if __name__ == "__main__":
    unittest.main()
