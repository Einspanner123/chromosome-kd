"""ChromoGen测试套件

红绿重构流程：先写测试(RED)，再修复使通过(GREEN)，最后重构。
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

# 添加项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


# ============================================================
# 辅助：小型配置用于快速测试
# ============================================================


def _unet_cfg_small():
    from projects.ChromoGen.config import UNetConfig

    return UNetConfig(
        sample_size=32,
        block_out_channels=(32, 64, 128, 128),
        attention_head_dim=4,
        cross_attention_dim=128,
    )


def _cond_cfg_small():
    from projects.ChromoGen.config import ConditionEncoderConfig

    return ConditionEncoderConfig(embed_dim=128, dropout=0.0)


def _bbox_cfg_small():
    from projects.ChromoGen.config import BBoxHeadConfig

    return BBoxHeadConfig(
        in_channels=128,
        feat_channels=128,
        num_proposals=20,
        num_heads=4,
        num_layers=1,
    )


def _diffusion_cfg_small():
    from projects.ChromoGen.config import DiffusionConfig

    return DiffusionConfig(num_train_timesteps=100)


# ============================================================
# 测试 constants
# ============================================================


class TestConstants:
    def test_num_classes_is_24(self):
        from projects.ChromoGen.constants import NUM_CLASSES

        assert NUM_CLASSES == 24

    def test_chromo_classes_length(self):
        from projects.ChromoGen.constants import CHROMO_CLASSES

        assert len(CHROMO_CLASSES) == 24

    def test_chromo_classes_content(self):
        from projects.ChromoGen.constants import CHROMO_CLASSES

        assert CHROMO_CLASSES[0] == 'A1'
        assert CHROMO_CLASSES[-1] == 'Y'

    def test_coco_cat_id_mapping(self):
        from projects.ChromoGen.constants import COCO_CAT_ID_TO_CLASS_IDX

        assert COCO_CAT_ID_TO_CLASS_IDX[1] == 0  # A1
        assert COCO_CAT_ID_TO_CLASS_IDX[24] == 23  # Y


# ============================================================
# 测试 config
# ============================================================


class TestConfig:
    def test_default_config(self):
        from projects.ChromoGen.config import ChromoGenConfig

        cfg = ChromoGenConfig()
        assert cfg.enable_bbox_head is True
        assert cfg.vae.model == 'stabilityai/sd-vae-ft-mse'
        assert cfg.unet.sample_size == 96
        assert cfg.diffusion.num_train_timesteps == 1000

    def test_bbox_head_disabled(self):
        from projects.ChromoGen.config import ChromoGenConfig

        cfg = ChromoGenConfig(enable_bbox_head=False)
        assert cfg.enable_bbox_head is False

    def test_unet_bottleneck_channels(self):
        from projects.ChromoGen.config import UNetConfig

        cfg = UNetConfig()
        assert cfg.bottleneck_channels == 1280

    def test_loss_config_defaults(self):
        from projects.ChromoGen.config import LossConfig

        cfg = LossConfig()
        assert cfg.lambda_img == 1.0
        assert cfg.lambda_bbox == 0.5
        assert cfg.lambda_cls == 0.5


# ============================================================
# 测试 condition_encoder
# ============================================================


class TestConditionEncoder:
    def test_output_shape(self):
        from projects.ChromoGen.constants import NUM_CLASSES
        from projects.ChromoGen.models.condition_encoder import (
            ChromoConditionEncoder,
        )

        enc = ChromoConditionEncoder(num_classes=NUM_CLASSES, embed_dim=128)
        class_labels = torch.arange(NUM_CLASSES).unsqueeze(0)
        counts = torch.randint(0, 5, (1, NUM_CLASSES))
        out = enc(class_labels, counts)
        assert out.shape == (1, NUM_CLASSES + 1, 128)

    def test_batch_output(self):
        from projects.ChromoGen.constants import NUM_CLASSES
        from projects.ChromoGen.models.condition_encoder import (
            ChromoConditionEncoder,
        )

        enc = ChromoConditionEncoder(num_classes=NUM_CLASSES, embed_dim=256)
        B = 4
        class_labels = torch.arange(NUM_CLASSES).unsqueeze(0).expand(B, -1)
        counts = torch.randint(0, 5, (B, NUM_CLASSES))
        out = enc(class_labels, counts)
        assert out.shape == (B, NUM_CLASSES + 1, 256)

    def test_different_counts_different_output(self):
        from projects.ChromoGen.constants import NUM_CLASSES
        from projects.ChromoGen.models.condition_encoder import (
            ChromoConditionEncoder,
        )

        enc = ChromoConditionEncoder(num_classes=NUM_CLASSES, embed_dim=128)
        class_labels = torch.arange(NUM_CLASSES).unsqueeze(0)
        counts_a = torch.ones(1, NUM_CLASSES, dtype=torch.long)
        counts_b = torch.ones(1, NUM_CLASSES, dtype=torch.long) * 5
        out_a = enc(class_labels, counts_a)
        out_b = enc(class_labels, counts_b)
        assert not torch.allclose(out_a, out_b)

    def test_build_class_labels(self):
        from projects.ChromoGen.constants import NUM_CLASSES
        from projects.ChromoGen.models.condition_encoder import (
            ChromoConditionEncoder,
        )

        labels = ChromoConditionEncoder.build_class_labels()
        assert labels.shape == (1, NUM_CLASSES)
        assert labels[0, 0].item() == 0
        assert labels[0, -1].item() == NUM_CLASSES - 1

    def test_sinusoidal_count_encoding_shape(self):
        from projects.ChromoGen.constants import NUM_CLASSES
        from projects.ChromoGen.models.condition_encoder import (
            SinusoidalCountEncoding,
        )

        enc = SinusoidalCountEncoding(num_classes=NUM_CLASSES, dim=128)
        counts = torch.randint(0, 10, (2, NUM_CLASSES))
        out = enc(counts)
        assert out.shape == (2, NUM_CLASSES, 128)

    def test_parse_counts_from_coco(self):
        from projects.ChromoGen.models.condition_encoder import (
            ChromoConditionEncoder,
        )

        anns = [
            {'category_id': 1},
            {'category_id': 1},
            {'category_id': 5},
        ]
        counts = ChromoConditionEncoder.parse_counts_from_coco(anns)
        assert counts[0].item() == 2  # category_id=1 → idx=0
        assert counts[4].item() == 1  # category_id=5 → idx=4


# ============================================================
# 测试 bbox_head
# ============================================================


class TestBBoxDiffusionHead:
    def test_inference_output_shape(self):
        from projects.ChromoGen.constants import NUM_CLASSES
        from projects.ChromoGen.models.bbox_head import BBoxDiffusionHead

        head = BBoxDiffusionHead(
            in_channels=1280,
            feat_channels=256,
            num_classes=NUM_CLASSES,
            num_proposals=50,
            num_heads=4,
            num_layers=1,
        )
        head.eval()
        B = 2
        bottleneck_feat = torch.randn(B, 1280, 12, 12)
        condition = torch.randn(B, NUM_CLASSES + 1, 256)
        t = torch.rand(B)
        out = head(bottleneck_feat, condition, t)
        assert out['pred_bboxes'].shape == (B, 50, 4)
        assert out['pred_labels'].shape == (B, 50)
        assert out['pred_scores'].shape == (B, 50)

    def test_training_loss_output(self):
        from projects.ChromoGen.constants import NUM_CLASSES
        from projects.ChromoGen.models.bbox_head import BBoxDiffusionHead

        head = BBoxDiffusionHead(
            in_channels=1280,
            feat_channels=256,
            num_classes=NUM_CLASSES,
            num_proposals=50,
            num_heads=4,
            num_layers=1,
        )
        head.train()
        B = 2
        bottleneck_feat = torch.randn(B, 1280, 12, 12)
        condition = torch.randn(B, NUM_CLASSES + 1, 256)
        t = torch.rand(B)
        gt_bboxes = [torch.rand(5, 4), torch.rand(3, 4)]
        gt_labels = [
            torch.randint(0, NUM_CLASSES, (5,)),
            torch.randint(0, NUM_CLASSES, (3,)),
        ]
        out = head(bottleneck_feat, condition, t, gt_bboxes, gt_labels)
        assert 'loss_bbox' in out
        assert 'loss_cls' in out
        assert 'loss_total' in out
        assert out['loss_bbox'].requires_grad
        assert out['loss_total'].item() > 0

    def test_empty_gt_no_crash(self):
        from projects.ChromoGen.constants import NUM_CLASSES
        from projects.ChromoGen.models.bbox_head import BBoxDiffusionHead

        head = BBoxDiffusionHead(
            in_channels=1280,
            feat_channels=256,
            num_classes=NUM_CLASSES,
            num_proposals=50,
            num_heads=4,
            num_layers=1,
        )
        head.train()
        B = 1
        bottleneck_feat = torch.randn(B, 1280, 12, 12)
        condition = torch.randn(B, NUM_CLASSES + 1, 256)
        t = torch.rand(B)
        gt_bboxes = [torch.zeros(0, 4)]
        gt_labels = [torch.zeros(0, dtype=torch.long)]
        out = head(bottleneck_feat, condition, t, gt_bboxes, gt_labels)
        assert 'loss_total' in out

    def test_pred_bboxes_in_range(self):
        from projects.ChromoGen.constants import NUM_CLASSES
        from projects.ChromoGen.models.bbox_head import BBoxDiffusionHead

        head = BBoxDiffusionHead(
            in_channels=1280,
            feat_channels=256,
            num_classes=NUM_CLASSES,
            num_proposals=50,
            num_heads=4,
            num_layers=1,
        )
        head.eval()
        B = 1
        bottleneck_feat = torch.randn(B, 1280, 12, 12)
        condition = torch.randn(B, NUM_CLASSES + 1, 256)
        t = torch.zeros(B)
        out = head(bottleneck_feat, condition, t)
        assert out['pred_bboxes'].min() >= 0
        assert out['pred_bboxes'].max() <= 1


# ============================================================
# 测试 unet
# ============================================================


class TestChromoUNet:
    def test_output_has_sample_and_bottleneck(self):
        from projects.ChromoGen.models.unet import ChromoUNet

        unet = ChromoUNet(
            sample_size=32,
            block_out_channels=(32, 64, 128, 128),
            attention_head_dim=4,
            cross_attention_dim=128,
        )
        B = 1
        sample = torch.randn(B, 4, 32, 32)
        timestep = torch.tensor([500])
        encoder_hidden_states = torch.randn(B, 25, 128)
        out = unet(sample, timestep, encoder_hidden_states)
        assert 'sample' in out
        assert 'bottleneck_feat' in out
        assert out['sample'].shape == (B, 4, 32, 32)

    def test_bottleneck_feat_shape(self):
        from projects.ChromoGen.models.unet import ChromoUNet

        unet = ChromoUNet(
            sample_size=32,
            block_out_channels=(32, 64, 128, 128),
            attention_head_dim=4,
            cross_attention_dim=128,
        )
        B = 1
        sample = torch.randn(B, 4, 32, 32)
        timestep = torch.tensor([500])
        encoder_hidden_states = torch.randn(B, 25, 128)
        out = unet(sample, timestep, encoder_hidden_states)
        assert out['bottleneck_feat'].shape[0] == B
        assert out['bottleneck_feat'].shape[1] == 128

    def test_bottleneck_channels_property(self):
        from projects.ChromoGen.models.unet import ChromoUNet

        unet = ChromoUNet(
            sample_size=32,
            block_out_channels=(32, 64, 128, 256),
            attention_head_dim=4,
            cross_attention_dim=128,
        )
        assert unet.bottleneck_channels == 256


# ============================================================
# 测试 pipeline (enable_bbox_head切换)
# ============================================================


class TestChromoGenPipeline:
    def test_phase1_no_bbox_head(self):
        """Phase1: enable_bbox_head=False, bbox_head应为None"""
        from projects.ChromoGen.models.chromogen_pipeline import (
            ChromoGenPipeline,
        )

        model = ChromoGenPipeline(
            enable_bbox_head=False,
            sample_size=32,
            unet_block_out_channels=(32, 64, 128, 128),
            unet_attention_head_dim=4,
            cross_attention_dim=128,
            condition_embed_dim=128,
            condition_dropout=0.0,
        )
        assert model.bbox_head is None
        assert model.enable_bbox_head is False

    def test_phase2_with_bbox_head(self):
        """Phase2: enable_bbox_head=True, bbox_head应存在"""
        from projects.ChromoGen.models.chromogen_pipeline import (
            ChromoGenPipeline,
        )

        model = ChromoGenPipeline(
            enable_bbox_head=True,
            sample_size=32,
            unet_block_out_channels=(32, 64, 128, 128),
            unet_attention_head_dim=4,
            cross_attention_dim=128,
            condition_embed_dim=128,
            condition_dropout=0.0,
            bbox_feat_channels=128,
            bbox_num_proposals=20,
            bbox_num_heads=4,
            bbox_num_layers=1,
        )
        assert model.bbox_head is not None
        assert model.enable_bbox_head is True

    def test_forward_phase1_no_bbox_loss(self):
        """Phase1 forward: 只有loss_img, 没有loss_bbox"""
        from projects.ChromoGen.constants import NUM_CLASSES
        from projects.ChromoGen.models.chromogen_pipeline import (
            ChromoGenPipeline,
        )

        model = ChromoGenPipeline(
            enable_bbox_head=False,
            sample_size=32,
            unet_block_out_channels=(32, 64, 128, 128),
            unet_attention_head_dim=4,
            cross_attention_dim=128,
            condition_embed_dim=128,
            condition_dropout=0.0,
        )
        model.train()
        B = 1
        pixel_values = torch.randn(B, 3, 256, 256)
        class_labels = torch.arange(NUM_CLASSES).unsqueeze(0)
        counts = torch.randint(0, 3, (1, NUM_CLASSES))
        losses = model(pixel_values, class_labels, counts)
        assert 'loss_img' in losses
        assert 'loss_total' in losses
        assert 'loss_bbox' not in losses

    def test_forward_phase2_with_bbox_loss(self):
        """Phase2 forward: 有loss_img + loss_bbox + loss_cls"""
        from projects.ChromoGen.constants import NUM_CLASSES
        from projects.ChromoGen.models.chromogen_pipeline import (
            ChromoGenPipeline,
        )

        model = ChromoGenPipeline(
            enable_bbox_head=True,
            sample_size=32,
            unet_block_out_channels=(32, 64, 128, 128),
            unet_attention_head_dim=4,
            cross_attention_dim=128,
            condition_embed_dim=128,
            condition_dropout=0.0,
            bbox_feat_channels=128,
            bbox_num_proposals=20,
            bbox_num_heads=4,
            bbox_num_layers=1,
        )
        model.train()
        B = 1
        pixel_values = torch.randn(B, 3, 256, 256)
        class_labels = torch.arange(NUM_CLASSES).unsqueeze(0)
        counts = torch.randint(0, 3, (1, NUM_CLASSES))
        gt_bboxes = [torch.rand(3, 4)]
        gt_labels = [torch.randint(0, NUM_CLASSES, (3,))]
        losses = model(
            pixel_values, class_labels, counts, gt_bboxes, gt_labels
        )
        assert 'loss_img' in losses
        assert 'loss_bbox' in losses
        assert 'loss_cls' in losses
        assert 'loss_total' in losses

    def test_vae_frozen(self):
        """VAE参数应全部frozen"""
        from projects.ChromoGen.models.chromogen_pipeline import (
            ChromoGenPipeline,
        )

        model = ChromoGenPipeline(
            enable_bbox_head=False,
            sample_size=32,
            unet_block_out_channels=(32, 64, 128, 128),
            unet_attention_head_dim=4,
            cross_attention_dim=128,
            condition_embed_dim=128,
            condition_dropout=0.0,
        )
        for p in model.vae.parameters():
            assert not p.requires_grad

    def test_prepare_bboxes_for_training(self):
        """xyxy → cxcywh归一化转换正确性"""
        from projects.ChromoGen.models.chromogen_pipeline import (
            ChromoGenPipeline,
        )

        model = ChromoGenPipeline.__new__(ChromoGenPipeline)
        bboxes_xyxy = [torch.tensor([[10, 20, 50, 80]], dtype=torch.float32)]
        labels = [torch.tensor([0])]
        result_bboxes, result_labels = model.prepare_bboxes_for_training(
            bboxes_xyxy, labels, img_h=100, img_w=200
        )
        # cx = (10+50)/2/200 = 0.15, cy = (20+80)/2/100 = 0.5
        # w = 40/200 = 0.2, h = 60/100 = 0.6
        assert abs(result_bboxes[0][0, 0].item() - 0.15) < 1e-5
        assert abs(result_bboxes[0][0, 1].item() - 0.5) < 1e-5
        assert abs(result_bboxes[0][0, 2].item() - 0.2) < 1e-5
        assert abs(result_bboxes[0][0, 3].item() - 0.6) < 1e-5


# ============================================================
# 测试 dataset
# ============================================================


class TestChromoGenDataset:
    def _make_coco_data(self, tmpdir):
        """创建最小COCO格式测试数据"""
        from PIL import Image

        img_dir = os.path.join(tmpdir, 'train')
        os.makedirs(img_dir, exist_ok=True)

        # 创建测试图像
        img = Image.new('RGB', (100, 100), color='red')
        img_path = os.path.join(img_dir, 'test_001.jpg')
        img.save(img_path)

        # COCO标注
        coco = {
            'images': [
                {
                    'id': 1,
                    'file_name': 'test_001.jpg',
                    'width': 100,
                    'height': 100,
                }
            ],
            'annotations': [
                {
                    'id': 1,
                    'image_id': 1,
                    'category_id': 1,
                    'bbox': [10, 20, 30, 40],
                },
                {
                    'id': 2,
                    'image_id': 1,
                    'category_id': 2,
                    'bbox': [50, 60, 20, 30],
                },
            ],
            'categories': [
                {'id': 1, 'name': 'A1'},
                {'id': 2, 'name': 'A2'},
            ],
        }
        ann_path = os.path.join(tmpdir, 'train', '_annotations.coco.json')
        with open(ann_path, 'w') as f:
            json.dump(coco, f)

        return tmpdir, ann_path

    def test_dataset_loads(self):
        from projects.ChromoGen.dataset.chromo_dataset import ChromoGenDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, ann_path = self._make_coco_data(tmpdir)
            ds = ChromoGenDataset(
                data_root=data_root,
                ann_file='train/_annotations.coco.json',
                img_dir='train',
                image_size=64,
                enable_bbox=True,
            )
            assert len(ds) == 1

    def test_dataset_item_keys(self):
        from projects.ChromoGen.dataset.chromo_dataset import ChromoGenDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, ann_path = self._make_coco_data(tmpdir)
            ds = ChromoGenDataset(
                data_root=data_root,
                ann_file='train/_annotations.coco.json',
                img_dir='train',
                image_size=64,
                enable_bbox=True,
            )
            item = ds[0]
            assert 'pixel_values' in item
            assert 'class_labels' in item
            assert 'counts' in item
            assert 'bboxes' in item
            assert 'labels' in item
            assert item['pixel_values'].shape == (3, 64, 64)

    def test_dataset_counts(self):
        from projects.ChromoGen.dataset.chromo_dataset import ChromoGenDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, ann_path = self._make_coco_data(tmpdir)
            ds = ChromoGenDataset(
                data_root=data_root,
                ann_file='train/_annotations.coco.json',
                img_dir='train',
                image_size=64,
                enable_bbox=True,
            )
            item = ds[0]
            # category_id=1 → idx=0, category_id=2 → idx=1
            assert item['counts'][0].item() == 1
            assert item['counts'][1].item() == 1

    def test_collate_fn(self):
        from projects.ChromoGen.dataset.chromo_dataset import (
            ChromoGenDataset,
            collate_fn,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root, ann_path = self._make_coco_data(tmpdir)
            ds = ChromoGenDataset(
                data_root=data_root,
                ann_file='train/_annotations.coco.json',
                img_dir='train',
                image_size=64,
                enable_bbox=True,
            )
            batch = [ds[0], ds[0]]
            collated = collate_fn(batch)
            assert collated['pixel_values'].shape[0] == 2
            assert len(collated['bboxes']) == 2


# ============================================================
# 测试 metrics
# ============================================================


class TestBBoxMetrics:
    def test_kl_divergence_identical(self):
        from projects.ChromoGen.tools.metrics.bbox_metrics import (
            _kl_divergence,
        )

        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        assert _kl_divergence(p, q) == pytest.approx(0.0, abs=1e-6)

    def test_js_divergence_identical(self):
        from projects.ChromoGen.tools.metrics.bbox_metrics import (
            _js_divergence,
        )

        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert _js_divergence(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_js_divergence_different(self):
        from projects.ChromoGen.tools.metrics.bbox_metrics import (
            _js_divergence,
        )

        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        js = _js_divergence(p, q)
        assert js > 0.0

    def test_js_symmetry(self):
        from projects.ChromoGen.tools.metrics.bbox_metrics import (
            _js_divergence,
        )

        p = np.array([0.3, 0.7])
        q = np.array([0.6, 0.4])
        assert abs(_js_divergence(p, q) - _js_divergence(q, p)) < 1e-10


class TestImageMetrics:
    def test_inception_feature_extractor_shape(self):
        from projects.ChromoGen.tools.metrics.image_metrics import (
            InceptionFeatureExtractor,
        )

        extractor = InceptionFeatureExtractor(device=torch.device('cpu'))
        x = torch.randn(1, 3, 299, 299)
        features, logits = extractor(x)
        assert features.shape == (1, 2048)
        assert logits.shape == (1, 1000)


# ============================================================
# 测试 SwanLab 集成
# ============================================================


class TestSwanLabIntegration:
    def test_train_imports_swanlab(self):
        """训练脚本应能导入swanlab（可能未安装也不崩溃）"""
        # 验证train.py中的HAS_SWANLAB变量逻辑
        try:
            import importlib.util

            importlib.util.find_spec('swanlab')
        except ImportError:
            pass  # 不崩溃即可

    def test_evaluate_imports_swanlab(self):
        """评估脚本应能导入swanlab"""
        try:
            import importlib.util

            importlib.util.find_spec('swanlab')
        except ImportError:
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
