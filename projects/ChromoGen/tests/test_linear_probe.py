"""测试 Phase 0 探针实验: Linear Probe

验证 ChromoGen 特征是否包含检测有用的信息。
在冻结特征上训练线性分类器, 预测 bbox 类别。
"""

import pytest
import torch
import torch.nn as nn

from projects.ChromoGen.evaluation.linear_probe import (
    FeatureExtractor,
    LinearProbe,
    RoIFeatureCollector,
)


class TestFeatureExtractor:
    """测试 ChromoGen 特征提取器 (用于探针实验)"""

    def test_extract_returns_multi_scale_features(self):
        """提取多尺度特征: VAE latent + UNet down/mid"""
        extractor = FeatureExtractor(
            checkpoint='work_dirs/chromogen_phase1/final_model.pt',
            vae_path='work_dirs/chromogen_phase1/vae',
            device='cpu',
        )
        img = torch.randn(1, 3, 768, 768)

        feats = extractor(img)

        assert isinstance(feats, dict)
        assert 'vae_latent' in feats  # (B, 4, 96, 96)
        assert 'down1' in feats  # (B, 320, 96, 96)
        assert 'down2' in feats  # (B, 640, 48, 48)
        assert 'down3' in feats  # (B, 1280, 24, 24)
        assert 'mid' in feats  # (B, 1280, 12, 12)

    def test_feature_channels(self):
        """特征通道数正确"""
        extractor = FeatureExtractor(
            checkpoint='work_dirs/chromogen_phase1/final_model.pt',
            vae_path='work_dirs/chromogen_phase1/vae',
            device='cpu',
        )
        img = torch.randn(1, 3, 768, 768)
        feats = extractor(img)

        assert feats['vae_latent'].shape[1] == 4
        assert feats['down1'].shape[1] == 320
        assert feats['down2'].shape[1] == 640
        assert feats['down3'].shape[1] == 1280
        assert feats['mid'].shape[1] == 1280

    def test_features_are_frozen(self):
        """提取的特征不需要梯度"""
        extractor = FeatureExtractor(
            checkpoint='work_dirs/chromogen_phase1/final_model.pt',
            vae_path='work_dirs/chromogen_phase1/vae',
            device='cpu',
        )
        img = torch.randn(1, 3, 768, 768)
        feats = extractor(img)

        for key, feat in feats.items():
            assert not feat.requires_grad, f'{key} should not require grad'

    def test_batch_extraction(self):
        """支持 batch 提取"""
        extractor = FeatureExtractor(
            checkpoint='work_dirs/chromogen_phase1/final_model.pt',
            vae_path='work_dirs/chromogen_phase1/vae',
            device='cpu',
        )
        img = torch.randn(2, 3, 768, 768)
        feats = extractor(img)

        for key, feat in feats.items():
            assert feat.shape[0] == 2


class TestRoIFeatureCollector:
    """测试 ROI 特征收集器"""

    def test_extract_roi_features(self):
        """从特征图提取 ROI 特征"""
        collector = RoIFeatureCollector(output_size=7)
        # 模拟特征图 (B=1, C=320, H=96, W=96)
        feats = torch.randn(1, 320, 96, 96)
        # bbox: [x1, y1, x2, y2] 归一化坐标
        bboxes = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.7, 0.7]])

        roi_feats = collector(feats, bboxes, image_size=(768, 768))

        assert roi_feats.shape == (2, 320, 7, 7)

    def test_different_feature_scales(self):
        """不同尺度的特征图都能提取 ROI"""
        collector = RoIFeatureCollector(output_size=7)

        # down1: 96x96, down2: 48x48, down3: 24x24
        for channels, h in [(320, 96), (640, 48), (1280, 24)]:
            feats = torch.randn(1, channels, h, h)
            bboxes = torch.tensor([[0.1, 0.1, 0.3, 0.3]])
            roi_feats = collector(feats, bboxes, image_size=(768, 768))
            assert roi_feats.shape == (1, channels, 7, 7)


class TestLinearProbe:
    """测试 Linear Probe 分类器"""

    def test_init(self):
        """初始化线性分类器"""
        probe = LinearProbe(in_channels=320, num_classes=24, roi_size=7)
        assert probe.classifier.in_features == 320 * 7 * 7
        assert probe.classifier.out_features == 24

    def test_forward(self):
        """前向传播"""
        probe = LinearProbe(in_channels=320, num_classes=24, roi_size=7)
        roi_feats = torch.randn(4, 320, 7, 7)

        logits = probe(roi_feats)
        assert logits.shape == (4, 24)

    def test_training_step(self):
        """训练步: 前向 + 计算 loss + 反向"""
        probe = LinearProbe(in_channels=320, num_classes=24, roi_size=7)
        roi_feats = torch.randn(4, 320, 7, 7)
        labels = torch.tensor([0, 5, 12, 23])

        logits = probe(roi_feats)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        # 检查梯度
        for param in probe.parameters():
            assert param.grad is not None

    def test_predict(self):
        """预测类别"""
        probe = LinearProbe(in_channels=320, num_classes=24, roi_size=7)
        roi_feats = torch.randn(4, 320, 7, 7)

        preds = probe.predict(roi_feats)
        assert preds.shape == (4,)
        assert preds.dtype == torch.long
        assert preds.min() >= 0 and preds.max() < 24
