"""测试 LDMDet detector 与 FeatureBridgeModule 集成

方向六: 将 FBM 集成到 PurePyTorchDiffusionDet, 在 extract_feat 后融合 ChromoGen 特征。
"""

import pytest
import torch
import torch.nn as nn

from ldmdet.feature_bridge import (
    ChromoGenFeatureExtractor,
    FeatureBridgeModule,
    UNetFeatureConfig,
)


class _MockUNet(nn.Module):
    """模拟 ChromoGen UNet"""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 320, 3, padding=1)
        self.down1 = _MockDownBlock(320, 320, downsample=False)
        self.down2 = _MockDownBlock(320, 640, downsample=True)
        self.down3 = _MockDownBlock(640, 1280, downsample=True)
        self.down4 = _MockDownBlock(1280, 1280, downsample=True)
        self.mid = _MockMidBlock(1280, 1280)

    def forward(self, sample, timestep, encoder_hidden_states, return_dict=True):
        h = self.conv_in(sample)
        feats = {}
        h = self.down1(h)
        feats['down1'] = h
        h = self.down2(h)
        feats['down2'] = h
        h = self.down3(h)
        feats['down3'] = h
        h = self.down4(h)
        h = self.mid(h)
        feats['mid'] = h
        self._feature_cache = feats
        return type('UNetOutput', (), {'sample': h})()


class _MockDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1) if downsample else None

    def forward(self, x):
        x = self.conv(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class _MockMidBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class _MockVAE(nn.Module):
    """模拟 VAE encoder: image (B,3,H,W) → latent (B,4,H/8,W/8)"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        # 下采样 8x (用 stride=8 的单次 conv 模拟)
        return nn.functional.avg_pool2d(self.encoder(x), 8)


class _StubBackbone(nn.Module):
    """模拟 ResNet-50 backbone 输出 4 个尺度 (stride 4/8/16/32)"""

    def __init__(self, channels=(256, 512, 1024, 2048)):
        super().__init__()
        # stride=[4, 8, 16, 32] 对应 ResNet out_indices=(0,1,2,3)
        strides = [4, 8, 16, 32]
        self.convs = nn.ModuleList([
            nn.Conv2d(3, c, 3, stride=s, padding=1) for c, s in zip(channels, strides)
        ])

    def forward(self, x):
        return [conv(x) for conv in self.convs]


class _StubFPN(nn.Module):
    """模拟 FPN: 4 个尺度输入 → 4 个 256ch 输出 (保持 stride 不变)"""

    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])

    def forward(self, feats):
        return [conv(f) for conv, f in zip(self.convs, feats)]


class TestDetectorFeatureBridgeIntegration:
    """验证 FBM 与 detector 集成的关键行为

    由于 PurePyTorchDiffusionDet 依赖 MMDetection 注册机制, 这里通过
    Mock 组件验证 FBM 在 detector pipeline 中的位置和正确性。
    """

    @pytest.fixture
    def components(self):
        """构建 backbone + FPN + extractor + FBM 组件"""
        backbone = _StubBackbone()
        neck = _StubFPN()
        unet = _MockUNet()
        vae = _MockVAE()
        extractor = ChromoGenFeatureExtractor(unet=unet)
        fbm = FeatureBridgeModule(cg_channels=[320, 640, 1280, 1280], ld_channels=256)
        return backbone, neck, vae, extractor, fbm

    def test_extract_feat_with_fbm(self, components):
        """完整 extract_feat 流程: img → backbone → neck → fbm(cg_feats)"""
        backbone, neck, vae, extractor, fbm = components

        bs, h, w = 1, 768, 768
        img = torch.randn(bs, 3, h, w)

        # 1. LDMDet 特征提取
        backbone_feats = backbone(img)
        ld_feats = neck(backbone_feats)

        # 2. ChromoGen 特征提取 (VAE encode → UNet extract)
        latent = vae(img)
        cg_feats = extractor(latent, torch.tensor([500]), torch.randn(bs, 1, 768))

        # 3. FBM 融合
        fused = fbm(ld_feats, cg_feats)

        # 验证形状
        assert len(fused) == 4
        for i, o in enumerate(fused):
            assert o.shape[1] == 256  # FPN out_channels

    def test_fbm_zero_init_no_baseline_regression(self, components):
        """零初始化: FBM 不引入任何 baseline 回退"""
        backbone, neck, vae, extractor, fbm = components

        bs, h, w = 1, 768, 768
        img = torch.randn(bs, 3, h, w)

        backbone_feats = backbone(img)
        ld_feats = neck(backbone_feats)
        latent = vae(img)
        cg_feats = extractor(latent, torch.tensor([500]), torch.randn(bs, 1, 768))

        # baseline (无 FBM)
        baseline = ld_feats

        # 融合后
        fused = fbm(ld_feats, cg_feats)

        # 应完全一致 (alpha=0)
        for i, (o, lf) in enumerate(zip(fused, baseline)):
            assert torch.allclose(o, lf, atol=1e-6), f'Level {i} regression'

    def test_fbm_trainable_params_only(self, components):
        """只有 FBM 参数可训练, ChromoGen 冻结"""
        _, _, _, extractor, fbm = components
        extractor.set_frozen()

        # ChromoGen 参数全部冻结
        for name, param in extractor.unet.named_parameters():
            assert not param.requires_grad, f'{name} should be frozen'

        # FBM 参数可训练
        for name, param in fbm.named_parameters():
            assert param.requires_grad, f'{name} should be trainable'

    def test_fbm_config_serialization(self):
        """FBM 配置可序列化为 dict (用于 mmengine Config)"""
        from ldmdet.feature_bridge import FeatureBridgeModule

        fbm_cfg = dict(
            type='FeatureBridgeModule',
            cg_channels=[320, 640, 1280, 1280],
            ld_channels=256,
        )

        # 验证配置可实例化
        fbm = FeatureBridgeModule(**{k: v for k, v in fbm_cfg.items() if k != 'type'})
        assert fbm.ld_channels == 256
        assert fbm.alpha.shape == (3,)

    def test_extractor_config_serialization(self):
        """Extractor 配置可序列化为 dict"""
        config = UNetFeatureConfig(
            extract_down1=True,
            extract_down2=True,
            extract_down3=True,
            extract_mid=True,
        )

        # 验证 dataclass 可转为 dict
        from dataclasses import asdict

        cfg_dict = asdict(config)
        assert cfg_dict['extract_down1'] is True
        assert cfg_dict['extract_mid'] is True
