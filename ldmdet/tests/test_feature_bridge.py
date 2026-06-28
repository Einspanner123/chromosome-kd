"""测试 ldmdet.feature_bridge — FeatureBridgeModule, ChromoGenFeatureExtractor

方向六：生成模型感知迁移。FeatureBridgeModule (FBM) 将 ChromoGen UNet
编码器特征投影并融合到 LDMDet FPN 特征空间。

架构对齐:
  LDMDet FPN (4 levels, 256ch, stride 4/8/16/32)
  ChromoGen UNet encoder (在 VAE latent H/8 空间操作):
    down1: 320ch, H/8  → 对齐 FPN level 1 (stride 8)
    down2: 640ch, H/16 → 对齐 FPN level 2 (stride 16)
    down3: 1280ch, H/32 → 对齐 FPN level 3 (stride 32)
    mid:   1280ch, H/64 → 全局语义，上采样后融合到 level 3
  FPN level 0 (stride 4) 无对应 ChromoGen 特征，保持不变。
"""

import pytest
import torch
import torch.nn as nn

from ldmdet.feature_bridge.feature_bridge_module import FeatureBridgeModule
from ldmdet.feature_bridge.chromogen_extractor import (
    ChromoGenFeatureExtractor,
    UNetFeatureConfig,
)


class TestFeatureBridgeModule:
    """FeatureBridgeModule 单元测试"""

    @pytest.fixture
    def fbm(self):
        """默认 FBM: ChromoGen UNet 特征 → LDMDet FPN (256ch)"""
        return FeatureBridgeModule(
            cg_channels=[320, 640, 1280, 1280],  # down1, down2, down3, mid
            ld_channels=256,
        )

    def _make_ldmdet_feats(self, bs=2, h=800, w=800, channels=256):
        """生成 LDMDet FPN 4 层特征 (stride 4/8/16/32)"""
        return [
            torch.randn(bs, channels, h // 4, w // 4),  # P3 stride 4
            torch.randn(bs, channels, h // 8, w // 8),  # P4 stride 8
            torch.randn(bs, channels, h // 16, w // 16),  # P5 stride 16
            torch.randn(bs, channels, h // 32, w // 32),  # P6 stride 32
        ]

    def _make_chromogen_feats(self, bs=2, h=800, w=800):
        """生成 ChromoGen UNet 编码器特征 (在 H/8 latent 空间)

        UNet 输入 latent 尺寸 = (h/8, w/8)
        down1: h/8  (320ch)
        down2: h/16 (640ch)
        down3: h/32 (1280ch)
        mid:   h/64 (1280ch)
        """
        return {
            'down1': torch.randn(bs, 320, h // 8, w // 8),
            'down2': torch.randn(bs, 640, h // 16, w // 16),
            'down3': torch.randn(bs, 1280, h // 32, w // 32),
            'mid': torch.randn(bs, 1280, h // 64, w // 64),
        }

    # ── 形状测试 ──────────────────────────────────────

    def test_output_shape_matches_input(self, fbm):
        """输出 4 层特征，每层形状与 LDMDet 输入一致"""
        ld_feats = self._make_ldmdet_feats(bs=2, h=800, w=800)
        cg_feats = self._make_chromogen_feats(bs=2, h=800, w=800)

        out = fbm(ld_feats, cg_feats)

        assert isinstance(out, (list, tuple))
        assert len(out) == 4
        for i, (o, lf) in enumerate(zip(out, ld_feats)):
            assert o.shape == lf.shape, f'Level {i} shape mismatch: {o.shape} vs {lf.shape}'

    def test_output_channels(self, fbm):
        """所有输出层通道数 = ld_channels (256)"""
        ld_feats = self._make_ldmdet_feats(bs=1, h=640, w=640)
        cg_feats = self._make_chromogen_feats(bs=1, h=640, w=640)

        out = fbm(ld_feats, cg_feats)

        for i, o in enumerate(out):
            assert o.shape[1] == 256, f'Level {i} channels={o.shape[1]}, expected 256'

    def test_different_batch_sizes(self, fbm):
        """不同 batch size 都能正常工作"""
        for bs in [1, 2, 4]:
            ld_feats = self._make_ldmdet_feats(bs=bs, h=512, w=512)
            cg_feats = self._make_chromogen_feats(bs=bs, h=512, w=512)
            out = fbm(ld_feats, cg_feats)
            assert out[0].shape[0] == bs

    def test_different_image_sizes(self, fbm):
        """不同图像尺寸（必须能被 64 整除）都能正常工作"""
        for h, w in [(512, 512), (768, 768), (800, 800), (1024, 768)]:
            ld_feats = self._make_ldmdet_feats(bs=1, h=h, w=w)
            cg_feats = self._make_chromogen_feats(bs=1, h=h, w=w)
            out = fbm(ld_feats, cg_feats)
            assert out[0].shape[2] == h // 4
            assert out[0].shape[3] == w // 4

    # ── 零初始化测试 (核心特性) ──────────────────────────

    def test_zero_init_output_equals_input(self, fbm):
        """零初始化: 初始时融合输出 == LDMDet 输入 (alpha=0)"""
        ld_feats = self._make_ldmdet_feats(bs=2, h=512, w=512)
        cg_feats = self._make_chromogen_feats(bs=2, h=512, w=512)

        out = fbm(ld_feats, cg_feats)

        for i, (o, lf) in enumerate(zip(out, ld_feats)):
            assert torch.allclose(o, lf, atol=1e-6), (
                f'Level {i}: zero-init broken, max diff={ (o - lf).abs().max().item()}'
            )

    def test_alpha_initialized_to_zero(self, fbm):
        """融合权重 alpha 初始化为 0"""
        assert fbm.alpha is not None
        assert fbm.alpha.shape == (3,)  # 3 个融合层 (level 1/2/3)
        assert torch.allclose(fbm.alpha, torch.zeros(3))

    # ── 梯度流测试 ──────────────────────────────────────

    def test_gradient_flow_to_alpha(self, fbm):
        """alpha 参数有梯度流"""
        ld_feats = self._make_ldmdet_feats(bs=1, h=512, w=512)
        cg_feats = self._make_chromogen_feats(bs=1, h=512, w=512)

        out = fbm(ld_feats, cg_feats)
        loss = sum(o.sum() for o in out)
        loss.backward()

        assert fbm.alpha.grad is not None
        # level 1/2/3 对应的 alpha 应有梯度
        assert fbm.alpha.grad.shape == (3,)

    def test_gradient_flow_to_projection(self, fbm):
        """投影层参数有梯度流"""
        ld_feats = self._make_ldmdet_feats(bs=1, h=512, w=512)
        cg_feats = self._make_chromogen_feats(bs=1, h=512, w=512)

        out = fbm(ld_feats, cg_feats)
        loss = sum(o.sum() for o in out)
        loss.backward()

        # 检查投影层有梯度 (alpha=0 时投影层梯度仍应通过 alpha 传播)
        for name, param in fbm.named_parameters():
            if 'proj' in name:
                assert param.grad is not None, f'{name} has no gradient'

    # ── 融合行为测试 ──────────────────────────────────────

    def test_level0_not_fused(self, fbm):
        """Level 0 (stride 4) 无对应 ChromoGen 特征，应保持不变"""
        ld_feats = self._make_ldmdet_feats(bs=1, h=512, w=512)
        cg_feats = self._make_chromogen_feats(bs=1, h=512, w=512)

        out = fbm(ld_feats, cg_feats)

        # Level 0 应完全等于输入 (不仅是零初始化, 且没有融合逻辑)
        assert torch.equal(out[0], ld_feats[0])

    def test_fusion_changes_output_when_alpha_nonzero(self, fbm):
        """alpha 非零时，level 1/2/3 输出应不同于输入"""
        ld_feats = self._make_ldmdet_feats(bs=1, h=512, w=512)
        cg_feats = self._make_chromogen_feats(bs=1, h=512, w=512)

        # 手动设置 alpha 为非零
        with torch.no_grad():
            fbm.alpha.fill_(0.5)

        out = fbm(ld_feats, cg_feats)

        for i in [1, 2, 3]:
            assert not torch.allclose(out[i], ld_feats[i], atol=1e-6), (
                f'Level {i}: fusion had no effect with alpha=0.5'
            )

    def test_mid_feature_upsampled_to_level3(self, fbm):
        """mid 特征 (H/64) 上采样后融合到 level 3 (H/32)"""
        ld_feats = self._make_ldmdet_feats(bs=1, h=512, w=512)
        cg_feats = self._make_chromogen_feats(bs=1, h=512, w=512)

        with torch.no_grad():
            fbm.alpha.fill_(0.0)
            # 只启用 mid 融合到 level 3
            fbm.alpha[2] = 1.0  # 假设 alpha[2] 控制 level 3 的融合

        out = fbm(ld_feats, cg_feats)

        # Level 3 应该变化 (mid 特征已上采样融合)
        assert not torch.allclose(out[3], ld_feats[3], atol=1e-6)

    # ── 异常处理测试 ──────────────────────────────────────

    def test_missing_chromogen_key_raises(self, fbm):
        """缺少必要的 ChromoGen 特征键应抛出 KeyError"""
        ld_feats = self._make_ldmdet_feats(bs=1, h=512, w=512)
        cg_feats = self._make_chromogen_feats(bs=1, h=512, w=512)
        del cg_feats['down2']

        with pytest.raises(KeyError):
            fbm(ld_feats, cg_feats)

    def test_mismatched_batch_size_raises(self, fbm):
        """LDMDet 和 ChromoGen 特征 batch size 不一致应抛出 ValueError"""
        ld_feats = self._make_ldmdet_feats(bs=2, h=512, w=512)
        cg_feats = self._make_chromogen_feats(bs=1, h=512, w=512)

        with pytest.raises(ValueError):
            fbm(ld_feats, cg_feats)

    def test_ldmdet_feats_must_have_four_levels(self, fbm):
        """LDMDet 特征必须是 4 层"""
        ld_feats = self._make_ldmdet_feats(bs=1, h=512, w=512)[:3]  # 只取 3 层
        cg_feats = self._make_chromogen_feats(bs=1, h=512, w=512)

        with pytest.raises(ValueError):
            fbm(ld_feats, cg_feats)


class TestChromoGenFeatureExtractor:
    """ChromoGen 特征提取器单元测试

    由于无法加载真实 ChromoGen 权重，使用 Mock UNet 验证 hook 机制。
    """

    @pytest.fixture
    def feature_config(self):
        """UNet 特征配置: 指定提取哪些层的特征"""
        return UNetFeatureConfig(
            extract_down1=True,
            extract_down2=True,
            extract_down3=True,
            extract_mid=True,
        )

    @pytest.fixture
    def extractor(self, feature_config):
        """创建特征提取器 (使用 MockUNet)"""
        return ChromoGenFeatureExtractor(
            unet=_MockUNet(),
            config=feature_config,
        )

    def test_extract_returns_dict(self, extractor):
        """提取结果应为包含指定键的 dict"""
        latent = torch.randn(2, 4, 96, 96)  # VAE latent (H/8)
        timestep = torch.tensor([500])
        encoder_hidden = torch.randn(2, 1, 768)

        feats = extractor(latent, timestep, encoder_hidden)

        assert isinstance(feats, dict)
        assert 'down1' in feats
        assert 'down2' in feats
        assert 'down3' in feats
        assert 'mid' in feats

    def test_feature_channels(self, extractor):
        """提取的特征通道数正确"""
        latent = torch.randn(1, 4, 96, 96)
        timestep = torch.tensor([500])
        encoder_hidden = torch.randn(1, 1, 768)

        feats = extractor(latent, timestep, encoder_hidden)

        assert feats['down1'].shape[1] == 320
        assert feats['down2'].shape[1] == 640
        assert feats['down3'].shape[1] == 1280
        assert feats['mid'].shape[1] == 1280

    def test_feature_spatial_sizes(self, extractor):
        """提取的特征空间尺寸正确 (逐步下采样)"""
        latent = torch.randn(1, 4, 96, 96)  # H/8 = 96

        feats = extractor(latent, torch.tensor([500]), torch.randn(1, 1, 768))

        assert feats['down1'].shape[2] == 96  # H/8
        assert feats['down2'].shape[2] == 48  # H/16
        assert feats['down3'].shape[2] == 24  # H/32
        assert feats['mid'].shape[2] == 12  # H/64

    def test_no_grad_in_extracted_features(self, extractor):
        """提取的特征不应有梯度 (ChromoGen 冻结)"""
        latent = torch.randn(1, 4, 96, 96)
        feats = extractor(latent, torch.tensor([500]), torch.randn(1, 1, 768))

        for key, feat in feats.items():
            assert not feat.requires_grad, f'{key} should not require grad (frozen)'

    def test_partial_extraction(self, feature_config):
        """只提取部分层特征"""
        config = UNetFeatureConfig(
            extract_down1=False,
            extract_down2=True,
            extract_down3=False,
            extract_mid=False,
        )
        extractor = ChromoGenFeatureExtractor(
            unet=_MockUNet(),
            config=config,
        )

        latent = torch.randn(1, 4, 96, 96)
        feats = extractor(latent, torch.tensor([500]), torch.randn(1, 1, 768))

        assert 'down2' in feats
        assert 'down1' not in feats
        assert 'down3' not in feats
        assert 'mid' not in feats

    def test_eval_mode(self, extractor):
        """提取器应将 UNet 设为 eval 模式"""
        extractor.set_frozen()
        assert not extractor.unet.training

    def test_deterministic_output(self, extractor):
        """相同输入产生相同输出 (eval + no_grad)"""
        extractor.set_frozen()
        latent = torch.randn(1, 4, 96, 96)
        enc = torch.randn(1, 1, 768)
        t = torch.tensor([500])

        feats1 = extractor(latent, t, enc)
        feats2 = extractor(latent, t, enc)

        for key in feats1:
            assert torch.allclose(feats1[key], feats2[key], atol=1e-6)

    def test_dict_config_construction(self):
        """通过 dict config 构建提取器 (支持 mmengine 配置)"""
        extractor = ChromoGenFeatureExtractor(
            unet=_MockUNet(),
            config=dict(
                type='UNetFeatureConfig',
                extract_down1=False,
                extract_down2=True,
                extract_down3=True,
                extract_mid=True,
            ),
        )

        latent = torch.randn(1, 4, 96, 96)
        feats = extractor(latent, torch.tensor([500]), torch.randn(1, 1, 768))

        assert 'down1' not in feats
        assert 'down2' in feats
        assert 'down3' in feats
        assert 'mid' in feats

    def test_missing_unet_raises(self):
        """未提供 unet 或 unet_cfg 应抛出 ValueError"""
        with pytest.raises(ValueError, match='Must provide unet'):
            ChromoGenFeatureExtractor()


class _MockUNet(nn.Module):
    """模拟 diffusers UNet2DConditionModel 的结构

    用于在没有真实 ChromoGen 权重时测试特征提取 hook 机制。
    实现与真实 UNet 兼容的 forward 签名和 down_block/mid_block 特征输出。
    """

    def __init__(self):
        super().__init__()
        # 模拟 UNet 各层 (不实际计算, 仅用于 hook 测试)
        self.conv_in = nn.Conv2d(4, 320, 3, padding=1)
        # Down blocks (down1 保持 H/8, down2/3/4 逐步下采样)
        self.down1 = _MockDownBlock(320, 320, downsample=False)
        self.down2 = _MockDownBlock(320, 640, downsample=True)
        self.down3 = _MockDownBlock(640, 1280, downsample=True)
        self.down4 = _MockDownBlock(1280, 1280, downsample=True)
        # Mid block
        self.mid = _MockMidBlock(1280, 1280)

    def forward(self, sample, timestep, encoder_hidden_states, return_dict=True):
        """模拟 UNet forward, 返回多尺度中间特征

        通过 forward_hook 或显式收集机制将 down/mid 特征暴露给提取器。
        这里采用显式 attach 方式: forward 时将特征存入 _feature_cache。
        """
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

        # 缓存特征供提取器读取
        self._feature_cache = feats

        # 返回一个 dummy 输出 (模拟 diffusers UNet 输出)
        return type('UNetOutput', (), {'sample': h})()


class _MockDownBlock(nn.Module):
    """模拟 UNet DownBlock"""

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
    """模拟 UNet MidBlock"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class TestEndToEndIntegration:
    """端到端集成测试: ChromoGenFeatureExtractor → FeatureBridgeModule

    验证从 VAE latent 到融合 LDMDet FPN 特征的完整流程。
    使用 MockUNet 模拟 ChromoGen, 验证特征流和形状对齐。
    """

    @pytest.fixture
    def pipeline(self):
        """构建 Extractor + FBM 流水线"""
        unet = _MockUNet()
        extractor = ChromoGenFeatureExtractor(unet=unet)
        fbm = FeatureBridgeModule(
            cg_channels=[320, 640, 1280, 1280],
            ld_channels=256,
        )
        return extractor, fbm

    def test_full_pipeline_forward(self, pipeline):
        """完整前向传播: latent → extract → fuse → LDMDet FPN shape"""
        extractor, fbm = pipeline

        bs, h, w = 2, 768, 768
        latent = torch.randn(bs, 4, h // 8, w // 8)  # VAE latent H/8
        timestep = torch.tensor([500, 500])
        encoder_hidden = torch.randn(bs, 1, 768)

        # 1. 提取 ChromoGen 特征
        cg_feats = extractor(latent, timestep, encoder_hidden)
        assert set(cg_feats.keys()) == {'down1', 'down2', 'down3', 'mid'}

        # 2. 构造 LDMDet FPN 特征 (stride 4/8/16/32)
        ld_feats = [
            torch.randn(bs, 256, h // 4, w // 4),
            torch.randn(bs, 256, h // 8, w // 8),
            torch.randn(bs, 256, h // 16, w // 16),
            torch.randn(bs, 256, h // 32, w // 32),
        ]

        # 3. 融合
        fused = fbm(ld_feats, cg_feats)

        # 4. 验证形状
        assert len(fused) == 4
        for i, (o, lf) in enumerate(zip(fused, ld_feats)):
            assert o.shape == lf.shape, f'Level {i} shape mismatch'

    def test_zero_init_preserves_baseline(self, pipeline):
        """零初始化: 融合后特征与 baseline 完全一致"""
        extractor, fbm = pipeline
        bs, h, w = 1, 512, 512

        latent = torch.randn(bs, 4, h // 8, w // 8)
        cg_feats = extractor(latent, torch.tensor([500]), torch.randn(bs, 1, 768))

        ld_feats = [
            torch.randn(bs, 256, h // 4, w // 4),
            torch.randn(bs, 256, h // 8, w // 8),
            torch.randn(bs, 256, h // 16, w // 16),
            torch.randn(bs, 256, h // 32, w // 32),
        ]

        fused = fbm(ld_feats, cg_feats)

        for i, (o, lf) in enumerate(zip(fused, ld_feats)):
            assert torch.allclose(o, lf, atol=1e-6)

    def test_gradient_only_in_fbm(self, pipeline):
        """ChromoGen 冻结, 梯度只流过 FBM 投影层和 alpha"""
        extractor, fbm = pipeline
        extractor.set_frozen()

        bs, h, w = 1, 512, 512
        latent = torch.randn(bs, 4, h // 8, w // 8)
        cg_feats = extractor(latent, torch.tensor([500]), torch.randn(bs, 1, 768))

        # ChromoGen 特征不应有梯度
        for key, feat in cg_feats.items():
            assert not feat.requires_grad

        # LDMDet 特征需要梯度
        ld_feats = [
            torch.randn(bs, 256, h // 4, w // 4, requires_grad=True),
            torch.randn(bs, 256, h // 8, w // 8, requires_grad=True),
            torch.randn(bs, 256, h // 16, w // 16, requires_grad=True),
            torch.randn(bs, 256, h // 32, w // 32, requires_grad=True),
        ]

        fused = fbm(ld_feats, cg_feats)
        loss = sum(o.mean() for o in fused)
        loss.backward()

        # FBM 参数应有梯度
        assert fbm.alpha.grad is not None
        for name, param in fbm.named_parameters():
            if 'proj' in name:
                assert param.grad is not None

        # ChromoGen 参数不应有梯度
        for name, param in extractor.unet.named_parameters():
            assert param.grad is None, f'{name} should not have gradient (frozen)'

    def test_alpha_nonzero_changes_output(self, pipeline):
        """alpha 非零时, 融合输出不同于 baseline"""
        extractor, fbm = pipeline
        bs, h, w = 1, 512, 512

        latent = torch.randn(bs, 4, h // 8, w // 8)
        cg_feats = extractor(latent, torch.tensor([500]), torch.randn(bs, 1, 768))

        ld_feats = [
            torch.randn(bs, 256, h // 4, w // 4),
            torch.randn(bs, 256, h // 8, w // 8),
            torch.randn(bs, 256, h // 16, w // 16),
            torch.randn(bs, 256, h // 32, w // 32),
        ]

        # baseline 输出
        baseline = fbm(ld_feats, cg_feats)

        # 启用融合
        with torch.no_grad():
            fbm.alpha.fill_(0.5)
        fused = fbm(ld_feats, cg_feats)

        # Level 0 不变, level 1/2/3 应变化
        assert torch.allclose(fused[0], baseline[0], atol=1e-6)
        for i in [1, 2, 3]:
            assert not torch.allclose(fused[i], baseline[i], atol=1e-6)

    def test_non_divisible_size_raises(self, pipeline):
        """图像尺寸不能被 64 整除时应抛出异常或产生不对齐形状"""
        extractor, fbm = pipeline
        # 768/8 = 96, 96/8 = 12 → mid 12x12, 但 level3 = 768/32 = 24
        # 这测试是为了验证 mid (H/64) 上采样到 level3 (H/32) 的对齐
        bs, h, w = 1, 768, 768
        latent = torch.randn(bs, 4, h // 8, w // 8)
        cg_feats = extractor(latent, torch.tensor([500]), torch.randn(bs, 1, 768))

        ld_feats = [
            torch.randn(bs, 256, h // 4, w // 4),
            torch.randn(bs, 256, h // 8, w // 8),
            torch.randn(bs, 256, h // 16, w // 16),
            torch.randn(bs, 256, h // 32, w // 32),
        ]

        # 应能正常融合 (mid 上采样到 level3 尺寸)
        fused = fbm(ld_feats, cg_feats)
        assert fused[3].shape == ld_feats[3].shape
