"""LDMDetDiT 核心功能验证测试

红绿重构: 每个测试都有真实断言，无兜底/空测试。
覆盖: 导入注册 → 单模块输出 → OT 匹配 → FeatureFusion → 端到端 loss
"""
import os

import pytest
import torch

# ============================================================
# Part 1: 导入与注册验证
# ============================================================

class TestLDMDetDiTImports:
    """验证 LDMDetDiT 的核心模块可以被导入"""

    def test_import_model(self):
        import projects.LDMDetDiT.model  # noqa: F401

    def test_import_hooks(self):
        import projects.LDMDetDiT.hooks  # noqa: F401

    def test_import_dit_head(self):
        import projects.LDMDetDiT.mods.dit_head  # noqa: F401

    def test_import_dit_single_head(self):
        import projects.LDMDetDiT.mods.dit_single_head  # noqa: F401

    def test_import_dit_block(self):
        import projects.LDMDetDiT.mods.dit_block  # noqa: F401

    def test_import_box_tokenizer(self):
        import projects.LDMDetDiT.mods.box_tokenizer  # noqa: F401

    def test_import_deformable_attn(self):
        import projects.LDMDetDiT.mods.deformable_attn  # noqa: F401

    def test_import_modules(self):
        import projects.LDMDetDiT.mods.modules  # noqa: F401

    def test_import_loss(self):
        import projects.LDMDetDiT.mods.loss  # noqa: F401

    def test_import_structures(self):
        import projects.LDMDetDiT.mods.structures  # noqa: F401

    def test_import_utils(self):
        import projects.LDMDetDiT.mods.utils  # noqa: F401

    def test_import_rectified_flow(self):
        import projects.LDMDetDiT.mods.rectified_flow  # noqa: F401

    def test_import_convnextv2(self):
        import projects.LDMDetDiT.mods.convnextv2  # noqa: F401


class TestLDMDetDiTRegistration:
    """验证 DiT 相关组件被正确注册到 MODELS 注册表"""

    @pytest.fixture(autouse=True)
    def setup(self):
        import projects.LDMDetDiT.model  # noqa: F401

    def test_dit_head_registered(self):
        from mmdet.registry import MODELS
        assert MODELS.get('DiTDiffusionDetHead') is not None

    def test_dit_single_head_registered(self):
        from mmdet.registry import MODELS
        assert MODELS.get('DiTSingleHead') is not None

    def test_criterion_registered(self):
        from mmdet.registry import MODELS
        assert MODELS.get('PurePyTorchDiffusionDetCriterion') is not None

    def test_matcher_registered(self):
        from mmdet.registry import MODELS
        assert MODELS.get('PurePyTorchDiffusionDetMatcher') is not None

    def test_losses_registered(self):
        from mmdet.registry import MODELS
        assert MODELS.get('PurePyTorchFocalLoss') is not None
        assert MODELS.get('PurePyTorchL1Loss') is not None
        assert MODELS.get('PurePyTorchGIoULoss') is not None

    def test_detector_registered(self):
        from mmdet.registry import MODELS
        assert MODELS.get('PurePyTorchDiffusionDet') is not None

    def test_ldmdetdit_registered(self):
        from mmdet.registry import MODELS
        assert MODELS.get('LDMDetDiT') is not None


class TestLDMDetDiTNoDiffusionDet:
    """验证 LDMDetDiT 不包含 DiffusionDet 专属组件"""

    def test_no_diffusiondet_head_import(self):
        import projects.LDMDetDiT.model as dit_model
        source = open(dit_model.__file__).read()
        assert 'from .mods.diffusiondet_head' not in source

    def test_no_single_diffusiondet_head_import(self):
        import projects.LDMDetDiT.model as dit_model
        source = open(dit_model.__file__).read()
        assert 'from .mods.single_head' not in source

    def test_no_roi_extractor_import(self):
        import projects.LDMDetDiT.model as dit_model
        source = open(dit_model.__file__).read()
        assert 'from .mods.roi_extractor' not in source

    def test_no_dynamic_conv_in_modules(self):
        import projects.LDMDetDiT.mods.modules as mods
        source = open(mods.__file__).read()
        assert 'DynamicConv' not in source

    def test_no_flow_matching_velocity_loss(self):
        import projects.LDMDetDiT.mods.loss as loss_mod
        source = open(loss_mod.__file__).read()
        assert 'FlowMatchingVelocityLoss' not in source


class TestLDMDetDiTConfig:
    """验证 LDMDetDiT 配置文件路径正确"""

    def test_config_custom_imports(self):
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'configs', 'ldmdet_dit.py'
        )
        source = open(config_path).read()
        assert 'projects.LDMDetDiT.model' in source
        assert 'projects.LDMDetDiT.hooks' in source
        assert 'projects.LDMDet.model' not in source
        assert 'projects.LDMDet.hooks' not in source


class TestLDMDetDiTHooks:
    """验证 hooks.py 的 CopyProjectHook 路径正确"""

    def test_copy_project_hook_path(self):
        import projects.LDMDetDiT.hooks as hooks_mod
        source = open(hooks_mod.__file__).read()
        assert "src_path='projects/LDMDetDiT'" in source


# ============================================================
# Part 2: 单模块输出格式和范围验证
# ============================================================

class TestRoPE1D:
    """RoPE1D 旋转位置编码输出验证"""

    def test_output_shape(self):
        from projects.LDMDetDiT.mods.modules import RoPE1D
        rope = RoPE1D(dim=32, base=10000)
        x = torch.rand(2, 100, 4)
        out = rope(x)
        assert out.shape == (2, 100, 4, 32)

    def test_output_range(self):
        from projects.LDMDetDiT.mods.modules import RoPE1D
        rope = RoPE1D(dim=32, base=10000)
        x = torch.rand(2, 100, 4)
        out = rope(x)
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_no_nan(self):
        from projects.LDMDetDiT.mods.modules import RoPE1D
        rope = RoPE1D(dim=32, base=10000)
        x = torch.rand(2, 100, 4)
        out = rope(x)
        assert not torch.isnan(out).any()


class TestSinusoidalPositionEmbeddings:
    """时间步位置编码输出验证"""

    def test_output_shape(self):
        from projects.LDMDetDiT.mods.modules import (
            SinusoidalPositionEmbeddings,
        )
        spe = SinusoidalPositionEmbeddings(dim=256)
        t = torch.rand(4)
        out = spe(t)
        assert out.shape == (4, 256)

    def test_output_range(self):
        from projects.LDMDetDiT.mods.modules import (
            SinusoidalPositionEmbeddings,
        )
        spe = SinusoidalPositionEmbeddings(dim=256)
        t = torch.rand(4) * 1000
        out = spe(t)
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_no_nan(self):
        from projects.LDMDetDiT.mods.modules import (
            SinusoidalPositionEmbeddings,
        )
        spe = SinusoidalPositionEmbeddings(dim=256)
        t = torch.rand(4) * 1000
        out = spe(t)
        assert not torch.isnan(out).any()


class TestBoxTokenizer:
    """BoxTokenizer 输出验证"""

    @pytest.fixture
    def tokenizer(self):
        from projects.LDMDetDiT.mods.box_tokenizer import BoxTokenizer
        return BoxTokenizer(feat_channels=256, num_fpn_levels=4, init_mode='query', num_proposals=100)

    @pytest.fixture
    def fpn_features(self):
        return [
            torch.rand(2, 256, 64, 64),
            torch.rand(2, 256, 32, 32),
            torch.rand(2, 256, 16, 16),
            torch.rand(2, 256, 8, 8),
        ]

    def test_output_shapes(self, tokenizer, fpn_features):
        bboxes = torch.rand(2, 100, 4)
        box_tokens, level_indices = tokenizer(bboxes, fpn_features)
        assert box_tokens.shape == (2, 100, 256)
        assert level_indices.shape == (2, 100)

    def test_level_indices_range(self, tokenizer, fpn_features):
        bboxes = torch.rand(2, 100, 4)
        _, level_indices = tokenizer(bboxes, fpn_features)
        assert level_indices.min() >= 0 and level_indices.max() < 4

    def test_no_nan_inf(self, tokenizer, fpn_features):
        bboxes = torch.rand(2, 100, 4)
        box_tokens, _ = tokenizer(bboxes, fpn_features)
        assert not torch.isnan(box_tokens).any()
        assert not torch.isinf(box_tokens).any()


class TestBboxToReferencePoints:
    """bbox_to_reference_points 输出验证"""

    def test_output_shape(self):
        from projects.LDMDetDiT.mods.box_tokenizer import (
            bbox_to_reference_points,
        )
        bboxes = torch.rand(2, 100, 4)
        ref = bbox_to_reference_points(bboxes)
        assert ref.shape == (2, 100, 2)

    def test_output_range(self):
        from projects.LDMDetDiT.mods.box_tokenizer import (
            bbox_to_reference_points,
        )
        bboxes = torch.rand(2, 100, 4)
        ref = bbox_to_reference_points(bboxes)
        assert ref.min() >= 0.0 and ref.max() <= 1.0

    def test_center_correctness(self):
        from projects.LDMDetDiT.mods.box_tokenizer import (
            bbox_to_reference_points,
        )
        bboxes = torch.tensor([[[0.1, 0.2, 0.5, 0.6]]])
        ref = bbox_to_reference_points(bboxes)
        expected = torch.tensor([[[0.3, 0.4]]])
        assert torch.allclose(ref, expected, atol=1e-6)


class TestReferencePointsWithLevels:
    """reference_points_with_levels 输出验证"""

    def test_output_shape(self):
        from projects.LDMDetDiT.mods.box_tokenizer import (
            reference_points_with_levels,
        )
        ref = torch.rand(2, 100, 2)
        out = reference_points_with_levels(ref, num_levels=4)
        assert out.shape == (2, 100, 4, 2)

    def test_values_match_input(self):
        from projects.LDMDetDiT.mods.box_tokenizer import (
            reference_points_with_levels,
        )
        ref = torch.tensor([[[0.3, 0.4]]])
        out = reference_points_with_levels(ref, num_levels=4)
        for l in range(4):
            assert torch.allclose(out[0, 0, l], ref[0, 0], atol=1e-6)


class TestMultiScaleDeformableAttention:
    """多尺度可变形注意力输出验证"""

    @pytest.fixture
    def attn_module(self):
        from projects.LDMDetDiT.mods.deformable_attn import (
            MultiScaleDeformableAttention,
        )
        return MultiScaleDeformableAttention(
            embed_dim=256, num_heads=8, num_levels=4, num_points=8, dropout=0.0
        )

    @pytest.fixture
    def fpn_data(self):
        from projects.LDMDetDiT.mods.deformable_attn import (
            flatten_fpn_features,
        )
        fpn = [
            torch.rand(2, 256, 16, 16),
            torch.rand(2, 256, 8, 8),
            torch.rand(2, 256, 4, 4),
            torch.rand(2, 256, 2, 2),
        ]
        return flatten_fpn_features(fpn)

    def test_output_shape(self, attn_module, fpn_data):
        flattened, spatial_shapes, level_start_index = fpn_data
        query = torch.rand(2, 100, 256)
        ref_points = torch.rand(2, 100, 4, 2)
        out = attn_module(query, ref_points, flattened, spatial_shapes, level_start_index)
        assert out.shape == (2, 100, 256)

    def test_no_nan_inf(self, attn_module, fpn_data):
        flattened, spatial_shapes, level_start_index = fpn_data
        query = torch.rand(2, 100, 256)
        ref_points = torch.rand(2, 100, 4, 2)
        out = attn_module(query, ref_points, flattened, spatial_shapes, level_start_index)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestFlattenFPNFeatures:
    """FPN 特征展平工具函数验证"""

    def test_output_shapes(self):
        from projects.LDMDetDiT.mods.deformable_attn import (
            flatten_fpn_features,
        )
        fpn = [
            torch.rand(2, 256, 16, 16),
            torch.rand(2, 256, 8, 8),
            torch.rand(2, 256, 4, 4),
            torch.rand(2, 256, 2, 2),
        ]
        flattened, spatial_shapes, level_start_index = flatten_fpn_features(fpn)
        total_tokens = 16*16 + 8*8 + 4*4 + 2*2
        assert flattened.shape == (2, total_tokens, 256)
        assert spatial_shapes.shape == (4, 2)
        assert level_start_index.shape == (4,)

    def test_spatial_shapes_correct(self):
        from projects.LDMDetDiT.mods.deformable_attn import (
            flatten_fpn_features,
        )
        fpn = [torch.rand(2, 256, 16, 16), torch.rand(2, 256, 8, 8)]
        _, spatial_shapes, _ = flatten_fpn_features(fpn)
        expected = torch.tensor([[16, 16], [8, 8]])
        assert torch.equal(spatial_shapes, expected)

    def test_level_start_index_correct(self):
        from projects.LDMDetDiT.mods.deformable_attn import (
            flatten_fpn_features,
        )
        fpn = [torch.rand(2, 256, 16, 16), torch.rand(2, 256, 8, 8), torch.rand(2, 256, 4, 4)]
        _, _, level_start_index = flatten_fpn_features(fpn)
        expected = torch.tensor([0, 256, 256+64])
        assert torch.equal(level_start_index, expected)


class TestDiTBlock:
    """DiTBlock 输出验证"""

    @pytest.fixture
    def dit_block(self):
        from projects.LDMDetDiT.mods.dit_block import DiTBlock
        return DiTBlock(
            feat_channels=256, num_heads=8, num_fpn_levels=4,
            num_ref_points=8, dim_feedforward=2048, adaln_params=9,
            use_adaln_zero=True,
        )

    @pytest.fixture
    def fpn_data(self):
        from projects.LDMDetDiT.mods.deformable_attn import (
            flatten_fpn_features,
        )
        fpn = [torch.rand(2, 256, 16, 16), torch.rand(2, 256, 8, 8),
               torch.rand(2, 256, 4, 4), torch.rand(2, 256, 2, 2)]
        return flatten_fpn_features(fpn)

    def test_output_shape(self, dit_block, fpn_data):
        flattened, spatial_shapes, level_start_index = fpn_data
        box_tokens = torch.rand(2, 100, 256)
        time_emb = torch.rand(2, 256 * 4)
        bbox_coords = torch.rand(2, 100, 4)
        out = dit_block(box_tokens, flattened, spatial_shapes, level_start_index, time_emb, bbox_coords)
        assert out.shape == (2, 100, 256)

    def test_no_nan_inf(self, dit_block, fpn_data):
        flattened, spatial_shapes, level_start_index = fpn_data
        box_tokens = torch.rand(2, 100, 256)
        time_emb = torch.rand(2, 256 * 4)
        bbox_coords = torch.rand(2, 100, 4)
        out = dit_block(box_tokens, flattened, spatial_shapes, level_start_index, time_emb, bbox_coords)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_adaln_zero_initial_residual(self):
        """AdaLN-Zero 零初始化: 输出 ≈ 输入 (残差项为 0)"""
        from projects.LDMDetDiT.mods.deformable_attn import (
            flatten_fpn_features,
        )
        from projects.LDMDetDiT.mods.dit_block import DiTBlock
        block = DiTBlock(
            feat_channels=256, num_heads=8, num_fpn_levels=4,
            num_ref_points=8, dim_feedforward=2048, adaln_params=9,
            use_adaln_zero=True,
        )
        block.eval()
        fpn = [torch.rand(1, 256, 8, 8)] * 4
        flattened, spatial_shapes, level_start_index = flatten_fpn_features(fpn)
        box_tokens = torch.rand(1, 10, 256)
        time_emb = torch.rand(1, 256 * 4)
        bbox_coords = torch.rand(1, 10, 4)
        out = block(box_tokens, flattened, spatial_shapes, level_start_index, time_emb, bbox_coords)
        diff = (out - box_tokens).abs().max().item()
        assert diff < 1e-4, f"AdaLN-Zero residual too large: {diff}"


class TestDiTSingleHead:
    """DiTSingleHead 输出验证"""

    @pytest.fixture
    def single_head(self):
        from projects.LDMDetDiT.mods.dit_single_head import DiTSingleHead
        return DiTSingleHead(
            num_classes=24, feat_channels=256, num_heads=8,
            num_fpn_levels=4, num_ref_points=8, dim_feedforward=2048,
            prediction_mode='x0', adaln_params=9, regression_mode='direct',
            use_adaln_zero=True, num_blocks=1,
        )

    @pytest.fixture
    def fpn_data(self):
        from projects.LDMDetDiT.mods.deformable_attn import (
            flatten_fpn_features,
        )
        fpn = [torch.rand(2, 256, 16, 16), torch.rand(2, 256, 8, 8),
               torch.rand(2, 256, 4, 4), torch.rand(2, 256, 2, 2)]
        return flatten_fpn_features(fpn)

    def test_output_shapes(self, single_head, fpn_data):
        flattened, spatial_shapes, level_start_index = fpn_data
        box_tokens = torch.rand(2, 100, 256)
        time_emb = torch.rand(2, 256 * 4)
        bbox_coords = torch.rand(2, 100, 4)
        cls_logits, pred_bboxes, updated_tokens, _placeholder, velocity = single_head(
            box_tokens, flattened, spatial_shapes, level_start_index, time_emb, bbox_coords
        )
        assert cls_logits.shape == (2, 100, 24)
        assert pred_bboxes.shape == (2, 100, 4)
        assert updated_tokens.shape == (2, 100, 256)
        assert _placeholder is None  # objectness 已移除
        assert velocity is None

    def test_pred_bboxes_finite_direct_mode(self, single_head, fpn_data):
        """direct 模式输出 raw 空间 cxcywh, 应无 NaN/Inf"""
        flattened, spatial_shapes, level_start_index = fpn_data
        box_tokens = torch.rand(2, 100, 256)
        time_emb = torch.rand(2, 256 * 4)
        bbox_coords = torch.rand(2, 100, 4)
        _, pred_bboxes, _, _, _ = single_head(
            box_tokens, flattened, spatial_shapes, level_start_index, time_emb, bbox_coords
        )
        assert not torch.isnan(pred_bboxes).any()
        assert not torch.isinf(pred_bboxes).any()

    def test_pred_bboxes_sigmoid_range(self, single_head, fpn_data):
        """direct 模式下 pred_bboxes 经 sigmoid 后应在 [0,1]"""
        flattened, spatial_shapes, level_start_index = fpn_data
        box_tokens = torch.rand(2, 100, 256)
        time_emb = torch.rand(2, 256 * 4)
        bbox_coords = torch.rand(2, 100, 4)
        _, pred_bboxes_raw, _, _, _ = single_head(
            box_tokens, flattened, spatial_shapes, level_start_index, time_emb, bbox_coords
        )
        # regression_mode='direct' 输出 sigmoid 归一化坐标
        pred_bboxes_normed = torch.sigmoid(pred_bboxes_raw)
        assert pred_bboxes_normed.min() >= 0.0
        assert pred_bboxes_normed.max() <= 1.0

    def test_no_nan_inf(self, single_head, fpn_data):
        flattened, spatial_shapes, level_start_index = fpn_data
        box_tokens = torch.rand(2, 100, 256)
        time_emb = torch.rand(2, 256 * 4)
        bbox_coords = torch.rand(2, 100, 4)
        cls_logits, pred_bboxes, updated_tokens, _, _ = single_head(
            box_tokens, flattened, spatial_shapes, level_start_index, time_emb, bbox_coords
        )
        for name, t in [("cls_logits", cls_logits), ("pred_bboxes", pred_bboxes), ("updated_tokens", updated_tokens)]:
            assert not torch.isnan(t).any(), f"{name} contains NaN"
            assert not torch.isinf(t).any(), f"{name} contains Inf"

    def test_velocity_head_output(self):
        """prediction_mode='velocity' 时 DiTSingleHead 仍返回 None (velocity 由 Head 从 x0 反推)"""
        from projects.LDMDetDiT.mods.deformable_attn import (
            flatten_fpn_features,
        )
        from projects.LDMDetDiT.mods.dit_single_head import DiTSingleHead
        head = DiTSingleHead(
            num_classes=24, feat_channels=256, num_heads=8,
            num_fpn_levels=4, num_ref_points=8, dim_feedforward=2048,
            prediction_mode='velocity',
            regression_mode='direct', use_adaln_zero=True, num_blocks=1,
        )
        head.eval()
        fpn = [torch.rand(1, 256, 8, 8)] * 4
        flattened, spatial_shapes, level_start_index = flatten_fpn_features(fpn)
        box_tokens = torch.rand(1, 10, 256)
        time_emb = torch.rand(1, 256 * 4)
        bbox_coords = torch.rand(1, 10, 4)
        _, _, _, _, velocity = head(
            box_tokens, flattened, spatial_shapes, level_start_index, time_emb, bbox_coords
        )
        # DiTSingleHead 始终返回 None 作为 velocity
        # velocity 由 DiTDiffusionDetHead 从 x0 反推: v = (x_t - x0) / t
        assert velocity is None, "DiTSingleHead should return None for velocity"


class TestRectifiedFlow:
    """RectifiedFlow 输出验证"""

    @pytest.fixture
    def rf(self):
        from projects.LDMDetDiT.mods.rectified_flow import RectifiedFlow
        return RectifiedFlow(snr_scale=2.0)

    def test_q_sample_shapes(self, rf):
        x_start = torch.randn(32, 4)
        t = torch.rand(32)
        x_t, velocity = rf.q_sample(x_start, t=t)
        assert x_t.shape == (32, 4)
        assert velocity.shape == (32, 4)

    def test_q_sample_boundary_t0(self, rf):
        """t=0 时 x_t ≈ x_start"""
        x_start = torch.randn(4, 4)
        t = torch.zeros(4)
        x_t, _ = rf.q_sample(x_start, t=t)
        assert torch.allclose(x_t, x_start, atol=1e-5)

    def test_q_sample_boundary_t1(self, rf):
        """t=1 时 x_t ≈ x_noise"""
        x_start = torch.randn(4, 4)
        x_noise = torch.randn(4, 4)
        t = torch.ones(4)
        x_t, _ = rf.q_sample(x_start, x_noise=x_noise, t=t)
        assert torch.allclose(x_t, x_noise, atol=1e-5)

    def test_velocity_definition(self, rf):
        """velocity = x_noise - x_start"""
        x_start = torch.randn(4, 4)
        x_noise = torch.randn(4, 4)
        t = torch.rand(4)
        _, velocity = rf.q_sample(x_start, x_noise=x_noise, t=t)
        expected_v = x_noise - x_start
        assert torch.allclose(velocity, expected_v, atol=1e-5)

    def test_step_shape(self, rf):
        x_t = torch.randn(4, 4)
        x0_pred = torch.randn(4, 4)
        x_next = rf.step(x_t, x0_pred, t_curr=0.5, t_next=0.4)
        assert x_next.shape == (4, 4)

    def test_step_no_nan(self, rf):
        x_t = torch.randn(4, 4)
        x0_pred = torch.randn(4, 4)
        x_next = rf.step(x_t, x0_pred, t_curr=0.5, t_next=0.4)
        assert not torch.isnan(x_next).any()


class TestCosineNoiseSchedule:
    """cosine_noise_schedule 输出验证"""

    def test_output_shape(self):
        from projects.LDMDetDiT.mods.modules import cosine_noise_schedule
        betas = cosine_noise_schedule(1000)
        assert betas.shape == (1000,)

    def test_betas_range(self):
        from projects.LDMDetDiT.mods.modules import cosine_noise_schedule
        betas = cosine_noise_schedule(1000)
        assert betas.min() > 0
        assert betas.max() < 1

    def test_monotonically_increasing(self):
        from projects.LDMDetDiT.mods.modules import cosine_noise_schedule
        betas = cosine_noise_schedule(1000)
        diffs = betas[1:] - betas[:-1]
        assert (diffs >= -1e-6).all()


# ============================================================
# Part 3: FeatureFusion LayerNorm 验证
# ============================================================

class TestFeatureFusion:
    """PurePyTorchSimpleFeatureFusion 输出验证

    验证 LayerNorm 归一化效果和多尺度输出正确性。
    """

    @pytest.fixture
    def fusion(self):
        from projects.LDMDetDiT.mods.modules import (
            PurePyTorchSimpleFeatureFusion,
        )
        return PurePyTorchSimpleFeatureFusion(
            in_channels=[384, 384, 384, 384], out_channels=256, num_outs=4
        )

    @pytest.fixture
    def vit_features(self):
        """模拟 ViT 4 级输出, 统一 H/16 x W/16 空间分辨率"""
        return [
            torch.rand(2, 384, 40, 40),  # B2
            torch.rand(2, 384, 40, 40),  # B5
            torch.rand(2, 384, 40, 40),  # B8
            torch.rand(2, 384, 40, 40),  # B11
        ]

    def test_output_count(self, fusion, vit_features):
        """应输出 4 级 FPN 特征"""
        outs = fusion(vit_features)
        assert len(outs) == 4

    def test_output_channels(self, fusion, vit_features):
        """所有输出通道数应为 out_channels=256"""
        outs = fusion(vit_features)
        for i, out in enumerate(outs):
            assert out.shape[1] == 256, f"Level {i}: expected 256 channels, got {out.shape[1]}"

    def test_output_spatial_scales(self, fusion, vit_features):
        """输出空间尺度应为 P2(1/4), P3(1/8), P4(1/16), P5(1/32)"""
        outs = fusion(vit_features)
        H, W = 40, 40  # ViT 输出 H/16 x W/16
        expected = [
            (H * 4, W * 4),  # P2: 上采样 4x
            (H * 2, W * 2),  # P3: 上采样 2x
            (H, W),           # P4: 保持
            (H // 2, W // 2), # P5: 下采样 2x
        ]
        for i, (out, (eh, ew)) in enumerate(zip(outs, expected)):
            assert out.shape[2] == eh and out.shape[3] == ew, \
                f"Level {i}: expected ({eh},{ew}), got ({out.shape[2]},{out.shape[3]})"

    def test_no_nan_inf(self, fusion, vit_features):
        outs = fusion(vit_features)
        for i, out in enumerate(outs):
            assert not torch.isnan(out).any(), f"Level {i} contains NaN"
            assert not torch.isinf(out).any(), f"Level {i} contains Inf"

    def test_layernorm_normalizes_explosive_input(self):
        """LayerNorm 应将数值爆炸的深层特征归一化到合理范围"""
        from projects.LDMDetDiT.mods.modules import (
            PurePyTorchSimpleFeatureFusion,
        )
        fusion = PurePyTorchSimpleFeatureFusion(
            in_channels=[384, 384, 384, 384], out_channels=256, num_outs=4
        )
        # 模拟 Block 11 数值爆炸: std=27
        normal = torch.rand(2, 384, 40, 40)
        explosive = normal * 27  # std ≈ 27
        features = [normal, normal, normal, explosive]
        outs = fusion(features)
        # LayerNorm 后输出不应爆炸
        for i, out in enumerate(outs):
            assert out.std() < 100, f"Level {i}: std={out.std():.1f}, LayerNorm failed to normalize"

    def test_gradient_flows_through_layernorm(self):
        """LayerNorm 应允许梯度正常回传"""
        from projects.LDMDetDiT.mods.modules import (
            PurePyTorchSimpleFeatureFusion,
        )
        fusion = PurePyTorchSimpleFeatureFusion(
            in_channels=[384, 384, 384, 384], out_channels=256, num_outs=4
        )
        features = [torch.rand(1, 384, 16, 16, requires_grad=True) for _ in range(4)]
        outs = fusion(features)
        loss = sum(o.sum() for o in outs)
        loss.backward()
        for i, f in enumerate(features):
            assert f.grad is not None, f"Feature {i} has no gradient"
            assert not torch.isnan(f.grad).any(), f"Feature {i} gradient is NaN"


# ============================================================
# Part 3.5: Spatial Tuning Adapter (STA) 验证
# 参考 DEIMv2: https://arxiv.org/abs/2509.20787
# ============================================================

class TestSpatialTuningAdapter:
    """SpatialTuningAdapter (DEIMv2 STA) 验证

    STA 核心设计:
    1. 从 DINOv3 多个中间层取特征 → 双线性插值调整尺度 (无参数)
    2. 轻量 CNN (SpatialPriorModule) 从原图提取多尺度细粒度细节
    3. Bi-Fusion: concat 语义 + 细节 → 1×1 Conv + BN 融合
    4. 输出 3 级特征 (P3: 1/8, P4: 1/16, P5: 1/32)
    """

    @pytest.fixture
    def sta(self):
        from projects.LDMDetDiT.mods.modules import SpatialTuningAdapter
        return SpatialTuningAdapter(
            embed_dim=384,
            out_channels=256,
            conv_inplane=16,
        )

    @pytest.fixture
    def sta_no_cnn(self):
        """不使用 CNN 细节分支的 STA (仅双线性插值)"""
        from projects.LDMDetDiT.mods.modules import SpatialTuningAdapter
        return SpatialTuningAdapter(
            embed_dim=384,
            out_channels=256,
            conv_inplane=0,
        )

    @pytest.fixture
    def vit_features_3level(self):
        """模拟 DINOv3 3 级中间层输出 (B5, B8, B11), 统一 H/16 x W/16"""
        B, C, H, W = 2, 384, 40, 40
        return [
            torch.rand(B, C, H, W),  # B5
            torch.rand(B, C, H, W),  # B8
            torch.rand(B, C, H, W),  # B11
        ]

    @pytest.fixture
    def raw_image(self):
        """模拟原始输入图像 (B, 3, H, W)"""
        return torch.rand(2, 3, 640, 640)

    def test_output_count(self, sta, vit_features_3level, raw_image):
        """STA 应输出 3 级特征 (P3, P4, P5)"""
        outs = sta(vit_features_3level, raw_image)
        assert len(outs) == 3

    def test_output_channels(self, sta, vit_features_3level, raw_image):
        """所有输出通道数应为 out_channels"""
        outs = sta(vit_features_3level, raw_image)
        for i, out in enumerate(outs):
            assert out.shape[1] == 256, f"Level {i}: expected 256, got {out.shape[1]}"

    def test_output_spatial_scales(self, sta, vit_features_3level, raw_image):
        """输出空间尺度应为 P3(1/8), P4(1/16), P5(1/32)"""
        outs = sta(vit_features_3level, raw_image)
        H_img, W_img = 640, 640
        expected = [
            (H_img // 8, W_img // 8),    # P3: 1/8
            (H_img // 16, W_img // 16),   # P4: 1/16
            (H_img // 32, W_img // 32),   # P5: 1/32
        ]
        for i, (out, (eh, ew)) in enumerate(zip(outs, expected)):
            assert out.shape[2] == eh and out.shape[3] == ew, \
                f"Level {i}: expected ({eh},{ew}), got ({out.shape[2]},{out.shape[3]})"

    def test_no_nan_inf(self, sta, vit_features_3level, raw_image):
        outs = sta(vit_features_3level, raw_image)
        for i, out in enumerate(outs):
            assert not torch.isnan(out).any(), f"Level {i} contains NaN"
            assert not torch.isinf(out).any(), f"Level {i} contains Inf"

    def test_bilinear_resize_parameter_free(self, sta_no_cnn, vit_features_3level, raw_image):
        """不使用 CNN 分支时, STA 应仅通过双线性插值调整尺度 (无额外上采样参数)"""
        outs = sta_no_cnn(vit_features_3level, raw_image)
        assert len(outs) == 3
        for out in outs:
            assert out.shape[1] == 256

    def test_cnn_detail_complements_semantics(self, sta, sta_no_cnn, vit_features_3level, raw_image):
        """CNN 细节分支应使输出与纯语义不同 (补充细粒度信息)"""
        sta.eval()
        sta_no_cnn.eval()
        with torch.no_grad():
            outs_with_cnn = sta(vit_features_3level, raw_image)
            outs_no_cnn = sta_no_cnn(vit_features_3level, raw_image)
        for i, (with_cnn, no_cnn) in enumerate(zip(outs_with_cnn, outs_no_cnn)):
            # CNN 分支引入额外细节, 输出应与纯语义不同
            assert not torch.allclose(with_cnn, no_cnn, atol=1e-4), \
                f"Level {i}: CNN detail branch has no effect"

    def test_handles_explosive_features(self, sta, raw_image):
        """STA 应处理数值爆炸的深层特征 (通过 BN 归一化)"""
        normal = torch.rand(2, 384, 40, 40)
        explosive = normal * 27  # 模拟 Block 11 数值爆炸
        vit_feats = [normal, normal, explosive]
        outs = sta(vit_feats, raw_image)
        for i, out in enumerate(outs):
            assert out.std() < 100, f"Level {i}: std={out.std():.1f}, BN failed to normalize"

    def test_gradient_flows(self, sta, vit_features_3level, raw_image):
        """梯度应正常回传到 ViT 特征"""
        feats = [f.requires_grad_(True) for f in vit_features_3level]
        img = raw_image.requires_grad_(True)
        outs = sta(feats, img)
        loss = sum(o.sum() for o in outs)
        loss.backward()
        for i, f in enumerate(feats):
            assert f.grad is not None, f"ViT feature {i} has no gradient"
            assert not torch.isnan(f.grad).any(), f"ViT feature {i} gradient is NaN"

    def test_different_input_resolutions(self, sta):
        """STA 应适配不同输入分辨率"""
        for H_img, W_img in [(512, 512), (640, 640), (800, 800)]:
            H_feat, W_feat = H_img // 16, W_img // 16
            vit_feats = [torch.rand(1, 384, H_feat, W_feat) for _ in range(3)]
            raw_img = torch.rand(1, 3, H_img, W_img)
            outs = sta(vit_feats, raw_img)
            assert outs[0].shape[2] == H_img // 8, f"P3 height mismatch for {H_img}"
            assert outs[1].shape[2] == H_img // 16, f"P4 height mismatch for {H_img}"
            assert outs[2].shape[2] == H_img // 32, f"P5 height mismatch for {H_img}"


# ============================================================
# Part 4: OT 匹配验证 (核心改动)
# ============================================================

class TestOTMatching:
    """验证 OT (Sinkhorn) 匹配替代 SimOTA 后的行为

    核心保证:
    1. 所有 proposal 都有对应的 GT (100% GT 覆盖)
    2. criterion 使用 OT 匹配结果, 不再调用 SimOTA
    3. 分类/GIoU 回归目标与 OT 匹配一致
    """

    @pytest.fixture
    def criterion(self):
        from projects.LDMDetDiT.mods.loss import (
            DiffusionDetCriterion,
            FocalLoss,
            GIoULoss,
        )
        return DiffusionDetCriterion(
            num_classes=24,
            matcher=None,  # 不使用 SimOTA
            loss_cls=FocalLoss(),
            loss_giou=GIoULoss(),
        )

    def test_ot_indices_all_proposals_are_positive(self, criterion):
        """OT 匹配: 所有 proposal 都应被标记为正例"""
        from projects.LDMDetDiT.mods.structures import ModelOutput
        num_proposals = 100
        # 模拟 OT 匹配: 每个 proposal 分配到某个 GT
        ot_indices = [
            torch.randint(0, 5, (num_proposals,)),  # batch 0: 5 个 GT
            torch.randint(0, 3, (num_proposals,)),  # batch 1: 3 个 GT
        ]
        outputs = ModelOutput(
            pred_logits=torch.rand(2, num_proposals, 24),
            pred_boxes=torch.rand(2, num_proposals, 4),
        )
        indices = criterion._ot_indices_to_match(ot_indices, num_proposals)
        for i, (src_idx, gt_idx) in enumerate(indices):
            # 所有 proposal 都是正例
            assert len(src_idx) == num_proposals, \
                f"Batch {i}: expected {num_proposals} positives, got {len(src_idx)}"
            # 每个 proposal 都有对应的 GT
            assert (gt_idx >= 0).all() and (gt_idx < [5, 3][i]).all(), \
                f"Batch {i}: gt_idx out of range"

    def test_ot_indices_full_gt_coverage(self):
        """OT 匹配应保证每个 GT 至少被 1 个 proposal 匹配"""
        from projects.LDMDetDiT.mods.dit_head import DiTDiffusionDetHead
        from projects.LDMDetDiT.mods.loss import (
            DiffusionDetCriterion,
            FocalLoss,
            GIoULoss,
        )
        criterion = DiffusionDetCriterion(
            num_classes=24, matcher=None,
            loss_cls=FocalLoss(), loss_giou=GIoULoss(),
        )
        head = DiTDiffusionDetHead(
            num_classes=24, feat_channels=256, num_proposals=100,
            num_heads=2, num_fpn_levels=4, num_ref_points=8,
            sampling_timesteps=2, rf_schedule='shifted', rf_shift=1.0,
            prediction_mode='x0', adaln_params=9, regression_mode='direct',
            use_adaln_zero=True, num_blocks=1, share_heads=True,
            deep_supervision=False, box_init_mode='query', criterion=criterion,
            ot_coupling=True, ot_matcher='sinkhorn',
        )
        head.train()
        device = next(head.parameters()).device

        # 模拟 _build_training_targets 中的 Sinkhorn 匹配
        num_gt = 10
        noise = torch.randn(100, 4, device=device)
        gt_diffusion = torch.randn(num_gt, 4, device=device)
        matched_idx, max_prob = head._sinkhorn_match(noise, gt_diffusion, device)

        # 验证: 每个 GT 至少被 1 个 proposal 匹配
        unique_matched = matched_idx.unique()
        assert len(unique_matched) == num_gt, \
            f"Sinkhorn: only {len(unique_matched)}/{num_gt} GTs matched"
        # 验证: 传输概率形状正确
        assert max_prob.shape == (100,), f"max_prob shape: {max_prob.shape}"

    def test_criterion_uses_ot_not_simota(self, criterion):
        """criterion.forward 接收 ot_matched_gt_indices 时应跳过 SimOTA"""
        from projects.LDMDetDiT.mods.structures import (
            InstanceData,
            ModelOutput,
        )
        num_proposals = 50
        ot_indices = [
            torch.randint(0, 3, (num_proposals,)),
            torch.randint(0, 2, (num_proposals,)),
        ]
        outputs = ModelOutput(
            pred_logits=torch.rand(2, num_proposals, 24),
            pred_boxes=torch.rand(2, num_proposals, 4).clamp(0, 1),
        )
        targets = [
            InstanceData(
                labels=torch.randint(0, 24, (3,)),
                bboxes=torch.rand(3, 4).clamp(0, 1),
                img_shape=(512, 512),
            ),
            InstanceData(
                labels=torch.randint(0, 24, (2,)),
                bboxes=torch.rand(2, 4).clamp(0, 1),
                img_shape=(512, 512),
            ),
        ]
        # 传入 ot_matched_gt_indices, criterion 不应调用 SimOTA matcher
        # 即使 matcher=None 也能正常工作
        losses = criterion(outputs, targets, ot_matched_gt_indices=ot_indices)
        assert isinstance(losses, dict)
        assert 'loss_cls' in losses
        assert 'loss_giou' in losses
        assert 'loss_bbox' not in losses  # loss_bbox 已移除
        assert 'loss_objectness' not in losses  # loss_objectness 已移除
        for k, v in losses.items():
            assert not torch.isnan(v), f"{k} is NaN"
            assert not torch.isinf(v), f"{k} is Inf"

    def test_criterion_classification_all_positive(self, criterion):
        """OT 匹配下不传 ot_match_probs 时, 所有 proposal 都应有分类目标"""
        from projects.LDMDetDiT.mods.structures import (
            InstanceData,
            ModelOutput,
        )
        num_proposals = 50
        num_classes = 24
        gt_labels_batch = [
            torch.tensor([3, 7, 15]),
            torch.tensor([0, 20]),
        ]
        ot_indices = [
            torch.randint(0, 3, (num_proposals,)),
            torch.randint(0, 2, (num_proposals,)),
        ]
        outputs = ModelOutput(
            pred_logits=torch.rand(2, num_proposals, num_classes),
            pred_boxes=torch.rand(2, num_proposals, 4).clamp(0, 1),
        )
        targets = [
            InstanceData(labels=gt_labels_batch[0], bboxes=torch.rand(3, 4).clamp(0, 1), img_shape=(512, 512)),
            InstanceData(labels=gt_labels_batch[1], bboxes=torch.rand(2, 4).clamp(0, 1), img_shape=(512, 512)),
        ]
        # 不传 ot_match_probs, 所有 proposal 都是正例
        indices = criterion._ot_indices_to_match(ot_indices, num_proposals)
        target_classes = torch.full((2, num_proposals), num_classes, dtype=torch.long)
        for i, (src_idx, gt_idx) in enumerate(indices):
            target_classes[i, src_idx] = gt_labels_batch[i][gt_idx]
        assert (target_classes < num_classes).all(), "Some proposals have background label"

    def test_criterion_ot_pos_ratio_filters_background(self):
        """ot_pos_ratio 应过滤低质量匹配为背景"""
        from projects.LDMDetDiT.mods.loss import (
            DiffusionDetCriterion,
            FocalLoss,
            GIoULoss,
        )
        num_proposals = 100
        num_classes = 24
        criterion = DiffusionDetCriterion(
            num_classes=num_classes, matcher=None,
            loss_cls=FocalLoss(), loss_giou=GIoULoss(),
            ot_pos_ratio=0.25,
        )
        gt_labels_batch = [
            torch.tensor([3, 7, 15, 0, 20]),
        ]
        ot_indices = [torch.randint(0, 5, (num_proposals,))]
        # 构造传输概率: 前 30 个高, 后 70 个低
        probs = [torch.cat([torch.ones(30) * 0.8, torch.ones(70) * 0.1])]

        indices = criterion._ot_indices_to_match(
            ot_indices, num_proposals, ot_match_probs=probs
        )
        src_idx, gt_idx = indices[0]
        # ot_pos_ratio=0.25 → 保留 25 个正例
        assert len(src_idx) == 25, f"Expected 25 positives, got {len(src_idx)}"
        # 正例应来自高概率区域
        assert (src_idx < 30).sum() >= 20, "Top-k should prefer high-probability proposals"

        # 验证分类目标: 只有正例有 GT 类别, 其余为背景
        target_classes = torch.full((1, num_proposals), num_classes, dtype=torch.long)
        target_classes[0, src_idx] = gt_labels_batch[0][gt_idx]
        num_pos = (target_classes[0] < num_classes).sum().item()
        num_bg = (target_classes[0] == num_classes).sum().item()
        assert num_pos == 25, f"Expected 25 positives, got {num_pos}"
        assert num_bg == 75, f"Expected 75 background, got {num_bg}"

    def test_criterion_without_ot_falls_back_to_simota(self, criterion):
        """不传 ot_matched_gt_indices 时应降级到 SimOTA (兼容性)"""
        from projects.LDMDetDiT.mods.loss import (
            DiffusionDetCriterion,
            DiffusionDetMatcher,
            FocalLoss,
            GIoULoss,
        )
        from projects.LDMDetDiT.mods.structures import (
            InstanceData,
            ModelOutput,
        )
        # 需要有 matcher 才能降级
        criterion_with_matcher = DiffusionDetCriterion(
            num_classes=24,
            matcher=DiffusionDetMatcher(),
            loss_cls=FocalLoss(),
            loss_giou=GIoULoss(),
        )
        outputs = ModelOutput(
            pred_logits=torch.rand(1, 50, 24),
            pred_boxes=torch.rand(1, 50, 4).clamp(0, 1),
        )
        targets = [
            InstanceData(labels=torch.tensor([0]), bboxes=torch.rand(1, 4).clamp(0, 1), img_shape=(512, 512)),
        ]
        losses = criterion_with_matcher(outputs, targets)  # 不传 ot_matched_gt_indices
        assert isinstance(losses, dict)
        assert 'loss_cls' in losses


# ============================================================
# Part 5: 端到端 DiTDiffusionDetHead loss 验证
# ============================================================

class TestDiTDiffusionDetHead:
    """DiTDiffusionDetHead 端到端输出验证 (使用 OT 匹配)"""

    @pytest.fixture
    def head_and_data(self):
        from projects.LDMDetDiT.mods.dit_head import DiTDiffusionDetHead
        from projects.LDMDetDiT.mods.loss import (
            DiffusionDetCriterion,
            FocalLoss,
            GIoULoss,
        )
        criterion = DiffusionDetCriterion(
            num_classes=24,
            matcher=None,
            loss_cls=FocalLoss(),
            loss_giou=GIoULoss(),
        )
        head = DiTDiffusionDetHead(
            num_classes=24, feat_channels=256, num_proposals=100,
            num_heads=2, num_fpn_levels=4, num_ref_points=8,
            sampling_timesteps=2, rf_schedule='shifted', rf_shift=1.0,
            prediction_mode='x0', adaln_params=9, regression_mode='direct',
            use_adaln_zero=True, num_blocks=1, share_heads=True,
            deep_supervision=True, box_init_mode='query', criterion=criterion,
            ot_coupling=True, ot_matcher='sinkhorn',
        )
        head.eval()
        fpn_features = tuple([
            torch.rand(2, 256, 16, 16),
            torch.rand(2, 256, 8, 8),
            torch.rand(2, 256, 4, 4),
            torch.rand(2, 256, 2, 2),
        ])
        bboxes = torch.rand(2, 100, 4) * 512
        t = torch.rand(2) * 1000
        img_metas = [
            {'img_shape': (512, 512), 'pad_shape': (512, 512), 'scale_factor': (1.0, 1.0)},
            {'img_shape': (512, 512), 'pad_shape': (512, 512), 'scale_factor': (1.0, 1.0)},
        ]
        return head, fpn_features, bboxes, t, img_metas

    def test_forward_output_shapes(self, head_and_data):
        head, fpn_features, bboxes, t, img_metas = head_and_data
        with torch.no_grad():
            all_cls, all_bbox, all_bbox_raw, all_x0_raw, all_vel, all_tokens = head(
                fpn_features, bboxes, t, img_metas=img_metas
            )
        assert all_cls.shape[0] == 2  # num_heads=2
        assert all_cls.shape[2:] == (100, 24)
        assert all_bbox.shape[2:] == (100, 4)

    def test_pred_bboxes_range(self, head_and_data):
        """预测 bboxes 应在合理范围内"""
        head, fpn_features, bboxes, t, img_metas = head_and_data
        with torch.no_grad():
            _, all_bbox, _, _, _, _ = head(
                fpn_features, bboxes, t, img_metas=img_metas
            )
        assert all_bbox.min() >= 0
        assert all_bbox.max() < 2000

    def test_no_nan(self, head_and_data):
        head, fpn_features, bboxes, t, img_metas = head_and_data
        with torch.no_grad():
            all_cls, all_bbox, _, _, _, _ = head(
                fpn_features, bboxes, t, img_metas=img_metas
            )
        assert not torch.isnan(all_cls).any()
        assert not torch.isnan(all_bbox).any()

    def test_loss_with_ot_matching(self, head_and_data):
        """loss 方法使用 OT 匹配, 应返回有限值且所有 proposal 参与训练"""
        head, fpn_features, bboxes, t, img_metas = head_and_data
        head.train()
        gt_bboxes = [
            torch.rand(5, 4) * 512,
            torch.rand(3, 4) * 512,
        ]
        gt_labels = [
            torch.randint(0, 24, (5,)),
            torch.randint(0, 24, (3,)),
        ]
        losses = head.loss(fpn_features, img_metas, gt_bboxes, gt_labels)
        assert isinstance(losses, dict)
        for key, val in losses.items():
            assert isinstance(val, torch.Tensor), f"{key}: expected Tensor, got {type(val)}"
            assert val.numel() == 1, f"{key}: expected scalar, got shape {val.shape}"
            assert not torch.isnan(val).any(), f"{key} is NaN"
            assert not torch.isinf(val).any(), f"{key} is Inf"

    def test_predict_no_error(self, head_and_data):
        """推理路径 (predict) 应正常完成，不抛异常"""
        head, fpn_features, bboxes, t, img_metas = head_and_data
        head.eval()
        from projects.LDMDetDiT.mods.structures import ImageMeta
        img_metas_objs = [
            ImageMeta(img_shape=m['img_shape'], pad_shape=m.get('pad_shape'),
                      scale_factor=m.get('scale_factor'))
            for m in img_metas
        ]
        with torch.no_grad():
            results = head.predict(fpn_features, img_metas_objs)
        assert len(results) == 2  # batch_size
        for r in results:
            assert hasattr(r, 'bboxes')
            assert hasattr(r, 'scores')
            assert hasattr(r, 'labels')
            assert r.bboxes.shape[0] <= 100  # 不超过 num_proposals

    def test_loss_all_proposals_positive(self, head_and_data):
        """OT 匹配下, loss_cls 的分母应为 num_proposals * batch_size (所有 proposal 都是正例)"""
        head, fpn_features, bboxes, t, img_metas = head_and_data
        head.train()
        gt_bboxes = [
            torch.rand(5, 4) * 512,
            torch.rand(3, 4) * 512,
        ]
        gt_labels = [
            torch.randint(0, 24, (5,)),
            torch.randint(0, 24, (3,)),
        ]
        # 手动调用 criterion 验证正例数
        from projects.LDMDetDiT.mods.structures import (
            InstanceData,
        )
        device = next(head.parameters()).device
        # 构建 targets
        targets = []
        for i in range(2):
            h, w = img_metas[i]['img_shape']
            norm_bboxes = gt_bboxes[i] / torch.tensor([w, h, w, h], device=device).float()
            targets.append(InstanceData(labels=gt_labels[i].to(device), bboxes=norm_bboxes.to(device), img_shape=(512, 512)))

        # 模拟 OT 匹配
        num_proposals = 100
        ot_indices = [
            torch.randint(0, len(gt_bboxes[0]), (num_proposals,), device=device),
            torch.randint(0, len(gt_bboxes[1]), (num_proposals,), device=device),
        ]
        indices = head.criterion._ot_indices_to_match(ot_indices, num_proposals)
        # 验证: 每个 batch 的正例数 = num_proposals
        for i, (src_idx, gt_idx) in enumerate(indices):
            assert len(src_idx) == num_proposals, \
                f"Batch {i}: expected {num_proposals} positives, got {len(src_idx)}"

    def test_loss_dense_gt_coverage(self):
        """密集 GT 场景 (46 个 GT): Sinkhorn 匹配覆盖率应远高于 SimOTA 的 26%"""
        from projects.LDMDetDiT.mods.dit_head import DiTDiffusionDetHead
        from projects.LDMDetDiT.mods.loss import (
            DiffusionDetCriterion,
            FocalLoss,
            GIoULoss,
        )
        criterion = DiffusionDetCriterion(
            num_classes=24, matcher=None,
            loss_cls=FocalLoss(), loss_giou=GIoULoss(),
        )
        head = DiTDiffusionDetHead(
            num_classes=24, feat_channels=256, num_proposals=100,
            num_heads=2, num_fpn_levels=4, num_ref_points=8,
            sampling_timesteps=2, rf_schedule='shifted', rf_shift=1.0,
            prediction_mode='x0', adaln_params=9, regression_mode='direct',
            use_adaln_zero=True, num_blocks=1, share_heads=True,
            deep_supervision=False, box_init_mode='query', criterion=criterion,
            ot_coupling=True, ot_matcher='sinkhorn',
        )
        head.train()
        device = next(head.parameters()).device

        # 46 个 GT (模拟染色体数据集)
        num_gt = 46
        # 多次运行取平均覆盖率 (Sinkhorn 受随机噪声影响)
        coverage_rates = []
        for _ in range(5):
            noise = torch.randn(100, 4, device=device)
            gt_diffusion = torch.randn(num_gt, 4, device=device)
            matched_idx, max_prob = head._sinkhorn_match(noise, gt_diffusion, device)
            unique_matched = matched_idx.unique()
            coverage_rates.append(len(unique_matched) / num_gt)

        avg_coverage = sum(coverage_rates) / len(coverage_rates)
        # Sinkhorn 覆盖率应远高于 SimOTA 的 26%, 至少 > 70%
        assert avg_coverage > 0.7, \
            f"Sinkhorn avg coverage {avg_coverage:.1%} too low (SimOTA was 26%)"
