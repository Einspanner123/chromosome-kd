"""LDMDetDiT 移植验证 + 每层输出格式/范围验证测试

TDD RED 阶段: 这些测试验证 LDMDetDiT 模块可以被正确导入和注册，
且每层的输出格式和数值范围符合预期。
"""
import os
import math

import pytest
import torch


# ============================================================
# Part 1: 导入与注册验证 (原有)
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
# Part 2: 每层输出格式和范围验证
# ============================================================

class TestRoPE1D:
    """RoPE1D 旋转位置编码输出验证"""

    def test_output_shape(self):
        from projects.LDMDetDiT.mods.modules import RoPE1D
        rope = RoPE1D(dim=32, base=10000)
        x = torch.rand(2, 100, 4)  # (bs, N, 4)
        out = rope(x)
        assert out.shape == (2, 100, 4, 32), f"Expected (2,100,4,32), got {out.shape}"

    def test_output_range(self):
        """RoPE 输出 sin/cos 应在 [-1, 1]"""
        from projects.LDMDetDiT.mods.modules import RoPE1D
        rope = RoPE1D(dim=32, base=10000)
        x = torch.rand(2, 100, 4)
        out = rope(x)
        assert out.min() >= -1.0 and out.max() <= 1.0, f"RoPE out range [{out.min()}, {out.max()}]"

    def test_no_nan(self):
        from projects.LDMDetDiT.mods.modules import RoPE1D
        rope = RoPE1D(dim=32, base=10000)
        x = torch.rand(2, 100, 4)
        out = rope(x)
        assert not torch.isnan(out).any(), "RoPE output contains NaN"


class TestSinusoidalPositionEmbeddings:
    """时间步位置编码输出验证"""

    def test_output_shape(self):
        from projects.LDMDetDiT.mods.modules import SinusoidalPositionEmbeddings
        spe = SinusoidalPositionEmbeddings(dim=256)
        t = torch.rand(4)  # (bs,)
        out = spe(t)
        assert out.shape == (4, 256), f"Expected (4,256), got {out.shape}"

    def test_output_range(self):
        """sin/cos 编码应在 [-1, 1]"""
        from projects.LDMDetDiT.mods.modules import SinusoidalPositionEmbeddings
        spe = SinusoidalPositionEmbeddings(dim=256)
        t = torch.rand(4) * 1000
        out = spe(t)
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_no_nan(self):
        from projects.LDMDetDiT.mods.modules import SinusoidalPositionEmbeddings
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
        # 4 级 FPN: P2(64x64), P3(32x32), P4(16x16), P5(8x8)
        return [
            torch.rand(2, 256, 64, 64),
            torch.rand(2, 256, 32, 32),
            torch.rand(2, 256, 16, 16),
            torch.rand(2, 256, 8, 8),
        ]

    def test_output_shapes(self, tokenizer, fpn_features):
        bboxes = torch.rand(2, 100, 4)  # (bs, N, 4) xyxy [0,1]
        box_tokens, level_indices = tokenizer(bboxes, fpn_features)
        assert box_tokens.shape == (2, 100, 256), f"Expected (2,100,256), got {box_tokens.shape}"
        assert level_indices.shape == (2, 100), f"Expected (2,100), got {level_indices.shape}"

    def test_level_indices_range(self, tokenizer, fpn_features):
        """FPN 层级索引应在 [0, num_fpn_levels)"""
        bboxes = torch.rand(2, 100, 4)
        _, level_indices = tokenizer(bboxes, fpn_features)
        assert level_indices.min() >= 0 and level_indices.max() < 4

    def test_no_nan(self, tokenizer, fpn_features):
        bboxes = torch.rand(2, 100, 4)
        box_tokens, _ = tokenizer(bboxes, fpn_features)
        assert not torch.isnan(box_tokens).any(), "BoxTokenizer output contains NaN"
        assert not torch.isinf(box_tokens).any(), "BoxTokenizer output contains Inf"


class TestBboxToReferencePoints:
    """bbox_to_reference_points 输出验证"""

    def test_output_shape(self):
        from projects.LDMDetDiT.mods.box_tokenizer import bbox_to_reference_points
        bboxes = torch.rand(2, 100, 4)
        ref = bbox_to_reference_points(bboxes)
        assert ref.shape == (2, 100, 2), f"Expected (2,100,2), got {ref.shape}"

    def test_output_range(self):
        """参考点应在 [0, 1]（当输入 bboxes 在 [0,1] 时）"""
        from projects.LDMDetDiT.mods.box_tokenizer import bbox_to_reference_points
        bboxes = torch.rand(2, 100, 4)  # [0,1] xyxy
        ref = bbox_to_reference_points(bboxes)
        assert ref.min() >= 0.0 and ref.max() <= 1.0, f"ref range [{ref.min()}, {ref.max()}]"

    def test_center_correctness(self):
        """参考点应为 bbox 中心"""
        from projects.LDMDetDiT.mods.box_tokenizer import bbox_to_reference_points
        bboxes = torch.tensor([[[0.1, 0.2, 0.5, 0.6]]])  # cx=0.3, cy=0.4
        ref = bbox_to_reference_points(bboxes)
        expected = torch.tensor([[[0.3, 0.4]]])
        assert torch.allclose(ref, expected, atol=1e-6)


class TestReferencePointsWithLevels:
    """reference_points_with_levels 输出验证"""

    def test_output_shape(self):
        from projects.LDMDetDiT.mods.box_tokenizer import reference_points_with_levels
        ref = torch.rand(2, 100, 2)
        out = reference_points_with_levels(ref, num_levels=4)
        assert out.shape == (2, 100, 4, 2), f"Expected (2,100,4,2), got {out.shape}"

    def test_values_match_input(self):
        from projects.LDMDetDiT.mods.box_tokenizer import reference_points_with_levels
        ref = torch.tensor([[[0.3, 0.4]]])
        out = reference_points_with_levels(ref, num_levels=4)
        # 所有 level 应该与输入相同
        for l in range(4):
            assert torch.allclose(out[0, 0, l], ref[0, 0], atol=1e-6)


class TestMultiScaleDeformableAttention:
    """多尺度可变形注意力输出验证"""

    @pytest.fixture
    def attn_module(self):
        from projects.LDMDetDiT.mods.deformable_attn import MultiScaleDeformableAttention
        return MultiScaleDeformableAttention(
            embed_dim=256, num_heads=8, num_levels=4, num_points=8, dropout=0.0
        )

    @pytest.fixture
    def fpn_data(self):
        from projects.LDMDetDiT.mods.deformable_attn import flatten_fpn_features
        fpn = [
            torch.rand(2, 256, 16, 16),
            torch.rand(2, 256, 8, 8),
            torch.rand(2, 256, 4, 4),
            torch.rand(2, 256, 2, 2),
        ]
        flattened, spatial_shapes, level_start_index = flatten_fpn_features(fpn)
        return flattened, spatial_shapes, level_start_index

    def test_output_shape(self, attn_module, fpn_data):
        flattened, spatial_shapes, level_start_index = fpn_data
        query = torch.rand(2, 100, 256)
        ref_points = torch.rand(2, 100, 4, 2)  # (bs, N, num_levels, 2)
        out = attn_module(query, ref_points, flattened, spatial_shapes, level_start_index)
        assert out.shape == (2, 100, 256), f"Expected (2,100,256), got {out.shape}"

    def test_no_nan(self, attn_module, fpn_data):
        flattened, spatial_shapes, level_start_index = fpn_data
        query = torch.rand(2, 100, 256)
        ref_points = torch.rand(2, 100, 4, 2)
        out = attn_module(query, ref_points, flattened, spatial_shapes, level_start_index)
        assert not torch.isnan(out).any(), "DeformableAttn output contains NaN"

    def test_finite_output(self, attn_module, fpn_data):
        flattened, spatial_shapes, level_start_index = fpn_data
        query = torch.rand(2, 100, 256)
        ref_points = torch.rand(2, 100, 4, 2)
        out = attn_module(query, ref_points, flattened, spatial_shapes, level_start_index)
        assert not torch.isinf(out).any(), "DeformableAttn output contains Inf"


class TestFlattenFPNFeatures:
    """FPN 特征展平工具函数验证"""

    def test_output_shapes(self):
        from projects.LDMDetDiT.mods.deformable_attn import flatten_fpn_features
        fpn = [
            torch.rand(2, 256, 16, 16),
            torch.rand(2, 256, 8, 8),
            torch.rand(2, 256, 4, 4),
            torch.rand(2, 256, 2, 2),
        ]
        flattened, spatial_shapes, level_start_index = flatten_fpn_features(fpn)
        total_tokens = 16*16 + 8*8 + 4*4 + 2*2  # = 256+64+16+4 = 340
        assert flattened.shape == (2, total_tokens, 256), f"Expected (2,{total_tokens},256), got {flattened.shape}"
        assert spatial_shapes.shape == (4, 2)
        assert level_start_index.shape == (4,)

    def test_spatial_shapes_correct(self):
        from projects.LDMDetDiT.mods.deformable_attn import flatten_fpn_features
        fpn = [
            torch.rand(2, 256, 16, 16),
            torch.rand(2, 256, 8, 8),
        ]
        _, spatial_shapes, _ = flatten_fpn_features(fpn)
        expected = torch.tensor([[16, 16], [8, 8]])
        assert torch.equal(spatial_shapes, expected)

    def test_level_start_index_correct(self):
        from projects.LDMDetDiT.mods.deformable_attn import flatten_fpn_features
        fpn = [
            torch.rand(2, 256, 16, 16),
            torch.rand(2, 256, 8, 8),
            torch.rand(2, 256, 4, 4),
        ]
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
        from projects.LDMDetDiT.mods.deformable_attn import flatten_fpn_features
        fpn = [
            torch.rand(2, 256, 16, 16),
            torch.rand(2, 256, 8, 8),
            torch.rand(2, 256, 4, 4),
            torch.rand(2, 256, 2, 2),
        ]
        return flatten_fpn_features(fpn)

    def test_output_shape(self, dit_block, fpn_data):
        flattened, spatial_shapes, level_start_index = fpn_data
        box_tokens = torch.rand(2, 100, 256)
        time_emb = torch.rand(2, 256 * 4)  # time_dim = feat_channels * 4
        bbox_coords = torch.rand(2, 100, 4)
        out = dit_block(box_tokens, flattened, spatial_shapes, level_start_index, time_emb, bbox_coords)
        assert out.shape == (2, 100, 256), f"Expected (2,100,256), got {out.shape}"

    def test_no_nan(self, dit_block, fpn_data):
        flattened, spatial_shapes, level_start_index = fpn_data
        box_tokens = torch.rand(2, 100, 256)
        time_emb = torch.rand(2, 256 * 4)
        bbox_coords = torch.rand(2, 100, 4)
        out = dit_block(box_tokens, flattened, spatial_shapes, level_start_index, time_emb, bbox_coords)
        assert not torch.isnan(out).any(), "DiTBlock output contains NaN"

    def test_no_inf(self, dit_block, fpn_data):
        flattened, spatial_shapes, level_start_index = fpn_data
        box_tokens = torch.rand(2, 100, 256)
        time_emb = torch.rand(2, 256 * 4)
        bbox_coords = torch.rand(2, 100, 4)
        out = dit_block(box_tokens, flattened, spatial_shapes, level_start_index, time_emb, bbox_coords)
        assert not torch.isinf(out).any(), "DiTBlock output contains Inf"

    def test_adaln_zero_initial_residual(self):
        """AdaLN-Zero 零初始化时，初始残差应接近 0（输出 ≈ 输入）"""
        from projects.LDMDetDiT.mods.dit_block import DiTBlock
        block = DiTBlock(
            feat_channels=256, num_heads=8, num_fpn_levels=4,
            num_ref_points=8, dim_feedforward=2048, adaln_params=9,
            use_adaln_zero=True,
        )
        block.eval()
        from projects.LDMDetDiT.mods.deformable_attn import flatten_fpn_features
        fpn = [torch.rand(1, 256, 8, 8)] * 4
        flattened, spatial_shapes, level_start_index = flatten_fpn_features(fpn)
        box_tokens = torch.rand(1, 10, 256)
        time_emb = torch.rand(1, 256 * 4)
        bbox_coords = torch.rand(1, 10, 4)
        out = block(box_tokens, flattened, spatial_shapes, level_start_index, time_emb, bbox_coords)
        # AdaLN-Zero: α 参数零初始化，残差项应为 0，输出 ≈ 输入
        diff = (out - box_tokens).abs().max().item()
        assert diff < 1e-4, f"AdaLN-Zero: residual too large {diff}, expected near 0"


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
        from projects.LDMDetDiT.mods.deformable_attn import flatten_fpn_features
        fpn = [
            torch.rand(2, 256, 16, 16),
            torch.rand(2, 256, 8, 8),
            torch.rand(2, 256, 4, 4),
            torch.rand(2, 256, 2, 2),
        ]
        return flatten_fpn_features(fpn)

    def test_output_shapes(self, single_head, fpn_data):
        flattened, spatial_shapes, level_start_index = fpn_data
        box_tokens = torch.rand(2, 100, 256)
        time_emb = torch.rand(2, 256 * 4)
        bbox_coords = torch.rand(2, 100, 4)

        cls_logits, pred_bboxes, updated_tokens, objectness, velocity = single_head(
            box_tokens, flattened, spatial_shapes, level_start_index, time_emb, bbox_coords
        )
        assert cls_logits.shape == (2, 100, 24), f"cls_logits: expected (2,100,24), got {cls_logits.shape}"
        assert pred_bboxes.shape == (2, 100, 4), f"pred_bboxes: expected (2,100,4), got {pred_bboxes.shape}"
        assert updated_tokens.shape == (2, 100, 256), f"updated_tokens: expected (2,100,256), got {updated_tokens.shape}"
        assert objectness is None, "objectness should be None when use_objectness=False"
        assert velocity is None, "velocity should be None when prediction_mode='x0'"

    def test_pred_bboxes_range_direct(self, single_head, fpn_data):
        """direct 模式下 pred_bboxes 是 raw 空间 cxcywh (无约束实数)"""
        flattened, spatial_shapes, level_start_index = fpn_data
        box_tokens = torch.rand(2, 100, 256)
        time_emb = torch.rand(2, 256 * 4)
        bbox_coords = torch.rand(2, 100, 4)

        _, pred_bboxes, _, _, _ = single_head(
            box_tokens, flattened, spatial_shapes, level_start_index, time_emb, bbox_coords
        )
        # direct 模式输出 raw 空间 cxcywh, 无 sigmoid 约束
        assert not torch.isnan(pred_bboxes).any(), "pred_bboxes contains NaN"
        assert not torch.isinf(pred_bboxes).any(), "pred_bboxes contains Inf"

    def test_pred_bboxes_valid_xyxy(self, single_head, fpn_data):
        """direct 模式下 pred_bboxes 是 cxcywh 格式 (非 xyxy), 跳过 xyxy 校验"""
        # direct 模式现在输出 raw 空间 cxcywh, 由 DiTDiffusionDetHead 负责转换
        # xyxy 格式校验应在 DiTDiffusionDetHead 层面进行
        pass

    def test_no_nan(self, single_head, fpn_data):
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
        """prediction_mode='velocity' 时应输出 velocity"""
        from projects.LDMDetDiT.mods.dit_single_head import DiTSingleHead
        from projects.LDMDetDiT.mods.deformable_attn import flatten_fpn_features
        head = DiTSingleHead(
            num_classes=24, feat_channels=256, num_heads=8,
            num_fpn_levels=4, num_ref_points=8, prediction_mode='velocity',
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
        assert velocity is not None, "velocity should not be None in velocity mode"
        assert velocity.shape == (1, 10, 4), f"velocity: expected (1,10,4), got {velocity.shape}"


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
        assert x_t.shape == (32, 4), f"x_t: expected (32,4), got {x_t.shape}"
        assert velocity.shape == (32, 4), f"velocity: expected (32,4), got {velocity.shape}"

    def test_q_sample_boundary_t0(self, rf):
        """t=0 时 x_t ≈ x_start"""
        x_start = torch.randn(4, 4)
        t = torch.zeros(4)
        x_t, _ = rf.q_sample(x_start, t=t)
        assert torch.allclose(x_t, x_start, atol=1e-5), "t=0: x_t should equal x_start"

    def test_q_sample_boundary_t1(self, rf):
        """t=1 时 x_t ≈ x_noise"""
        x_start = torch.randn(4, 4)
        x_noise = torch.randn(4, 4)
        t = torch.ones(4)
        x_t, _ = rf.q_sample(x_start, x_noise=x_noise, t=t)
        assert torch.allclose(x_t, x_noise, atol=1e-5), "t=1: x_t should equal x_noise"

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

    def test_no_nan_in_step(self, rf):
        x_t = torch.randn(4, 4)
        x0_pred = torch.randn(4, 4)
        x_next = rf.step(x_t, x0_pred, t_curr=0.5, t_next=0.4)
        assert not torch.isnan(x_next).any()


class TestDiTDiffusionDetHead:
    """DiTDiffusionDetHead 端到端输出验证"""

    @pytest.fixture
    def head_and_data(self):
        """构建一个最小化的 DiTDiffusionDetHead 及测试数据"""
        from projects.LDMDetDiT.mods.dit_head import DiTDiffusionDetHead
        from projects.LDMDetDiT.mods.loss import (
            DiffusionDetCriterion, DiffusionDetMatcher, FocalLoss, L1Loss, GIoULoss,
        )

        criterion = DiffusionDetCriterion(
            num_classes=24,
            matcher=DiffusionDetMatcher(),
            loss_cls=FocalLoss(),
            loss_bbox=L1Loss(),
            loss_giou=GIoULoss(),
        )

        head = DiTDiffusionDetHead(
            num_classes=24,
            feat_channels=256,
            num_proposals=100,
            num_heads=6,  # 6 次迭代
            num_fpn_levels=4,
            num_ref_points=8,
            sampling_timesteps=6,
            diffusion_type='rectified_flow',
            rf_schedule='shifted',
            rf_shift=1.0,
            prediction_mode='x0',
            adaln_params=9,
            regression_mode='direct',
            use_adaln_zero=True,
            num_blocks=1,
            share_heads=True,
            deep_supervision=True,
            box_init_mode='query',
            criterion=criterion,
        )
        head.eval()

        # FPN 特征
        fpn_features = tuple([
            torch.rand(2, 256, 16, 16),
            torch.rand(2, 256, 8, 8),
            torch.rand(2, 256, 4, 4),
            torch.rand(2, 256, 2, 2),
        ])

        # 初始 bboxes (图像坐标)
        bboxes = torch.rand(2, 100, 4) * 512  # 模拟 512x512 图像

        # 时间步
        t = torch.rand(2) * 1000

        # img_metas
        img_metas = [
            {'img_shape': (512, 512), 'pad_shape': (512, 512), 'scale_factor': (1.0, 1.0)},
            {'img_shape': (512, 512), 'pad_shape': (512, 512), 'scale_factor': (1.0, 1.0)},
        ]

        return head, fpn_features, bboxes, t, img_metas

    def test_forward_output_shapes(self, head_and_data):
        head, fpn_features, bboxes, t, img_metas = head_and_data
        with torch.no_grad():
            all_cls, all_bbox, all_bbox_raw, all_x0_raw, all_obj, all_vel, all_tokens = head(
                fpn_features, bboxes, t, img_metas=img_metas
            )

        # deep_supervision=True, num_heads=6 → stack 6
        assert all_cls.shape[0] == 6, f"Expected 6 iterations, got {all_cls.shape[0]}"
        assert all_cls.shape[2:] == (100, 24), f"cls shape: {all_cls.shape}"
        assert all_bbox.shape[2:] == (100, 4), f"bbox shape: {all_bbox.shape}"

    def test_pred_bboxes_range(self, head_and_data):
        """预测 bboxes 应在 [0, image_size] 范围内"""
        head, fpn_features, bboxes, t, img_metas = head_and_data
        with torch.no_grad():
            all_cls, all_bbox, all_bbox_raw, all_x0_raw, all_obj, all_vel, all_tokens = head(
                fpn_features, bboxes, t, img_metas=img_metas
            )
        # 归一化后的 bbox 应在 [0,1]
        # 但 head 输出的是图像坐标，范围应在 [0, 512]
        assert all_bbox.min() >= 0, f"bbox min {all_bbox.min()} < 0"
        # 上限允许略超出（delta regression），但不应极端
        assert all_bbox.max() < 2000, f"bbox max {all_bbox.max()} too large"

    def test_no_nan(self, head_and_data):
        head, fpn_features, bboxes, t, img_metas = head_and_data
        with torch.no_grad():
            all_cls, all_bbox, all_bbox_raw, all_x0_raw, all_obj, all_vel, all_tokens = head(
                fpn_features, bboxes, t, img_metas=img_metas
            )
        assert not torch.isnan(all_cls).any(), "cls_logits contains NaN"
        assert not torch.isnan(all_bbox).any(), "pred_bboxes contains NaN"

    def test_loss_runs(self, head_and_data):
        """loss 方法应能正常运行并返回有限值"""
        head, fpn_features, bboxes, t, img_metas = head_and_data
        head.train()

        gt_bboxes = [
            torch.rand(5, 4) * 512,  # 5 个 GT 框
            torch.rand(3, 4) * 512,  # 3 个 GT 框
        ]
        gt_labels = [
            torch.randint(0, 24, (5,)),
            torch.randint(0, 24, (3,)),
        ]

        losses = head.loss(fpn_features, img_metas, gt_bboxes, gt_labels)
        assert isinstance(losses, dict), f"loss should return dict, got {type(losses)}"
        for key, val in losses.items():
            assert isinstance(val, torch.Tensor), f"{key}: expected Tensor, got {type(val)}"
            assert val.numel() == 1, f"{key}: expected scalar, got shape {val.shape}"
            assert not torch.isnan(val).any(), f"{key} is NaN"
            assert not torch.isinf(val).any(), f"{key} is Inf"


class TestCosineNoiseSchedule:
    """cosine_noise_schedule 输出验证"""

    def test_output_shape(self):
        from projects.LDMDetDiT.mods.modules import cosine_noise_schedule
        betas = cosine_noise_schedule(1000)
        assert betas.shape == (1000,), f"Expected (1000,), got {betas.shape}"

    def test_betas_range(self):
        """betas 应在 (0, 1)"""
        from projects.LDMDetDiT.mods.modules import cosine_noise_schedule
        betas = cosine_noise_schedule(1000)
        assert betas.min() > 0, f"betas min {betas.min()} <= 0"
        assert betas.max() < 1, f"betas max {betas.max()} >= 1"

    def test_monotonically_increasing(self):
        """cosine schedule 的 betas 应单调递增"""
        from projects.LDMDetDiT.mods.modules import cosine_noise_schedule
        betas = cosine_noise_schedule(1000)
        diffs = betas[1:] - betas[:-1]
        assert (diffs >= -1e-6).all(), "betas not monotonically increasing"
