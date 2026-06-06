"""LDMDetDiT 移植验证测试

TDD RED 阶段: 这些测试验证 LDMDetDiT 模块可以被正确导入和注册。
在移植完成前，这些测试应该全部失败。
"""
import os

import pytest


class TestLDMDetDiTImports:
    """验证 LDMDetDiT 的核心模块可以被导入"""

    def test_import_model(self):
        """LDMDetDiT.model 可以被导入"""
        import projects.LDMDetDiT.model  # noqa: F401

    def test_import_hooks(self):
        """LDMDetDiT.hooks 可以被导入"""
        import projects.LDMDetDiT.hooks  # noqa: F401

    def test_import_dit_head(self):
        """LDMDetDiT.mods.dit_head 可以被导入"""
        import projects.LDMDetDiT.mods.dit_head  # noqa: F401

    def test_import_dit_single_head(self):
        """LDMDetDiT.mods.dit_single_head 可以被导入"""
        import projects.LDMDetDiT.mods.dit_single_head  # noqa: F401

    def test_import_dit_block(self):
        """LDMDetDiT.mods.dit_block 可以被导入"""
        import projects.LDMDetDiT.mods.dit_block  # noqa: F401

    def test_import_box_tokenizer(self):
        """LDMDetDiT.mods.box_tokenizer 可以被导入"""
        import projects.LDMDetDiT.mods.box_tokenizer  # noqa: F401

    def test_import_deformable_attn(self):
        """LDMDetDiT.mods.deformable_attn 可以被导入"""
        import projects.LDMDetDiT.mods.deformable_attn  # noqa: F401

    def test_import_modules(self):
        """LDMDetDiT.mods.modules 可以被导入"""
        import projects.LDMDetDiT.mods.modules  # noqa: F401

    def test_import_loss(self):
        """LDMDetDiT.mods.loss 可以被导入"""
        import projects.LDMDetDiT.mods.loss  # noqa: F401

    def test_import_structures(self):
        """LDMDetDiT.mods.structures 可以被导入"""
        import projects.LDMDetDiT.mods.structures  # noqa: F401

    def test_import_utils(self):
        """LDMDetDiT.mods.utils 可以被导入"""
        import projects.LDMDetDiT.mods.utils  # noqa: F401

    def test_import_rectified_flow(self):
        """LDMDetDiT.mods.rectified_flow 可以被导入"""
        import projects.LDMDetDiT.mods.rectified_flow  # noqa: F401

    def test_import_convnextv2(self):
        """LDMDetDiT.mods.convnextv2 可以被导入"""
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
        """LDMDetDiT.model 不应 import DiffusionDetHead"""
        import projects.LDMDetDiT.model as dit_model
        source = open(dit_model.__file__).read()
        # 不应出现 DiffusionDetHead 的 import（DiTDiffusionDetHead 除外）
        assert 'from .mods.diffusiondet_head' not in source

    def test_no_single_diffusiondet_head_import(self):
        """LDMDetDiT.model 不应 import SingleDiffusionDetHead"""
        import projects.LDMDetDiT.model as dit_model
        source = open(dit_model.__file__).read()
        assert 'from .mods.single_head' not in source

    def test_no_roi_extractor_import(self):
        """LDMDetDiT.model 不应 import SingleRoIExtractor"""
        import projects.LDMDetDiT.model as dit_model
        source = open(dit_model.__file__).read()
        assert 'from .mods.roi_extractor' not in source

    def test_no_dynamic_conv_in_modules(self):
        """LDMDetDiT/mods/modules.py 不应包含 DynamicConv"""
        import projects.LDMDetDiT.mods.modules as mods
        source = open(mods.__file__).read()
        assert 'DynamicConv' not in source

    def test_no_flow_matching_velocity_loss(self):
        """LDMDetDiT/mods/loss.py 不应包含 FlowMatchingVelocityLoss"""
        import projects.LDMDetDiT.mods.loss as loss_mod
        source = open(loss_mod.__file__).read()
        assert 'FlowMatchingVelocityLoss' not in source


class TestLDMDetDiTConfig:
    """验证 LDMDetDiT 配置文件路径正确"""

    def test_config_custom_imports(self):
        """ldmdet_dit.py 的 custom_imports 应指向 LDMDetDiT"""
        # 配置文件使用 mmengine _base_ 机制，不能直接 import
        # 直接读取文件内容验证
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'configs', 'ldmdet_dit.py'
        )
        source = open(config_path).read()
        assert 'projects.LDMDetDiT.model' in source
        assert 'projects.LDMDetDiT.hooks' in source
        # 不应包含旧的 LDMDet 路径
        assert 'projects.LDMDet.model' not in source
        assert 'projects.LDMDet.hooks' not in source


class TestLDMDetDiTHooks:
    """验证 hooks.py 的 CopyProjectHook 路径正确"""

    def test_copy_project_hook_path(self):
        """CopyProjectHook 默认 src_path 应为 projects/LDMDetDiT"""
        import projects.LDMDetDiT.hooks as hooks_mod
        source = open(hooks_mod.__file__).read()
        assert "src_path='projects/LDMDetDiT'" in source
