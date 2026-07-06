"""SD-1.5 v2 fine-tune 测试套件：分阶段训练 + 分组 LR + EDM Karras + EMA Warmup

红绿重构流程：先写测试(RED)，再修复使通过(GREEN)，最后重构。
覆盖：
  1. Pipeline freeze_unet/unfreeze_unet 切换
  2. Pipeline get_param_groups 支持分组 LR
  3. KDPM2DiscreteScheduler 替换 DDPMScheduler
  4. EMAModel 支持 warmup（decay 从低到高渐进）
  5. train.py 支持 freeze_unet_epochs 参数
  6. v2 配置 chromogen_phase1_sd15_24obj_v2.py 加载正确
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


# ============================================================
# 测试 Pipeline freeze/unfreeze UNet
# ============================================================


class TestFreezeUnet:
    def test_freeze_unet_sets_requires_grad_false(self):
        """freeze_unet() 应将 unet 参数 requires_grad 设为 False"""
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
        # 默认 UNet 可训练
        unet_trainable_before = any(
            p.requires_grad for p in model.unet.parameters()
        )
        assert unet_trainable_before, 'UNet 默认应可训练'

        # 调用 freeze_unet
        model.freeze_unet()

        unet_trainable_after = any(
            p.requires_grad for p in model.unet.parameters()
        )
        assert not unet_trainable_after, 'freeze_unet 后 UNet 应不可训练'

        # Condition Encoder 仍可训练
        cond_trainable = any(
            p.requires_grad for p in model.condition_encoder.parameters()
        )
        assert cond_trainable, 'Condition Encoder 应保持可训练'

    def test_unfreeze_unet_restores_requires_grad(self):
        """unfreeze_unet() 应恢复 UNet 参数可训练"""
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
        model.freeze_unet()
        model.unfreeze_unet()

        unet_trainable = any(p.requires_grad for p in model.unet.parameters())
        assert unet_trainable, 'unfreeze_unet 后 UNet 应可训练'


# ============================================================
# 测试 Pipeline get_param_groups 分组 LR
# ============================================================


class TestParamGroups:
    def test_get_param_groups_returns_two_groups(self):
        """get_param_groups 应返回两组参数：unet 组和 condition_encoder 组"""
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
        groups = model.get_param_groups(unet_lr=5e-5, cond_lr=5e-4)
        assert isinstance(groups, list)
        assert len(groups) == 2, '应返回 2 组参数'
        # 每组应有 lr 和 params
        for g in groups:
            assert 'params' in g
            assert 'lr' in g
        # 两组 lr 应不同
        lrs = [g['lr'] for g in groups]
        assert 5e-5 in lrs
        assert 5e-4 in lrs

    def test_get_param_groups_unet_group_has_unet_params(self):
        """unet 组应包含 unet 参数"""
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
        groups = model.get_param_groups(unet_lr=5e-5, cond_lr=5e-4)
        # 找到 unet 组
        unet_group = next(g for g in groups if g['lr'] == 5e-5)
        # 验证包含 unet 参数
        unet_group_params = unet_group['params']
        unet_total = sum(p.numel() for p in unet_group_params)
        unet_expected = sum(
            p.numel() for p in model.unet.parameters() if p.requires_grad
        )
        assert unet_total == unet_expected, (
            f'unet 组参数量不匹配: {unet_total} vs {unet_expected}'
        )


# ============================================================
# 测试 EDM Karras Schedule
# ============================================================


class TestEDMKarrasSchedule:
    def test_pipeline_accepts_karras_schedule(self):
        """Pipeline 应接受 noise_schedule='karras' 并使用 DDPMScheduler with scaled_linear

        注意：KDPM2DiscreteScheduler 是推理专用，没有 get_velocity 方法不能用于训练。
        所以 karras 训练时改用 DDPMScheduler with scaled_linear（Karras 风格的 beta schedule）。
        """
        from diffusers import DDPMScheduler

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
            noise_schedule='karras',
            prediction_type='v_prediction',
        )
        # 验证 scheduler 类型：应为 DDPMScheduler（训练用，支持 get_velocity）
        assert isinstance(model.noise_scheduler, DDPMScheduler), (
            'noise_schedule=karras 应使用 DDPMScheduler（训练可用的 Karras 近似）'
        )
        # 验证 beta_schedule 解析为 scaled_linear
        assert model.noise_scheduler.config.beta_schedule == 'scaled_linear', (
            'karras 应解析为 scaled_linear beta_schedule'
        )

    def test_pipeline_default_still_ddpm(self):
        """默认 noise_schedule='linear' 仍应使用 DDPMScheduler"""
        from diffusers import DDPMScheduler

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
        assert isinstance(model.noise_scheduler, DDPMScheduler)


# ============================================================
# 测试 EMA Warmup
# ============================================================


class TestEMAWarmup:
    def test_emamodel_accepts_warmup_params(self):
        """EMAModel 应接受 decay_start 和 warmup_steps 参数"""
        from projects.ChromoGen.models.ema import EMAModel

        model = torch.nn.Linear(4, 4)
        ema = EMAModel(
            model,
            decay=0.9999,
            decay_start=0.999,
            warmup_steps=100,
        )
        # warmup 期间 decay 应从 decay_start 渐进到 decay
        assert hasattr(ema, 'decay_start')
        assert hasattr(ema, 'warmup_steps')
        assert ema.warmup_steps == 100
        assert ema.decay_start == 0.999

    def test_emamodel_warmup_decay_increases_with_step(self):
        """EMA warmup 期间，decay 应从 decay_start 渐进到 decay"""
        from projects.ChromoGen.models.ema import EMAModel

        model = torch.nn.Linear(4, 4)
        ema = EMAModel(
            model,
            decay=0.9999,
            decay_start=0.999,
            warmup_steps=100,
        )
        # step 0: decay ≈ decay_start
        ema.cur_step = 0
        decay_0 = ema.get_current_decay()
        # step 100: decay ≈ decay
        ema.cur_step = 100
        decay_100 = ema.get_current_decay()
        # step 50: decay 在中间
        ema.cur_step = 50
        decay_50 = ema.get_current_decay()
        assert decay_0 < decay_50 < decay_100, (
            f'warmup decay 应递增: {decay_0} < {decay_50} < {decay_100}'
        )

    def test_emamodel_default_no_warmup(self):
        """默认不传 warmup 参数时，行为应与原版一致"""
        from projects.ChromoGen.models.ema import EMAModel

        model = torch.nn.Linear(4, 4)
        ema = EMAModel(model, decay=0.9999)
        # 不应触发 warmup 逻辑
        ema.cur_step = 0
        decay = ema.get_current_decay()
        assert decay == 0.9999, (
            f'默认无 warmup 时 decay 应等于 decay，当前 {decay}'
        )


# ============================================================
# 测试 train.py 分阶段训练支持
# ============================================================


class TestStagedTraining:
    def test_config_supports_freeze_unet_epochs(self):
        """配置应支持 freeze_unet_epochs 字段"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'configs',
            'chromogen_phase1_sd15_24obj_v2.py',
        )
        if not os.path.exists(path):
            pytest.skip('GREEN 阶段未完成')
        import sys

        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), '..', 'tools')
        )
        from train import load_config

        cfg = load_config(path)
        assert 'freeze_unet_epochs' in cfg, '配置应指定 freeze_unet_epochs'
        assert cfg['freeze_unet_epochs'] > 0

    def test_config_supports_param_group_lr(self):
        """配置应支持 unet_learning_rate 和 cond_encoder_learning_rate"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'configs',
            'chromogen_phase1_sd15_24obj_v2.py',
        )
        if not os.path.exists(path):
            pytest.skip('GREEN 阶段未完成')
        import sys

        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), '..', 'tools')
        )
        from train import load_config

        cfg = load_config(path)
        assert 'unet_learning_rate' in cfg
        assert 'cond_encoder_learning_rate' in cfg
        assert cfg['cond_encoder_learning_rate'] > cfg['unet_learning_rate'], (
            'CondEncoder LR 应高于 UNet LR'
        )


# ============================================================
# 测试 v2 配置文件加载
# ============================================================


class TestV2Config:
    def _load_config(self, config_path: str) -> dict:
        import sys

        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), '..', 'tools')
        )
        from train import load_config

        return load_config(config_path)

    def test_v2_config_exists(self):
        """v2 配置文件应存在"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'configs',
            'chromogen_phase1_sd15_24obj_v2.py',
        )
        assert os.path.exists(path), f'v2 配置文件不存在: {path}'

    def test_v2_config_uses_karras(self):
        """v2 配置应使用 Karras schedule"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'configs',
            'chromogen_phase1_sd15_24obj_v2.py',
        )
        if not os.path.exists(path):
            pytest.skip('GREEN 阶段未完成')
        cfg = self._load_config(path)
        assert cfg.get('noise_schedule') == 'karras'

    def test_v2_config_v_prediction(self):
        """v2 配置应使用 v_prediction（EDM 推荐）"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'configs',
            'chromogen_phase1_sd15_24obj_v2.py',
        )
        if not os.path.exists(path):
            pytest.skip('GREEN 阶段未完成')
        cfg = self._load_config(path)
        assert cfg.get('prediction_type') == 'v_prediction'

    def test_v2_config_cfg_dropout_0_2(self):
        """v2 配置 CFG Dropout 应为 0.2"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'configs',
            'chromogen_phase1_sd15_24obj_v2.py',
        )
        if not os.path.exists(path):
            pytest.skip('GREEN 阶段未完成')
        cfg = self._load_config(path)
        assert cfg.get('cfg_dropout') == 0.2

    def test_v2_config_adam_beta2_95(self):
        """v2 配置 AdamW β2 应为 0.95"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'configs',
            'chromogen_phase1_sd15_24obj_v2.py',
        )
        if not os.path.exists(path):
            pytest.skip('GREEN 阶段未完成')
        cfg = self._load_config(path)
        assert cfg.get('adam_beta2') == 0.95

    def test_v2_config_ema_warmup(self):
        """v2 配置应指定 ema_decay_start 和 ema_warmup_steps"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'configs',
            'chromogen_phase1_sd15_24obj_v2.py',
        )
        if not os.path.exists(path):
            pytest.skip('GREEN 阶段未完成')
        cfg = self._load_config(path)
        assert cfg.get('ema_decay_start') is not None
        assert cfg.get('ema_warmup_steps') is not None

    def test_v2_config_larger_batch(self):
        """v2 配置 batch_size 应大于 8（效率优化）"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'configs',
            'chromogen_phase1_sd15_24obj_v2.py',
        )
        if not os.path.exists(path):
            pytest.skip('GREEN 阶段未完成')
        cfg = self._load_config(path)
        assert cfg.get('batch_size', 8) > 8, (
            'v2 应提高 batch_size 优化训练效率'
        )

    def test_v2_config_output_dir_distinct(self):
        """v2 配置应使用独立 output_dir"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'configs',
            'chromogen_phase1_sd15_24obj_v2.py',
        )
        if not os.path.exists(path):
            pytest.skip('GREEN 阶段未完成')
        cfg = self._load_config(path)
        assert cfg.get('output_dir') not in (
            'work_dirs/chromogen_phase1_sd15_24obj',
            'work_dirs/chromogen_phase1_sd15',
        ), 'v2 应使用独立目录'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
