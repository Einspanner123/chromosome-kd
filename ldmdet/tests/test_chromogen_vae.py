"""测试 ChromoGenVAE wrapper — 包装 diffusers AutoencoderKL

方向六 Phase 1: 让 VAE 可通过 mmengine MODELS.build 加载, 用于 image → latent 编码。
"""

import pytest
import torch
import torch.nn as nn

from ldmdet.feature_bridge.chromogen_vae import ChromoGenVAE


class TestChromoGenVAE:
    """测试 ChromoGenVAE wrapper"""

    def test_init_with_local_path(self):
        """通过本地 VAE 路径初始化"""
        vae = ChromoGenVAE(
            vae_model='work_dirs/chromogen_phase1/vae',
            scaling_factor=0.18215,
        )
        assert isinstance(vae.vae, nn.Module)
        assert vae.scaling_factor == 0.18215

    def test_forward_returns_latent(self):
        """前向传播返回潜变量"""
        vae = ChromoGenVAE(
            vae_model='work_dirs/chromogen_phase1/vae',
            scaling_factor=0.18215,
        )
        img = torch.randn(1, 3, 768, 768)
        latent = vae(img)
        # VAE f8: 768/8 = 96
        assert latent.shape == (1, 4, 96, 96)

    def test_forward_batch(self):
        """支持 batch 前向"""
        vae = ChromoGenVAE(
            vae_model='work_dirs/chromogen_phase1/vae',
        )
        img = torch.randn(2, 3, 512, 512)
        latent = vae(img)
        assert latent.shape == (2, 4, 64, 64)

    def test_scaling_factor_applied(self):
        """scaling_factor 正确应用"""
        vae = ChromoGenVAE(
            vae_model='work_dirs/chromogen_phase1/vae',
            scaling_factor=0.18215,
        )
        img = torch.randn(1, 3, 256, 256)
        latent = vae(img)
        # 检查 latent 数值范围被缩放
        assert latent.abs().max() < 10  # scaling 后应在合理范围

    def test_frozen_by_default(self):
        """默认冻结参数"""
        vae = ChromoGenVAE(
            vae_model='work_dirs/chromogen_phase1/vae',
        )
        for param in vae.parameters():
            assert not param.requires_grad

    def test_eval_mode(self):
        """默认 eval 模式"""
        vae = ChromoGenVAE(
            vae_model='work_dirs/chromogen_phase1/vae',
        )
        assert not vae.training

    def test_load_from_chromogen_checkpoint(self):
        """从 ChromoGen pipeline checkpoint 加载 VAE 权重"""
        vae = ChromoGenVAE.from_chromogen_checkpoint(
            checkpoint='work_dirs/chromogen_phase1/final_model.pt',
            vae_model='work_dirs/chromogen_phase1/vae',
        )
        assert isinstance(vae.vae, nn.Module)
        img = torch.randn(1, 3, 256, 256)
        latent = vae(img)
        assert latent.shape == (1, 4, 32, 32)

    def test_no_grad_in_forward(self):
        """前向传播不需要梯度"""
        vae = ChromoGenVAE(
            vae_model='work_dirs/chromogen_phase1/vae',
        )
        img = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            latent = vae(img)
        assert latent.shape == (1, 4, 32, 32)

    def test_denormalize_from_imagenet(self):
        """从 ImageNet 归一化输入反归一化后编码"""
        vae = ChromoGenVAE(
            vae_model='work_dirs/chromogen_phase1/vae',
            input_mean=[123.675, 116.28, 103.53],
            input_std=[58.395, 57.12, 57.375],
        )
        # 模拟 ImageNet 归一化后的输入
        img = torch.randn(1, 3, 256, 256)
        latent = vae(img)
        assert latent.shape == (1, 4, 32, 32)
