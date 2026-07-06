"""Stage 1 (frozen UNet) + gradient_checkpointing 集成测试

补充覆盖：之前 smoke test 没有在 Stage 1 + gradient_checkpointing=True
场景下验证 forward+backward，导致实际训练报错：
  element 0 of tensors does not require grad and does not have a grad_fn

根因：当 UNet 所有参数 requires_grad=False 且启用 gradient_checkpointing 时，
PyTorch autograd 优化会导致 unet_out 丢失 grad_fn，backward 失败。

红绿重构纪律：测试必须覆盖实际训练配置（gradient_checkpointing=True），
而非只用简化配置（gradient_checkpointing=False）。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


class TestStage1GradientCheckpointing:
    """Stage 1 (frozen UNet) + gradient_checkpointing=True 场景测试

    这是真实训练配置：UNet 冻结、gradient_checkpointing 启用、CondEncoder 可训练。
    之前 smoke test 用 gradient_checkpointing=False 跑过了，但实际训练用 True 失败。
    """

    def _build_model_with_gc(self):
        """构建启用 gradient_checkpointing 的小尺寸模型"""
        from projects.ChromoGen.models.chromogen_pipeline import (
            ChromoGenPipeline,
        )

        return ChromoGenPipeline(
            enable_bbox_head=False,
            sample_size=32,
            unet_block_out_channels=(32, 64, 128, 128),
            unet_attention_head_dim=4,
            cross_attention_dim=128,
            condition_embed_dim=128,
            condition_dropout=0.0,
            cfg_dropout=0.0,  # 显式关闭 CFG dropout，避免 flaky 测试
            #  CFG dropout 的梯度问题由 test_cfg_dropout_gradient.py 专门覆盖
            noise_schedule='karras',
            prediction_type='v_prediction',
            gradient_checkpointing=True,  # 真实训练配置
        )

    def _make_dummy_batch(self):
        return {
            'pixel_values': torch.randn(2, 3, 256, 256),
            'class_labels': torch.randint(0, 24, (2, 24)),
            'counts': torch.randint(1, 5, (2, 24)),
        }

    def test_stage1_gc_enabled_forward_backward_no_error(self):
        """Stage 1 + gradient_checkpointing=True 时 forward+backward 应正常

        重现实际训练报错场景：UNet 冻结 + GC 启用 → backward 失败。
        修复后应能正常 backward。
        """
        model = self._build_model_with_gc()
        model.train()

        # Stage 1: 冻结 UNet
        model.freeze_unet()

        # 验证 UNet 冻结
        unet_trainable = any(p.requires_grad for p in model.unet.parameters())
        assert not unet_trainable, 'UNet 应已冻结'

        # 验证 CondEncoder 仍可训练
        cond_trainable = any(
            p.requires_grad for p in model.condition_encoder.parameters()
        )
        assert cond_trainable, 'CondEncoder 应可训练'

        batch = self._make_dummy_batch()
        losses = model(
            pixel_values=batch['pixel_values'],
            class_labels=batch['class_labels'],
            counts=batch['counts'],
        )
        loss = losses['loss_img']

        # loss 应有 grad_fn（依赖 CondEncoder 参数）
        assert loss.grad_fn is not None, (
            'loss 应有 grad_fn，否则 backward 会报错 '
            '（根因：UNet 冻结 + gradient_checkpointing 导致 grad_fn 丢失）'
        )

        # backward 应能正常完成
        loss.backward()

        # CondEncoder 应有梯度
        has_cond_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.condition_encoder.parameters()
        )
        assert has_cond_grad, 'CondEncoder 应有梯度（Stage 1 训练目标）'

    def test_stage2_gc_enabled_forward_backward_no_error(self):
        """Stage 2 + gradient_checkpointing=True 时 forward+backward 应正常"""
        model = self._build_model_with_gc()
        model.train()

        # Stage 2: UNet 解冻
        model.unfreeze_unet()

        batch = self._make_dummy_batch()
        losses = model(
            pixel_values=batch['pixel_values'],
            class_labels=batch['class_labels'],
            counts=batch['counts'],
        )
        loss = losses['loss_img']
        assert loss.grad_fn is not None
        loss.backward()

        # UNet 和 CondEncoder 都应有梯度
        has_unet_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.unet.parameters()
        )
        has_cond_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.condition_encoder.parameters()
        )
        assert has_unet_grad, 'UNet 应有梯度'
        assert has_cond_grad, 'CondEncoder 应有梯度'


class TestFreezeUnetDisablesGradientCheckpointing:
    """验证 freeze_unet 时自动关闭 gradient_checkpointing

    修复方案：UNet 冻结时无需 gradient_checkpointing（不省显存反而破坏梯度），
    freeze_unet() 应自动调用 unet.disable_gradient_checkpointing()。
    unfreeze_unet() 应根据需要恢复。
    """

    def test_freeze_unet_disables_gc(self):
        """freeze_unet() 应关闭 UNet 的 gradient_checkpointing"""
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
            cfg_dropout=0.0,  # 显式关闭，避免 flaky
            gradient_checkpointing=True,
        )
        # 验证初始 GC 启用（UNet2DConditionModel 用 is_gradient_checkpointing）
        assert model.unet.unet.is_gradient_checkpointing

        # freeze_unet 应自动关闭 GC
        model.freeze_unet()
        assert not model.unet.unet.is_gradient_checkpointing, (
            'freeze_unet 后应关闭 gradient_checkpointing，'
            '否则 backward 会因 UNet 参数 frozen + GC 而失败'
        )

    def test_unfreeze_unet_restores_gc(self):
        """unfreeze_unet() 应恢复 UNet 的 gradient_checkpointing"""
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
            cfg_dropout=0.0,  # 显式关闭，避免 flaky
            gradient_checkpointing=True,
        )
        model.freeze_unet()
        assert not model.unet.unet.is_gradient_checkpointing

        model.unfreeze_unet()
        assert model.unet.unet.is_gradient_checkpointing, (
            'unfreeze_unet 后应恢复 gradient_checkpointing'
        )


class TestStage1Fp16GradScaler:
    """Stage 1 + fp16 autocast + GradScaler 场景测试

    这是实际训练报错的精确复现场景：
      UNet 冻结 + fp16 autocast + GradScaler + gradient_checkpointing
      报错：element 0 of tensors does not require grad and does not have a grad_fn
    """

    def test_stage1_fp16_gradscaler_backward_no_error(self):
        """Stage 1 + fp16 + GradScaler 时 loss 应保留 grad_fn

        精确复现实际训练报错场景：
          with autocast(fp16):
              losses = model(batch)
              loss = losses['loss_img']
          scaler.scale(loss).backward()  # 报错：no grad_fn

        CPU 测试不用 GradScaler（CUDA 专用），只验证 loss 保留了 grad_fn。
        """
        from projects.ChromoGen.models.chromogen_pipeline import (
            ChromoGenPipeline,
        )

        # 防御测试间状态污染：确保 grad 全局开启
        prev_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        try:
            model = ChromoGenPipeline(
                enable_bbox_head=False,
                sample_size=32,
                unet_block_out_channels=(32, 64, 128, 128),
                unet_attention_head_dim=4,
                cross_attention_dim=128,
                condition_embed_dim=128,
                condition_dropout=0.0,
                cfg_dropout=0.0,  # 显式关闭，避免 flaky
                noise_schedule='karras',
                prediction_type='v_prediction',
                gradient_checkpointing=True,
            )
            model.train()
            model.freeze_unet()  # Stage 1

            batch = {
                'pixel_values': torch.randn(2, 3, 256, 256),
                'class_labels': torch.randint(0, 24, (2, 24)),
                'counts': torch.randint(1, 5, (2, 24)),
            }

            # 模拟 fp16 autocast（CPU 用 bf16 替代）
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                losses = model(
                    pixel_values=batch['pixel_values'],
                    class_labels=batch['class_labels'],
                    counts=batch['counts'],
                )
                loss = losses['loss_img']

            # 核心验证：loss 应有 grad_fn
            assert loss.requires_grad, (
                'bf16 + GC + frozen UNet 下 loss 应仍 requires_grad=True'
            )
            assert loss.grad_fn is not None, (
                'bf16 + GC + frozen UNet 下 loss 应有 grad_fn，'
                '否则 scaler.scale(loss).backward() 报 "does not have a grad_fn"'
            )

            # 普通 backward 也应正常
            loss.backward()
            has_cond_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.condition_encoder.parameters()
            )
            assert has_cond_grad, 'bf16 + GC 下 CondEncoder 应有梯度'
        finally:
            torch.set_grad_enabled(prev_grad)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
