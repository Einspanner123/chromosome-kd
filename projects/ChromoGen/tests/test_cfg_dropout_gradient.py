"""CFG Dropout + frozen UNet 梯度图完整性测试

根因分析：
  当 cfg_dropout 触发时，原实现用 torch.zeros() 替换 condition（不要求梯度）。
  在 Stage 1（UNet 冻结 + VAE no_grad 编码）下，整个计算图无梯度入口，
  loss 无 grad_fn，backward 报错：
    "element 0 of tensors does not require grad and does not have a grad_fn"

  此 bug 是间歇性的：cfg_dropout=0.1 → 每步 10% 概率崩溃。
  这解释了之前测试的 flaky 行为（"单独跑通过，完整套件中失败"）：
  完整套件改变了全局随机状态，使 10% 概率更容易命中。

修复方案：
  CFG dropout 时仍运行 condition_encoder，然后置零（condition * 0.0），
  保留 grad_fn 使梯度可回传到 CondEncoder（梯度值为0，不影响训练语义）。

红绿重构纪律：
  RED  — 本测试文件应在修复前失败（loss 无 grad_fn）
  GREEN— 修复 chromogen_pipeline.py 后本测试应全部通过
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


class TestCfgDropoutGradient:
    """CFG Dropout 梯度图完整性测试

    覆盖实际训练中触发 backward 崩溃的场景：
      Stage 1 (frozen UNet) + cfg_dropout 触发 → loss 无 grad_fn
    """

    def _build_small_model(self, cfg_dropout=0.0, gc=True):
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
            condition_dropout=0.0,  # encoder 内部 dropout，与 CFG dropout 无关
            cfg_dropout=cfg_dropout,
            noise_schedule='scaled_linear',
            prediction_type='v_prediction',
            gradient_checkpointing=gc,
        )

    def _make_dummy_batch(self):
        return {
            'pixel_values': torch.randn(2, 3, 256, 256),
            'class_labels': torch.randint(0, 24, (2, 24)),
            'counts': torch.randint(1, 5, (2, 24)),
        }

    # ----------------------------------------------------------
    # RED 测试：以下测试在修复前应失败（loss 无 grad_fn）
    # ----------------------------------------------------------

    def test_cfg_dropout_frozen_unet_loss_has_grad_fn(self):
        """Stage 1 + CFG dropout=1.0 时 loss 应保留 grad_fn

        精确复现实际训练报错：
          cfg_dropout 触发 → condition=torch.zeros() 无梯度
          → frozen UNet + no-grad latents → loss 无 grad_fn
          → backward 报 "does not have a grad_fn"
        """
        model = self._build_small_model(cfg_dropout=1.0)  # 强制 CFG dropout
        model.train()
        model.freeze_unet()  # Stage 1

        batch = self._make_dummy_batch()
        losses = model(
            pixel_values=batch['pixel_values'],
            class_labels=batch['class_labels'],
            counts=batch['counts'],
        )
        loss = losses['loss_img']

        # 核心验证：loss 应有 grad_fn
        assert loss.requires_grad, (
            'CFG dropout + frozen UNet 下 loss 应 requires_grad=True，'
            '否则 scaler.scale(loss).backward() 报错'
        )
        assert loss.grad_fn is not None, (
            'CFG dropout + frozen UNet 下 loss 应有 grad_fn，'
            '否则 backward 报 "element 0 of tensors does not require grad '
            'and does not have a grad_fn"'
        )

        # backward 应能正常完成
        loss.backward()

        # CondEncoder 应有梯度（虽然 CFG dropout 置零，梯度回传应为 0，
        # 但 backward 不应报错）
        for p in model.condition_encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None, (
                    'CondEncoder 可训练参数 backward 后应有 .grad（值可为0）'
                )

    def test_cfg_dropout_frozen_unet_bf16_loss_has_grad_fn(self):
        """bf16 autocast + Stage 1 + CFG dropout=1.0 应保留 grad_fn

        实际训练使用 fp16 autocast + GradScaler。
        CPU 测试用 bf16 替代（CPU 不支持 fp16 autocast）。
        """
        model = self._build_small_model(cfg_dropout=1.0)
        model.train()
        model.freeze_unet()

        batch = self._make_dummy_batch()

        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            losses = model(
                pixel_values=batch['pixel_values'],
                class_labels=batch['class_labels'],
                counts=batch['counts'],
            )
            loss = losses['loss_img']

        assert loss.requires_grad, (
            'bf16 + CFG dropout + frozen UNet 下 loss 应 requires_grad=True'
        )
        assert loss.grad_fn is not None, (
            'bf16 + CFG dropout + frozen UNet 下 loss 应有 grad_fn'
        )
        loss.backward()

    def test_cfg_dropout_unfrozen_unet_loss_has_grad_fn(self):
        """Stage 2 + CFG dropout=1.0 时 loss 应保留 grad_fn

        Stage 2 UNet 解冻，即使 condition=zeros 也能通过 UNet 参数
        维持梯度图。但修复后行为应一致（condition * 0.0 保留 grad_fn）。
        """
        model = self._build_small_model(cfg_dropout=1.0)
        model.train()
        model.unfreeze_unet()  # Stage 2

        batch = self._make_dummy_batch()
        losses = model(
            pixel_values=batch['pixel_values'],
            class_labels=batch['class_labels'],
            counts=batch['counts'],
        )
        loss = losses['loss_img']

        assert loss.grad_fn is not None
        loss.backward()

    # ----------------------------------------------------------
    # 正确性验证：CFG dropout 置零语义
    # ----------------------------------------------------------

    def test_cfg_dropout_condition_is_zeroed(self):
        """CFG dropout 触发时传入 UNet 的 condition 应全为 0

        修复后 condition = encoder(...) * 0.0，值全为 0 但保留 grad_fn。
        """
        model = self._build_small_model(cfg_dropout=1.0)
        model.train()

        # 用 forward hook 捕获传入 UNet 的 encoder_hidden_states
        captured = {}

        original_unet_forward = model.unet.unet.forward

        def hooked_forward(*args, **kwargs):
            # UNet2DConditionModel.forward 接收 encoder_hidden_states
            if 'encoder_hidden_states' in kwargs:
                captured['condition'] = kwargs['encoder_hidden_states']
            elif len(args) >= 3:
                captured['condition'] = args[2]
            return original_unet_forward(*args, **kwargs)

        model.unet.unet.forward = hooked_forward
        try:
            batch = self._make_dummy_batch()
            model(
                pixel_values=batch['pixel_values'],
                class_labels=batch['class_labels'],
                counts=batch['counts'],
            )
        finally:
            model.unet.unet.forward = original_unet_forward

        cond = captured.get('condition')
        assert cond is not None, '应捕获到传入 UNet 的 condition'
        # 值应全为 0（CFG dropout 置零）
        assert torch.all(cond == 0), 'CFG dropout 触发时 condition 应全为 0'
        # 但应保留 grad_fn（修复后）
        assert cond.requires_grad, (
            'CFG dropout 置零的 condition 应保留 requires_grad，'
            '否则 frozen UNet 下梯度图断裂'
        )

    def test_no_cfg_dropout_condition_not_zeroed(self):
        """cfg_dropout=0.0 时 condition 不应被置零"""
        model = self._build_small_model(cfg_dropout=0.0)
        model.train()

        captured = {}
        original_unet_forward = model.unet.unet.forward

        def hooked_forward(*args, **kwargs):
            if 'encoder_hidden_states' in kwargs:
                captured['condition'] = kwargs['encoder_hidden_states']
            elif len(args) >= 3:
                captured['condition'] = args[2]
            return original_unet_forward(*args, **kwargs)

        model.unet.unet.forward = hooked_forward
        try:
            batch = self._make_dummy_batch()
            model(
                pixel_values=batch['pixel_values'],
                class_labels=batch['class_labels'],
                counts=batch['counts'],
            )
        finally:
            model.unet.unet.forward = original_unet_forward

        cond = captured.get('condition')
        assert cond is not None
        assert not torch.all(cond == 0), (
            'cfg_dropout=0 时 condition 不应全为 0'
        )

    def test_cfg_dropout_grad_is_zero_for_encoder(self):
        """CFG dropout 触发时 CondEncoder 梯度应为 0

        语义验证：condition = encoder_output * 0.0，
        d(loss)/d(encoder_params) = 0（因为 condition 被置零）。
        这符合 CFG dropout 的训练语义——dropped-out step 不更新 encoder。
        """
        model = self._build_small_model(cfg_dropout=1.0)
        model.train()
        model.freeze_unet()  # Stage 1

        batch = self._make_dummy_batch()
        losses = model(
            pixel_values=batch['pixel_values'],
            class_labels=batch['class_labels'],
            counts=batch['counts'],
        )
        losses['loss_img'].backward()

        # 所有 CondEncoder 可训练参数的梯度应为 0
        for name, p in model.condition_encoder.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f'{name} backward 后应有 .grad'
                assert p.grad.abs().sum() == 0, (
                    f'{name} 梯度应为 0（CFG dropout 置零，'
                    f'此 step 不应更新 encoder）'
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
