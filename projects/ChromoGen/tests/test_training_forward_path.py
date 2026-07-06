"""v2 训练前向路径集成测试

补充覆盖：之前 RED 阶段只测 scheduler 类型，没测 model.forward() 在 v2 配置下能否完整运行。
导致 GREEN 阶段测试全绿但实际训练报错：
  'KDPM2DiscreteScheduler' object has no attribute 'get_velocity'

红绿重构纪律要求：测试必须覆盖实际代码路径，包括训练前向。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


# ============================================================
# 训练前向路径集成测试（覆盖各种 noise_schedule）
# ============================================================


class TestTrainingForwardPath:
    """验证 model.forward() 在不同 noise_schedule 下都能正常运行

    之前只测 scheduler 类型，没测 forward 路径，
    导致 KDPM2DiscreteScheduler 缺 get_velocity 方法未被早期发现。
    """

    def _build_small_model(
        self, noise_schedule: str, prediction_type: str = 'v_prediction'
    ):
        """构建小尺寸模型用于前向测试

        Args:
            noise_schedule: 'linear' / 'scaled_linear' / 'karras'
            prediction_type: 'epsilon' / 'v_prediction'
        """
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
            cfg_dropout=0.0,  # 显式关闭，避免 flaky
            noise_schedule=noise_schedule,
            prediction_type=prediction_type,
        )

    def _make_dummy_batch(self, device='cpu'):
        """构造小尺寸 dummy batch 用于前向测试

        注意：ChromoGenPipeline.forward() 的签名是
            forward(pixel_values, class_labels, counts, gt_bboxes=None, gt_labels=None)
        不接受 dict batch，需要按位置传参。

        ConditionEncoder 期望：
            class_labels: [B, num_classes=24] 各类别的索引 (0-23)
            counts: [B, num_classes=24] 各类别的数量
        """
        batch_size = 2
        return {
            'pixel_values': torch.randn(
                batch_size, 3, 256, 256, device=device
            ),
            'class_labels': torch.randint(
                0, 24, (batch_size, 24), device=device
            ),
            'counts': torch.randint(1, 5, (batch_size, 24), device=device),
        }

    def _call_forward(self, model, batch):
        """按 forward 签名调用，而非传 dict"""
        return model(
            pixel_values=batch['pixel_values'],
            class_labels=batch['class_labels'],
            counts=batch['counts'],
        )

    @pytest.mark.parametrize(
        'noise_schedule',
        ['linear', 'scaled_linear', 'karras'],
    )
    def test_forward_v_prediction_runs_without_error(self, noise_schedule):
        """v_prediction 模式下，所有 noise_schedule 都应支持 forward

        这是之前漏测的关键场景：KDPM2 在 v_prediction 下调用 get_velocity 会失败。
        """
        model = self._build_small_model(
            noise_schedule=noise_schedule, prediction_type='v_prediction'
        )
        model.train()
        batch = self._make_dummy_batch()

        # 应能正常运行 forward，不抛异常
        losses = self._call_forward(model, batch)
        assert 'loss_img' in losses
        assert torch.isfinite(losses['loss_img']), (
            f'{noise_schedule} v_prediction loss 应为有限值，'
            f'实际: {losses["loss_img"]}'
        )

    @pytest.mark.parametrize(
        'noise_schedule',
        ['linear', 'scaled_linear', 'karras'],
    )
    def test_forward_epsilon_runs_without_error(self, noise_schedule):
        """epsilon 模式下，所有 noise_schedule 都应支持 forward"""
        model = self._build_small_model(
            noise_schedule=noise_schedule, prediction_type='epsilon'
        )
        model.train()
        batch = self._make_dummy_batch()

        losses = self._call_forward(model, batch)
        assert 'loss_img' in losses
        assert torch.isfinite(losses['loss_img'])

    def test_karras_scheduler_has_get_velocity_or_equivalent(self):
        """Karras scheduler 应支持训练所需的 get_velocity（或等效方法）

        如果用 KDPM2DiscreteScheduler，它不支持训练，应该用 DDPMScheduler 替代。
        这个测试验证 Karras 模式下 scheduler 真的能用于训练。
        """
        model = self._build_small_model(
            noise_schedule='karras', prediction_type='v_prediction'
        )
        # 必须有 get_velocity 方法（训练用）
        assert hasattr(model.noise_scheduler, 'get_velocity'), (
            f'{type(model.noise_scheduler).__name__} 必须支持 get_velocity 用于训练'
        )

    def test_karras_scheduler_has_add_noise(self):
        """Karras scheduler 应支持训练所需的 add_noise"""
        model = self._build_small_model(
            noise_schedule='karras', prediction_type='v_prediction'
        )
        assert hasattr(model.noise_scheduler, 'add_noise'), (
            f'{type(model.noise_scheduler).__name__} 必须支持 add_noise 用于训练'
        )


# ============================================================
# 完整训练 step 模拟（模拟 train.py 中的实际 forward + backward）
# ============================================================


class TestFullTrainingStepSimulation:
    """模拟 train.py 中的完整训练 step：forward + backward + optimizer.step"""

    def test_karras_v_prediction_full_step_no_error(self):
        """Karras + v_prediction 配置下，完整训练 step 不应报错

        模拟 train.py 中：
            losses = model(batch)
            loss = losses['loss_img']
            loss.backward()
            optimizer.step()
        """
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
            noise_schedule='karras',
            prediction_type='v_prediction',
        )
        model.train()

        batch = {
            'pixel_values': torch.randn(2, 3, 256, 256),
            'class_labels': torch.randint(0, 24, (2, 24)),
            'counts': torch.randint(1, 5, (2, 24)),
        }

        losses = model(
            pixel_values=batch['pixel_values'],
            class_labels=batch['class_labels'],
            counts=batch['counts'],
        )
        loss = losses['loss_img']
        # backward 不应报错
        loss.backward()
        # 梯度应存在
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, '反向传播后应至少有一个参数有梯度'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
