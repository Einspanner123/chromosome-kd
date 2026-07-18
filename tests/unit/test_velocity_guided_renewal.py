"""测试 Velocity-Guided Adaptive Renewal (VGAR)

方向4: box_renewal×RF速度场耦合
将纯随机 renewal 替换为基于 v_θ 预测 x0 的自适应 renewal。

核心思想: renewal 时不完全重置为随机噪声，而是部分保留 v_θ 预测的 x0 方向 (利用)，
同时保持随机扰动 (探索)，且利用比例随时间步自适应 (早期更随机，后期更确定)。
"""

import torch

from ldmdet.diffusion.sampling import DiffusionSampler


class TestVelocityGuidedRenewal:
    """VGAR: 利用 v_θ 的 x0 预测指导 box renewal"""

    def _build_sampler(self, velocity_guided_renewal=False):
        """构建测试用 sampler"""
        return DiffusionSampler(
            diffusion_type='rectified_flow',
            timesteps=1000,
            sampling_timesteps=4,
            solver_type='dpm_solver_pp',
            ddim_sampling_eta=1.0,
            rf_schedule='shifted',
            rf_power=1.0,
            rf_shift=3.0,
            snr_scale=2.0,
            box_renewal=True,
            use_ensemble=True,
            use_nms=True,
            nms_thr=0.5,
            score_thr=0.05,
            min_keep=10,
            velocity_guided_renewal=velocity_guided_renewal,  # 新参数
        )

    def test_vgar_parameter_exists_when_enabled(self):
        """启用 VGAR 时, sampler 应记录该配置"""
        sampler = self._build_sampler(velocity_guided_renewal=True)
        assert sampler.velocity_guided_renewal is True

    def test_vgar_disabled_by_default(self):
        """默认不启用 VGAR, 保持向后兼容"""
        sampler = self._build_sampler()
        assert sampler.velocity_guided_renewal is False

    def test_vgar_uses_x0_pred_when_enabled(self):
        """VGAR 启用时, renewal 应使用 x0_pred 而非纯随机"""
        sampler = self._build_sampler(velocity_guided_renewal=True)
        bs, n = 2, 20
        x_raw = torch.randn(bs, n, 4)
        # 构造低置信度 logits, 使部分框被 renewal
        cls_logits = torch.full(
            (bs, n, 24), -10.0
        )  # sigmoid(-10)≈0, 所有框低置信度
        cls_logits[:, :5, :] = 10.0  # 前 5 个框高置信度
        x0_pred = torch.ones(bs, n, 4) * 3.0  # 明显的 x0 预测值

        t_curr = 0.25  # 后期 step, alpha 应该较大
        result = sampler.apply_box_renewal(
            x_raw, cls_logits, x0_pred=x0_pred, t_curr=t_curr
        )

        # 被 renewal 的框 (索引 5-19) 应该接近 x0_pred (=3.0) 而非纯随机
        renewed_boxes = result[:, 5:, :]
        # alpha(0.25) = 0.2 + 0.6*sigmoid(5*(0.5-0.25)) ≈ 0.2 + 0.6*0.78 ≈ 0.67
        # renewed ≈ 0.67*3.0 + 0.33*randn ≈ 2.0 + noise
        # 均值应明显偏向 3.0 而非 0 (纯随机均值 ≈ 0)
        assert renewed_boxes.mean() > 1.0

    def test_vgar_backward_compatible_without_x0(self):
        """VGAR 禁用或不传 x0_pred 时, 行为与原始完全一致 (纯随机)"""
        sampler = self._build_sampler(velocity_guided_renewal=False)
        bs, n = 2, 20
        x_raw = torch.randn(bs, n, 4)
        cls_logits = torch.full((bs, n, 24), -10.0)
        cls_logits[:, :5, :] = 10.0

        # 不传 x0_pred 时, 应该走纯随机路径
        result = sampler.apply_box_renewal(x_raw, cls_logits)
        renewed_boxes = result[:, 5:, :]
        # 纯随机 renewal, 均值应接近 0
        assert abs(renewed_boxes.mean().item()) < 0.5

    def test_vgar_time_adaptive_alpha(self):
        """VGAR 的 alpha 应随时间步自适应: 早期 (t 大) 更随机, 后期 (t 小) 更确定"""
        sampler = self._build_sampler(velocity_guided_renewal=True)
        bs, n = 1, 20
        x_raw = torch.zeros(bs, n, 4)
        cls_logits = torch.full((bs, n, 24), -10.0)  # 全部低置信度
        cls_logits[:, :10, :] = 10.0  # 前 10 个保留
        x0_pred = torch.ones(bs, n, 4) * 5.0

        # 早期 step (t=0.75): alpha 小, 更随机, renewed 均值远离 x0_pred
        result_early = sampler.apply_box_renewal(
            x_raw, cls_logits, x0_pred=x0_pred, t_curr=0.75
        )
        renewed_early = result_early[:, 10:, :]

        # 后期 step (t=0.25): alpha 大, 更确定, renewed 均值接近 x0_pred
        result_late = sampler.apply_box_renewal(
            x_raw, cls_logits, x0_pred=x0_pred, t_curr=0.25
        )
        renewed_late = result_late[:, 10:, :]

        # 后期 renewal 应更接近 x0_pred (=5.0)
        assert abs(renewed_late.mean().item() - 5.0) < abs(
            renewed_early.mean().item() - 5.0
        )

    def test_vgar_preserves_high_confidence_boxes(self):
        """VGAR 不应修改高置信度框"""
        sampler = self._build_sampler(velocity_guided_renewal=True)
        bs, n = 1, 20
        x_raw = torch.randn(bs, n, 4)
        cls_logits = torch.full((bs, n, 24), -10.0)
        cls_logits[:, :10, :] = 10.0  # 前 10 个高置信度
        x0_pred = torch.ones(bs, n, 4) * 3.0

        result = sampler.apply_box_renewal(
            x_raw, cls_logits, x0_pred=x0_pred, t_curr=0.25
        )
        # 高置信度框 (前 10 个) 不应被修改
        assert torch.allclose(result[:, :10, :], x_raw[:, :10, :])
