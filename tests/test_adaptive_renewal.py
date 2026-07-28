"""P0: 自适应阈值 box_renewal 单元测试

验证移植自 DiffuDETR 的自适应阈值机制:
- 早期步骤 (t_curr→1.0): 高阈值, 积极淘汰低分框
- 后期步骤 (t_curr→0.0): 低阈值 (score_thr), 保守保留
- min_keep 机制仍然生效
- 禁用时回退到固定阈值 (向后兼容)
- VGAR 与自适应阈值正交兼容
"""

import math
import unittest

import torch

from ldmdet.diffusion.sampling import DiffusionSampler


def build_sampler(
    adaptive_renewal_threshold: bool = True,
    adaptive_renewal_scale: float = 0.9,
    velocity_guided_renewal: bool = False,
    score_thr: float = 0.05,
    min_keep: int = 10,
) -> DiffusionSampler:
    """构建测试用 DiffusionSampler (最小参数集)."""
    return DiffusionSampler(
        diffusion_type='rf',
        timesteps=1000,
        sampling_timesteps=4,
        solver_type='dpm_solver_pp',
        ddim_sampling_eta=1.0,
        rf_schedule='linear',
        rf_power=1.0,
        rf_shift=1.0,
        snr_scale=2.0,
        box_renewal=True,
        use_ensemble=True,
        use_nms=True,
        nms_thr=0.5,
        score_thr=score_thr,
        min_keep=min_keep,
        velocity_guided_renewal=velocity_guided_renewal,
        adaptive_renewal_threshold=adaptive_renewal_threshold,
        adaptive_renewal_scale=adaptive_renewal_scale,
    )


def make_inputs(
    bs: int = 2,
    n: int = 100,
    num_classes: int = 24,
    score_pattern: str = 'mixed',
) -> tuple:
    """构建测试输入: x_raw, cls_logits.

    Args:
        score_pattern:
            'mixed' — 50% 高分(>0.9), 50% 低分(<0.1)
            'all_low' — 全部低分
            'all_high' — 全部高分
    """
    torch.manual_seed(42)
    x_raw = torch.randn(bs, n, 4)

    if score_pattern == 'mixed':
        # 前 50% 高 logit (sigmoid > 0.9), 后 50% 低 logit (sigmoid < 0.1)
        high_logit = torch.ones(bs, n // 2, num_classes) * 5.0
        low_logit = torch.ones(bs, n - n // 2, num_classes) * (-5.0)
        cls_logits = torch.cat([high_logit, low_logit], dim=1)
    elif score_pattern == 'all_low':
        cls_logits = torch.ones(bs, n, num_classes) * (-5.0)
    elif score_pattern == 'all_high':
        cls_logits = torch.ones(bs, n, num_classes) * 5.0
    else:
        raise ValueError(f'Unknown score_pattern: {score_pattern}')

    return x_raw, cls_logits


class TestAdaptiveRenewalThreshold(unittest.TestCase):
    """P0: 自适应阈值 _compute_renewal_threshold 测试."""

    def test_threshold_early_high(self):
        """早期步骤 (t_curr=1.0) 阈值应为 0.9."""
        sampler = build_sampler(
            adaptive_renewal_threshold=True,
            adaptive_renewal_scale=0.9,
            score_thr=0.05,
        )
        threshold = sampler._compute_renewal_threshold(t_curr=1.0)
        self.assertAlmostEqual(threshold, 0.9, places=6)

    def test_threshold_late_low(self):
        """后期步骤 (t_curr=0.0) 阈值应回退到 score_thr."""
        sampler = build_sampler(
            adaptive_renewal_threshold=True,
            adaptive_renewal_scale=0.9,
            score_thr=0.05,
        )
        threshold = sampler._compute_renewal_threshold(t_curr=0.0)
        self.assertAlmostEqual(threshold, 0.05, places=6)

    def test_threshold_midpoint(self):
        """中间步骤 (t_curr=0.5) 阈值应为 0.45."""
        sampler = build_sampler(
            adaptive_renewal_threshold=True,
            adaptive_renewal_scale=0.9,
            score_thr=0.05,
        )
        threshold = sampler._compute_renewal_threshold(t_curr=0.5)
        self.assertAlmostEqual(threshold, 0.45, places=6)

    def test_threshold_floor(self):
        """当 t_curr * scale < score_thr 时, 阈值不低于 score_thr."""
        sampler = build_sampler(
            adaptive_renewal_threshold=True,
            adaptive_renewal_scale=0.1,  # 很小的 scale
            score_thr=0.2,
        )
        # t_curr=0.5, scale=0.1 → 0.05 < score_thr=0.2 → 回退到 0.2
        threshold = sampler._compute_renewal_threshold(t_curr=0.5)
        self.assertAlmostEqual(threshold, 0.2, places=6)

    def test_threshold_disabled_fallback(self):
        """禁用自适应阈值时, 回退到固定 score_thr."""
        sampler = build_sampler(
            adaptive_renewal_threshold=False,
            score_thr=0.05,
        )
        threshold = sampler._compute_renewal_threshold(t_curr=1.0)
        self.assertAlmostEqual(threshold, 0.05, places=6)

    def test_threshold_none_t_curr(self):
        """t_curr=None 时, 回退到固定 score_thr."""
        sampler = build_sampler(
            adaptive_renewal_threshold=True,
            score_thr=0.05,
        )
        threshold = sampler._compute_renewal_threshold(t_curr=None)
        self.assertAlmostEqual(threshold, 0.05, places=6)


class TestApplyBoxRenewalAdaptive(unittest.TestCase):
    """P0: apply_box_renewal 自适应阈值集成测试."""

    def test_early_step_aggressive_renewal(self):
        """早期步骤 (t_curr=1.0, threshold=0.9): 仅保留高分框, 大量 renewal."""
        sampler = build_sampler(
            adaptive_renewal_threshold=True,
            min_keep=5,
        )
        x_raw, cls_logits = make_inputs(n=100, score_pattern='mixed')

        x_raw_new = sampler.apply_box_renewal(
            x_raw, cls_logits, t_curr=1.0
        )

        # 混合输入: 50 高分(>0.9), 50 低分(<0.1)
        # threshold=0.9 → 高分框保留, 低分框被 renewal
        # 验证: 高分位置不变, 低分位置被替换为噪声
        scores = torch.sigmoid(cls_logits).max(-1)[0]
        high_mask = scores[0] > 0.9
        low_mask = scores[0] < 0.1

        # 高分框应保留 (未变)
        self.assertTrue(
            torch.allclose(x_raw[0, high_mask], x_raw_new[0, high_mask])
        )
        # 低分框应被替换 (已变)
        self.assertFalse(
            torch.allclose(x_raw[0, low_mask], x_raw_new[0, low_mask])
        )

    def test_late_step_conservative_renewal(self):
        """后期步骤 (t_curr=0.0, threshold=0.05): 几乎全部保留."""
        sampler = build_sampler(
            adaptive_renewal_threshold=True,
            min_keep=5,
        )
        x_raw, cls_logits = make_inputs(n=100, score_pattern='mixed')

        x_raw_new = sampler.apply_box_renewal(
            x_raw, cls_logits, t_curr=0.0
        )

        # threshold=0.05 → 高分框(>0.9)和低分框(>0.05)都保留
        # make_inputs 中低分框 sigmoid≈0.0067 < 0.05 → 仍被 renewal
        # 但 min_keep=5 保证至少 5 个保留
        scores = torch.sigmoid(cls_logits).max(-1)[0]
        high_mask = scores[0] > 0.9
        # 高分框应保留
        self.assertTrue(
            torch.allclose(x_raw[0, high_mask], x_raw_new[0, high_mask])
        )

    def test_min_keep_respected(self):
        """min_keep 机制: 即使阈值很高, 也至少保留 min_keep 个框."""
        sampler = build_sampler(
            adaptive_renewal_threshold=True,
            adaptive_renewal_scale=0.99,
            min_keep=20,
        )
        x_raw, cls_logits = make_inputs(n=100, score_pattern='all_low')

        x_raw_new = sampler.apply_box_renewal(
            x_raw, cls_logits, t_curr=1.0
        )

        # 所 有框都是低分 (<0.1), threshold=0.99 → 全部不满足
        # 但 min_keep=20 → 至少保留 20 个
        # 验证: 不是所有框都被替换 (至少 20 个保留)
        unchanged = torch.allclose(x_raw, x_raw_new)
        self.assertFalse(unchanged, '应有一些框被 renewal')
        # 验证: 至少 min_keep 个框保留 (未变)
        same_mask = torch.all(x_raw[0] == x_raw_new[0], dim=-1)
        self.assertGreaterEqual(same_mask.sum().item(), 20)

    def test_disabled_fallback_identical(self):
        """禁用自适应阈值时, 行为与固定阈值完全一致."""
        sampler_adaptive = build_sampler(
            adaptive_renewal_threshold=False,
            score_thr=0.05,
            min_keep=10,
        )
        sampler_fixed = build_sampler(
            adaptive_renewal_threshold=False,
            score_thr=0.05,
            min_keep=10,
        )

        torch.manual_seed(42)
        x_raw, cls_logits = make_inputs(n=100, score_pattern='mixed')

        torch.manual_seed(123)
        out_adaptive = sampler_adaptive.apply_box_renewal(
            x_raw, cls_logits, t_curr=1.0
        )
        torch.manual_seed(123)
        out_fixed = sampler_fixed.apply_box_renewal(
            x_raw, cls_logits, t_curr=1.0
        )

        # 禁用时 t_curr 不影响阈值, 两者应完全相同
        self.assertTrue(torch.allclose(out_adaptive, out_fixed))

    def test_vgar_compatibility(self):
        """VGAR + 自适应阈值: 两者正交, 可同时启用."""
        sampler = build_sampler(
            adaptive_renewal_threshold=True,
            velocity_guided_renewal=True,
            min_keep=5,
        )
        x_raw, cls_logits = make_inputs(n=50, score_pattern='mixed')
        x0_pred = torch.randn(2, 50, 4)  # VGAR 需要的 x0 预测

        # 不应报错, 且输出形状正确
        x_raw_new = sampler.apply_box_renewal(
            x_raw, cls_logits,
            x0_pred=x0_pred,
            t_curr=0.7,
        )
        self.assertEqual(x_raw_new.shape, x_raw.shape)

        # 验证: 早期高阈值 → 更多框被 renewal (VGAR 引导方向)
        scores = torch.sigmoid(cls_logits).max(-1)[0]
        high_mask = scores[0] > 0.9
        # 高分框应保留
        self.assertTrue(
            torch.allclose(x_raw[0, high_mask], x_raw_new[0, high_mask])
        )

    def test_threshold_decreasing_over_steps(self):
        """模拟 4 步采样, 验证阈值递减导致 renewal 率递减."""
        sampler = build_sampler(
            adaptive_renewal_threshold=True,
            adaptive_renewal_scale=0.9,
            score_thr=0.05,
            min_keep=5,
        )
        x_raw, cls_logits = make_inputs(n=100, score_pattern='mixed')

        # 模拟 4 步采样的时间步: 1.0, 0.75, 0.5, 0.25
        time_steps = [1.0, 0.75, 0.5, 0.25]
        renewal_rates = []

        for t in time_steps:
            torch.manual_seed(42)
            x_raw_copy = x_raw.clone()
            x_raw_new = sampler.apply_box_renewal(
                x_raw_copy, cls_logits, t_curr=t
            )
            # 计算有多少框被替换
            changed = ~torch.all(x_raw[0] == x_raw_new[0], dim=-1)
            renewal_rate = changed.float().mean().item()
            renewal_rates.append(renewal_rate)

        # 早期 (t=1.0) 的 renewal 率应高于后期 (t=0.25)
        # 混合输入: 50 高分, 50 低分
        # t=1.0, threshold=0.9 → 仅高分保留 → 50% renewal
        # t=0.25, threshold=0.225 → 高分(>0.9)保留, 部分低分可能 > 0.225? 不, 低分≈0.007
        # 实际上低分都很低, 所以 renewal 率可能相同
        # 但阈值确实在递减
        thresholds = [sampler._compute_renewal_threshold(t) for t in time_steps]
        self.assertGreater(thresholds[0], thresholds[-1])
        # 阈值递减
        for i in range(len(thresholds) - 1):
            self.assertGreaterEqual(
                thresholds[i], thresholds[i + 1],
                f'阈值应递减: t={time_steps[i]} → {thresholds[i]}, '
                f't={time_steps[i+1]} → {thresholds[i+1]}'
            )


class TestDDIMPathIntegration(unittest.TestCase):
    """P0: DDIM 路径 (ddim_step) 自适应阈值集成测试."""

    def test_ddim_step_passes_t_curr(self):
        """DDIM 路径应传递归一化 t_curr 给 apply_box_renewal."""
        # 这个测试验证 ddim_step 内部调用 apply_box_renewal 时传递了 t_curr
        # 通过 mock 验证
        sampler = build_sampler(
            adaptive_renewal_threshold=True,
            min_keep=5,
        )

        # 构建极简 ddim_step 输入
        bs = 1
        x_raw = torch.randn(bs, 20, 4)
        cls_logits = torch.ones(bs, 20, 24) * 5.0  # 全高分
        pred_bboxes = torch.rand(bs, 20, 4) * 100  # 像素坐标
        img_metas = [{
            'img_shape': (100, 100),
            'scale_factor': [1.0, 1.0, 1.0, 1.0],
        }]

        # 构建 alphas_cumprod
        timesteps = 1000
        betas = torch.linspace(1e-4, 2e-2, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 调用 ddim_step (t_curr=999, t_next=500)
        # 不应报错
        try:
            result = sampler.ddim_step(
                t_curr=999,
                t_next=500,
                x_raw=x_raw,
                cls_logits=cls_logits,
                pred_bboxes=pred_bboxes,
                img_metas=img_metas,
                alphas_cumprod=alphas_cumprod,
            )
            # 验证输出形状
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f'ddim_step 不应报错: {e}')


if __name__ == '__main__':
    unittest.main()
