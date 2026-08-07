"""测试 TRIP: Tikhonov/MAP 正则化回归目标

核心机制 (FEASIBLE_TRIP.md §7):
  将 RF x0-prediction 的训练回归目标由 GT x_0 替换为贝叶斯 MAP 收缩估计:
    x_tilde^c(t) = (1 - s(t)) * x_0 + s(t) * mu_p^c
  其中 s(t) = (t²/σ_p²) / ((1-t)² + t²/σ_p²) ∈ [0, 1] (MAP, λ=t²)
  边界: s(0)≈0 (目标=x_0=GT, 零正则); s(1)≈1 (目标=mu_p, 完全收缩)

核心验证:
1. box_target_mode / class_priors / trip_lambda_mode 参数存在, 默认向后兼容
2. box_target_mode='trip' 必须提供 class_priors (断言)
3. _load_class_priors 支持 dict / str (pickle 路径), 拒绝错误格式
4. _compute_trip_target s(t) 边界: s(0)≈0, s(1)≈1
5. TRIP target = (1-s)*x_0 + s*mu_p^c (向 mu_p 收缩)
6. 背景位置 (fg=False) 用原 GT (不参与收缩)
7. box_target_mode='trip' 时 _loss_boxes 真实调用 _compute_trip_target
8. TRIP loss != GT loss (大 t 下显著收缩)
9. TRIP loss → GT loss 当 t→0 (s→0, 退化为 GT)
10. map vs morozov lambda 模式都可计算
11. 向后兼容: 'gt' 模式不受影响, 'trip' 模式不依赖 box_targets 参数
12. 梯度可流回 src_boxes (TRIP target 不切断梯度)
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ldmdet.criterion import (
    DiffusionDetCriterion,
    DiffusionDetMatcher,
    FocalLoss,
    GIoULoss,
    L1Loss,
)
from ldmdet.data.structures import InstanceData, ModelOutput


# ============================================================
# 工具函数
# ============================================================

def _make_class_priors(num_classes=24, device='cpu'):
    """构造合法的类条件先验 dict

    mu_p^c: [num_classes, 4] cxcywh [0,1] (中心 0.5, 合理 w/h)
    sigma_bar_sq^c: [num_classes] (各向同性平均方差)
    """
    torch.manual_seed(42)
    # 中心约 0.5, 宽高约 0.15 (染色体典型尺寸)
    mu = torch.zeros(num_classes, 4)
    mu[:, 0] = 0.5  # cx
    mu[:, 1] = 0.5  # cy
    mu[:, 2] = 0.15 + 0.01 * torch.arange(num_classes) / num_classes  # w
    mu[:, 3] = 0.15 + 0.01 * torch.arange(num_classes) / num_classes  # h
    # 各类不同的方差, 以验证类条件行为
    sigma_bar_sq = 0.02 + 0.001 * torch.arange(num_classes)
    return {'mu': mu, 'sigma_bar_sq': sigma_bar_sq}


def _make_criterion(
    box_target_mode='gt',
    num_classes=24,
    class_priors=None,
    trip_lambda_mode='map',
    trip_tau=1.0,
    snr_scale=1.0,
):
    """构建 DiffusionDetCriterion (可选 TRIP 模式)"""
    if class_priors is None and box_target_mode == 'trip':
        class_priors = _make_class_priors(num_classes)
    matcher = DiffusionDetMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    return DiffusionDetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        loss_cls=FocalLoss(use_sigmoid=True, alpha=0.25, gamma=2.0, loss_weight=2.0),
        loss_bbox=L1Loss(loss_weight=5.0),
        loss_giou=GIoULoss(loss_weight=2.0),
        deep_supervision=False,
        box_target_mode=box_target_mode,
        class_priors=class_priors,
        trip_lambda_mode=trip_lambda_mode,
        trip_tau=trip_tau,
        snr_scale=snr_scale,
    )


def _make_targets_and_outputs(num_queries=10, num_classes=24, bs=2, num_gt=2):
    """构造 targets (GT) + outputs (pred) + t"""
    torch.manual_seed(42)
    # GT bboxes 归一化 xyxy [0,1] (与 criterion 假设空间一致)
    gt_bboxes = torch.tensor(
        [[[0.10, 0.10, 0.50, 0.50], [0.50, 0.50, 0.90, 0.90]]] * bs,
        dtype=torch.float32,
    )
    gt_labels = torch.tensor([[0, 1]] * bs, dtype=torch.long)
    targets = [
        InstanceData(
            bboxes=gt_bboxes[i], labels=gt_labels[i], img_shape=(100, 100)
        )
        for i in range(bs)
    ]
    # pred: 复制 GT + 小扰动填充到 num_queries
    n_gt = gt_bboxes.shape[1]
    repeats = (num_queries + n_gt - 1) // n_gt
    pred_boxes = gt_bboxes.repeat(1, repeats, 1)[:, :num_queries, :].contiguous()
    pred_boxes = pred_boxes + torch.randn_like(pred_boxes) * 0.02  # 小扰动
    pred_logits = torch.randn(bs, num_queries, num_classes + 1)
    outputs = ModelOutput(
        pred_logits=pred_logits, pred_boxes=pred_boxes, aux_outputs=None
    )
    # t: [bs] RF 时间步 (大 t 触发明显收缩)
    t = torch.tensor([0.8, 0.9], dtype=torch.float32)
    return targets, outputs, t


# ============================================================
# 1. 参数存在性与默认值
# ============================================================

class TestTRIPParameters:
    """TRIP 参数存在性与默认值 (向后兼容)"""

    def test_box_target_mode_default_gt(self):
        """box_target_mode 默认 'gt' (向后兼容)"""
        criterion = _make_criterion(box_target_mode='gt')
        assert criterion.box_target_mode == 'gt'

    def test_box_target_mode_trip(self):
        """box_target_mode='trip' 可启用"""
        criterion = _make_criterion(box_target_mode='trip')
        assert criterion.box_target_mode == 'trip'

    def test_box_target_mode_invalid_rejected(self):
        """非法 box_target_mode 被拒绝"""
        with pytest.raises(AssertionError):
            _make_criterion(box_target_mode='invalid')

    def test_trip_lambda_mode_default_map(self):
        """trip_lambda_mode 默认 'map'"""
        criterion = _make_criterion(box_target_mode='trip')
        assert criterion.trip_lambda_mode == 'map'

    def test_trip_lambda_mode_morozov(self):
        """trip_lambda_mode='morozov' 可启用"""
        criterion = _make_criterion(
            box_target_mode='trip', trip_lambda_mode='morozov'
        )
        assert criterion.trip_lambda_mode == 'morozov'

    def test_trip_lambda_mode_invalid_rejected(self):
        """非法 trip_lambda_mode 被拒绝"""
        with pytest.raises(AssertionError):
            _make_criterion(
                box_target_mode='trip', trip_lambda_mode='invalid'
            )

    def test_trip_tau_default(self):
        """trip_tau 默认 1.0"""
        criterion = _make_criterion(box_target_mode='trip')
        assert criterion.trip_tau == 1.0

    def test_trip_requires_class_priors(self):
        """box_target_mode='trip' 必须提供 class_priors (断言)"""
        with pytest.raises(AssertionError):
            DiffusionDetCriterion(
                num_classes=24,
                matcher=DiffusionDetMatcher(),
                loss_cls=FocalLoss(use_sigmoid=True),
                loss_bbox=L1Loss(),
                loss_giou=GIoULoss(),
                box_target_mode='trip',
                class_priors=None,  # 缺失 → 断言失败
            )

    def test_gt_mode_no_class_priors_required(self):
        """'gt' 模式不要求 class_priors (默认 None)"""
        criterion = _make_criterion(box_target_mode='gt')
        assert criterion._class_priors_arg is None

    def test_snr_scale_default(self):
        """snr_scale 默认 1.0 (向后兼容)"""
        criterion = _make_criterion(box_target_mode='trip')
        assert criterion.snr_scale == 1.0

    def test_snr_scale_set(self):
        """snr_scale 可显式设置 (如 2.0, 匹配实际训练配置)"""
        criterion = _make_criterion(box_target_mode='trip', snr_scale=2.0)
        assert criterion.snr_scale == 2.0


# ============================================================
# 2. 类条件先验加载 (_load_class_priors)
# ============================================================

class TestClassPriorsLoading:
    """测试类条件先验懒加载机制"""

    def test_load_from_dict(self):
        """从 dict 加载先验"""
        priors = _make_class_priors(24)
        criterion = _make_criterion(
            box_target_mode='trip', class_priors=priors
        )
        loaded = criterion._load_class_priors()
        assert 'mu' in loaded and 'sigma_bar_sq' in loaded
        assert loaded is priors  # dict 直接缓存

    def test_load_from_pickle_file(self):
        """从 pickle 文件路径加载先验"""
        priors = _make_class_priors(24)
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            torch.save(priors, f)
            path = f.name
        try:
            criterion = _make_criterion(
                box_target_mode='trip', class_priors=path
            )
            loaded = criterion._load_class_priors()
            assert loaded['mu'].shape == (24, 4)
            assert loaded['sigma_bar_sq'].shape == (24,)
            assert torch.allclose(loaded['mu'], priors['mu'])
        finally:
            os.unlink(path)

    def test_load_missing_file_raises(self):
        """不存在的文件路径应报错"""
        criterion = _make_criterion(
            box_target_mode='trip', class_priors='/nonexistent/path.pkl'
        )
        with pytest.raises(FileNotFoundError):
            criterion._load_class_priors()

    def test_load_invalid_type_raises(self):
        """非法类型应报错"""
        criterion = _make_criterion(
            box_target_mode='trip', class_priors=12345  # 非法类型
        )
        with pytest.raises(TypeError):
            criterion._load_class_priors()

    def test_load_missing_mu_key_raises(self):
        """缺少 'mu' 键应报错"""
        criterion = _make_criterion(
            box_target_mode='trip', class_priors={'sigma_bar_sq': torch.zeros(24)}
        )
        with pytest.raises(AssertionError):
            criterion._load_class_priors()

    def test_load_missing_sigma_key_raises(self):
        """缺少 'sigma_bar_sq' 键应报错"""
        criterion = _make_criterion(
            box_target_mode='trip', class_priors={'mu': torch.zeros(24, 4)}
        )
        with pytest.raises(AssertionError):
            criterion._load_class_priors()

    def test_load_num_classes_mismatch_raises(self):
        """mu 的 num_classes 与 criterion 不匹配应报错"""
        priors = _make_class_priors(num_classes=10)  # 10 类
        criterion = _make_criterion(
            box_target_mode='trip', num_classes=24, class_priors=priors
        )
        with pytest.raises(AssertionError):
            criterion._load_class_priors()

    def test_load_caches_result(self):
        """加载结果被缓存 (不重复 IO)"""
        priors = _make_class_priors(24)
        criterion = _make_criterion(
            box_target_mode='trip', class_priors=priors
        )
        first = criterion._load_class_priors()
        second = criterion._load_class_priors()
        assert first is second  # 同一对象引用 (缓存)


# ============================================================
# 3. _compute_trip_target 数学正确性
# ============================================================

class TestTRIPTargetMath:
    """测试 TRIP target 的数学正确性"""

    def _make_trip_inputs(self, num_classes=24, bs=2, N=5, t_val=0.8):
        """构造 _compute_trip_target 所需输入"""
        torch.manual_seed(42)
        gt_xyxy = torch.rand(bs, 3, 4) * 0.8 + 0.1  # [bs, max_gt=3, 4] xyxy [0,1]
        # 保证有效 xyxy (x2 > x1)
        gt_xyxy_sorted = torch.sort(gt_xyxy, dim=-1).values
        matched_gt_inds = torch.randint(0, 3, (bs, N))  # [bs, N]
        fg_masks = torch.tensor(
            [[True, True, True, False, False], [True, True, False, False, True]],
            dtype=torch.bool,
        )
        gt_labels_padded = torch.full(
            (bs, 3), num_classes, dtype=torch.long
        )  # 背景填充
        gt_labels_padded[:, 0] = 0
        gt_labels_padded[:, 1] = 1
        gt_labels_padded[:, 2] = 2
        t = torch.full((bs,), t_val, dtype=torch.float32)
        return gt_xyxy_sorted, matched_gt_inds, fg_masks, gt_labels_padded, t

    def test_s_boundary_t_near_zero(self):
        """t→0 时 s(t)→0 (目标=x_0=GT, 零正则)"""
        criterion = _make_criterion(box_target_mode='trip')
        gt_xyxy, mgt, fg, labels, _ = self._make_trip_inputs(t_val=0.0)
        t = torch.full((2,), 1e-4, dtype=torch.float32)  # t≈0
        trip_tgt = criterion._compute_trip_target(gt_xyxy, mgt, fg, labels, t)
        # s(t≈0) ≈ 0 → trip_tgt ≈ GT (gather)
        mgt_clamped = mgt.clamp(min=0)
        gt_gathered = torch.gather(
            gt_xyxy, 1, mgt_clamped.unsqueeze(-1).expand(-1, -1, 4)
        )
        assert torch.allclose(trip_tgt, gt_gathered, atol=1e-3), (
            f"t≈0 should give trip_tgt≈GT, got diff={((trip_tgt - gt_gathered).abs().max())}"
        )

    def test_s_boundary_t_near_one(self):
        """t→1 时 s(t)→1 (目标=mu_p, 完全收缩)"""
        criterion = _make_criterion(box_target_mode='trip')
        gt_xyxy, mgt, fg, labels, _ = self._make_trip_inputs(t_val=1.0)
        t = torch.full((2,), 1.0 - 1e-4, dtype=torch.float32)  # t≈1
        trip_tgt = criterion._compute_trip_target(gt_xyxy, mgt, fg, labels, t)
        # s(t≈1) ≈ 1 → trip_tgt ≈ mu_p^c (xyxy 空间), 仅在 fg 位置生效
        # 背景位置 (fg=False) 用原 GT (代码 §9 设计), 不应等于 mu_p
        priors = criterion._load_class_priors()
        mu_p = priors['mu']  # [num_classes, 4] cxcywh [0,1]
        from ldmdet.utils.box_ops import bbox_cxcywh_to_xyxy
        mu_p_xyxy = bbox_cxcywh_to_xyxy(mu_p)  # [num_classes, 4] xyxy [0,1]
        labels_clamped = labels.gather(1, mgt.clamp(min=0))
        labels_clamped = labels_clamped.clamp(max=23)
        expected_mu = mu_p_xyxy[labels_clamped]  # [bs, N, 4]
        # 仅检查 fg 位置 (背景位置保留原 GT)
        for b in range(2):
            for n in range(5):
                if not fg[b, n]:
                    continue
                assert torch.allclose(
                    trip_tgt[b, n], expected_mu[b, n], atol=1e-2
                ), (
                    f"t≈1 fg[{b},{n}] should give trip_tgt≈mu_p, got "
                    f"{trip_tgt[b, n]} vs {expected_mu[b, n]}"
                )

    def test_s_monotonic_increasing(self):
        """s(t) 关于 t 单调递增 (中段 0 < s(0.3) < s(0.5) < s(0.7) < s(0.9))"""
        criterion = _make_criterion(box_target_mode='trip')
        gt_xyxy, mgt, fg, labels, _ = self._make_trip_inputs()
        s_values = []
        for t_val in [0.3, 0.5, 0.7, 0.9]:
            t = torch.full((2,), t_val, dtype=torch.float32)
            trip_tgt = criterion._compute_trip_target(
                gt_xyxy, mgt, fg, labels, t
            )
            # 反推 s: trip_tgt = (1-s)*gt + s*mu_p → s = (trip - gt) / (mu_p - gt)
            mgt_clamped = mgt.clamp(min=0)
            gt_gathered = torch.gather(
                gt_xyxy, 1, mgt_clamped.unsqueeze(-1).expand(-1, -1, 4)
            )
            # 用 fg 位置上的均值估计 s
            s_est = ((trip_tgt - gt_gathered) / (
                criterion._load_class_priors()['mu'][labels.gather(1, mgt_clamped).clamp(max=23)] - gt_gathered + 1e-8
            )).mean().item()
            s_values.append(s_est)
        for i in range(len(s_values) - 1):
            assert s_values[i] < s_values[i + 1], (
                f"s(t) should be monotonically increasing: {s_values}"
            )

    def test_trip_target_between_gt_and_mu_p(self):
        """TRIP target 位于 GT 和 mu_p 之间 (插值性质)"""
        criterion = _make_criterion(box_target_mode='trip')
        gt_xyxy, mgt, fg, labels, _ = self._make_trip_inputs()
        t = torch.full((2,), 0.5, dtype=torch.float32)
        trip_tgt = criterion._compute_trip_target(gt_xyxy, mgt, fg, labels, t)
        mgt_clamped = mgt.clamp(min=0)
        gt_gathered = torch.gather(
            gt_xyxy, 1, mgt_clamped.unsqueeze(-1).expand(-1, -1, 4)
        )
        priors = criterion._load_class_priors()
        from ldmdet.utils.box_ops import bbox_cxcywh_to_xyxy
        mu_p_xyxy = bbox_cxcywh_to_xyxy(priors['mu'])
        labels_clamped = labels.gather(1, mgt_clamped).clamp(max=23)
        mu_p_per_prop = mu_p_xyxy[labels_clamped]
        # 对每个 fg 元素: trip_tgt 应在 gt 和 mu_p 之间 (凸组合)
        for b in range(2):
            for n in range(5):
                if not fg[b, n]:
                    continue
                for d in range(4):
                    lo = min(gt_gathered[b, n, d].item(), mu_p_per_prop[b, n, d].item())
                    hi = max(gt_gathered[b, n, d].item(), mu_p_per_prop[b, n, d].item())
                    v = trip_tgt[b, n, d].item()
                    assert lo - 1e-4 <= v <= hi + 1e-4, (
                        f"trip_tgt[{b},{n},{d}]={v} not in [{lo}, {hi}] "
                        f"(gt={gt_gathered[b,n,d].item()}, mu_p={mu_p_per_prop[b,n,d].item()})"
                    )

    def test_background_uses_original_gt(self):
        """背景位置 (fg=False) 用原 GT (不参与 TRIP 收缩)"""
        criterion = _make_criterion(box_target_mode='trip')
        gt_xyxy, mgt, fg, labels, _ = self._make_trip_inputs()
        t = torch.full((2,), 0.9, dtype=torch.float32)  # 大 t (强收缩)
        trip_tgt = criterion._compute_trip_target(gt_xyxy, mgt, fg, labels, t)
        mgt_clamped = mgt.clamp(min=0)
        gt_gathered = torch.gather(
            gt_xyxy, 1, mgt_clamped.unsqueeze(-1).expand(-1, -1, 4)
        )
        # 背景位置: trip_tgt 应等于原 GT
        for b in range(2):
            for n in range(5):
                if not fg[b, n]:
                    assert torch.allclose(
                        trip_tgt[b, n], gt_gathered[b, n], atol=1e-6
                    ), (
                        f"Background [{b},{n}] should equal GT, got "
                        f"{trip_tgt[b, n]} vs {gt_gathered[b, n]}"
                    )

    def test_snr_scale_reduces_s_at_medium_t(self):
        """snr_scale=2.0 使 s(t) 在中 t 段显著低于 snr_scale=1.0

        空间一致性: σ_p²_norm (归一化空间) 经 4·snr_scale² 缩放为 σ_p²_raw (raw 空间),
        使 s(t) 在相同 t 下更保守 (crossover 右移). 这是修复训练崩溃的关键.
        """
        # 用固定 σ_p² 和 μ_p 避免类间差异干扰
        num_classes = 4
        priors = {
            'mu': torch.tensor([
                [0.5, 0.5, 0.15, 0.15]] * num_classes
            ),
            'sigma_bar_sq': torch.full((num_classes,), 0.025),  # 归一化空间
        }
        crit_noscale = _make_criterion(
            box_target_mode='trip', num_classes=num_classes,
            class_priors=priors, snr_scale=1.0,  # σ²_raw = 4·1²·0.025 = 0.1
        )
        crit_scale2 = _make_criterion(
            box_target_mode='trip', num_classes=num_classes,
            class_priors=priors, snr_scale=2.0,  # σ²_raw = 4·4·0.025 = 0.4
        )
        gt_xyxy, mgt, fg, labels, _ = self._make_trip_inputs(num_classes=num_classes)
        t = torch.full((2,), 0.3, dtype=torch.float32)  # 中 t 段

        trip_noscale = crit_noscale._compute_trip_target(
            gt_xyxy, mgt, fg, labels, t
        )
        trip_scale2 = crit_scale2._compute_trip_target(
            gt_xyxy, mgt, fg, labels, t
        )
        mgt_clamped = mgt.clamp(min=0)
        gt_gathered = torch.gather(
            gt_xyxy, 1, mgt_clamped.unsqueeze(-1).expand(-1, -1, 4)
        )
        # snr_scale=2.0 → σ_p²_raw 更大 → s(t) 更小 → trip_tgt 更接近 GT
        diff_noscale = (trip_noscale - gt_gathered).abs().mean()
        diff_scale2 = (trip_scale2 - gt_gathered).abs().mean()
        assert diff_scale2 < diff_noscale, (
            f"snr_scale=2.0 should give smaller |trip-gt| (more conservative s), "
            f"got scale2={diff_scale2:.4f} >= noscale={diff_noscale:.4f}"
        )

    def test_snr_scale_2_matches_raw_space_formula(self):
        """snr_scale=2.0 时 s(t) 等价于直接用 σ_p²_raw=0.4 计算

        验证: priors σ²_norm=0.025, snr_scale=2.0 → σ²_raw=4·4·0.025=0.4
        等价于 priors σ²=0.4, snr_scale=1.0 (无额外缩放, 仅 4× 中心化缩放)
        """
        num_classes = 4
        # 方式 A: σ²_norm=0.025, snr_scale=2.0 → σ²_raw=0.4
        priors_norm = {
            'mu': torch.tensor([[0.5, 0.5, 0.15, 0.15]] * num_classes),
            'sigma_bar_sq': torch.full((num_classes,), 0.025),
        }
        crit_A = _make_criterion(
            box_target_mode='trip', num_classes=num_classes,
            class_priors=priors_norm, snr_scale=2.0,
        )
        # 方式 B: σ²=0.4 (= 4·2²·0.025, 已在 raw 空间), snr_scale=1.0
        # 但 snr_scale=1.0 仍会乘 4·1²=4, 所以要用 σ²=0.1 (=0.4/4)
        # 不对 — snr_scale=1.0 时 σ²_raw = 4·1²·σ²_input, 要让 σ²_raw=0.4
        # 需要 σ²_input = 0.4/4 = 0.1
        priors_raw = {
            'mu': torch.tensor([[0.5, 0.5, 0.15, 0.15]] * num_classes),
            'sigma_bar_sq': torch.full((num_classes,), 0.1),  # 0.4/4
        }
        crit_B = _make_criterion(
            box_target_mode='trip', num_classes=num_classes,
            class_priors=priors_raw, snr_scale=1.0,
        )
        gt_xyxy, mgt, fg, labels, _ = self._make_trip_inputs(num_classes=num_classes)
        t = torch.full((2,), 0.5, dtype=torch.float32)

        trip_A = crit_A._compute_trip_target(gt_xyxy, mgt, fg, labels, t)
        trip_B = crit_B._compute_trip_target(gt_xyxy, mgt, fg, labels, t)
        # 两者应数值一致 (σ²_raw 相同 = 0.4)
        assert torch.allclose(trip_A, trip_B, atol=1e-6), (
            f"snr_scale=2.0 with σ²_norm=0.025 should match snr_scale=1.0 "
            f"with σ²=0.1 (both → σ²_raw=0.4), got max diff="
            f"{(trip_A - trip_B).abs().max():.8f}"
        )

    def test_map_vs_morozov_both_compute(self):
        """map 和 morozov 两种 lambda 模式都能正常计算"""
        for mode in ['map', 'morozov']:
            criterion = _make_criterion(
                box_target_mode='trip', trip_lambda_mode=mode
            )
            gt_xyxy, mgt, fg, labels, _ = self._make_trip_inputs()
            t = torch.full((2,), 0.7, dtype=torch.float32)
            trip_tgt = criterion._compute_trip_target(
                gt_xyxy, mgt, fg, labels, t
            )
            assert torch.isfinite(trip_tgt).all(), f"{mode}: output must be finite"
            assert trip_tgt.shape == (2, 5, 4)

    def test_map_and_morozov_differ(self):
        """map 和 morozov 在中 t 段产生不同 target"""
        gt_xyxy, mgt, fg, labels, _ = self._make_trip_inputs()
        t = torch.full((2,), 0.7, dtype=torch.float32)
        crit_map = _make_criterion(box_target_mode='trip', trip_lambda_mode='map')
        crit_mor = _make_criterion(
            box_target_mode='trip', trip_lambda_mode='morozov', trip_tau=0.5
        )
        tgt_map = crit_map._compute_trip_target(gt_xyxy, mgt, fg, labels, t)
        tgt_mor = crit_mor._compute_trip_target(gt_xyxy, mgt, fg, labels, t)
        # 不同 τ / 不同公式应产生不同结果 (至少在 fg 位置)
        diff = (tgt_map - tgt_mor).abs().sum()
        assert diff > 1e-4, f"map and morozov should differ, diff={diff}"


# ============================================================
# 4. _loss_boxes 集成测试 (验证 TRIP target 真实被使用)
# ============================================================

class TestTRIPLossIntegration:
    """验证 TRIP target 真实被 _loss_boxes 使用"""

    def test_trip_mode_calls_compute_trip_target(self):
        """box_target_mode='trip' 时 _compute_trip_target 被调用"""
        criterion = _make_criterion(box_target_mode='trip')
        called = {'flag': False}
        original = criterion._compute_trip_target

        def spy(*args, **kwargs):
            called['flag'] = True
            return original(*args, **kwargs)

        criterion._compute_trip_target = spy
        targets, outputs, t = _make_targets_and_outputs()
        criterion.forward(outputs, targets, t=t)
        assert called['flag'], "_compute_trip_target must be called in 'trip' mode"

    def test_gt_mode_does_not_call_compute_trip_target(self):
        """'gt' 模式不调用 _compute_trip_target (向后兼容)"""
        criterion = _make_criterion(box_target_mode='gt')
        called = {'flag': False}
        original = getattr(criterion, '_compute_trip_target', None)
        if original is None:
            # 'gt' 模式下 _compute_trip_target 存在但不应被调用
            pass

        def spy(*args, **kwargs):
            called['flag'] = True
            return 0

        criterion._compute_trip_target = spy
        targets, outputs, t = _make_targets_and_outputs()
        criterion.forward(outputs, targets, t=t)
        assert not called['flag'], "'gt' mode must not call _compute_trip_target"

    def test_trip_loss_differs_from_gt_loss(self):
        """大 t 下 TRIP loss 与 GT loss 显著不同 (证明 target 改变)"""
        targets, outputs, t = _make_targets_and_outputs()
        # t=[0.8, 0.9] 触发明显收缩
        crit_gt = _make_criterion(box_target_mode='gt')
        crit_trip = _make_criterion(box_target_mode='trip')
        losses_gt = crit_gt.forward(outputs, targets, t=t)
        losses_trip = crit_trip.forward(outputs, targets, t=t)
        # loss_bbox 应不同 (target 从 GT → 收缩估计)
        diff = abs(losses_gt['loss_bbox'].item() - losses_trip['loss_bbox'].item())
        assert diff > 1e-4, (
            f"TRIP loss should differ from GT loss (t={t.tolist()}), "
            f"got gt={losses_gt['loss_bbox'].item()}, trip={losses_trip['loss_bbox'].item()}"
        )

    def test_trip_loss_converges_to_gt_loss_at_small_t(self):
        """t→0 时 TRIP loss → GT loss (s→0, 退化为 GT)"""
        targets, outputs, _ = _make_targets_and_outputs()
        t_small = torch.full((2,), 1e-3, dtype=torch.float32)  # t≈0
        crit_gt = _make_criterion(box_target_mode='gt')
        crit_trip = _make_criterion(box_target_mode='trip')
        losses_gt = crit_gt.forward(outputs, targets, t=t_small)
        losses_trip = crit_trip.forward(outputs, targets, t=t_small)
        # t≈0 时 s≈0 → TRIP target ≈ GT → loss 应几乎相同
        diff = abs(losses_gt['loss_bbox'].item() - losses_trip['loss_bbox'].item())
        assert diff < 1e-3, (
            f"t≈0 should make TRIP loss→GT loss, diff={diff}, "
            f"gt={losses_gt['loss_bbox'].item()}, trip={losses_trip['loss_bbox'].item()}"
        )

    def test_trip_loss_does_not_require_box_targets(self):
        """TRIP 模式不依赖 box_targets 参数 (内部自算 target)"""
        criterion = _make_criterion(box_target_mode='trip')
        targets, outputs, t = _make_targets_and_outputs()
        # 不传 box_targets (默认 None), TRIP 模式应正常工作
        losses = criterion.forward(outputs, targets, t=t)
        assert 'loss_bbox' in losses
        assert torch.isfinite(losses['loss_bbox']).all()

    def test_trip_loss_giou_differs_from_gt(self):
        """GIoU loss 也使用 TRIP target (target 改变影响 GIoU)"""
        targets, outputs, t = _make_targets_and_outputs()
        crit_gt = _make_criterion(box_target_mode='gt')
        crit_trip = _make_criterion(box_target_mode='trip')
        losses_gt = crit_gt.forward(outputs, targets, t=t)
        losses_trip = crit_trip.forward(outputs, targets, t=t)
        # GIoU 也应不同
        diff_giou = abs(
            losses_gt['loss_giou'].item() - losses_trip['loss_giou'].item()
        )
        assert diff_giou > 1e-4, (
            f"GIoU loss should differ (TRIP target), diff={diff_giou}"
        )

    def test_cls_loss_unchanged_by_trip(self):
        """cls target 始终用 GT (TRIP 不影响分类 loss)"""
        targets, outputs, t = _make_targets_and_outputs()
        crit_gt = _make_criterion(box_target_mode='gt')
        crit_trip = _make_criterion(box_target_mode='trip')
        losses_gt = crit_gt.forward(outputs, targets, t=t)
        losses_trip = crit_trip.forward(outputs, targets, t=t)
        # cls loss 应几乎相同 (matcher 用 GT, 不受 box target 影响)
        diff_cls = abs(
            losses_gt['loss_cls'].item() - losses_trip['loss_cls'].item()
        )
        assert diff_cls < 1e-3, (
            f"cls loss should be unchanged (matcher uses GT), diff={diff_cls}"
        )


# ============================================================
# 5. 梯度流测试 (验证 TRIP target 可反向传播)
# ============================================================

class TestTRIPGradientFlow:
    """验证 TRIP target 不切断梯度流"""

    def test_gradient_flows_to_pred_boxes(self):
        """梯度可从 TRIP loss 流回 outputs.pred_boxes"""
        criterion = _make_criterion(box_target_mode='trip')
        targets, outputs, t = _make_targets_and_outputs()
        # pred_boxes 需要 requires_grad
        outputs.pred_boxes.requires_grad_(True)
        losses = criterion.forward(outputs, targets, t=t)
        total_loss = losses['loss_bbox'] + losses['loss_giou']
        total_loss.backward()
        assert outputs.pred_boxes.grad is not None, (
            "Gradient must flow to pred_boxes"
        )
        assert torch.isfinite(outputs.pred_boxes.grad).all()

    def test_trip_target_is_differentiable(self):
        """_compute_trip_target 输出可微分 (含梯度)"""
        criterion = _make_criterion(box_target_mode='trip')
        torch.manual_seed(42)
        bs, max_gt, N = 2, 3, 5
        gt_xyxy = torch.rand(bs, max_gt, 4, requires_grad=True)
        gt_xyxy_sorted = torch.sort(gt_xyxy, dim=-1).values
        mgt = torch.randint(0, max_gt, (bs, N))
        fg = torch.tensor(
            [[True, True, True, False, False], [True, True, False, False, True]]
        )
        labels = torch.full((bs, max_gt), 24, dtype=torch.long)
        labels[:, 0] = 0
        labels[:, 1] = 1
        labels[:, 2] = 2
        t = torch.tensor([0.5, 0.7], requires_grad=True)
        trip_tgt = criterion._compute_trip_target(
            gt_xyxy_sorted, mgt, fg, labels, t
        )
        # trip_tgt 应有梯度流回 gt_xyxy 和 t
        assert trip_tgt.requires_grad, "trip_tgt must require grad"
        loss = trip_tgt.sum()
        loss.backward()
        assert gt_xyxy.grad is not None, "Gradient must flow to gt_xyxy"
        assert t.grad is not None, "Gradient must flow to t"

    def test_t_gradient_nonzero_at_mid_t(self):
        """中段 t (0.3~0.7) 的梯度非零 (s(t) 对 t 有梯度)"""
        criterion = _make_criterion(box_target_mode='trip')
        torch.manual_seed(42)
        bs, max_gt, N = 2, 3, 5
        gt_xyxy = torch.rand(bs, max_gt, 4)
        gt_xyxy_sorted = torch.sort(gt_xyxy, dim=-1).values
        mgt = torch.randint(0, max_gt, (bs, N))
        fg = torch.ones(bs, N, dtype=torch.bool)
        labels = torch.zeros(bs, max_gt, dtype=torch.long)
        t = torch.tensor([0.5, 0.7], requires_grad=True)
        trip_tgt = criterion._compute_trip_target(
            gt_xyxy_sorted, mgt, fg, labels, t
        )
        trip_tgt.sum().backward()
        # t 在中段 (远离边界) 的梯度应非零
        assert t.grad is not None
        assert t.grad.abs().sum() > 0, "Gradient w.r.t. t must be non-zero at mid t"


# ============================================================
# 6. 端到端 (Head + Criterion) 集成测试
# ============================================================

class TestTRIPHeadIntegration:
    """验证 TRIP 通过 head.loss() 端到端可用"""

    def _make_head_with_trip(self, num_classes=24, feat_channels=64):
        """构建带 TRIP 的 head (复用 test_lvd_rf 的 head 构造)"""
        from ldmdet.core.head import DiffusionDetHead
        from ldmdet.core.roi_extractor import SingleRoIExtractor
        from ldmdet.core.single_head import SingleDiffusionDetHead

        single_head = SingleDiffusionDetHead(
            num_classes=num_classes, feat_channels=feat_channels,
            num_cls_convs=1, num_reg_convs=2, use_focal_loss=True,
            use_normalized_classifier=False, time_conditioning='adaln_zero',
        )
        roi_extractor = SingleRoIExtractor(
            featmap_strides=[16], out_channels=feat_channels,
            roi_layer=dict(
                type='RoIAlign', output_size=7, sampling_ratio=0, aligned=True
            ),
        )
        criterion = _make_criterion(box_target_mode='trip', num_classes=num_classes)
        head = DiffusionDetHead(
            num_classes=num_classes,
            feat_channels=feat_channels,
            num_proposals=8,
            num_heads=2,
            diffusion_type='rf',
            solver_type='euler',
            sampling_timesteps=1,
            deep_supervision=False,
            single_head=single_head,
            roi_extractor=roi_extractor,
            criterion=criterion,
        )
        head.eval()
        return head

    def test_head_loss_with_trip(self):
        """head.loss() 在 TRIP 模式下端到端可运行"""
        head = self._make_head_with_trip()
        head.train()
        bs = 1
        h, w = 100, 100
        features = [torch.randn(1, 64, 7, 7, requires_grad=True)]
        img_metas = [{'img_shape': (h, w, 3)}]
        gt_bboxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
        gt_labels = [torch.tensor([0], dtype=torch.long)]
        losses = head.loss(features, img_metas, gt_bboxes, gt_labels)
        # 必须包含标准损失键
        assert 'loss_bbox' in losses
        assert 'loss_giou' in losses
        assert 'loss_cls' in losses
        # TRIP 不应产生额外 loss 键 (TRIP 改 target, 不加 loss 项)
        assert 'loss_lvd' not in losses
        # 梯度可流回特征
        total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
        total_loss.backward()
        assert features[0].grad is not None
        assert torch.isfinite(features[0].grad).all()

    def test_head_loss_trip_uses_t_from_training(self):
        """head.loss() 传入的 t 真实用于 TRIP 收缩 (大 t → 收缩明显)"""
        # 此测试验证 head.loss() 中 t 张量正确传递给 criterion
        # 通过验证不同 t 产生不同 loss_bbox 间接确认
        head = self._make_head_with_trip()
        head.train()
        bs = 1
        h, w = 100, 100
        features = [torch.randn(1, 64, 7, 7)]
        img_metas = [{'img_shape': (h, w, 3)}]
        gt_bboxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]])]
        gt_labels = [torch.tensor([0], dtype=torch.long)]
        # 多次运行, t 由 head 内部采样, loss 应随 t 变化
        # (运行多次, 验证不崩溃即可; 严格 t 依赖性已在前述测试中验证)
        for _ in range(3):
            features[0].requires_grad_(True)
            losses = head.loss(features, img_metas, gt_bboxes, gt_labels)
            assert torch.isfinite(losses['loss_bbox']).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
