"""测试 LVD-RF: Lyapunov Velocity Direction Regularization

核心机制 (FEASIBLE_LVD_RF.md §7):
  在训练损失中增加 Lyapunov 方向余弦正则项 (默认 sin² 形式),
  利用 d=4 低维优势以零额外前向传播计算方向余弦.

核心验证:
1. use_lvd 参数存在 (DiffusionDetHead), 默认 False 向后兼容
2. lvd_form 参数存在, 支持 'sin2' / 'cos' / 'sqrt' 三种形式
3. use_lvd=True 时 loss() 返回 loss_lvd 键
4. loss_lvd 是标量且 require_grad=True (梯度可流向 x_hat_0)
5. 空间转换正确: xyxy 像素 → raw cxcywh (与 _sampler.raw_to_xyxy 互逆)
6. t < lvd_t_threshold 时跳过 (cos_sim 不稳定)
7. 无效 xyxy 框 (x2<=x1) 不贡献 loss
8. 理想对齐 (x_hat_0 = x_0) 时 loss ≈ 0
9. use_lvd=False 时不影响 loss (向后兼容)
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ldmdet.core.head import DiffusionDetHead
from ldmdet.core.roi_extractor import SingleRoIExtractor
from ldmdet.core.single_head import SingleDiffusionDetHead
from ldmdet.criterion import (
    DiffusionDetCriterion,
    DiffusionDetMatcher,
    FocalLoss,
    GIoULoss,
    L1Loss,
)
from ldmdet.data.structures import InstanceData
from ldmdet.diagnostics.instrumentation import probe
from ldmdet.diffusion.sampling import _get_img_shape


def _make_single_head(num_classes=24, feat_channels=64):
    return SingleDiffusionDetHead(
        num_classes=num_classes, feat_channels=feat_channels,
        num_cls_convs=1, num_reg_convs=2, use_focal_loss=True,
        use_normalized_classifier=False, time_conditioning='adaln_zero',
    )


def _make_roi_extractor(out_channels=64):
    return SingleRoIExtractor(
        featmap_strides=[16], out_channels=out_channels,
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0, aligned=True),
    )


def _make_criterion(num_classes=24):
    matcher = DiffusionDetMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    return DiffusionDetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        loss_cls=FocalLoss(use_sigmoid=True, alpha=0.25, gamma=2.0, loss_weight=2.0),
        loss_bbox=L1Loss(loss_weight=5.0),
        loss_giou=GIoULoss(loss_weight=2.0),
        deep_supervision=False,
    )


def _make_head(
    use_lvd=False, lvd_form='sin2', lvd_lambda=0.1,
    lvd_t_threshold=0.05, lvd_space='raw_cxcywh',
    num_classes=24, feat_channels=64,
):
    """构建带 LVD-RF 的 DiffusionDetHead (num_heads=2 以加速测试)"""
    head = DiffusionDetHead(
        num_classes=num_classes,
        feat_channels=feat_channels,
        num_proposals=8,
        num_heads=2,
        diffusion_type='rf',
        solver_type='euler',
        sampling_timesteps=1,
        deep_supervision=False,
        single_head=_make_single_head(num_classes, feat_channels),
        roi_extractor=_make_roi_extractor(feat_channels),
        criterion=_make_criterion(num_classes),
        use_lvd=use_lvd,
        lvd_lambda=lvd_lambda,
        lvd_form=lvd_form,
        lvd_t_threshold=lvd_t_threshold,
        lvd_space=lvd_space,
    )
    head.eval()
    return head


def _make_img_metas(bs=2, h=100, w=100):
    """构造 img_metas (dict 格式, 与 _get_img_shape 兼容)"""
    return [{'img_shape': (h, w, 3)} for _ in range(bs)]


def _make_gt(bs=2, num_gt=2, h=100, w=100):
    """构造 GT bboxes (xyxy 像素) 和 labels"""
    torch.manual_seed(42)
    gt_bboxes = []
    gt_labels = []
    for i in range(bs):
        # 在图像中心区域放置 GT
        cx, cy = w / 2, h / 2
        bw, bh = 30, 30
        boxes = torch.tensor(
            [[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
             [10, 10, 40, 40]],
            dtype=torch.float32,
        )
        gt_bboxes.append(boxes)
        gt_labels.append(torch.tensor([0, 1], dtype=torch.long))
    return gt_bboxes, gt_labels


class TestLVDRFParameters:
    """测试 LVD-RF 参数存在性和默认值"""

    def test_use_lvd_default_false(self):
        """use_lvd 默认 False (向后兼容)"""
        head = _make_head()
        assert head.use_lvd is False

    def test_use_lvd_true(self):
        """use_lvd=True 可启用"""
        head = _make_head(use_lvd=True)
        assert head.use_lvd is True

    def test_lvd_lambda_default(self):
        """lvd_lambda 默认 0.1"""
        head = _make_head()
        assert head.lvd_lambda == 0.1

    def test_lvd_form_options(self):
        """lvd_form 支持 sin2 / cos / sqrt"""
        for form in ['sin2', 'cos', 'sqrt']:
            head = _make_head(use_lvd=True, lvd_form=form)
            assert head.lvd_form == form

    def test_lvd_form_invalid_rejected(self):
        """lvd_form 非法值应被拒绝"""
        with pytest.raises(AssertionError):
            _make_head(use_lvd=True, lvd_form='invalid')

    def test_lvd_t_threshold_default(self):
        """lvd_t_threshold 默认 0.05"""
        head = _make_head()
        assert head.lvd_t_threshold == 0.05

    def test_cos_sim_high_count_init_zero(self):
        """自适应切换计数器初始化为 0"""
        head = _make_head(use_lvd=True)
        assert head._cos_sim_high_count == 0


class TestLVDLossComputation:
    """测试 LVD 损失计算核心逻辑"""

    def test_lvd_loss_returns_scalar_with_grad(self):
        """LVD loss 是标量且 require_grad=True (梯度可流向 x_hat_0)"""
        head = _make_head(use_lvd=True)
        head.train()
        bs, N = 2, 8
        h, w = 100, 100
        # x_t, x_0: raw cxcywh [-snr_scale, snr_scale]
        snr = head.snr_scale
        x_boxes = [torch.randn(N, 4, requires_grad=False) * snr for _ in range(bs)]
        x_starts = [torch.randn(N, 4) * snr for _ in range(bs)]
        # all_pred_bboxes: [num_heads, bs, N, 4] xyxy 像素 (requires_grad=True)
        all_pred = torch.rand(2, bs, N, 4, requires_grad=True) * torch.tensor(
            [w, h, w, h]
        )
        t = torch.tensor([0.5, 0.7])
        img_metas = _make_img_metas(bs, h, w)

        lvd_loss = head._compute_lvd_loss(
            x_boxes=x_boxes, x_starts=x_starts,
            all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
        )
        assert lvd_loss.dim() == 0, "LVD loss must be scalar"
        assert lvd_loss.requires_grad, "LVD loss must require grad (flow to x_hat_0)"
        assert lvd_loss.item() >= 0, "LVD loss must be non-negative"

    def test_lvd_loss_ideal_alignment_near_zero(self):
        """理想对齐 (x_hat_0 ≈ x_0) 时 loss ≈ 0"""
        head = _make_head(use_lvd=True, lvd_form='sin2')
        bs, N = 2, 8
        h, w = 100, 100
        snr = head.snr_scale
        # x_0: raw cxcywh
        x_starts = [torch.randn(N, 4) * snr for _ in range(bs)]
        # x_t = (1-t)*x_0 + t*noise, 用 t=0.5
        torch.manual_seed(0)
        noise = [torch.randn(N, 4) * snr for _ in range(bs)]
        t_val = 0.5
        x_boxes = [(1 - t_val) * x_starts[i] + t_val * noise[i] for i in range(bs)]
        # x_hat_0 ≈ x_0 (理想预测): 转换 raw cxcywh → xyxy 像素
        # raw cxcywh → norm cxcywh → norm xyxy → xyxy 像素
        from ldmdet.utils.box_ops import bbox_cxcywh_to_xyxy
        x_hat_0_raw = torch.stack(x_starts)  # [bs, N, 4] raw cxcywh
        norm_cxcywh = (x_hat_0_raw / snr + 1) / 2  # → [0, 1]
        norm_xyxy = bbox_cxcywh_to_xyxy(norm_cxcywh)
        scales = torch.tensor([w, h, w, h], dtype=torch.float32)
        pred_xyxy = norm_xyxy * scales  # xyxy 像素
        all_pred = pred_xyxy.unsqueeze(0).repeat(2, 1, 1, 1)  # [num_heads, bs, N, 4]
        t = torch.tensor([t_val, t_val])
        img_metas = _make_img_metas(bs, h, w)

        lvd_loss = head._compute_lvd_loss(
            x_boxes=x_boxes, x_starts=x_starts,
            all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
        )
        # cos_sim ≈ 1 (理想对齐), sin²(0) = 0
        assert lvd_loss.item() < 0.01, (
            f"Ideal alignment should give near-zero loss, got {lvd_loss.item()}"
        )

    def test_lvd_loss_gradient_flows_to_prediction(self):
        """梯度能从 LVD loss 流回 all_pred_bboxes (x_hat_0)"""
        head = _make_head(use_lvd=True)
        head.train()
        bs, N = 2, 8
        h, w = 100, 100
        snr = head.snr_scale
        x_boxes = [torch.randn(N, 4) * snr for _ in range(bs)]
        x_starts = [torch.randn(N, 4) * snr for _ in range(bs)]
        # 用 leaf tensor + clamp 保证有效 xyxy (x2 > x1, y2 > y1)
        all_pred = torch.rand(2, bs, N, 4, requires_grad=True)
        # 保证 x2 > x1: x1 ∈ [0, 50), x2 ∈ [50, 100); y1 ∈ [0, 50), y2 ∈ [50, 100)
        scale_tensor = torch.tensor([w, h, w, h], dtype=torch.float32)
        all_pred_scaled = all_pred * scale_tensor
        # 用 retain_grad 捕获非叶子张量梯度
        all_pred_scaled.retain_grad()
        t = torch.tensor([0.5, 0.7])
        img_metas = _make_img_metas(bs, h, w)

        lvd_loss = head._compute_lvd_loss(
            x_boxes=x_boxes, x_starts=x_starts,
            all_pred_bboxes=all_pred_scaled, t=t, img_metas=img_metas,
        )
        lvd_loss.backward()
        assert all_pred_scaled.grad is not None, (
            "Gradient must flow to all_pred_bboxes"
        )
        assert torch.isfinite(all_pred_scaled.grad).all(), (
            "Gradient must be finite"
        )

    def test_lvd_loss_different_forms(self):
        """三种 lvd_form (sin2/cos/sqrt) 都能计算且非负"""
        for form in ['sin2', 'cos', 'sqrt']:
            head = _make_head(use_lvd=True, lvd_form=form)
            head.train()
            bs, N = 2, 8
            h, w = 100, 100
            snr = head.snr_scale
            x_boxes = [torch.randn(N, 4) * snr for _ in range(bs)]
            x_starts = [torch.randn(N, 4) * snr for _ in range(bs)]
            all_pred = torch.rand(2, bs, N, 4, requires_grad=True) * torch.tensor(
                [w, h, w, h]
            )
            t = torch.tensor([0.5, 0.7])
            img_metas = _make_img_metas(bs, h, w)
            lvd_loss = head._compute_lvd_loss(
                x_boxes=x_boxes, x_starts=x_starts,
                all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
            )
            assert lvd_loss.item() >= 0, f"{form}: loss must be non-negative"
            assert torch.isfinite(lvd_loss), f"{form}: loss must be finite"


class TestLVDThresholdMasking:
    """测试 t_threshold 和 xyxy 有效性掩码"""

    def test_small_t_masked(self):
        """t < lvd_t_threshold 时样本被跳过 (loss=0)"""
        head = _make_head(use_lvd=True, lvd_t_threshold=0.5)
        head.train()
        bs, N = 2, 8
        h, w = 100, 100
        snr = head.snr_scale
        x_boxes = [torch.randn(N, 4) * snr for _ in range(bs)]
        x_starts = [torch.randn(N, 4) * snr for _ in range(bs)]
        all_pred = torch.rand(2, bs, N, 4, requires_grad=True) * torch.tensor(
            [w, h, w, h]
        )
        # 两个样本 t 都 < 0.5 → 都被跳过 → loss=0
        t = torch.tensor([0.1, 0.2])
        img_metas = _make_img_metas(bs, h, w)
        lvd_loss = head._compute_lvd_loss(
            x_boxes=x_boxes, x_starts=x_starts,
            all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
        )
        assert lvd_loss.item() == 0.0, (
            f"t < threshold should give zero loss, got {lvd_loss.item()}"
        )

    def test_mixed_t_only_large_contributes(self):
        """混合 t: 仅大 t 样本贡献 loss"""
        head = _make_head(use_lvd=True, lvd_t_threshold=0.5)
        head.train()
        bs, N = 2, 8
        h, w = 100, 100
        snr = head.snr_scale
        x_boxes = [torch.randn(N, 4) * snr for _ in range(bs)]
        x_starts = [torch.randn(N, 4) * snr for _ in range(bs)]
        all_pred = torch.rand(2, bs, N, 4, requires_grad=True) * torch.tensor(
            [w, h, w, h]
        )
        # 样本 0: t=0.1 (跳过), 样本 1: t=0.8 (参与)
        t = torch.tensor([0.1, 0.8])
        img_metas = _make_img_metas(bs, h, w)
        lvd_loss = head._compute_lvd_loss(
            x_boxes=x_boxes, x_starts=x_starts,
            all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
        )
        # 应为非零 (样本 1 贡献)
        assert lvd_loss.item() > 0, "Large-t sample should contribute to loss"

    def test_invalid_xyxy_masked(self):
        """无效 xyxy 框 (x2<=x1) cos_sim 设为 1 (loss=0)"""
        head = _make_head(use_lvd=True)
        head.train()
        bs, N = 2, 4
        h, w = 100, 100
        snr = head.snr_scale
        x_boxes = [torch.randn(N, 4) * snr for _ in range(bs)]
        x_starts = [torch.randn(N, 4) * snr for _ in range(bs)]
        # 构造全部无效的 xyxy 框 (x2 < x1)
        all_pred = torch.zeros(2, bs, N, 4, requires_grad=True)
        # x1=50, x2=10 (x2 < x1, 无效)
        all_pred.data[..., 0] = 50  # x1
        all_pred.data[..., 2] = 10  # x2 (无效)
        all_pred.data[..., 1] = 50  # y1
        all_pred.data[..., 3] = 10  # y2 (无效)
        t = torch.tensor([0.5, 0.7])
        img_metas = _make_img_metas(bs, h, w)
        lvd_loss = head._compute_lvd_loss(
            x_boxes=x_boxes, x_starts=x_starts,
            all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
        )
        assert lvd_loss.item() == 0.0, (
            f"Invalid xyxy should give zero loss, got {lvd_loss.item()}"
        )


class TestLVDSpaceConversion:
    """测试空间转换正确性 (xyxy 像素 ↔ raw cxcywh)"""

    def test_space_conversion_roundtrip(self):
        """xyxy 像素 → raw cxcywh → xyxy 像素 应可逆 (与 raw_to_xyxy 互逆)"""
        head = _make_head(use_lvd=True)
        bs, N = 1, 4
        h, w = 100, 200
        snr = head.snr_scale
        # 构造有效 xyxy 框 (像素)
        from ldmdet.utils.box_ops import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
        # raw cxcywh in [-snr, snr]
        raw_cxcywh_orig = torch.tensor(
            [[0.5, -0.3, 1.0, 0.8], [-1.0, 0.5, 0.5, 0.5]]
        ) * snr
        # raw → norm xyxy → xyxy 像素 (复用 _sampler.raw_to_xyxy 逻辑)
        norm_cxcywh = (raw_cxcywh_orig / snr + 1) / 2
        norm_xyxy = bbox_cxcywh_to_xyxy(norm_cxcywh)
        scales = torch.tensor([w, h, w, h], dtype=torch.float32)
        xyxy_pixel = norm_xyxy * scales
        # xyxy 像素 → raw cxcywh (LVD 内部转换)
        norm_xyxy_back = xyxy_pixel / scales.unsqueeze(0)
        norm_cxcywh_back = bbox_xyxy_to_cxcywh(norm_xyxy_back)
        raw_cxcywh_back = (norm_cxcywh_back * 2 - 1) * snr
        assert torch.allclose(raw_cxcywh_orig, raw_cxcywh_back, atol=1e-5), (
            f"Roundtrip failed: {raw_cxcywh_orig} vs {raw_cxcywh_back}"
        )


class TestLVDIntegration:
    """测试 LVD-RF 与 loss() 的集成"""

    def test_loss_with_lvd_disabled_no_loss_lvd_key(self):
        """use_lvd=False 时 losses 不含 loss_lvd (向后兼容)"""
        head = _make_head(use_lvd=False)
        head.train()
        bs = 1
        h, w = 100, 100
        # 构造简单特征
        features = [torch.randn(1, 64, 7, 7, requires_grad=True)]
        img_metas = _make_img_metas(bs, h, w)
        gt_bboxes, gt_labels = _make_gt(bs, h=h, w=w)
        losses = head.loss(features, img_metas, gt_bboxes, gt_labels)
        assert 'loss_lvd' not in losses, "use_lvd=False should not produce loss_lvd"

    def test_loss_with_lvd_enabled_has_loss_lvd_key(self):
        """use_lvd=True 时 losses 含 loss_lvd 键"""
        head = _make_head(use_lvd=True)
        head.train()
        bs = 1
        h, w = 100, 100
        features = [torch.randn(1, 64, 7, 7, requires_grad=True)]
        img_metas = _make_img_metas(bs, h, w)
        gt_bboxes, gt_labels = _make_gt(bs, h=h, w=w)
        losses = head.loss(features, img_metas, gt_bboxes, gt_labels)
        assert 'loss_lvd' in losses, "use_lvd=True should produce loss_lvd"
        assert isinstance(losses['loss_lvd'], torch.Tensor)
        assert losses['loss_lvd'].item() >= 0

    def test_loss_lvd_gradient_flows_to_backbone(self):
        """loss_lvd 梯度能流回 backbone 特征"""
        head = _make_head(use_lvd=True)
        head.train()
        bs = 1
        h, w = 100, 100
        features = [torch.randn(1, 64, 7, 7, requires_grad=True)]
        img_metas = _make_img_metas(bs, h, w)
        gt_bboxes, gt_labels = _make_gt(bs, h=h, w=w)
        losses = head.loss(features, img_metas, gt_bboxes, gt_labels)
        total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
        total_loss.backward()
        assert features[0].grad is not None, "Gradient must flow to features"
        assert torch.isfinite(features[0].grad).all()


class TestLVDSpaceParameter:
    """测试 lvd_space 参数 (方案 §7.2: 计算空间, 默认 raw_cxcywh)

    方案 §3.2 明确: LVD-RF 仅在 raw cxcywh 空间计算 (xyxy 像素 → 归一化 →
    raw cxcywh 转换链). 当前仅支持 'raw_cxcywh' 一个空间, 非该值应直接报错退出
    (符合 "遇到和方案设计不一致的行为直接报错退出" 的实现约束).
    """

    def test_lvd_space_default_raw_cxcywh(self):
        """lvd_space 默认 'raw_cxcywh' (方案 §7.2)"""
        head = _make_head(use_lvd=True)
        assert head.lvd_space == 'raw_cxcywh'

    def test_lvd_space_explicit_raw_cxcywh(self):
        """显式传 lvd_space='raw_cxcywh' 可接受"""
        head = _make_head(use_lvd=True, lvd_space='raw_cxcywh')
        assert head.lvd_space == 'raw_cxcywh'

    def test_lvd_space_invalid_rejected(self):
        """lvd_space 非 'raw_cxcywh' 应直接报错 (仅支持唯一计算空间)"""
        with pytest.raises(AssertionError):
            _make_head(use_lvd=True, lvd_space='xyxy_pixel')

    def test_lvd_space_disabled_head_still_has_attr(self):
        """use_lvd=False 时 lvd_space 属性仍存在且为默认值"""
        head = _make_head(use_lvd=False)
        assert head.lvd_space == 'raw_cxcywh'


class TestLVDAdaptiveSwitching:
    """测试自适应切换 sin2 → sqrt (方案 §7.2 注 2.3, §9 风险缓解)

    机制: cos_sim_mean > 0.99 持续 1000 iter → 自动将 lvd_form 从 'sin2'
    切换为 'sqrt' (sqrt 形式在 cos_sim→1 时具有非零梯度 0.5/sqrt(ε),
    避免梯度消失). 一旦 cos_sim_mean ≤ 0.99, 计数器立即归零.
    """

    def _make_ideal_alignment_inputs(self, bs=1, N=2, h=100, w=100):
        """构造理想对齐输入 (x_hat_0 ≈ x_0 → cos_sim ≈ 1.0 > 0.99)"""
        snr = 2.0
        torch.manual_seed(0)
        # x_0: raw cxcywh
        x_starts = [torch.randn(N, 4) * snr for _ in range(bs)]
        # x_t = (1-t)*x_0 + t*noise, t=0.5
        noise = [torch.randn(N, 4) * snr for _ in range(bs)]
        t_val = 0.5
        x_boxes = [(1 - t_val) * x_starts[i] + t_val * noise[i]
                   for i in range(bs)]
        # x_hat_0 = x_0 (理想): raw cxcywh → xyxy 像素 (走 _compute_lvd_loss 逆转换)
        from ldmdet.utils.box_ops import bbox_cxcywh_to_xyxy
        x_hat_0_raw = torch.stack(x_starts)
        norm_cxcywh = (x_hat_0_raw / snr + 1) / 2
        norm_xyxy = bbox_cxcywh_to_xyxy(norm_cxcywh)
        scales = torch.tensor([w, h, w, h], dtype=torch.float32)
        pred_xyxy = norm_xyxy * scales
        all_pred = pred_xyxy.unsqueeze(0).repeat(2, 1, 1, 1)
        t = torch.tensor([t_val] * bs)
        img_metas = _make_img_metas(bs, h, w)
        return x_boxes, x_starts, all_pred, t, img_metas

    def test_switch_sin2_to_sqrt_after_threshold(self):
        """cos_sim>0.99 持续 >1000 iter → lvd_form 由 sin2 切换为 sqrt"""
        head = _make_head(use_lvd=True, lvd_form='sin2')
        head.train()
        x_boxes, x_starts, all_pred, t, img_metas = (
            self._make_ideal_alignment_inputs()
        )
        assert head.lvd_form == 'sin2'
        assert head._cos_sim_high_count == 0

        # 调用 1001 次 (cos_sim ≈ 1.0 > 0.99, 每次计数器 +1)
        for _ in range(1001):
            head._compute_lvd_loss(
                x_boxes=x_boxes, x_starts=x_starts,
                all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
            )
        # 第 1001 次后 _cos_sim_high_count = 1001 > 1000 → 触发切换
        assert head.lvd_form == 'sqrt', (
            f"应切换为 sqrt, got {head.lvd_form}"
        )

    def test_no_switch_below_threshold_iters(self):
        """cos_sim>0.99 持续 ≤1000 iter 不切换 (边界: 1000 次仍为 sin2)"""
        head = _make_head(use_lvd=True, lvd_form='sin2')
        head.train()
        x_boxes, x_starts, all_pred, t, img_metas = (
            self._make_ideal_alignment_inputs()
        )
        # 调用 1000 次: 计数器 = 1000, 条件是 > 1000, 不触发
        for _ in range(1000):
            head._compute_lvd_loss(
                x_boxes=x_boxes, x_starts=x_starts,
                all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
            )
        assert head.lvd_form == 'sin2', "1000 次不应触发切换 (条件 > 1000)"
        assert head._cos_sim_high_count == 1000

    def test_low_cos_sim_resets_counter(self):
        """cos_sim_mean ≤ 0.99 时计数器立即归零"""
        head = _make_head(use_lvd=True, lvd_form='sin2')
        head.train()
        # 先用理想对齐累计计数器
        x_boxes, x_starts, all_pred, t, img_metas = (
            self._make_ideal_alignment_inputs()
        )
        for _ in range(500):
            head._compute_lvd_loss(
                x_boxes=x_boxes, x_starts=x_starts,
                all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
            )
        assert head._cos_sim_high_count == 500

        # 再用随机错对齐 (cos_sim 通常 < 0.99) 调用一次 → 计数器归零
        bs, N = 1, 8
        h, w = 100, 100
        snr = head.snr_scale
        x_boxes_rand = [torch.randn(N, 4) * snr for _ in range(bs)]
        x_starts_rand = [torch.randn(N, 4) * snr for _ in range(bs)]
        all_pred_rand = torch.rand(2, bs, N, 4) * torch.tensor([w, h, w, h])
        t_rand = torch.tensor([0.5])
        img_metas_rand = _make_img_metas(bs, h, w)
        head._compute_lvd_loss(
            x_boxes=x_boxes_rand, x_starts=x_starts_rand,
            all_pred_bboxes=all_pred_rand, t=t_rand,
            img_metas=img_metas_rand,
        )
        # 若该次 cos_sim_mean ≤ 0.99 → 计数器归零
        # (随机输入 cos_sim_mean 极大概率 < 0.99)
        assert head._cos_sim_high_count == 0, (
            f"低 cos_sim 应归零计数器, got {head._cos_sim_high_count}"
        )
        assert head.lvd_form == 'sin2'

    def test_no_switch_when_form_already_not_sin2(self):
        """lvd_form='cos' 时即使 cos_sim>0.99 持续也不切换 (条件需 sin2)"""
        head = _make_head(use_lvd=True, lvd_form='cos')
        head.train()
        x_boxes, x_starts, all_pred, t, img_metas = (
            self._make_ideal_alignment_inputs()
        )
        for _ in range(1001):
            head._compute_lvd_loss(
                x_boxes=x_boxes, x_starts=x_starts,
                all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
            )
        assert head.lvd_form == 'cos', "cos 形式不应被切换"
        assert head._cos_sim_high_count == 1001


class TestLVDCosSimMasking:
    """测试 cos_sim 诊断/切换均应仅在有效样本 (effective_mask) 上统计

    背景 (审计问题 2+4):
      - L829 将无效框 cos_sim 置 1.0 (loss 占位符, 使其 sin²=0 不贡献 loss)
      - 但 L855/L856 诊断与 L861 切换判断用全量 cos_sim.mean()/min()
      - 无效框 (cos=1.0 占位) 与小 t 样本 (||x_t-x_0||→0, cos 为噪声) 混入均值,
        会虚高 cos_sim_mean, 训练早期无效框占比高时可误触发 sin2→sqrt 切换.
      - Loss 归一化已用 effective_mask (L848), 诊断/切换应保持一致.
    """

    def test_invalid_boxes_do_not_false_trigger_switching(self):
        """无效框占比高时不应误触发 sin2→sqrt 切换

        场景: 100 个 proposal, 99 个 xyxy 无效 (x2<x1, cos_sim 被置 1.0),
        1 个有效框真实 cos_sim≈0.70 (对齐差). 全量 mean=(99+0.70)/100=0.997>0.99
        会误增计数器; 修复后应仅统计有效框, 均值=0.70<0.99, 不触发.
        """
        head = _make_head(use_lvd=True, lvd_form='sin2')
        head.train()
        bs, N = 1, 100
        h, w = 100, 100
        snr = head.snr_scale
        t_val = 0.5  # 大 t → t_mask=1 (隔离小 t 因素, 只测无效框污染)
        t = torch.tensor([t_val])
        img_metas = _make_img_metas(bs, h, w)

        # x_0, x_t: raw cxcywh
        x_starts = [torch.zeros(N, 4)]
        x_boxes = [torch.zeros(N, 4)]
        # box 99: 确定性构造 cos_sim≈0.707 (对齐差但 cos>0, 确保全量均值>0.99)
        #   x_0=[0,0,0.4,0.4], x_t=[0.2,0,0.4,0.4] → d_gt=[0.2,0,0,0] (沿 cx)
        #   x_hat_0=[0,0.2,0.4,0.4] → d_pred=[0.2,-0.2,0,0] → cos=0.707
        x_starts[0][99] = torch.tensor([0.0, 0.0, 0.4, 0.4]) * snr / snr  # = [0,0,0.4,0.4]
        x_boxes[0][99] = torch.tensor([0.2, 0.0, 0.4, 0.4])
        # box 0..98: 随机 x_0/x_t (cos_sim 不重要, 会被置 1.0)
        torch.manual_seed(7)
        x_starts[0][:99] = torch.randn(99, 4) * snr
        x_boxes[0][:99] = torch.randn(99, 4) * snr

        # all_pred: [num_heads=2, bs, N, 4] xyxy 像素
        all_pred = torch.zeros(2, bs, N, 4)
        # box 0..98: 无效 xyxy (x1=80 > x2=20)
        all_pred[:, :, :99, 0] = 80  # x1
        all_pred[:, :, :99, 2] = 20  # x2 (无效)
        all_pred[:, :, :99, 1] = 80  # y1
        all_pred[:, :, :99, 3] = 20  # y2 (无效)
        # box 99: 有效 xyxy, 对应 x_hat_0=[0,0.2,0.4,0.4] raw cxcywh
        #   raw→norm: ([0,0.2,0.4,0.4]/snr+1)/2 = [0.5,0.55,0.6,0.6]
        #   norm cxcywh→norm xyxy: [0.2,0.25,0.8,0.85] → pixel [20,25,80,85]
        all_pred[:, :, 99, 0] = 20
        all_pred[:, :, 99, 1] = 25
        all_pred[:, :, 99, 2] = 80
        all_pred[:, :, 99, 3] = 85
        all_pred.requires_grad_(True)

        head._compute_lvd_loss(
            x_boxes=x_boxes, x_starts=x_starts,
            all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
        )
        # 有效框真实 cos_sim≈0.707 < 0.99 → 不应触发计数器
        assert head._cos_sim_high_count == 0, (
            f"无效框(占位 cos=1.0)不应污染切换判断; 有效框 cos≈0.707 < 0.99, "
            f"但计数器={head._cos_sim_high_count} (说明全量均值被无效框虚高误触发)"
        )

    def test_small_t_excluded_from_cos_sim_diagnostic(self):
        """小 t 样本不应污染 train/cos_sim_mean 诊断

        场景: bs=2. 图 0 t=0.01 (<阈值, t_mask=0), 理想对齐 cos=1.0;
        图 1 t=0.5 (t_mask=1), cos=0.0 (正交).
        全量 mean=(1.0+0.0)/2=0.5; 修复后应仅统计图 1 (大 t), 均值=0.0.
        """
        head = _make_head(use_lvd=True, lvd_form='sin2')
        head.train()
        bs, N = 2, 2
        h, w = 100, 100
        img_metas = _make_img_metas(bs, h, w)
        t = torch.tensor([0.01, 0.5])  # 图 0 小 t, 图 1 大 t

        # 图 0: x_hat_0 = x_0 (理想, cos=1.0), x_t = 0.99*x_0 + 0.01*noise
        x0_img0 = torch.tensor([0.0, 0.0, 0.4, 0.4])
        xt_img0 = 0.99 * x0_img0 + 0.01 * torch.tensor([0.5, -0.3, 0.2, 0.1])
        # 图 1: x_0=[0,0,0.4,0.4], x_t=[0.2,0,0.4,0.4] → d_gt 沿 cx
        #   x_hat_0=[0.2,-0.2,0.4,0.4] → d_pred=[0,0.2,0,0] 沿 cy → cos=0.0
        x0_img1 = torch.tensor([0.0, 0.0, 0.4, 0.4])
        xt_img1 = torch.tensor([0.2, 0.0, 0.4, 0.4])

        x_starts = [x0_img0.unsqueeze(0).repeat(N, 1),
                    x0_img1.unsqueeze(0).repeat(N, 1)]
        x_boxes = [xt_img0.unsqueeze(0).repeat(N, 1),
                   xt_img1.unsqueeze(0).repeat(N, 1)]

        # all_pred xyxy 像素
        all_pred = torch.zeros(2, bs, N, 4)
        # 图 0: x_hat_0 = x_0 = [0,0,0.4,0.4] → norm [0.5,0.5,0.6,0.6]
        #   → norm xyxy [0.2,0.2,0.8,0.8] → pixel [20,20,80,80]
        all_pred[:, 0, :, :] = torch.tensor([20.0, 20.0, 80.0, 80.0])
        # 图 1: x_hat_0 = [0.2,-0.2,0.4,0.4] → norm ([0.2,-0.2,0.4,0.4]/2+1)/2
        #   = [0.55,0.45,0.6,0.6] → norm xyxy [0.25,0.15,0.85,0.75] → pixel [25,15,85,75]
        all_pred[:, 1, :, :] = torch.tensor([25.0, 15.0, 85.0, 75.0])
        all_pred.requires_grad_(True)

        # 启用 probe 采集本次诊断值 (finally 中关闭, 避免污染其他测试)
        probe.enable(train_interval=1)
        probe._should_collect = True
        probe._scalars['train'].clear()
        try:
            head._compute_lvd_loss(
                x_boxes=x_boxes, x_starts=x_starts,
                all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
            )
            recorded = probe._scalars['train'].get('train/cos_sim_mean')
            assert recorded is not None, "应记录 train/cos_sim_mean"
            # 修复后: 仅图 1 (大 t) 进均值 → cos=0.0; 不应是混入图 0 的 0.5
            assert recorded < 0.1, (
                f"小 t 样本 (cos=1.0) 不应污染诊断均值; 修复后应≈0.0 (仅大 t 图), "
                f"got {recorded} (全量均值=0.5 说明被污染)"
            )
        finally:
            probe.disable()


class TestLVDGradientStrength:
    """验证方案 §2.6 核心论断: sin² 梯度比 1-cos 强 2×

    数学 (方案 §2.6):
      - 1 - cos(α):  dL/dα = sin(α) ≈ α        (线性消失)
      - sin²(α):     dL/dα = sin(2α) ≈ 2α      (2× 强梯度)
    近而对齐时 (α→0): sin²α = (1-cosα)(1+cosα) ≈ 2(1-cosα),
    即 sin² 形式的损失约为 1-cos 形式的 2 倍. 这正是默认采用 sin² 的动机.
    """

    def test_sin2_loss_approx_double_cos_loss_near_alignment(self):
        """近对齐时 sin²_loss ≈ 2 × cos_loss (方案 §2.6 2× 论断)"""
        head_sin2 = _make_head(use_lvd=True, lvd_form='sin2')
        head_cos = _make_head(use_lvd=True, lvd_form='cos')
        bs, N = 1, 8
        h, w = 100, 100
        snr = head_sin2.snr_scale
        # x_0, x_t (raw cxcywh)
        torch.manual_seed(123)
        x_starts = [torch.randn(N, 4) * snr for _ in range(bs)]
        noise = [torch.randn(N, 4) * snr for _ in range(bs)]
        t_val = 0.5
        x_boxes = [(1 - t_val) * x_starts[i] + t_val * noise[i]
                   for i in range(bs)]
        # x_hat_0 = x_0 + 小扰动 → d_pred 与 d_gt 有小夹角 α
        # 扰动需让 cos_sim 落在 [0.9, 0.99] (小角近似成立且 loss 非数值噪声)
        perturb = torch.randn(N, 4) * 0.15 * snr
        x_hat_0_raw = (torch.stack(x_starts) + perturb)
        from ldmdet.utils.box_ops import bbox_cxcywh_to_xyxy
        norm_cxcywh = (x_hat_0_raw / snr + 1) / 2
        norm_xyxy = bbox_cxcywh_to_xyxy(norm_cxcywh)
        scales = torch.tensor([w, h, w, h], dtype=torch.float32)
        pred_xyxy = norm_xyxy * scales
        all_pred = pred_xyxy.unsqueeze(0).repeat(2, 1, 1, 1)
        t = torch.tensor([t_val] * bs)
        img_metas = _make_img_metas(bs, h, w)

        loss_sin2 = head_sin2._compute_lvd_loss(
            x_boxes=x_boxes, x_starts=x_starts,
            all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
        )
        loss_cos = head_cos._compute_lvd_loss(
            x_boxes=x_boxes, x_starts=x_starts,
            all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
        )
        # 近对齐: sin² ≈ 2 × (1-cos), 比值应在 [1.5, 2.5] 内
        ratio = loss_sin2.item() / max(loss_cos.item(), 1e-8)
        assert 1.5 < ratio < 2.5, (
            f"sin²/cos 损失比应 ≈ 2 (方案 §2.6), got ratio={ratio:.4f}, "
            f"sin2={loss_sin2.item():.6f}, cos={loss_cos.item():.6f}"
        )

    def test_sin2_loss_zero_at_exact_alignment(self):
        """精确对齐 (α=0) 时 sin²=0 且 1-cos=0 (两种形式都收敛到 0)"""
        head_sin2 = _make_head(use_lvd=True, lvd_form='sin2')
        head_cos = _make_head(use_lvd=True, lvd_form='cos')
        bs, N = 1, 4
        h, w = 100, 100
        snr = head_sin2.snr_scale
        torch.manual_seed(0)
        x_starts = [torch.randn(N, 4) * snr for _ in range(bs)]
        noise = [torch.randn(N, 4) * snr for _ in range(bs)]
        t_val = 0.5
        x_boxes = [(1 - t_val) * x_starts[i] + t_val * noise[i]
                   for i in range(bs)]
        # x_hat_0 = x_0 精确对齐
        from ldmdet.utils.box_ops import bbox_cxcywh_to_xyxy
        x_hat_0_raw = torch.stack(x_starts)
        norm_cxcywh = (x_hat_0_raw / snr + 1) / 2
        norm_xyxy = bbox_cxcywh_to_xyxy(norm_cxcywh)
        scales = torch.tensor([w, h, w, h], dtype=torch.float32)
        pred_xyxy = norm_xyxy * scales
        all_pred = pred_xyxy.unsqueeze(0).repeat(2, 1, 1, 1)
        t = torch.tensor([t_val] * bs)
        img_metas = _make_img_metas(bs, h, w)
        loss_sin2 = head_sin2._compute_lvd_loss(
            x_boxes=x_boxes, x_starts=x_starts,
            all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
        )
        loss_cos = head_cos._compute_lvd_loss(
            x_boxes=x_boxes, x_starts=x_starts,
            all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
        )
        assert loss_sin2.item() < 1e-3, f"sin² at α=0 ≈ 0, got {loss_sin2.item()}"
        assert loss_cos.item() < 1e-3, f"1-cos at α=0 ≈ 0, got {loss_cos.item()}"


def _collect_lvd_probe(head, x_boxes, x_starts, all_pred, t, img_metas, key):
    """启用 probe 调用一次 _compute_lvd_loss, 返回指定诊断键值 (调用后关闭 probe)."""
    probe.enable(train_interval=1)
    probe._should_collect = True
    probe._scalars['train'].clear()
    try:
        head._compute_lvd_loss(
            x_boxes=x_boxes, x_starts=x_starts,
            all_pred_bboxes=all_pred, t=t, img_metas=img_metas,
        )
        return probe._scalars['train'].get(key)
    finally:
        probe.disable()


class TestLVDGradientProbe:
    """测试 LVD 梯度量级探针 (监测 sqrt 切换后梯度放大, 审计问题 3)

    探针记录 LVD 损失对 cos_sim 的解析梯度量级 (×λ 后实际尺度):
      - train/lvd_grad_mean: 有效样本梯度均值
      - train/lvd_grad_max:  有效样本梯度最大值
      - train/lvd_grad_deadzone_ratio (仅 sqrt): 1-cos≤ε 梯度归 0 的样本占比

    数学 (方案 §2.6):
      sin2:  |d(1-cos²)/d(cos)| = 2|cos|             (近对齐 →2)
      cos:   |d(1-cos)/d(cos)|   = 1                 (恒定)
      sqrt:  |d(√((1-cos)∧ε))/d(cos)| = 0.5/√(1-cos) (cos→1 →0.5/√ε=500)
            clamp 边界下方 (1-cos≤ε) 梯度归 0 (死区)
    """

    def _make_near_alignment_inputs(self, perturb_scale=0.1, bs=1, N=8):
        """构造近对齐输入 (x_hat_0 = x_0 + 小扰动, cos 较高但 <1)"""
        h, w = 100, 100
        snr = 2.0
        torch.manual_seed(42)
        x_starts = [torch.randn(N, 4) * snr for _ in range(bs)]
        noise = [torch.randn(N, 4) * snr for _ in range(bs)]
        t_val = 0.5
        x_boxes = [(1 - t_val) * x_starts[i] + t_val * noise[i]
                   for i in range(bs)]
        from ldmdet.utils.box_ops import bbox_cxcywh_to_xyxy
        x_hat_0_raw = torch.stack(x_starts) + torch.randn(N, 4) * perturb_scale * snr
        norm_cxcywh = (x_hat_0_raw / snr + 1) / 2
        norm_xyxy = bbox_cxcywh_to_xyxy(norm_cxcywh)
        scales = torch.tensor([w, h, w, h], dtype=torch.float32)
        pred_xyxy = norm_xyxy * scales
        all_pred = pred_xyxy.unsqueeze(0).repeat(2, 1, 1, 1)
        t = torch.tensor([t_val] * bs)
        img_metas = _make_img_metas(bs, h, w)
        return x_boxes, x_starts, all_pred, t, img_metas

    def test_grad_probes_recorded(self):
        """train/lvd_grad_mean 与 train/lvd_grad_max 应被记录"""
        head = _make_head(use_lvd=True, lvd_form='sin2')
        head.train()
        inputs = self._make_near_alignment_inputs()
        grad_mean = _collect_lvd_probe(head, *inputs, 'train/lvd_grad_mean')
        grad_max = _collect_lvd_probe(head, *inputs, 'train/lvd_grad_max')
        assert grad_mean is not None, "应记录 train/lvd_grad_mean"
        assert grad_max is not None, "应记录 train/lvd_grad_max"
        assert grad_mean >= 0
        assert grad_max >= grad_mean, "max 应 ≥ mean"

    def test_sqrt_grad_larger_than_sin2_near_alignment(self):
        """近对齐时 sqrt 形式梯度 > sin2 形式 (验证探针能监测 sqrt 放大)

        数学: cos>0.9375 时 0.5/√(1-cos) > 2·cos, 故 sqrt 梯度 > sin2 梯度.
        近对齐 (cos≈0.97+) 时 sqrt 放大明显, 探针应反映这一差异.
        """
        inputs = self._make_near_alignment_inputs(perturb_scale=0.08)
        head_sin2 = _make_head(use_lvd=True, lvd_form='sin2')
        head_sin2.train()
        head_sqrt = _make_head(use_lvd=True, lvd_form='sqrt')
        head_sqrt.train()
        sin2_grad_max = _collect_lvd_probe(
            head_sin2, *inputs, 'train/lvd_grad_max'
        )
        sqrt_grad_max = _collect_lvd_probe(
            head_sqrt, *inputs, 'train/lvd_grad_max'
        )
        assert sin2_grad_max is not None and sqrt_grad_max is not None
        # sqrt 梯度应显著大于 sin2 (近对齐放大, 方案 §2.6 注 2.3 核心动机)
        assert sqrt_grad_max > sin2_grad_max * 1.2, (
            f"sqrt 梯度应 > sin2×1.2 (近对齐放大), "
            f"got sqrt={sqrt_grad_max:.4f}, sin2={sin2_grad_max:.4f}"
        )

    def test_sqrt_deadzone_ratio_recorded(self):
        """sqrt 形式应记录 train/lvd_grad_deadzone_ratio (clamp 死区占比)"""
        head = _make_head(use_lvd=True, lvd_form='sqrt')
        head.train()
        inputs = self._make_near_alignment_inputs()
        ratio = _collect_lvd_probe(head, *inputs, 'train/lvd_grad_deadzone_ratio')
        assert ratio is not None, "sqrt 形式应记录 train/lvd_grad_deadzone_ratio"
        assert 0.0 <= ratio <= 1.0, f"死区占比应在 [0,1], got {ratio}"

    def test_sin2_does_not_record_deadzone(self):
        """sin2 形式无 clamp, 不应记录 deadzone_ratio"""
        head = _make_head(use_lvd=True, lvd_form='sin2')
        head.train()
        inputs = self._make_near_alignment_inputs()
        ratio = _collect_lvd_probe(head, *inputs, 'train/lvd_grad_deadzone_ratio')
        assert ratio is None, "sin2 无 clamp 死区, 不应记录 deadzone_ratio"

    def test_grad_max_bounded_by_sqrt_upper_limit(self):
        """sqrt 梯度最大值有上界 0.5/√ε ×λ (方案 §2.6: 有界非零)"""
        head = _make_head(use_lvd=True, lvd_form='sqrt')
        head.train()
        inputs = self._make_near_alignment_inputs()
        grad_max = _collect_lvd_probe(head, *inputs, 'train/lvd_grad_max')
        # 上界: 0.5/√ε × λ = 0.5/√(1e-6) × 0.1 = 500 × 0.1 = 50
        upper = 0.5 / (head.lvd_eps ** 0.5) * head.lvd_lambda
        assert grad_max <= upper + 1e-6, (
            f"sqrt 梯度应 ≤ 0.5/√ε×λ = {upper}, got {grad_max}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
