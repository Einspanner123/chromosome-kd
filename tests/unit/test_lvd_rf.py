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
    lvd_t_threshold=0.05, num_classes=24, feat_channels=64,
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
