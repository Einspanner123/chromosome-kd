"""红绿重构测试: 验证 RoI Align 内容注入修复

验证清单:
  [ ] SingleRoIExtractor 正确创建并提取特征
  [ ] BoxTokenizer 接受 roi_content 参数，输出形状正确
  [ ] DiTDiffusionDetHead forward 端到端通过
  [ ] DiTDiffusionDetHead loss 端到端通过
  [ ] 反向传播无错误
  [ ] 分类 logits 不为全负 (验证 prior_prob=0.5 生效)
"""

import torch
import torch.nn as nn

# 添加项目路径
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mods.roi_extractor import SingleRoIExtractor
from mods.box_tokenizer import BoxTokenizer
from mods.dit_head import DiTDiffusionDetHead
from mods.structures import ImageMeta


def create_mock_image_metas(batch_size=2, img_size=(800, 1200)):
    """创建模拟图像元数据"""
    metas = []
    for i in range(batch_size):
        meta = ImageMeta(
            img_shape=(img_size[0], img_size[1], 3),
            pad_shape=(img_size[0], img_size[1], 3),
            scale_factor=[1.0, 1.0, 1.0, 1.0],
        )
        metas.append(meta)
    return metas


def create_mock_fpn_features(batch_size=2, feat_channels=384):
    """创建模拟 FPN 特征 [P2, P3, P4, P5]"""
    strides = [4, 8, 16, 32]
    base_size = 800
    feats = []
    for s in strides:
        h = base_size // s
        w = 1200 // s
        feats.append(torch.randn(batch_size, feat_channels, h, w))
    return feats


def create_mock_gt(batch_size=2, num_gt_per_image=5, img_size=(800, 1200)):
    """创建模拟 GT bboxes 和 labels"""
    gt_bboxes = []
    gt_labels = []
    for _ in range(batch_size):
        # 生成随机非空 GT 框 (xyxy, 图像坐标)
        bboxes = torch.rand(num_gt_per_image, 4)
        # 确保 x2 > x1, y2 > y1
        bboxes[:, [0, 2]] = (
            bboxes[:, [0, 2]].sort(dim=1)[0] * img_size[1]
        )
        bboxes[:, [1, 3]] = (
            bboxes[:, [1, 3]].sort(dim=1)[0] * img_size[0]
        )
        bboxes[:, 2] = torch.min(
            bboxes[:, 2], bboxes.new_full((num_gt_per_image,), float(img_size[1]))
        )
        bboxes[:, 2] = torch.max(bboxes[:, 2], bboxes[:, 0] + 10)
        bboxes[:, 3] = torch.min(
            bboxes[:, 3], bboxes.new_full((num_gt_per_image,), float(img_size[0]))
        )
        bboxes[:, 3] = torch.max(bboxes[:, 3], bboxes[:, 1] + 10)
        gt_bboxes.append(bboxes)
        gt_labels.append(torch.randint(0, 24, (num_gt_per_image,)))
    return gt_bboxes, gt_labels


# ============================================================
# Test 1: SingleRoIExtractor 正确性
# ============================================================
def test_roi_extractor():
    """测试 ROI 提取器形状正确"""
    print("\n=== Test 1: SingleRoIExtractor ===")
    roi_extractor = SingleRoIExtractor(
        roi_layer={'type': 'RoIAlign', 'output_size': (7, 7), 'sampling_ratio': 2},
        out_channels=384,
        featmap_strides=[4, 8, 16, 32],
    )
    fpn = create_mock_fpn_features()
    bs, N = 2, 100
    # 模拟 proposal bboxes (图像坐标)
    bboxes_img = torch.rand(bs, N, 4)
    bboxes_img[..., [0, 2]] *= 1200
    bboxes_img[..., [1, 3]] *= 800
    # 确保 x2 > x1, y2 > y1
    bboxes_img[..., [0, 2]] = bboxes_img[..., [0, 2]].sort(dim=-1)[0]
    bboxes_img[..., [1, 3]] = bboxes_img[..., [1, 3]].sort(dim=-1)[0]

    batch_idx = torch.arange(bs).unsqueeze(1).expand(bs, N).flatten()
    rois = torch.stack([
        batch_idx.float(),
        bboxes_img[..., 0].flatten(),
        bboxes_img[..., 1].flatten(),
        bboxes_img[..., 2].flatten(),
        bboxes_img[..., 3].flatten(),
    ], dim=-1)

    roi_feats = roi_extractor(tuple(fpn), rois)
    assert roi_feats.shape == (bs * N, 384, 7, 7), (
        f"Expected ({bs*N}, 384, 7, 7), got {roi_feats.shape}"
    )
    assert not torch.isnan(roi_feats).any(), "NaN in ROI features"
    assert not torch.isinf(roi_feats).any(), "Inf in ROI features"
    print(f"  PASS: ROI features shape = {tuple(roi_feats.shape)}")
    return roi_extractor


# ============================================================
# Test 2: BoxTokenizer 接受 roi_content
# ============================================================
def test_box_tokenizer_with_roi():
    """测试 BoxTokenizer 接受 roi_content 参数"""
    print("\n=== Test 2: BoxTokenizer with roi_content ===")
    bs, N, C = 2, 100, 384
    box_size = (7, 7)

    tokenizer = BoxTokenizer(
        feat_channels=C,
        num_fpn_levels=4,
        init_mode='spatial_prior',
        num_proposals=N,
    )

    # 正常化的 bboxes [0,1]
    bboxes = torch.rand(bs, N, 4)
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]].sort(dim=-1)[0]
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]].sort(dim=-1)[0]

    fpn = create_mock_fpn_features(batch_size=bs, feat_channels=C)

    # Test A: 无 roi_content (降级路径)
    tokens_a, levels_a = tokenizer(bboxes, fpn)
    assert tokens_a.shape == (bs, N, C), (
        f"Expected tokens ({bs}, {N}, {C}), got {tokens_a.shape}"
    )
    print(f"  PASS (no roi): tokens shape = {tuple(tokens_a.shape)}")

    # Test B: 有 roi_content (主路径)
    roi_content = torch.randn(bs * N, C, *box_size)
    tokens_b, levels_b = tokenizer(bboxes, fpn, roi_content=roi_content)
    assert tokens_b.shape == (bs, N, C), (
        f"Expected tokens ({bs}, {N}, {C}), got {tokens_b.shape}"
    )
    print(f"  PASS (with roi): tokens shape = {tuple(tokens_b.shape)}")

    # Test C: roi_content 与无 roi 的输出应不同
    diff = (tokens_a - tokens_b).abs().mean()
    assert diff > 1e-6, f"roi_content should change output, but diff={diff:.6f}"
    print(f"  PASS: roi vs no-roi difference = {diff:.6f} (expected > 0)")


# ============================================================
# Test 3: DiTDiffusionDetHead forward 端到端
# ============================================================
def test_dit_head_forward():
    """测试 DiT Head forward 端到端 (含 roi_extractor)"""
    print("\n=== Test 3: DiTDiffusionDetHead forward ===")
    bs, N, C = 2, 100, 384
    num_classes = 24

    head = DiTDiffusionDetHead(
        num_classes=num_classes,
        feat_channels=C,
        num_proposals=N,
        num_heads=3,
        num_blocks=1,
        prior_prob=0.5,
        deep_supervision=True,
        box_init_mode='spatial_prior',
        use_adaln_zero=False,
        share_heads=False,
        roi_extractor={
            'roi_layer': {'output_size': (7, 7), 'sampling_ratio': 2},
            'featmap_strides': [4, 8, 16, 32],
        },
    )

    fpn = create_mock_fpn_features(batch_size=bs, feat_channels=C)
    img_metas = create_mock_image_metas(batch_size=bs)

    # 模拟当前 bboxes (图像坐标)
    bboxes = torch.rand(bs, N, 4) * 800
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]].sort(dim=-1)[0]
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]].sort(dim=-1)[0]

    t = torch.full((bs,), 500.0, dtype=torch.float32)  # t*1000 对应 RF

    (
        all_cls_logits,
        all_pred_bboxes,
        all_objectness,
        all_velocity,
        all_curr_proposals,
    ) = head(features=tuple(fpn), bboxes=bboxes, t=t, img_metas=img_metas)

    # 验证形状
    assert all_cls_logits.shape[0] == 3, (
        f"Expected 3 heads, got {all_cls_logits.shape[0]}"
    )
    assert all_cls_logits.shape[1:] == (bs, N, num_classes), (
        f"Expected (bs, N, C)={(bs, N, num_classes)}, got {all_cls_logits.shape[1:]}"
    )
    assert all_pred_bboxes.shape[1:] == (bs, N, 4), (
        f"Expected (bs, N, 4), got {all_pred_bboxes.shape[1:]}"
    )

    # 检查 cls_logits 范围 (prior_prob=0.5, bias_value=0)
    sigmoid_scores = torch.sigmoid(all_cls_logits[-1])
    max_score = sigmoid_scores.max().item()
    print(f"  sigmoid max score = {max_score:.4f} (prior_prob=0.5, "
          f"expected ~0.5 initially)")
    assert max_score >= 0.3, (
        f"prior_prob=0.5 should give sigmoid around 0.5, got max={max_score:.4f}"
    )

    # 检查 NaN
    assert not torch.isnan(all_cls_logits).any(), "NaN in cls_logits"
    assert not torch.isnan(all_pred_bboxes).any(), "NaN in pred_bboxes"

    print(f"  PASS: cls_logits shape = {tuple(all_cls_logits.shape)}")
    print(f"  PASS: pred_bboxes shape = {tuple(all_pred_bboxes.shape)}")


# ============================================================
# Test 4: DiTDiffusionDetHead loss + backward
# ============================================================
def test_dit_head_loss():
    """测试 DiT Head loss 计算 + 反向传播"""
    print("\n=== Test 4: DiTDiffusionDetHead loss + backward ===")
    bs, N, C = 2, 100, 384
    num_classes = 24

    # 需要 criterion，构造最小 criterion
    from mods.loss import (
        DiffusionDetCriterion,
        DiffusionDetMatcher,
        FocalLoss,
        FocalLossCost,
        L1Loss,
        GIoULoss,
        RelativeL1Cost,
        IoUCost,
    )

    criterion = DiffusionDetCriterion(
        num_classes=num_classes,
        bbox_loss_mode='relative_l1',
        matcher=DiffusionDetMatcher(
            match_costs=[
                FocalLossCost(weight=2.0),
                RelativeL1Cost(weight=5.0),
                IoUCost(iou_mode='giou', weight=2.0),
            ],
            center_radius=5.0,
            candidate_topk=12,
        ),
        loss_cls=FocalLoss(loss_weight=2.0),
        loss_bbox=L1Loss(loss_weight=5.0),
        loss_giou=GIoULoss(loss_weight=2.0),
    )

    head = DiTDiffusionDetHead(
        num_classes=num_classes,
        feat_channels=C,
        num_proposals=N,
        num_heads=3,
        num_blocks=1,
        prior_prob=0.5,
        deep_supervision=True,
        diffusion_type='rectified_flow',
        box_init_mode='spatial_prior',
        use_adaln_zero=False,
        share_heads=False,
        criterion=criterion,
        roi_extractor={
            'roi_layer': {'output_size': (7, 7), 'sampling_ratio': 2},
            'featmap_strides': [4, 8, 16, 32],
        },
    )

    fpn = create_mock_fpn_features(batch_size=bs, feat_channels=C)
    img_metas = create_mock_image_metas(batch_size=bs)
    gt_bboxes, gt_labels = create_mock_gt(batch_size=bs, num_gt_per_image=3)

    # 正向: 计算 loss
    losses = head.loss(
        features=tuple(fpn),
        img_metas=img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
    )

    assert isinstance(losses, dict), f"Expected dict, got {type(losses)}"
    assert len(losses) > 0, "Losses dict is empty"

    print(f"  Loss keys: {list(losses.keys())}")
    for k, v in losses.items():
        print(f"    {k}: {v.item():.4f}")
        assert not torch.isnan(v).any(), f"NaN in {k}"
        assert not torch.isinf(v).any(), f"Inf in {k}"

    # 反向: 验证梯度可传播
    total_loss = sum(losses.values())
    total_loss.backward()

    has_grad = False
    for name, param in head.named_parameters():
        if param.grad is not None:
            has_grad = True
            assert not torch.isnan(param.grad).any(), (
                f"NaN grad in {name}"
            )
    assert has_grad, "No parameters received gradients!"
    print(f"  PASS: backward successful, all grads clean")


# ============================================================
# Test 5: 无 roi_extractor 降级路径
# ============================================================
def test_dit_head_without_roi():
    """测试无 roi_extractor 时的降级路径 (向后兼容)"""
    print("\n=== Test 5: DiTDiffusionDetHead WITHOUT roi_extractor ===")
    bs, N, C = 2, 100, 384
    num_classes = 24

    head = DiTDiffusionDetHead(
        num_classes=num_classes,
        feat_channels=C,
        num_proposals=N,
        num_heads=2,
        num_blocks=1,
        prior_prob=0.5,
        deep_supervision=True,
        box_init_mode='spatial_prior',
        use_adaln_zero=False,
        share_heads=False,
        roi_extractor=None,  # 无 RoI Extractor
    )

    fpn = create_mock_fpn_features(batch_size=bs, feat_channels=C)
    img_metas = create_mock_image_metas(batch_size=bs)
    bboxes = torch.rand(bs, N, 4) * 800
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]].sort(dim=-1)[0]
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]].sort(dim=-1)[0]
    t = torch.full((bs,), 500.0)

    all_cls_logits, all_pred_bboxes, _, _, _ = head(
        features=tuple(fpn), bboxes=bboxes, t=t, img_metas=img_metas
    )
    assert all_cls_logits.shape == (2, bs, N, num_classes)
    assert all_pred_bboxes.shape == (2, bs, N, 4)
    print(f"  PASS: fallback path works, cls_logits={tuple(all_cls_logits.shape)}")


# ============================================================
# Test 6: prior_prob 值验证
# ============================================================
def test_prior_prob_effect():
    """验证 prior_prob=0.01 vs 0.5 对分类 logits 初始值的影响"""
    print("\n=== Test 6: prior_prob effect ===")
    bs, N, C = 2, 100, 384
    num_classes = 24

    def get_max_sigmoid(prior_prob):
        head = DiTDiffusionDetHead(
            num_classes=num_classes,
            feat_channels=C,
            num_proposals=N,
            num_heads=1,
            num_blocks=1,
            prior_prob=prior_prob,
            deep_supervision=False,
            box_init_mode='spatial_prior',
            use_adaln_zero=False,
            share_heads=False,
            roi_extractor={
                'roi_layer': {'output_size': (7, 7), 'sampling_ratio': 2},
                'featmap_strides': [4, 8, 16, 32],
            },
        )
        fpn = create_mock_fpn_features(batch_size=bs, feat_channels=C)
        img_metas = create_mock_image_metas(batch_size=bs)
        bboxes = torch.rand(bs, N, 4) * 800
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]].sort(dim=-1)[0]
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]].sort(dim=-1)[0]
        t = torch.full((bs,), 500.0)
        cls_logits, _, _, _, _ = head(
            features=tuple(fpn), bboxes=bboxes, t=t, img_metas=img_metas
        )
        return torch.sigmoid(cls_logits[0]).max().item()

    # Fix randomness
    torch.manual_seed(42)

    max_001 = get_max_sigmoid(0.01)
    max_05 = get_max_sigmoid(0.5)

    print(f"  prior_prob=0.01 → max sigmoid = {max_001:.4f}")
    print(f"  prior_prob=0.5  → max sigmoid = {max_05:.4f}")
    assert max_05 > max_001 * 1.5, (
        f"prior_prob=0.5 should give higher initial sigmoid than 0.01 "
        f"({max_05:.4f} vs {max_001:.4f})"
    )
    print("  PASS: prior_prob=0.5 gives higher initial sigmoid scores")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("RoI Content Injection — Red-Green Tests")
    print("=" * 70)

    all_passed = True
    tests = [
        test_roi_extractor,
        test_box_tokenizer_with_roi,
        test_dit_head_forward,
        test_dit_head_loss,
        test_dit_head_without_roi,
        test_prior_prob_effect,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"\n  FAIL: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)