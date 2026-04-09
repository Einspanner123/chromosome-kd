"""
FlowDet 全组件测试

验证所有新增组件的基本功能:
- Phase 1: AdaLN-Zero 条件化
- Phase 2: Sinkhorn OT 匹配器
- Phase 3A: 结构化噪声采样
- Phase 3B: Objectness 预测头
- Phase 4: Velocity 预测模式
- Phase 5: Reflow / 一致性蒸馏
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/home/linkst/workplace/chromo/chromosome-kd")


def test_adaln_zero():
    """Phase 1: AdaLN-Zero 零初始化 → 初始输出应接近恒等映射"""
    from projects.LDMDet.mods.single_head import SingleDiffusionDetHead
    from projects.LDMDet.mods.modules import DynamicConv

    head = SingleDiffusionDetHead(
        num_classes=24,
        feat_channels=256,
        time_conditioning="adaln_zero",
        use_objectness=False,
        prediction_mode="x0",
    )

    # 验证 adaln_mlp 最后一层权重为零
    assert head.adaln_mlp is not None, "adaln_mlp should exist"
    last_layer = head.adaln_mlp[-1]
    assert (last_layer.weight == 0).all(), "AdaLN last layer weight should be zero"
    assert (last_layer.bias == 0).all(), "AdaLN last layer bias should be zero"

    print("[PASS] test_adaln_zero: Zero-init verified")


def test_scale_shift_backward_compat():
    """Phase 1: scale-shift 模式应向后兼容"""
    from projects.LDMDet.mods.single_head import SingleDiffusionDetHead

    head = SingleDiffusionDetHead(
        num_classes=24,
        feat_channels=256,
        time_conditioning="scale_shift",
    )
    assert hasattr(head, "time_mlp"), "scale_shift mode should have time_mlp"
    assert not hasattr(head, "adaln_mlp") or head.time_conditioning != "adaln_zero"
    print("[PASS] test_scale_shift_backward_compat")


def test_sinkhorn_matcher():
    """Phase 2: Sinkhorn OT 匹配器基本功能"""
    from projects.LDMDet.mods.sinkhorn import SinkhornOTMatcher
    from projects.LDMDet.mods.structures import InstanceData, ModelOutput

    matcher = SinkhornOTMatcher(epsilon=0.05, num_iters=50, dustbin_cost=1.0)

    # 模拟数据
    bs, N, C = 2, 100, 24
    pred_logits = torch.randn(bs, N, C)
    pred_bboxes = torch.rand(bs, N, 4)
    # 确保 xyxy 格式有效
    pred_bboxes[:, :, 2:] = pred_bboxes[:, :, :2] + pred_bboxes[:, :, 2:].abs() + 0.01

    targets = [
        InstanceData(
            bboxes=torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]]),
            labels=torch.tensor([0, 1]),
            img_shape=(800, 800),
        ),
        InstanceData(
            bboxes=torch.tensor([[0.2, 0.2, 0.4, 0.6]]),
            labels=torch.tensor([2]),
            img_shape=(800, 800),
        ),
    ]

    outputs = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_bboxes)
    indices = matcher(outputs, targets)

    assert len(indices) == bs
    for src_idx, gt_idx in indices:
        assert src_idx.dtype == torch.long
        assert gt_idx.dtype == torch.long
        assert len(src_idx) == len(gt_idx)
        assert len(src_idx) > 0, "Should match at least some proposals"

    print(f"[PASS] test_sinkhorn_matcher: matched {[len(i[0]) for i in indices]} proposals")


def test_sinkhorn_empty_gt():
    """Phase 2: Sinkhorn 处理空 GT 的情况"""
    from projects.LDMDet.mods.sinkhorn import SinkhornOTMatcher
    from projects.LDMDet.mods.structures import InstanceData, ModelOutput

    matcher = SinkhornOTMatcher(epsilon=0.05, num_iters=20)
    pred_logits = torch.randn(1, 50, 24)
    pred_bboxes = torch.rand(1, 50, 4)
    pred_bboxes[:, :, 2:] = pred_bboxes[:, :, :2] + pred_bboxes[:, :, 2:].abs() + 0.01

    targets = [
        InstanceData(
            bboxes=torch.zeros(0, 4),
            labels=torch.zeros(0, dtype=torch.long),
            img_shape=(800, 800),
        ),
    ]
    outputs = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_bboxes)
    indices = matcher(outputs, targets)
    assert len(indices[0][0]) == 0
    print("[PASS] test_sinkhorn_empty_gt")


def test_structured_noise():
    """Phase 3A: 结构化噪声采样"""
    from projects.LDMDet.mods.noise_sampler import StructuredNoiseSampler

    for strategy in ["pure", "grid", "grid_multiscale"]:
        sampler = StructuredNoiseSampler(
            num_proposals=200, noise_scale=1.0, strategy=strategy, snr_scale=2.0
        )
        noise = sampler.sample(batch_size=2, device=torch.device("cpu"))
        assert noise.shape == (2, 200, 4), f"{strategy}: shape mismatch {noise.shape}"
        assert torch.isfinite(noise).all(), f"{strategy}: non-finite values"

    print("[PASS] test_structured_noise: all strategies work")


def test_objectness_head():
    """Phase 3B: Objectness 预测头"""
    from projects.LDMDet.mods.single_head import SingleDiffusionDetHead

    head = SingleDiffusionDetHead(
        num_classes=24,
        feat_channels=256,
        time_conditioning="adaln_zero",
        use_objectness=True,
    )
    assert head.objectness_head is not None
    print("[PASS] test_objectness_head: head exists")


def test_velocity_head():
    """Phase 4: Velocity 预测头"""
    from projects.LDMDet.mods.single_head import SingleDiffusionDetHead

    head = SingleDiffusionDetHead(
        num_classes=24,
        feat_channels=256,
        time_conditioning="adaln_zero",
        prediction_mode="velocity",
    )
    assert head.velocity_head is not None
    print("[PASS] test_velocity_head: head exists")


def test_velocity_loss():
    """Phase 4: Flow Matching Velocity Loss"""
    from projects.LDMDet.mods.loss import FlowMatchingVelocityLoss

    loss_fn = FlowMatchingVelocityLoss(loss_weight=5.0)

    v_pred = torch.randn(4, 100, 4)
    v_target = torch.randn(4, 100, 4)
    fg_mask = torch.rand(4, 100) > 0.5

    loss = loss_fn(v_pred, v_target, fg_mask)
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"

    # 测试空 mask
    empty_mask = torch.zeros(4, 100, dtype=torch.bool)
    loss_empty = loss_fn(v_pred, v_target, empty_mask)
    assert loss_empty.item() == 0, "Empty mask should give zero loss"

    print("[PASS] test_velocity_loss")


def test_objectness_loss():
    """Phase 3B: Objectness Loss 在 Criterion 中"""
    from projects.LDMDet.mods.loss import (
        DiffusionDetCriterion,
        DiffusionDetMatcher,
        FocalLoss,
        GIoULoss,
        L1Loss,
    )
    from projects.LDMDet.mods.structures import InstanceData, ModelOutput

    matcher = DiffusionDetMatcher()
    criterion = DiffusionDetCriterion(
        num_classes=24,
        matcher=matcher,
        loss_cls=FocalLoss(),
        loss_bbox=L1Loss(),
        loss_giou=GIoULoss(),
        loss_objectness_weight=1.0,
    )

    N = 50
    pred_logits = torch.randn(1, N, 24)
    pred_bboxes = torch.rand(1, N, 4)
    pred_bboxes[:, :, 2:] = pred_bboxes[:, :, :2] + pred_bboxes[:, :, 2:].abs() + 0.01
    pred_obj = torch.randn(1, N, 1)

    outputs = ModelOutput(
        pred_logits=pred_logits,
        pred_boxes=pred_bboxes,
        pred_objectness=pred_obj,
    )
    targets = [
        InstanceData(
            bboxes=torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]]),
            labels=torch.tensor([0, 1]),
            img_shape=(800, 800),
        ),
    ]

    losses = criterion(outputs, targets)
    assert "loss_objectness" in losses, "Objectness loss should be present"
    assert losses["loss_objectness"].item() > 0
    print(f"[PASS] test_objectness_loss: {losses['loss_objectness'].item():.4f}")


def test_reflow():
    """Phase 5A: Reflow 基本结构"""
    from projects.LDMDet.mods.reflow import DetectionReflow

    # 只验证类可以实例化和方法存在
    assert hasattr(DetectionReflow, "generate_pairs")
    assert hasattr(DetectionReflow, "reflow_loss")
    assert hasattr(DetectionReflow, "save_pairs")
    assert hasattr(DetectionReflow, "load_pairs")
    print("[PASS] test_reflow: API complete")


def test_consistency():
    """Phase 5B: 一致性蒸馏基本结构"""
    from projects.LDMDet.mods.consistency import ConsistencyDetDistillation

    assert hasattr(ConsistencyDetDistillation, "update_ema")
    assert hasattr(ConsistencyDetDistillation, "consistency_loss")
    assert hasattr(ConsistencyDetDistillation, "train_step")
    print("[PASS] test_consistency: API complete")


def test_sinkhorn_convergence():
    """Phase 2: 验证 Sinkhorn 收敛到有效分配"""
    from projects.LDMDet.mods.sinkhorn import SinkhornOTMatcher

    matcher = SinkhornOTMatcher(epsilon=0.1, num_iters=100)

    N, K = 50, 5
    cost = torch.rand(N, K + 1)  # +1 for dustbin
    a = torch.ones(N) / N
    proposals_per_gt = N // K
    gt_mass = torch.full((K,), proposals_per_gt / N)
    dustbin_mass = max(1.0 - gt_mass.sum().item(), 0.01)
    b = torch.cat([gt_mass, torch.tensor([dustbin_mass])])
    b = b / b.sum()

    plan = matcher.sinkhorn_log_domain(cost, a, b)

    # 检查行和近似等于 a
    row_sums = plan.sum(dim=1)
    row_err = (row_sums - a).abs().max().item()

    # 检查列和近似等于 b
    col_sums = plan.sum(dim=0)
    col_err = (col_sums - b).abs().max().item()

    assert row_err < 0.01, f"Row constraint violated: max err = {row_err}"
    assert col_err < 0.01, f"Col constraint violated: max err = {col_err}"
    assert (plan >= 0).all(), "Plan should be non-negative"

    print(f"[PASS] test_sinkhorn_convergence: row_err={row_err:.6f}, col_err={col_err:.6f}")


if __name__ == "__main__":
    tests = [
        test_adaln_zero,
        test_scale_shift_backward_compat,
        test_sinkhorn_matcher,
        test_sinkhorn_empty_gt,
        test_sinkhorn_convergence,
        test_structured_noise,
        test_objectness_head,
        test_velocity_head,
        test_velocity_loss,
        test_objectness_loss,
        test_reflow,
        test_consistency,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*60}")
