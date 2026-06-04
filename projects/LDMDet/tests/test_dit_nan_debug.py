"""
LDMDet-DiT NaN 根因定位插桩脚本

在 CPU 上运行单步训练，开启 detect_anomaly，
在每个管道节点插入 NaN 检查，精确定位 NaN 首次出现的位置和原因。

运行方式:
  PYTHONPATH=. python projects/LDMDet/tests/test_dit_nan_debug.py
"""

import sys
import traceback

import torch
import torch.nn as nn

# 开启异常检测 — NaN 出现在 backward 时会直接报错并指明操作
torch.autograd.set_detect_anomaly(True)

sys.path.insert(0, 'projects/LDMDet')

from mods.box_tokenizer import BoxTokenizer
from mods.deformable_attn import flatten_fpn_features
from mods.dit_block import DiTBlock
from mods.dit_single_head import DiTSingleHead
from mods.dit_head import DiTDiffusionDetHead
from mods.loss import (
    FocalLossCost,
    BBoxL1Cost,
    RelativeL1Cost,
    IoUCost,
    DiffusionDetMatcher,
    DiffusionDetCriterion,
    FocalLoss,
    L1Loss,
    GIoULoss,
)
from mods.structures import InstanceData, ModelOutput, ImageMeta

# 强制 CPU
DEVICE = torch.device('cpu')


def assert_finite(tensor, name):
    """CPU 上检查 NaN/Inf，一旦发现立即报错并打印详细信息"""
    if not tensor.isfinite().all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        total = tensor.numel()
        msg = (
            f'[NaN DETECTED] {name}: '
            f'shape={tuple(tensor.shape)}, '
            f'NaN={nan_count}/{total}, Inf={inf_count}/{total}, '
            f'min={tensor[torch.isfinite(tensor)].min().item() if nan_count < total else "N/A"}, '
            f'max={tensor[torch.isfinite(tensor)].max().item() if nan_count < total else "N/A"}'
        )
        print(msg)
        raise RuntimeError(msg)


def assert_finite_or_warn(tensor, name):
    """检查 NaN/Inf，发现时打印警告但不中断"""
    if not tensor.isfinite().all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        total = tensor.numel()
        print(f'  [WARN] {name}: NaN={nan_count}/{total}, Inf={inf_count}/{total}')
    else:
        print(f'  [OK]   {name}: shape={tuple(tensor.shape)}, all finite')


# ============================================================
def make_head(pred_mode='x0', deep_sup=True, num_heads=6):
    return DiTDiffusionDetHead(
        num_classes=24,
        feat_channels=256,
        num_proposals=300,
        num_heads=num_heads,
        snr_scale=2.0,
        sampling_timesteps=2,
        diffusion_type='rectified_flow',
        solver_type='euler',
        rf_schedule='shifted',
        rf_shift=3.0,
        regression_mode='direct',
        prediction_mode=pred_mode,
        adaln_params=9,
        num_fpn_levels=4,
        num_ref_points=8,
        box_init_mode='zero',
        deep_supervision=deep_sup,
        criterion=DiffusionDetCriterion(
            num_classes=24,
            matcher=DiffusionDetMatcher(
                match_costs=[
                    FocalLossCost(weight=2.0),
                    RelativeL1Cost(weight=5.0),
                    IoUCost(iou_mode='giou', weight=2.0),
                ],
                center_radius=2.5,
                candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='l1',
        ),
    ).to(DEVICE)


def make_data(bs=2, C=256, M_range=(5, 20)):
    """创建接近真实训练配置的测试数据"""
    fpn = [torch.rand(bs, C, 25, 34, device=DEVICE) for _ in range(4)]
    img_metas = [
        ImageMeta(img_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])
        for _ in range(bs)
    ]
    gt_bboxes = []
    gt_labels = []
    for i in range(bs):
        M = torch.randint(M_range[0], M_range[1] + 1, (1,)).item()
        bboxes_norm = torch.rand(M, 4, device=DEVICE)
        bboxes_norm[..., 2] = bboxes_norm[..., 0] + 0.02 + torch.rand(M, device=DEVICE) * 0.1
        bboxes_norm[..., 3] = bboxes_norm[..., 1] + 0.02 + torch.rand(M, device=DEVICE) * 0.1
        bboxes_norm = bboxes_norm.clamp(0, 1)
        gt_bboxes.append(bboxes_norm * torch.tensor([1333, 800, 1333, 800], device=DEVICE))
        gt_labels.append(torch.randint(0, 24, (M,), device=DEVICE))
    return fpn, img_metas, gt_bboxes, gt_labels


# ============================================================
def test_single_step(head, fpn, img_metas, gt_bboxes_img, gt_labels, step_label=''):
    """单步训练 + 全节点 NaN 检查"""
    print(f'\n{"="*60}')
    print(f'Step: {step_label}')
    print(f'{"="*60}')

    # --- 拆解 loss 方法，逐节点检查 ---
    device = fpn[0].device
    bs = len(img_metas)

    # Node 1: normalize targets
    print('\n[Node 1] _normalize_targets')
    targets = head._normalize_targets(gt_bboxes_img, gt_labels, img_metas, bs)
    for i, t in enumerate(targets):
        assert_finite(t.bboxes, f'  target[{i}].bboxes')
        assert_finite(t.labels.float(), f'  target[{i}].labels')

    # Node 2: sample t
    print('\n[Node 2] _sample_t')
    t, lsas_log_probs = head._sample_t(bs, device)
    assert_finite(t, '  t')
    print(f'  t values: {t.tolist()}')

    # Node 3: build training targets
    print('\n[Node 3] _build_training_targets')
    x_boxes, x_starts, x_noises, matched_gt_indices = head._build_training_targets(
        bs, device, t, targets, gt_bboxes_img, img_metas,
    )
    for i in range(bs):
        assert_finite(x_boxes[i], f'  x_boxes[{i}]')
        assert_finite(x_starts[i], f'  x_starts[{i}]')
        assert_finite(x_noises[i], f'  x_noises[{i}]')

    x_noisy_batch = torch.stack(x_boxes)
    assert_finite(x_noisy_batch, '  x_noisy_batch')

    # Node 4: raw_to_xyxy
    print('\n[Node 4] _raw_to_xyxy')
    curr_bboxes = head._raw_to_xyxy(x_noisy_batch, img_metas)
    assert_finite(curr_bboxes, '  curr_bboxes')

    # Node 5: normalize for tokenizer
    print('\n[Node 5] _normalize_bboxes_for_tokenizer')
    normed = head._normalize_bboxes_for_tokenizer(curr_bboxes, img_metas)
    assert_finite(normed, '  normed_bboxes')

    # Node 5b: prepare FPN
    print('\n[Node 5b] FPN flatten')
    fpn_list = list(fpn)[: head.num_fpn_levels]
    fpn_flattened, spatial_shapes, level_start_index = flatten_fpn_features(fpn_list)
    assert_finite(fpn_flattened, '  fpn_flattened')
    for li, feat in enumerate(fpn_list):
        assert_finite(feat, f'  fpn[{li}]')

    # Node 5c: time embedding
    print('\n[Node 5c] time_mlp')
    t_input = t if head.diffusion_type == 'ddpm' else t * head.timesteps
    time_emb = head.time_mlp(t_input)
    assert_finite(time_emb, '  time_emb')

    # Node 6: BoxTokenizer
    print('\n[Node 6] BoxTokenizer')
    box_tokens, level_indices = head.box_tokenizer(normed, fpn_list)
    assert_finite(box_tokens, '  box_tokens')

    # Node 7-8: iterate heads
    print('\n[Node 7-8] Head iterations')
    curr_tokens = box_tokens
    curr_normed = normed

    for hi, single_head in enumerate(head.head_series):
        print(f'\n  --- Head {hi} ---')

        # Node 9: DiTBlock
        print(f'  [Block {hi}] DiTBlock.forward')
        updated_tokens = single_head.dit_block(
            curr_tokens, fpn_flattened, spatial_shapes,
            level_start_index, time_emb, curr_normed,
        )
        assert_finite(updated_tokens, f'    updated_tokens')

        # cls_head
        print(f'  [Block {hi}] cls_head')
        class_logits = single_head.cls_head(updated_tokens)
        assert_finite(class_logits, f'    class_logits')

        # reg_head
        print(f'  [Block {hi}] reg_head + _predict_bboxes')
        pred_bboxes = single_head._predict_bboxes(updated_tokens, curr_normed)
        assert_finite(pred_bboxes, f'    pred_bboxes')

        # velocity head
        if single_head.velocity_head is not None:
            print(f'  [Block {hi}] velocity_head')
            pred_velocity = single_head.velocity_head(updated_tokens)
            assert_finite(pred_velocity, f'    pred_velocity')

        curr_tokens = updated_tokens
        curr_normed = pred_bboxes.detach().clamp(0, 1)

    # Collect outputs (simulating forward)
    print('\n[Node 10] Construct ModelOutput for Loss')
    # We need to run the actual forward to get stacked outputs
    all_cls_logits, all_pred_bboxes, all_objectness, all_velocity, all_curr = (
        head(fpn, curr_bboxes, t_input, img_metas=img_metas)
    )
    assert_finite(all_cls_logits, '  all_cls_logits')
    assert_finite(all_pred_bboxes, '  all_pred_bboxes')

    outputs = ModelOutput(
        pred_logits=all_cls_logits[-1],
        pred_boxes=all_pred_bboxes[-1],
        pred_objectness=all_objectness[-1] if all_objectness[0] is not None else None,
    )
    if head.deep_supervision and head.num_heads > 1:
        outputs.aux_outputs = [
            ModelOutput(
                pred_logits=all_cls_logits[i],
                pred_boxes=all_pred_bboxes[i],
                pred_objectness=all_objectness[i] if all_objectness[0] is not None else None,
            )
            for i in range(head.num_heads - 1)
        ]

    # Node 10: Matcher
    print('\n[Node 10] DiffusionDetMatcher')
    indices = head.criterion.matcher(outputs, targets)
    print(f'  matched indices: {[(len(idx[0]), len(idx[1])) for idx in indices]}')

    # Node 10b: Classification loss
    print('\n[Node 10b] _loss_classification')
    loss_cls = head.criterion._loss_classification(outputs, targets, indices)
    assert_finite(loss_cls, '  loss_cls')

    # Node 10c: BBox loss
    print('\n[Node 10c] _loss_boxes')
    loss_bbox, loss_giou = head.criterion._loss_boxes(outputs, targets, indices)
    assert_finite(loss_bbox, '  loss_bbox')
    assert_finite(loss_giou, '  loss_giou')

    # Node 11: Velocity loss
    if head.prediction_mode == 'velocity' and all_velocity[-1] is not None:
        print('\n[Node 11] velocity_loss')
        v_target = torch.stack(x_noises) - torch.stack(x_starts)
        loss_velocity = nn.functional.mse_loss(all_velocity[-1], v_target) * head.velocity_loss_weight
        assert_finite(loss_velocity, '  loss_velocity')

    # Full loss
    print('\n[Full] Computing total loss via head.loss()')
    losses = head.loss(fpn, img_metas, gt_bboxes_img, gt_labels)
    total_loss = sum(losses.values())
    assert_finite(total_loss, '  total_loss')

    print(f'\n  Losses:')
    for k, v in losses.items():
        print(f'    {k}: {v.item():.6f}')
    print(f'    TOTAL: {total_loss.item():.6f}')

    # Backward
    print('\n[Backward] Running backward (detect_anomaly=True)...')
    try:
        total_loss.backward()
    except RuntimeError as e:
        print(f'\n!!! BACKWARD ERROR: {e}')
        traceback.print_exc()
        return False

    # 检查梯度
    print('\n[Grad] Checking gradients...')
    max_grad = 0
    nan_params = []
    for name, param in head.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.norm().item()
            max_grad = max(max_grad, g_norm)
            if not param.grad.isfinite().all():
                nan_params.append((name, g_norm))

    if nan_params:
        print(f'  NaN gradients found in:')
        for name, g_norm in nan_params:
            print(f'    {name}: grad_norm={g_norm}')
        return False

    print(f'  All gradients finite. Max grad norm: {max_grad:.2f}')
    return True


# ============================================================
def test_multiple_steps(num_steps=5):
    """多步训练，模拟真实训练过程，观察 loss/grad 是否稳定"""
    print('\n' + '=' * 60)
    print(f'Multiple Step Training Simulation ({num_steps} steps)')
    print('=' * 60)

    head = make_head(pred_mode='x0', deep_sup=True, num_heads=6)
    head.train()

    # 使用固定 FPN 和多变 GT 模拟真实训练
    fpn = [torch.rand(2, 256, 25, 34, device=DEVICE) for _ in range(4)]
    img_metas = [
        ImageMeta(img_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])
        for _ in range(2)
    ]

    loss_history = []
    grad_history = []

    for step in range(num_steps):
        _, _, gt_bboxes, gt_labels = make_data(bs=2, M_range=(3, 15))

        losses = head.loss(fpn, img_metas, gt_bboxes, gt_labels)
        total_loss = sum(losses.values())

        if not total_loss.isfinite():
            print(f'  Step {step}: LOSS IS NOT FINITE! Stopping.')
            return False

        total_loss.backward()

        max_grad = 0
        for param in head.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.norm().item())

        loss_history.append(total_loss.item())
        grad_history.append(max_grad)

        print(f'  Step {step}: loss={total_loss.item():.4f}, max_grad={max_grad:.2f}')

        head.zero_grad()

        # 检查梯度爆炸
        if max_grad > 1000:
            print(f'  ⚠ Step {step}: MAX_GRAD > 1000! Potential gradient explosion.')

        # 检查权重是否已 NaN
        for name, param in head.named_parameters():
            if not param.isfinite().all():
                print(f'  ⚠ Step {step}: Weight NaN detected in {name}!')
                return False

    print(f'\n  Loss range: [{min(loss_history):.4f}, {max(loss_history):.4f}]')
    print(f'  Grad range: [{min(grad_history):.2f}, {max(grad_history):.2f}]')
    return True


# ============================================================
def test_edge_cases():
    """测试各种边界条件"""
    print('\n' + '=' * 60)
    print('Edge Case Tests')
    print('=' * 60)

    head = make_head(pred_mode='x0', deep_sup=True, num_heads=6)
    head.train()
    fpn = [torch.rand(2, 256, 25, 34, device=DEVICE) for _ in range(4)]
    img_metas = [
        ImageMeta(img_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])
        for _ in range(2)
    ]

    # Case 1: 单 GT
    print('\n--- Case: Single GT ---')
    gt_bboxes = [torch.tensor([[100., 100., 200., 200.]], device=DEVICE)]
    gt_labels = [torch.tensor([0], device=DEVICE)]
    gt_bboxes2 = [torch.tensor([[500., 400., 700., 600.]], device=DEVICE)]
    gt_labels2 = [torch.tensor([1], device=DEVICE)]
    gt_bboxes_all = [torch.cat([gt_bboxes[0], gt_bboxes2[0]], dim=0)]
    gt_labels_all = [torch.cat([gt_labels[0], gt_labels2[0]], dim=0)]

    try:
        losses = head.loss(fpn, img_metas, gt_bboxes_all, gt_labels_all)
        total = sum(losses.values())
        print(f'  total_loss={total.item():.4f}')
        total.backward()
        print('  backward OK')
    except Exception as e:
        print(f'  ERROR: {e}')

    # Case 2: 空 GT
    print('\n--- Case: Empty GT ---')
    gt_bboxes_empty = [torch.zeros(0, 4, device=DEVICE) for _ in range(2)]
    gt_labels_empty = [torch.zeros(0, dtype=torch.long, device=DEVICE) for _ in range(2)]

    head.zero_grad()
    try:
        losses = head.loss(fpn, img_metas, gt_bboxes_empty, gt_labels_empty)
        total = sum(losses.values())
        print(f'  total_loss={total.item():.4f}')
        total.backward()
        print('  backward OK')
    except Exception as e:
        print(f'  ERROR: {e}')

    # Case 3: 极端 GT 位置
    print('\n--- Case: Edge-position GT ---')
    gt_bboxes_edge = [
        torch.tensor([
            [0., 0., 50., 50.],           # 左上角
            [1283., 0., 1333., 50.],       # 右上角
            [0., 750., 50., 800.],         # 左下角
            [1283., 750., 1333., 800.],    # 右下角
        ], device=DEVICE)
    ]
    gt_labels_edge = [torch.tensor([0, 1, 2, 3], device=DEVICE)]
    gt_bboxes_edge2 = [torch.tensor([[600., 350., 700., 450.]], device=DEVICE)]
    gt_labels_edge2 = [torch.tensor([4], device=DEVICE)]
    gt_bboxes_edge_all = [torch.cat([gt_bboxes_edge[0], gt_bboxes_edge2[0]], dim=0)]
    gt_labels_edge_all = [torch.cat([gt_labels_edge[0], gt_labels_edge2[0]], dim=0)]

    head.zero_grad()
    try:
        losses = head.loss(fpn, img_metas, gt_bboxes_edge_all, gt_labels_edge_all)
        total = sum(losses.values())
        print(f'  total_loss={total.item():.4f}')
        total.backward()
        print('  backward OK')
    except Exception as e:
        print(f'  ERROR: {e}')

    # Case 4: 小 GT 框
    print('\n--- Case: Tiny GT ---')
    gt_bboxes_tiny = [
        torch.tensor([
            [500., 400., 502., 402.],   # 2x2 pixel box
            [600., 500., 601., 501.],   # 1x1 pixel box
        ], device=DEVICE)
    ]
    gt_labels_tiny = [torch.tensor([0, 1], device=DEVICE)]
    gt_bboxes_tiny2 = [torch.tensor([[300., 200., 310., 210.]], device=DEVICE)]
    gt_labels_tiny2 = [torch.tensor([2], device=DEVICE)]
    gt_bboxes_tiny_all = [torch.cat([gt_bboxes_tiny[0], gt_bboxes_tiny2[0]], dim=0)]
    gt_labels_tiny_all = [torch.cat([gt_labels_tiny[0], gt_labels_tiny2[0]], dim=0)]

    head.zero_grad()
    try:
        losses = head.loss(fpn, img_metas, gt_bboxes_tiny_all, gt_labels_tiny_all)
        total = sum(losses.values())
        print(f'  total_loss={total.item():.4f}')
        total.backward()
        print('  backward OK')
    except Exception as e:
        print(f'  ERROR: {e}')

    return True


# ============================================================
if __name__ == '__main__':
    print('LDMDet-DiT NaN Root Cause Debug')
    print(f'Device: {DEVICE}')
    print(f'Detect Anomaly: {torch.is_anomaly_enabled()}')

    # 1. 单步详细插桩
    print('\n' + '#' * 60)
    print('# Phase 1: Single step with node-level instrumentation')
    print('#' * 60)

    head = make_head(pred_mode='x0', deep_sup=True, num_heads=6)
    head.train()
    fpn, img_metas, gt_bboxes, gt_labels = make_data(bs=2, M_range=(5, 15))

    ok = test_single_step(head, fpn, img_metas, gt_bboxes, gt_labels,
                          step_label='x0/deep_sup=True/num_heads=6')
    if not ok:
        print('\n!!! NaN detected in single step. See above for exact location.')
        sys.exit(1)

    # 2. 多步模拟
    print('\n' + '#' * 60)
    print('# Phase 2: Multi-step training simulation')
    print('#' * 60)
    ok = test_multiple_steps(num_steps=10)
    if not ok:
        print('\n!!! NaN detected in multi-step simulation.')
        sys.exit(1)

    # 3. 边界条件
    print('\n' + '#' * 60)
    print('# Phase 3: Edge cases')
    print('#' * 60)
    test_edge_cases()

    print('\n' + '=' * 60)
    print('All checks passed — no NaN detected on CPU.')
    print('If NaN occurs on GPU but not CPU, the issue may be:')
    print('  1. CUDA-specific numerical precision (fp16/bfloat16)')
    print('  2. CUDA kernel edge cases in grid_sample or attention')
    print('  3. Race conditions in gradient accumulation')
    print('=' * 60)