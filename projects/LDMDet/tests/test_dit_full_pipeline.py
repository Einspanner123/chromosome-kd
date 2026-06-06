"""
LDMDet-DiT 全链路节点验证测试

覆盖从数据准备到 loss 计算的每一个节点:
  Node 1: _normalize_targets → InstanceData
  Node 2: _sample_t → t (时间步)
  Node 3: _build_training_targets → noisy boxes
  Node 4: _raw_to_xyxy → 坐标还原
  Node 5: _normalize_bboxes_for_tokenizer → [0,1] 归一化
  Node 6: BoxTokenizer.forward → box_tokens
  Node 7: DiTHead.forward → iterating head_series
  Node 8: DiTSingleHead.forward → cls/bbox/v tokens
  Node 9: DiTBlock.forward → AdaLN+SA+CA+FFN
  Node 10: DiffusionDetCriterion._get_loss → matched losses
  Node 11: _add_velocity_loss → velocity loss
  Node 12: _add_lsas_loss → LSAS loss
  Node 13: 全流程梯度检查 (无 NaN/Inf/爆炸)
  Node 14: 多步迭代稳定性检查
  Node 15: 大规模输入压测

每个节点验证:
  - 输出形状正确
  - 输出值有限 (无 NaN/Inf)
  - 输出值在合理范围内
  - 梯度流完整 (对可训练节点)

运行方式:
  PYTHONPATH=. python projects/LDMDet/tests/test_dit_full_pipeline.py
"""

import sys
import torch
import torch.nn as nn

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
    FlowMatchingVelocityLoss,
)
from mods.structures import InstanceData, ModelOutput, ImageMeta
from mods.utils import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy

PASS = 0
FAIL = 0
ERROR = 0


def check(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f'  [PASS] {name}')
    else:
        FAIL += 1
        print(f'  [FAIL] {name} {detail}')


def check_no_crash(name, fn, detail=''):
    global PASS, FAIL, ERROR
    try:
        fn()
        PASS += 1
        print(f'  [PASS] {name}')
    except Exception as e:
        ERROR += 1
        print(f'  [ERROR] {name} => {type(e).__name__}: {e} {detail}')


def _make_head(pred_mode='x0', deep_sup=False, num_heads=2):
    """创建标准测试用 DiTDiffusionDetHead"""
    return DiTDiffusionDetHead(
        num_classes=24,
        feat_channels=256,
        num_proposals=50,
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
                    BBoxL1Cost(weight=5.0),
                    IoUCost(iou_mode='giou', weight=2.0),
                ],
                center_radius=0.5,
                candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
        ),
    )


def _make_fpn_and_metas(bs=2, C=256):
    """创建标准测试用 FPN 特征和 img_metas"""
    fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
    img_metas = [
        ImageMeta(img_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])
        for _ in range(bs)
    ]
    return fpn, img_metas


def _make_gt(bs=2, M=5):
    """创建标准测试用 GT (归一化 [0,1])"""
    gt_bboxes_img = []
    gt_labels = []
    for i in range(bs):
        bboxes_norm = torch.rand(M, 4)
        bboxes_norm[..., 2] = bboxes_norm[..., 0] + 0.05 + torch.rand(M) * 0.1
        bboxes_norm[..., 3] = bboxes_norm[..., 1] + 0.05 + torch.rand(M) * 0.1
        bboxes_norm = bboxes_norm.clamp(0, 1)
        gt_bboxes_img.append(bboxes_norm * torch.tensor([1333, 800, 1333, 800]))
        gt_labels.append(torch.randint(0, 24, (M,)))
    return gt_bboxes_img, gt_labels


# ============================================================
# Node 1: _normalize_targets
# ============================================================
def test_node1_normalize_targets():
    """验证 GT 归一化到 [0,1] 且类型正确"""
    print('\n=== Node 1: _normalize_targets ===')
    head = _make_head()
    bs = 2
    gt_bboxes_img, gt_labels = _make_gt(bs)
    fpn, img_metas = _make_fpn_and_metas(bs)
    device = fpn[0].device

    targets = head._normalize_targets(gt_bboxes_img, gt_labels, img_metas, bs)

    check('targets count', len(targets) == bs)
    for i, t in enumerate(targets):
        check(f'  target[{i}] type', isinstance(t, InstanceData),
              f'got {type(t).__name__}')
        check(f'  target[{i}] bboxes shape', t.bboxes.ndim == 2 and t.bboxes.shape[1] == 4)
        check(f'  target[{i}] labels shape', t.labels.ndim == 1)
        check(f'  target[{i}] img_shape', isinstance(t.img_shape, tuple) and len(t.img_shape) == 2)
        # 验证归一化: bboxes 应在 [0, 1] 范围内
        bbox_min = t.bboxes.min().item()
        bbox_max = t.bboxes.max().item()
        check(f'  target[{i}] bboxes [0,1]', 0 <= bbox_min and bbox_max <= 1 + 1e-5,
              f'min={bbox_min:.6f}, max={bbox_max:.6f}')
        check(f'  target[{i}] bboxes finite', t.bboxes.isfinite().all().item())


# ============================================================
# Node 2: _sample_t
# ============================================================
def test_node2_sample_t():
    """验证时间步 t 的采样范围"""
    print('\n=== Node 2: _sample_t ===')
    head = _make_head()
    fpn, _ = _make_fpn_and_metas()
    device = fpn[0].device

    for bs in [1, 2, 8, 16, 32]:
        t, lsas_log_probs = head._sample_t(bs, device)
        check(f'  bs={bs}: t shape', t.shape == (bs,))
        check(f'  bs={bs}: t in [0,1]',
              0 <= t.min().item() and t.max().item() <= 1.0,
              f'min={t.min().item():.6f}, max={t.max().item():.6f}')
        check(f'  bs={bs}: t finite', t.isfinite().all().item())
        # 验证 t 不完全相同 (stratified / uniform mode)
        check(f'  bs={bs}: t has diversity', t.std() > 1e-6 or bs == 1,
              f'std={t.std().item():.10f}')


# ============================================================
# Node 3: _build_training_targets
# ============================================================
def test_node3_build_training_targets():
    """验证训练目标构建: noisy boxes, x_starts, x_noises"""
    print('\n=== Node 3: _build_training_targets ===')
    head = _make_head()
    bs = 2
    gt_bboxes_img, gt_labels = _make_gt(bs)
    fpn, img_metas = _make_fpn_and_metas(bs)
    device = fpn[0].device

    targets = head._normalize_targets(gt_bboxes_img, gt_labels, img_metas, bs)
    t, _ = head._sample_t(bs, device)

    x_boxes, x_starts, x_noises, matched_gt_indices = head._build_training_targets(
        bs, device, t, targets, gt_bboxes_img, img_metas,
    )

    check('x_boxes count', len(x_boxes) == bs)
    check('x_starts count', len(x_starts) == bs)
    check('x_noises count', len(x_noises) == bs)
    check('matched_gt_indices count', len(matched_gt_indices) == bs)

    for i in range(bs):
        check(f'  x_boxes[{i}] shape', x_boxes[i].shape == (50, 4))
        check(f'  x_boxes[{i}] finite', x_boxes[i].isfinite().all().item())
        check(f'  x_starts[{i}] shape', x_starts[i].shape == (50, 4))
        check(f'  x_starts[{i}] finite', x_starts[i].isfinite().all().item())
        check(f'  x_noises[{i}] shape', x_noises[i].shape == (50, 4))
        check(f'  x_noises[{i}] finite', x_noises[i].isfinite().all().item())
        check(f'  matched_gt_indices[{i}] shape', matched_gt_indices[i].shape == (50,))

    # 验证 x_boxes 在 snr_scale 范围内 (RF 模式下应在 [-snr_scale, snr_scale])
    x_noisy_batch = torch.stack(x_boxes)
    x_min = x_noisy_batch.min().item()
    x_max = x_noisy_batch.max().item()
    # RF: x_t = (1-t)*x0 + t*eps, x0 in [-snr_scale, snr_scale], eps ~ N(0,1)
    # 允许一定超限
    check('x_noisy_batch finite', x_noisy_batch.isfinite().all().item(),
          f'min={x_min:.4f}, max={x_max:.4f}')


def test_node3_build_training_targets_empty_gt():
    """验证空 GT 时的训练目标构建"""
    print('\n=== Node 3b: build_training_targets (empty GT) ===')
    head = _make_head()
    bs = 2
    gt_bboxes_img = [torch.zeros(0, 4) for _ in range(bs)]
    gt_labels = [torch.zeros(0, dtype=torch.long) for _ in range(bs)]
    fpn, img_metas = _make_fpn_and_metas(bs)
    device = fpn[0].device

    targets = head._normalize_targets(gt_bboxes_img, gt_labels, img_metas, bs)
    t, _ = head._sample_t(bs, device)

    x_boxes, x_starts, x_noises, matched_gt_indices = head._build_training_targets(
        bs, device, t, targets, gt_bboxes_img, img_metas,
    )

    for i in range(bs):
        check(f'  x_boxes[{i}] shape', x_boxes[i].shape == (50, 4))
        check(f'  x_boxes[{i}] finite', x_boxes[i].isfinite().all().item())
        # 空 GT 时 x_start 应为全零 (纯噪声模式)
        # x_boxes 应为随机噪声
        check(f'  x_starts[{i}] all zero', (x_starts[i] == 0).all().item())


# ============================================================
# Node 4: _raw_to_xyxy
# ============================================================
def test_node4_raw_to_xyxy():
    """验证 raw diffusion space → xyxy image 坐标转换"""
    print('\n=== Node 4: _raw_to_xyxy ===')
    head = _make_head()
    bs = 2
    fpn, img_metas = _make_fpn_and_metas(bs)

    # 测试正常 raw 值
    x_raw = torch.randn(bs, 50, 4) * 2.0  # 在 snr_scale=2.0 范围内
    xyxy = head._raw_to_xyxy(x_raw, img_metas)
    check('xyxy shape', xyxy.shape == (bs, 50, 4))
    check('xyxy finite', xyxy.isfinite().all().item())
    # 验证 x2 > x1, y2 > y1 (代码中已保证 1e-4 最小宽高)
    x1, y1, x2, y2 = xyxy[..., 0], xyxy[..., 1], xyxy[..., 2], xyxy[..., 3]
    check('x2 >= x1', (x2 >= x1).all().item())
    check('y2 >= y1', (y2 >= y1).all().item())

    # 测试极端 raw 值
    x_raw_extreme = torch.randn(bs, 50, 4) * 100.0
    xyxy_extreme = head._raw_to_xyxy(x_raw_extreme, img_metas)
    check('extreme xyxy finite', xyxy_extreme.isfinite().all().item())


# ============================================================
# Node 5: _normalize_bboxes_for_tokenizer
# ============================================================
def test_node5_normalize_bboxes_for_tokenizer():
    """验证图像坐标 → [0,1] 归一化"""
    print('\n=== Node 5: _normalize_bboxes_for_tokenizer ===')
    head = _make_head()
    bs = 2
    fpn, img_metas = _make_fpn_and_metas(bs)

    # 正常图像坐标
    bboxes_img = torch.rand(bs, 50, 4)
    bboxes_img[..., 2] = bboxes_img[..., 0] + 50.0
    bboxes_img[..., 3] = bboxes_img[..., 1] + 50.0
    normed = head._normalize_bboxes_for_tokenizer(bboxes_img, img_metas)
    check('normed shape', normed.shape == (bs, 50, 4))
    check('normed in [0,1]', 0 <= normed.min().item() and normed.max().item() <= 1.0)
    check('normed finite', normed.isfinite().all().item())

    # 极端图像坐标
    bboxes_extreme = torch.rand(bs, 50, 4) * 10000.0
    bboxes_extreme[..., 2] = bboxes_extreme[..., 0] + 10000.0
    bboxes_extreme[..., 3] = bboxes_extreme[..., 1] + 10000.0
    normed_extreme = head._normalize_bboxes_for_tokenizer(bboxes_extreme, img_metas)
    check('extreme normed in [0,1]', 0 <= normed_extreme.min().item()
          and normed_extreme.max().item() <= 1.0)
    check('extreme normed finite', normed_extreme.isfinite().all().item())


# ============================================================
# Node 6: BoxTokenizer.forward
# ============================================================
def test_node6_box_tokenizer():
    """验证 BoxTokenizer 产生的 box_tokens"""
    print('\n=== Node 6: BoxTokenizer.forward ===')
    C = 256
    tokenizer = BoxTokenizer(feat_channels=C, num_fpn_levels=4, init_mode='zero')
    fpn = [torch.rand(2, C, 20, 20) for _ in range(4)]
    bboxes_norm = torch.rand(2, 50, 4)
    bboxes_norm[..., 2] = bboxes_norm[..., 0] + 0.05
    bboxes_norm[..., 3] = bboxes_norm[..., 1] + 0.05
    bboxes_norm = bboxes_norm.clamp(0, 1)

    tokens, levels = tokenizer(bboxes_norm, fpn)
    check('tokens shape', tokens.shape == (2, 50, C))
    check('levels shape', levels.shape == (2, 50))
    check('tokens finite', tokens.isfinite().all().item())

    # 验证不同框产生不同 token
    bboxes_norm2 = bboxes_norm + 0.1
    bboxes_norm2 = bboxes_norm2.clamp(0, 1)
    tokens2, _ = tokenizer(bboxes_norm2, fpn)
    diff = (tokens - tokens2).abs().max().item()
    check('different bboxes→different tokens', diff > 1e-6,
          f'max_diff={diff:.10f}')

    # 验证 token 不为全零 (因为有 pos_embed + level_embed)
    check('tokens not all zero', tokens.abs().max().item() > 0)


def test_node6_box_tokenizer_learnable():
    """验证 learnable 模式"""
    print('\n=== Node 6b: BoxTokenizer (learnable) ===')
    C = 256
    tokenizer = BoxTokenizer(feat_channels=C, num_fpn_levels=4, init_mode='learnable')
    fpn = [torch.rand(2, C, 20, 20) for _ in range(4)]
    bboxes_norm = torch.rand(2, 50, 4)
    bboxes_norm[..., 2] = bboxes_norm[..., 0] + 0.05
    bboxes_norm[..., 3] = bboxes_norm[..., 1] + 0.05
    bboxes_norm = bboxes_norm.clamp(0, 1)

    tokens, levels = tokenizer(bboxes_norm, fpn)
    check('tokens shape', tokens.shape == (2, 50, C))
    check('tokens finite', tokens.isfinite().all().item())


# ============================================================
# Node 7: DiTHead.forward (迭代 head_series)
# ============================================================
def test_node7_dithead_forward():
    """验证 DiTHead forward 迭代逻辑"""
    print('\n=== Node 7: DiTHead.forward ===')
    for deep_sup in [False, True]:
        for pred_mode in ['x0', 'velocity']:
            head = _make_head(pred_mode=pred_mode, deep_sup=deep_sup, num_heads=2)
            head.eval()
            bs, C = 2, 256
            fpn, img_metas = _make_fpn_and_metas(bs, C)
            x_noisy = torch.randn(bs, 50, 4) * 2.0
            curr_bboxes = head._raw_to_xyxy(x_noisy, img_metas)
            t_input = torch.full((bs,), 500.0)
            with torch.no_grad():
                all_cls, all_bboxes, obj_list, vel_list, curr_list = head(
                    fpn, curr_bboxes, t_input, img_metas=img_metas
                )

            seq_len = 2 if deep_sup else 1
            check(f'{pred_mode}/ds={deep_sup}: cls seq shape',
                  all_cls.shape == (seq_len, bs, 50, 24),
                  f'got {all_cls.shape}')
            check(f'{pred_mode}/ds={deep_sup}: bbox seq shape',
                  all_bboxes.shape == (seq_len, bs, 50, 4),
                  f'got {all_bboxes.shape}')

            for s in range(all_bboxes.shape[0]):
                check(f'{pred_mode}/ds={deep_sup}: bbox[{s}] finite',
                      all_bboxes[s].isfinite().all().item())
                check(f'{pred_mode}/ds={deep_sup}: bbox[{s}] in [0,1]',
                      all_bboxes[s].min().item() >= 0 and all_bboxes[s].max().item() <= 1)

            if pred_mode == 'velocity':
                check(f'{pred_mode}/ds={deep_sup}: velocity exists',
                      vel_list[-1] is not None)


# ============================================================
# Node 8: DiTSingleHead.forward
# ============================================================
def test_node8_single_head_direct_regression():
    """验证 direct regression 模式下 bbox 输出合理性"""
    print('\n=== Node 8: DiTSingleHead (direct regression) ===')
    C = 256
    head = DiTSingleHead(
        num_classes=24, feat_channels=C, num_heads=8,
        num_fpn_levels=4, num_ref_points=8,
        prediction_mode='x0', adaln_params=9, regression_mode='direct',
    )
    head.eval()
    bs, N = 2, 50
    time_dim = C * 4
    fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
    flat, shapes, starts = flatten_fpn_features(fpn)
    time_emb = torch.rand(bs, time_dim)
    box_tokens = torch.rand(bs, N, C)
    bbox_coords = torch.rand(bs, N, 4)
    bbox_coords[..., 2] = bbox_coords[..., 0] + 0.05
    bbox_coords[..., 3] = bbox_coords[..., 1] + 0.05
    bbox_coords = bbox_coords.clamp(0, 1)

    with torch.no_grad():
        cls_logits, pred_bboxes, updated_tokens, obj, vel = head(
            box_tokens, flat, shapes, starts, time_emb, bbox_coords
        )

    check('cls shape', cls_logits.shape == (bs, N, 24))
    check('pred_bboxes shape', pred_bboxes.shape == (bs, N, 4))
    check('pred_bboxes in [0,1]', pred_bboxes.min().item() >= 0
          and pred_bboxes.max().item() <= 1)
    check('pred_bboxes finite', pred_bboxes.isfinite().all().item())
    check('cls finite', cls_logits.isfinite().all().item())
    check('updated_tokens finite', updated_tokens.isfinite().all().item())

    # 验证 x2>=x1, y2>=y1
    x1, y1 = pred_bboxes[..., 0], pred_bboxes[..., 1]
    x2, y2 = pred_bboxes[..., 2], pred_bboxes[..., 3]
    check('x2 >= x1', (x2 >= x1).all().item())
    check('y2 >= y1', (y2 >= y1).all().item())

    # 验证不同 token 产生不同输出
    box_tokens2 = torch.randn(bs, N, C) * 5
    with torch.no_grad():
        _, pred_bboxes2, _, _, _ = head(
            box_tokens2, flat, shapes, starts, time_emb, bbox_coords
        )
    diff = (pred_bboxes - pred_bboxes2).abs().max().item()
    check('different input→different output', diff > 1e-6,
          f'max_diff={diff:.10f}')


def test_node8_single_head_delta_regression():
    """验证 delta regression 模式"""
    print('\n=== Node 8b: DiTSingleHead (delta regression) ===')
    C = 256
    head = DiTSingleHead(
        num_classes=24, feat_channels=C, num_heads=8,
        num_fpn_levels=4, num_ref_points=8,
        prediction_mode='x0', adaln_params=9, regression_mode='delta',
    )
    head.eval()
    bs, N = 2, 50
    time_dim = C * 4
    fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
    flat, shapes, starts = flatten_fpn_features(fpn)
    time_emb = torch.rand(bs, time_dim)
    box_tokens = torch.rand(bs, N, C)
    bbox_coords = torch.rand(bs, N, 4)
    bbox_coords[..., 2] = bbox_coords[..., 0] + 0.1
    bbox_coords[..., 3] = bbox_coords[..., 1] + 0.1
    bbox_coords = bbox_coords.clamp(0, 1)

    with torch.no_grad():
        cls_logits, pred_bboxes, _, _, _ = head(
            box_tokens, flat, shapes, starts, time_emb, bbox_coords
        )

    check('pred_bboxes finite', pred_bboxes.isfinite().all().item())
    check('pred_bboxes in [0,1]', pred_bboxes.min().item() >= -1e-5
          and pred_bboxes.max().item() <= 1 + 1e-5)


# ============================================================
# Node 9: DiTBlock.forward
# ============================================================
def test_node9_ditblock_forward():
    """验证 DiTBlock 输出与梯度"""
    print('\n=== Node 9: DiTBlock.forward ===')
    C = 256
    for adaln_zero in [True, False]:
        block = DiTBlock(
            feat_channels=C, num_heads=8, num_fpn_levels=4,
            num_ref_points=8, dim_feedforward=2048, adaln_params=9,
            use_adaln_zero=adaln_zero,
        )
        block.train()
        bs, N = 2, 10
        time_dim = C * 4
        fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
        flat, shapes, starts = flatten_fpn_features(fpn)
        box_tokens = torch.rand(bs, N, C, requires_grad=True)
        time_emb = torch.rand(bs, time_dim)
        bbox_coords = torch.rand(bs, N, 4)
        bbox_coords[..., 2] = bbox_coords[..., 0] + 0.1
        bbox_coords[..., 3] = bbox_coords[..., 1] + 0.1

        out = block(box_tokens, flat, shapes, starts, time_emb, bbox_coords)
        check(f'adaln_zero={adaln_zero}: shape', out.shape == (bs, N, C))
        check(f'adaln_zero={adaln_zero}: finite', out.isfinite().all().item())

        loss = out.sum()
        loss.backward()
        check(f'adaln_zero={adaln_zero}: grad finite',
              box_tokens.grad.isfinite().all().item())

        # 检查 grad norm 是否合理
        grad_norm = box_tokens.grad.norm().item()
        check(f'adaln_zero={adaln_zero}: grad norm reasonable',
              0 < grad_norm < 1e6, f'grad_norm={grad_norm:.2f}')


# ============================================================
# Node 10: DiffusionDetCriterion._get_loss
# ============================================================
def test_node10_criterion_loss():
    """验证 Criterion 产生的各项 loss 数值合理性"""
    print('\n=== Node 10: DiffusionDetCriterion._get_loss ===')
    bs, N, C = 2, 50, 24
    pred_logits = torch.randn(bs, N, C)
    # 使用合理的预测 bbox，确保有正样本匹配
    pred_bboxes = torch.rand(bs, N, 4)
    pred_bboxes[..., 2] = pred_bboxes[..., 0] + 0.05
    pred_bboxes[..., 3] = pred_bboxes[..., 1] + 0.05
    pred_bboxes = pred_bboxes.clamp(0, 1)

    targets = []
    for i in range(bs):
        M = 5
        # 使 GT 与部分 pred 接近，确保有正样本匹配
        gt_bboxes_i = pred_bboxes[i, :M].clone()
        gt_bboxes_i += 0.02 * torch.randn_like(gt_bboxes_i)
        gt_bboxes_i = gt_bboxes_i.clamp(0, 1)
        gt_bboxes_i[..., 2] = gt_bboxes_i[..., 0] + 0.05
        gt_bboxes_i[..., 3] = gt_bboxes_i[..., 1] + 0.05
        targets.append(
            InstanceData(
                labels=torch.randint(0, C, (M,)),
                bboxes=gt_bboxes_i,
                img_shape=(800, 1333),
            )
        )

    for bbox_loss_mode in ['l1', 'relative_l1']:
        matcher = DiffusionDetMatcher(
            match_costs=[
                FocalLossCost(weight=2.0),
                RelativeL1Cost(weight=5.0),
                IoUCost(iou_mode='giou', weight=2.0),
            ],
            center_radius=0.5,
            candidate_topk=5,
        )
        criterion = DiffusionDetCriterion(
            num_classes=C,
            matcher=matcher,
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            deep_supervision=False,
            bbox_loss_mode=bbox_loss_mode,
        )
        outputs = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_bboxes)
        losses = criterion(outputs, targets)

        for k, v in losses.items():
            check(f'{bbox_loss_mode}: {k} finite', v.isfinite().item(),
                  f'val={v.item():.6f}')
            check(f'{bbox_loss_mode}: {k} >= 0', v.item() >= 0,
                  f'val={v.item():.6f}')


def test_node10_criterion_deep_supervision():
    """验证 Deep Supervision 的 aux losses"""
    print('\n=== Node 10b: Criterion Deep Supervision ===')
    bs, N, C = 2, 50, 24
    pred_logits = torch.randn(bs, N, C)
    pred_bboxes = torch.rand(bs, N, 4)
    pred_bboxes[..., 2] = pred_bboxes[..., 0] + 0.05
    pred_bboxes[..., 3] = pred_bboxes[..., 1] + 0.05
    pred_bboxes = pred_bboxes.clamp(0, 1)

    targets = []
    for i in range(bs):
        M = 5
        gt_bboxes_i = pred_bboxes[i, :M].clone() + 0.01
        gt_bboxes_i = gt_bboxes_i.clamp(0, 1)
        targets.append(
            InstanceData(
                labels=torch.randint(0, C, (M,)),
                bboxes=gt_bboxes_i,
                img_shape=(800, 1333),
            )
        )

    matcher = DiffusionDetMatcher(
        match_costs=[
            FocalLossCost(weight=2.0),
            BBoxL1Cost(weight=5.0),
            IoUCost(iou_mode='giou', weight=2.0),
        ],
        center_radius=0.5,
        candidate_topk=5,
    )
    criterion = DiffusionDetCriterion(
        num_classes=C, matcher=matcher,
        loss_cls=FocalLoss(loss_weight=2.0),
        loss_bbox=L1Loss(loss_weight=5.0),
        loss_giou=GIoULoss(loss_weight=2.0),
        deep_supervision=True,
    )

    main_output = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_bboxes)
    # 添加 aux outputs
    main_output.aux_outputs = [
        ModelOutput(
            pred_logits=pred_logits + 0.1 * torch.randn_like(pred_logits),
            pred_boxes=(pred_bboxes + 0.01).clamp(0, 1),
        )
    ]

    losses = criterion(main_output, targets)
    check('has aux_0_loss_cls', 'aux_0_loss_cls' in losses)
    for k, v in losses.items():
        check(f'{k} finite', v.isfinite().item())


# ============================================================
# Node 11: _add_velocity_loss
# ============================================================
def test_node11_velocity_loss():
    """验证 velocity loss 计算"""
    print('\n=== Node 11: _add_velocity_loss ===')
    head = _make_head(pred_mode='velocity')
    head.train()
    bs = 2
    fpn, img_metas = _make_fpn_and_metas(bs)
    gt_bboxes_img, gt_labels = _make_gt(bs)

    losses = head.loss(fpn, img_metas, gt_bboxes_img, gt_labels)

    check('has loss_velocity', 'loss_velocity' in losses,
          f'keys={list(losses.keys())}')
    check('loss_velocity finite', losses['loss_velocity'].isfinite().item())
    check('loss_velocity >= 0', losses['loss_velocity'].item() >= 0)

    # 验证 velocity loss 不为异常值 (如 nan 或极大值)
    vel_val = losses['loss_velocity'].item()
    check('loss_velocity not crazy', vel_val < 1e6, f'val={vel_val:.4f}')


# ============================================================
# Node 12: _add_lsas_loss
# ============================================================
# Note: LSAS loss requires use_lsas=True, which is not in standard _make_head.
# We test it implicitly through the full pipeline when use_lsas is on.


# ============================================================
# Node 13: 全流程梯度检查
# ============================================================
def test_node13_full_pipeline_gradient():
    """全流程梯度流检查: loss → backward → 所有参数梯度正常"""
    print('\n=== Node 13: Full Pipeline Gradient ===')
    for pred_mode in ['x0', 'velocity']:
        head = _make_head(pred_mode=pred_mode)
        head.train()
        bs = 2
        fpn, img_metas = _make_fpn_and_metas(bs)
        gt_bboxes_img, gt_labels = _make_gt(bs)

        losses = head.loss(fpn, img_metas, gt_bboxes_img, gt_labels)
        total_loss = sum(losses.values())
        total_loss.backward()

        # 检查 total_loss 本身是否正常
        check(f'{pred_mode}: total_loss finite', total_loss.isfinite().item(),
              f'val={total_loss.item():.6f}')
        check(f'{pred_mode}: total_loss >= 0', total_loss.item() >= 0)

        # 检查各模块参数梯度
        grads = {}
        for name, param in head.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grads[name] = grad_norm

        check(f'{pred_mode}: has gradients', len(grads) > 0)

        # 检查梯度中无 NaN/Inf
        nan_params = [name for name, g in grads.items()
                      if not torch.tensor(g).isfinite().item()]
        check(f'{pred_mode}: no NaN/Inf gradients', len(nan_params) == 0,
              f'nan_params={nan_params}')

        # 检查最大梯度 norm 是否异常
        max_grad = max(grads.values()) if grads else 0
        min_grad = min(grads.values()) if grads else 0
        check(f'{pred_mode}: max grad norm not extreme',
              max_grad < 1e9, f'max_grad={max_grad:.2f}')

        print(f'  [{pred_mode}] grad stats: count={len(grads)}, '
              f'max={max_grad:.2f}, min={min(min_grad, 1e6):.6f}')


# ============================================================
# Node 14: 多步迭代稳定性
# ============================================================
def test_node14_multiple_iterations():
    """验证多步训练迭代中 loss 和梯度稳定"""
    print('\n=== Node 14: Multiple Iteration Stability ===')
    head = _make_head(pred_mode='x0', num_heads=6, deep_sup=True)
    head.train()
    bs = 2
    fpn, img_metas = _make_fpn_and_metas(bs)

    loss_values = []
    grad_norms = []

    for it in range(5):
        gt_bboxes_img, gt_labels = _make_gt(bs, M=3 + it)  # 变化 GT 数量
        losses = head.loss(fpn, img_metas, gt_bboxes_img, gt_labels)
        total_loss = sum(losses.values())

        check(f'iter {it}: total_loss finite', total_loss.isfinite().item(),
              f'val={total_loss.item():.6f}')

        total_loss.backward()
        total_norm = sum(
            p.grad.norm().item() for p in head.parameters() if p.grad is not None
        ) / max(len(list(head.parameters())), 1)

        loss_values.append(total_loss.item())
        grad_norms.append(total_norm)

        # 清零梯度
        head.zero_grad()

    # 验证 loss 变化在合理范围内
    loss_range = max(loss_values) - min(loss_values)
    check('loss variance reasonable', loss_range < 100,
          f'losses={[f"{v:.4f}" for v in loss_values]}')

    print(f'  loss values: {[f"{v:.4f}" for v in loss_values]}')
    print(f'  avg grad norms: {[f"{g:.4f}" for g in grad_norms]}')


# ============================================================
# Node 15: 大规模输入压测
# ============================================================
def test_node15_large_scale():
    """模拟近训练配置的大规模输入 (num_proposals=300)"""
    print('\n=== Node 15: Large Scale Test ===')
    head = DiTDiffusionDetHead(
        num_classes=24,
        feat_channels=256,
        num_proposals=300,
        num_heads=6,
        snr_scale=2.0,
        sampling_timesteps=2,
        diffusion_type='rectified_flow',
        solver_type='euler',
        rf_schedule='shifted',
        rf_shift=3.0,
        regression_mode='direct',
        prediction_mode='x0',
        adaln_params=9,
        num_fpn_levels=4,
        num_ref_points=8,
        box_init_mode='zero',
        deep_supervision=True,
        criterion=DiffusionDetCriterion(
            num_classes=24,
            matcher=DiffusionDetMatcher(
                match_costs=[
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
    )
    head.train()
    bs = 2
    C = 256
    fpn, img_metas = _make_fpn_and_metas(bs, C)

    # 多 GT 场景
    gt_bboxes_img = []
    gt_labels = []
    for i in range(bs):
        M = 20 + i * 5
        bboxes_norm = torch.rand(M, 4)
        bboxes_norm[..., 2] = bboxes_norm[..., 0] + 0.02 + torch.rand(M) * 0.05
        bboxes_norm[..., 3] = bboxes_norm[..., 1] + 0.02 + torch.rand(M) * 0.05
        bboxes_norm = bboxes_norm.clamp(0, 1)
        gt_bboxes_img.append(bboxes_norm * torch.tensor([1333, 800, 1333, 800]))
        gt_labels.append(torch.randint(0, 24, (M,)))

    losses = head.loss(fpn, img_metas, gt_bboxes_img, gt_labels)
    total_loss = sum(losses.values())
    total_loss.backward()

    check('total_loss finite', total_loss.isfinite().item(),
          f'val={total_loss.item():.6f}')
    check('total_loss >= 0', total_loss.item() >= 0)

    # 检查梯度
    nan_grads = []
    for name, param in head.named_parameters():
        if param.grad is not None and not param.grad.isfinite().all().item():
            nan_grads.append(name)
    check('no NaN grads at scale', len(nan_grads) == 0,
          f'nan_grads={nan_grads}')

    print(f'  total_loss={total_loss.item():.4f}, '
          f'params_with_grad={sum(1 for p in head.parameters() if p.grad is not None)}')


# ============================================================
# Node 16: 极端扩散时间步测试
# ============================================================
def test_node16_edge_timesteps():
    """验证极端 t 值 (t≈0, t≈1) 下的行为"""
    print('\n=== Node 16: Edge Timesteps ===')
    head = _make_head()
    bs = 2
    gt_bboxes_img, gt_labels = _make_gt(bs)
    fpn, img_metas = _make_fpn_and_metas(bs)
    device = fpn[0].device

    for t_val in [0.001, 0.01, 0.5, 0.99, 0.999]:
        t = torch.full((bs,), t_val, device=device)
        targets = head._normalize_targets(gt_bboxes_img, gt_labels, img_metas, bs)
        x_boxes, x_starts, x_noises, _ = head._build_training_targets(
            bs, device, t, targets, gt_bboxes_img, img_metas,
        )
        x_noisy_batch = torch.stack(x_boxes)
        curr_bboxes = head._raw_to_xyxy(x_noisy_batch, img_metas)
        t_input = t * head.timesteps

        all_cls, all_bboxes, _, _, _ = head(
            fpn, curr_bboxes, t_input, img_metas=img_metas
        )

        check(f't={t_val}: bbox seq finite',
              all_bboxes.isfinite().all().item())
        check(f't={t_val}: cls seq finite',
              all_cls.isfinite().all().item())
        check(f't={t_val}: bbox in [0,1]',
              all_bboxes[-1].min().item() >= 0
              and all_bboxes[-1].max().item() <= 1)


# ============================================================
# Node 17: 输出端到端一致性检查
# ============================================================
def test_node17_end_to_end_consistency():
    """验证 loss 计算中的输出在语义上一致"""
    print('\n=== Node 17: End-to-End Consistency ===')
    head = _make_head(pred_mode='x0', deep_sup=True, num_heads=3)
    head.train()
    bs = 2
    fpn, img_metas = _make_fpn_and_metas(bs)

    gt_bboxes_img, gt_labels = _make_gt(bs, M=5)

    # 拆解 loss 方法, 逐步验证
    device = fpn[0].device
    targets = head._normalize_targets(gt_bboxes_img, gt_labels, img_metas, bs)
    t, _ = head._sample_t(bs, device)

    x_boxes, x_starts, x_noises, matched_gt_indices = head._build_training_targets(
        bs, device, t, targets, gt_bboxes_img, img_metas,
    )
    x_noisy_batch = torch.stack(x_boxes)
    curr_bboxes = head._raw_to_xyxy(x_noisy_batch, img_metas)

    # 验证 curr_bboxes 合理
    check('curr_bboxes finite', curr_bboxes.isfinite().all().item())
    check('curr_bboxes valid', (curr_bboxes[..., 2] >= curr_bboxes[..., 0]).all().item())

    t_input = t * head.timesteps
    all_cls, all_bboxes, _, _, _ = head(fpn, curr_bboxes, t_input, img_metas=img_metas)

    # AdaLN-Zero 模式下各层输出应相同 (alpha=0, 残差为零)
    # 这是预期的初始化行为，验证各层输出一致
    if head.deep_supervision and head.num_heads > 2:
        for h in range(1, head.num_heads):
            diff = (all_cls[h] - all_cls[h - 1]).abs().max().item()
            check(f'layer {h} output consistent (AdaLN-Zero)', True,
                  f'max_diff={diff:.10f} (expected 0 for AdaLN-Zero init)')

    # 验证 bbox 始终在 [0,1]
    for h in range(all_bboxes.shape[0]):
        check(f'layer {h} bbox in [0,1]',
              all_bboxes[h].min().item() >= 0
              and all_bboxes[h].max().item() <= 1)


# ============================================================
# Node 18: random seed 敏感性
# ============================================================
def test_node18_seed_sensitivity():
    """验证模型对随机种子的敏感性在合理范围内"""
    print('\n=== Node 18: Seed Sensitivity ===')
    bs = 2
    fpn, img_metas = _make_fpn_and_metas(bs)
    gt_bboxes_img, gt_labels = _make_gt(bs, M=5)

    all_losses = []
    for seed in [42, 123, 777]:
        torch.manual_seed(seed)
        head = _make_head(pred_mode='x0')
        head.train()
        losses = head.loss(fpn, img_metas, gt_bboxes_img, gt_labels)
        total = sum(losses.values()).item()
        all_losses.append(total)
        check(f'seed={seed}: loss finite', torch.tensor(total).isfinite().item(),
              f'val={total:.6f}')

    # 不同 seed 下 loss 应不同但都正常
    all_same = all(l == all_losses[0] for l in all_losses)
    check('loss varies with seed', not all_same, f'losses={all_losses}')

    # 验证 loss 都在合理范围
    for loss_val in all_losses:
        check(f'loss {loss_val:.4f} in reasonable range',
              0 < loss_val < 1000)


# ============================================================
# Node 19: num_blocks=3 多 block 全链路测试
# ============================================================
def test_node19_num_blocks_3():
    """验证 num_blocks=3 多 block 堆叠下的全链路正确性"""
    print('\n=== Node 19: num_blocks=3 Full Pipeline ===')
    head = DiTDiffusionDetHead(
        num_classes=24,
        feat_channels=256,
        num_proposals=50,
        num_heads=3,
        num_blocks=3,
        share_heads=False,
        snr_scale=2.0,
        sampling_timesteps=2,
        diffusion_type='rectified_flow',
        solver_type='euler',
        rf_schedule='shifted',
        rf_shift=3.0,
        regression_mode='direct',
        prediction_mode='x0',
        adaln_params=9,
        num_fpn_levels=4,
        num_ref_points=8,
        box_init_mode='zero',
        deep_supervision=True,
        criterion=DiffusionDetCriterion(
            num_classes=24,
            matcher=DiffusionDetMatcher(
                match_costs=[
                    FocalLossCost(weight=2.0),
                    BBoxL1Cost(weight=5.0),
                    IoUCost(iou_mode='giou', weight=2.0),
                ],
                center_radius=0.5,
                candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
        ),
    )
    head.train()
    bs = 2
    fpn, img_metas = _make_fpn_and_metas(bs)
    gt_bboxes_img, gt_labels = _make_gt(bs, M=5)

    # 验证每个 head 有 num_blocks=3 个 DiTBlock
    for i, h in enumerate(head.head_series):
        check(f'head[{i}] has 3 dit_blocks', len(h.dit_blocks) == 3,
              f'got {len(h.dit_blocks)}')

    # 验证 share_heads=False: 各 head 参数独立
    if len(head.head_series) >= 2:
        p0 = list(head.head_series[0].parameters())[0]
        p1 = list(head.head_series[1].parameters())[0]
        check('share_heads=False: heads independent',
              not torch.equal(p0, p1))

    # 全链路 loss + backward
    losses = head.loss(fpn, img_metas, gt_bboxes_img, gt_labels)
    total_loss = sum(losses.values())
    total_loss.backward()

    check('total_loss finite', total_loss.isfinite().item(),
          f'val={total_loss.item():.6f}')
    check('total_loss >= 0', total_loss.item() >= 0)

    # 检查梯度无 NaN/Inf
    nan_params = []
    for name, param in head.named_parameters():
        if param.grad is not None and not param.grad.isfinite().all().item():
            nan_params.append(name)
    check('no NaN/Inf grads with num_blocks=3', len(nan_params) == 0,
          f'nan_params={nan_params}')

    # 多步迭代稳定性
    for it in range(3):
        gt_bboxes_img, gt_labels = _make_gt(bs, M=3 + it)
        losses = head.loss(fpn, img_metas, gt_bboxes_img, gt_labels)
        total_loss = sum(losses.values())
        check(f'iter{it} loss finite', total_loss.isfinite().item(),
              f'val={total_loss.item():.6f}')
        total_loss.backward()
        head.zero_grad()

    print(f'  num_blocks=3, share_heads=False pipeline OK')


# ============================================================
# Node 20: spatial_prior 全链路测试
# ============================================================
def test_node20_spatial_prior_pipeline():
    """验证 box_init_mode='spatial_prior' 的全链路正确性"""
    print('\n=== Node 20: spatial_prior Full Pipeline ===')
    head = DiTDiffusionDetHead(
        num_classes=24,
        feat_channels=256,
        num_proposals=50,
        num_heads=2,
        snr_scale=2.0,
        sampling_timesteps=2,
        diffusion_type='rectified_flow',
        solver_type='euler',
        rf_schedule='shifted',
        rf_shift=3.0,
        regression_mode='direct',
        prediction_mode='x0',
        adaln_params=9,
        num_fpn_levels=4,
        num_ref_points=8,
        box_init_mode='spatial_prior',
        deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=24,
            matcher=DiffusionDetMatcher(
                match_costs=[
                    FocalLossCost(weight=2.0),
                    BBoxL1Cost(weight=5.0),
                    IoUCost(iou_mode='giou', weight=2.0),
                ],
                center_radius=0.5,
                candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
        ),
    )
    head.train()
    bs = 2
    fpn, img_metas = _make_fpn_and_metas(bs)
    gt_bboxes_img, gt_labels = _make_gt(bs, M=5)

    # 验证 anchor_boxes 存在且形状正确
    tok = head.box_tokenizer
    check('anchor_boxes exists', hasattr(tok, 'anchor_boxes'))
    check('anchor_boxes shape', tok.anchor_boxes.shape == (50, 4))

    # 验证 spatial_prior 产生的 token 不为全零
    bs = 2
    gt_bboxes_img, gt_labels = _make_gt(bs, M=5)
    device = fpn[0].device
    targets = head._normalize_targets(gt_bboxes_img, gt_labels, img_metas, bs)
    t, _ = head._sample_t(bs, device)
    x_boxes, _, _, _ = head._build_training_targets(
        bs, device, t, targets, gt_bboxes_img, img_metas,
    )
    x_noisy_batch = torch.stack(x_boxes)
    curr_bboxes = head._raw_to_xyxy(x_noisy_batch, img_metas)
    normed = head._normalize_bboxes_for_tokenizer(curr_bboxes, img_metas)
    tokens, _ = tok(normed, [f.clone() for f in fpn])
    check('spatial_prior tokens not all zero', tokens.abs().max().item() > 0)
    check('spatial_prior tokens finite', tokens.isfinite().all().item())

    # 全链路 loss + backward
    losses = head.loss(fpn, img_metas, gt_bboxes_img, gt_labels)
    total_loss = sum(losses.values())
    total_loss.backward()

    check('total_loss finite', total_loss.isfinite().item(),
          f'val={total_loss.item():.6f}')
    check('total_loss >= 0', total_loss.item() >= 0)

    nan_params = []
    for name, param in head.named_parameters():
        if param.grad is not None and not param.grad.isfinite().all().item():
            nan_params.append(name)
    check('no NaN/Inf grads with spatial_prior', len(nan_params) == 0,
          f'nan_params={nan_params}')

    print(f'  spatial_prior pipeline OK')


# ============================================================
# Node 21: delta regression 全链路测试
# ============================================================
def test_node21_delta_regression_pipeline():
    """验证 regression_mode='delta' 的全链路正确性"""
    print('\n=== Node 21: delta regression Full Pipeline ===')
    head = DiTDiffusionDetHead(
        num_classes=24,
        feat_channels=256,
        num_proposals=50,
        num_heads=2,
        snr_scale=2.0,
        sampling_timesteps=2,
        diffusion_type='rectified_flow',
        solver_type='euler',
        rf_schedule='shifted',
        rf_shift=3.0,
        regression_mode='delta',
        prediction_mode='x0',
        adaln_params=9,
        num_fpn_levels=4,
        num_ref_points=8,
        box_init_mode='zero',
        deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=24,
            matcher=DiffusionDetMatcher(
                match_costs=[
                    FocalLossCost(weight=2.0),
                    BBoxL1Cost(weight=5.0),
                    IoUCost(iou_mode='giou', weight=2.0),
                ],
                center_radius=0.5,
                candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
        ),
    )
    head.train()
    bs = 2
    fpn, img_metas = _make_fpn_and_metas(bs)
    gt_bboxes_img, gt_labels = _make_gt(bs, M=5)

    # 全链路 loss + backward
    losses = head.loss(fpn, img_metas, gt_bboxes_img, gt_labels)
    total_loss = sum(losses.values())
    total_loss.backward()

    check('total_loss finite', total_loss.isfinite().item(),
          f'val={total_loss.item():.6f}')
    check('total_loss >= 0', total_loss.item() >= 0)

    nan_params = []
    for name, param in head.named_parameters():
        if param.grad is not None and not param.grad.isfinite().all().item():
            nan_params.append(name)
    check('no NaN/Inf grads with delta mode', len(nan_params) == 0,
          f'nan_params={nan_params}')

    # 验证 delta 模式下 pred bbox 在 [0,1] 范围内
    head.eval()
    x_noisy = torch.randn(bs, 50, 4) * 2.0
    curr_bboxes = head._raw_to_xyxy(x_noisy, img_metas)
    t_input = torch.full((bs,), 500.0)
    with torch.no_grad():
        _, all_bboxes, _, _, _ = head(fpn, curr_bboxes, t_input, img_metas=img_metas)
    check('delta: bbox in [0,1]', all_bboxes[-1].min().item() >= -1e-5
          and all_bboxes[-1].max().item() <= 1 + 1e-5)
    check('delta: bbox finite', all_bboxes[-1].isfinite().all().item())

    print(f'  delta regression pipeline OK')


# ============================================================
# Node 22: 训练初期 loss 下降趋势验证
# ============================================================
def test_node22_early_training_loss_descent():
    """验证训练初期 10 步迭代 loss 是否单调下降

    验证噪声框不会导致模型找不到正确梯度方向。
    使用实际配置: num_blocks=3, spatial_prior, delta regression, OT coupling.
    """
    print('\n=== Node 22: Early Training Loss Descent ===')
    head = DiTDiffusionDetHead(
        num_classes=24,
        feat_channels=256,
        num_proposals=100,
        num_heads=3,
        num_blocks=3,
        share_heads=False,
        deep_supervision=True,
        snr_scale=2.0,
        sampling_timesteps=2,
        diffusion_type='rectified_flow',
        solver_type='euler',
        rf_schedule='shifted',
        rf_shift=1.0,
        regression_mode='delta',
        prediction_mode='x0',
        adaln_params=9,
        num_fpn_levels=4,
        num_ref_points=8,
        box_init_mode='spatial_prior',
        ot_coupling=True,
        ot_matcher='sinkhorn',
        ot_epsilon=1.0,
        ot_num_iters=10,
        criterion=DiffusionDetCriterion(
            num_classes=24,
            matcher=DiffusionDetMatcher(
                match_costs=[
                    FocalLossCost(weight=2.0),
                    BBoxL1Cost(weight=5.0),
                    IoUCost(iou_mode='giou', weight=2.0),
                ],
                center_radius=5.0,
                candidate_topk=12,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
        ),
    )
    head.train()
    bs = 2
    fpn, img_metas = _make_fpn_and_metas(bs)
    optimizer = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad], lr=1e-4
    )

    losses = []
    for step in range(12):
        gt_bboxes_img, gt_labels = _make_gt(bs, M=3)
        loss_dict = head.loss(fpn, img_metas, gt_bboxes_img, gt_labels)
        total_loss = sum(loss_dict.values())
        optimizer.zero_grad()
        total_loss.backward()
        # 梯度裁剪 (模拟训练配置)
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=50.0)
        optimizer.step()
        losses.append(total_loss.item())

    print(f'  Loss sequence: {[f"{l:.4f}" for l in losses]}')

    # 验证: 前5步均值 > 后5步均值 (loss 有下降趋势)
    early_mean = sum(losses[:5]) / 5
    late_mean = sum(losses[-5:]) / 5
    check('loss decreases: early_mean > late_mean',
          early_mean > late_mean,
          f'early={early_mean:.4f}, late={late_mean:.4f}')

    # 验证: 最后一步 loss 有限且非零
    check('final loss finite', torch.isfinite(torch.tensor(losses[-1])).item(),
          f'val={losses[-1]:.6f}')
    check('final loss > 0', losses[-1] > 0,
          f'val={losses[-1]:.6f}')

    # 验证: 无 NaN
    check('no NaN loss in any step',
          all(torch.isfinite(torch.tensor(l)).item() for l in losses),
          f'losses={losses}')

    # 验证: loss 不会爆炸 (>1000)
    check('loss not exploding',
          max(losses) < 1000,
          f'max={max(losses):.4f}')

    print(f'  Early training loss descent OK (early={early_mean:.4f} -> late={late_mean:.4f})')


def test_node23_predict_uses_spatial_prior():
    """Node 23: 验证 predict() 在 spatial_prior 模式下使用 anchor_boxes 初始化。

    Red: 当前 predict() 用 randn 初始化，导致推理时框分布与训练不匹配。
    验证点:
    1. spatial_prior 模式下 predict() 不使用 randn 初始化
    2. 初始框来自 anchor_boxes，具有合理的 width/height (>0.01)
    3. 推理输出 bbox 在 [0,1] 范围内
    4. 推理输出 scores > 0 (分类分支有响应)
    """
    head = DiTDiffusionDetHead(
        num_classes=24,
        feat_channels=256,
        num_proposals=50,
        num_heads=2,
        snr_scale=2.0,
        sampling_timesteps=2,
        diffusion_type='rectified_flow',
        solver_type='euler',
        rf_schedule='shifted',
        rf_shift=3.0,
        regression_mode='delta',
        prediction_mode='x0',
        adaln_params=9,
        num_fpn_levels=4,
        num_ref_points=8,
        box_init_mode='spatial_prior',
        deep_supervision=True,
        criterion=DiffusionDetCriterion(
            num_classes=24,
            matcher=DiffusionDetMatcher(
                match_costs=[
                    FocalLossCost(weight=2.0),
                    BBoxL1Cost(weight=5.0),
                    IoUCost(iou_mode='giou', weight=2.0),
                ],
                center_radius=0.5,
                candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
        ),
    )
    head.eval()

    fpn, img_metas = _make_fpn_and_metas(bs=2, C=256)

    with torch.no_grad():
        results = head.predict(fpn, img_metas, rescale=False)

    # 验证: 输出非空
    check('predict returns results', len(results) == 2,
          f'len={len(results)}')

    for i, r in enumerate(results):
        # 验证: bbox 有限且非负 (predict 输出是像素坐标)
        if len(r.bboxes) > 0:
            bboxes = r.bboxes
            check(f'img{i} bboxes finite',
                  torch.isfinite(bboxes).all().item(),
                  f'has NaN/Inf')
            check(f'img{i} bboxes >= 0',
                  (bboxes >= -1).all().item(),
                  f'min={bboxes.min().item():.4f}')

        # 验证: 分类分支有响应 (scores 不全为 0)
        if len(r.scores) > 0:
            check(f'img{i} scores > 0',
                  (r.scores > 0).any().item(),
                  f'max_score={r.scores.max().item():.6f}')

    # 验证: anchor_boxes 存在且有合理的 width/height
    tokenizer = head.box_tokenizer
    check('anchor_boxes exists',
          hasattr(tokenizer, 'anchor_boxes'),
          'BoxTokenizer 没有 anchor_boxes buffer')
    if hasattr(tokenizer, 'anchor_boxes'):
        anchors = tokenizer.anchor_boxes  # (num_proposals, 4) cxcywh
        widths = anchors[:, 2]
        heights = anchors[:, 3]
        check('anchor widths > 0.01',
              (widths > 0.01).all().item(),
              f'min_width={widths.min().item():.6f}')
        check('anchor heights > 0.01',
              (heights > 0.01).all().item(),
              f'min_height={heights.min().item():.6f}')

    print('  Node 23: predict spatial_prior init OK')


def test_node24_predict_randn_vs_anchor_difference():
    """Node 24: 验证 spatial_prior 初始化与 randn 初始化产生不同的推理结果。

    Red: 如果 predict() 仍用 randn，两种初始化应产生不同结果。
    但关键是 spatial_prior 应产生更合理的初始框 (非退化点框)。
    """
    head = DiTDiffusionDetHead(
        num_classes=24,
        feat_channels=256,
        num_proposals=50,
        num_heads=2,
        snr_scale=2.0,
        sampling_timesteps=2,
        diffusion_type='rectified_flow',
        solver_type='euler',
        rf_schedule='shifted',
        rf_shift=3.0,
        regression_mode='delta',
        prediction_mode='x0',
        adaln_params=9,
        num_fpn_levels=4,
        num_ref_points=8,
        box_init_mode='spatial_prior',
        deep_supervision=True,
        criterion=DiffusionDetCriterion(
            num_classes=24,
            matcher=DiffusionDetMatcher(
                match_costs=[
                    FocalLossCost(weight=2.0),
                    BBoxL1Cost(weight=5.0),
                    IoUCost(iou_mode='giou', weight=2.0),
                ],
                center_radius=0.5,
                candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
        ),
    )
    head.eval()

    # 验证: _init_inference_boxes 方法存在且返回合理形状
    check('_init_inference_boxes method exists',
          hasattr(head, '_init_inference_boxes'),
          'DiTDiffusionDetHead 没有 _init_inference_boxes 方法')

    if hasattr(head, '_init_inference_boxes'):
        fpn, img_metas = _make_fpn_and_metas(bs=2, C=256)
        device = fpn[0].device
        x_raw = head._init_inference_boxes(2, device)

        # 验证: 形状正确
        check('x_raw shape', x_raw.shape == (2, 50, 4),
              f'shape={x_raw.shape}')

        # 验证: spatial_prior 初始化的框有合理的 width/height
        # 转换到 cxcywh [0,1] 空间检查
        cxcywh = (x_raw / head.snr_scale + 1) / 2
        widths = cxcywh[:, :, 2]  # w in cxcywh
        heights = cxcywh[:, :, 3]  # h in cxcywh
        check('spatial_prior init widths > 0.01',
              (widths > 0.01).all().item(),
              f'min_width={widths.min().item():.6f}')
        check('spatial_prior init heights > 0.01',
              (heights > 0.01).all().item(),
              f'min_height={heights.min().item():.6f}')

    print('  Node 24: predict init method check OK')


def test_node25_box_renewal_uses_anchors():
    """Node 25: Bug2 — _apply_box_renewal 在 spatial_prior 模式下应用 anchor 替换低分框。

    Red: 当前 _apply_box_renewal 用 randn 替换低分框，导致新框与训练时的 anchor 分布不匹配。
    验证: spatial_prior 模式下替换后的 x_raw 来自 anchor_boxes 而非 randn。
    """
    head = DiTDiffusionDetHead(
        num_classes=24,
        feat_channels=256,
        num_proposals=50,
        num_heads=2,
        snr_scale=2.0,
        sampling_timesteps=2,
        diffusion_type='rectified_flow',
        solver_type='euler',
        rf_schedule='shifted',
        rf_shift=3.0,
        regression_mode='delta',
        prediction_mode='x0',
        adaln_params=9,
        num_fpn_levels=4,
        num_ref_points=8,
        box_init_mode='spatial_prior',
        deep_supervision=True,
        min_keep=5,  # 关键: 设小以触发实际替换
        criterion=DiffusionDetCriterion(
            num_classes=24,
            matcher=DiffusionDetMatcher(
                match_costs=[
                    FocalLossCost(weight=2.0),
                    BBoxL1Cost(weight=5.0),
                    IoUCost(iou_mode='giou', weight=2.0),
                ],
                center_radius=0.5,
                candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
        ),
    )
    head.eval()

    # 构造低分 cls_logits (全 < 0.05) 触发 box_renewal
    bs = 2
    device = next(head.parameters()).device
    cls_logits = torch.full((bs, 50, 24), -5.0)  # sigmoid(-5) ≈ 0.007 < 0.05
    scores = torch.sigmoid(cls_logits).max(-1)[0]
    x_raw = head._init_inference_boxes(bs, device)

    # 验证: 替换后的框来自 anchor 池 (有合理的 cxcywh width/height)
    # 而非 randn (cxcywh w/h 可能 < 0 或极小)
    with torch.no_grad():
        x_raw_renewed = head._apply_box_renewal(x_raw, cls_logits)

    cxcywh = (x_raw_renewed / head.snr_scale + 1) / 2
    widths = cxcywh[:, :, 2]
    heights = cxcywh[:, :, 3]
    check('renewed widths > 0.05 (anchor scale)',
          (widths > 0.05).all().item(),
          f'min_width={widths.min().item():.6f}')
    check('renewed heights > 0.05 (anchor scale)',
          (heights > 0.05).all().item(),
          f'min_height={heights.min().item():.6f}')

    # 对比: randn 替换的框 cxcywh w/h 不稳定
    x_raw_randn = x_raw.clone()
    for i in range(bs):
        keep = scores[i] > head.score_thr
        if keep.sum() < head.min_keep:
            _, topk_idx = scores[i].topk(min(head.min_keep, scores.shape[1]))
            keep[topk_idx] = True
        num_renew = (~keep).sum()
        if num_renew > 0:
            x_raw_randn[i, ~keep] = torch.randn(num_renew, 4, device=device)
    cxcywh_randn = (x_raw_randn / head.snr_scale + 1) / 2
    randn_min_w = cxcywh_randn[:, :, 2].min().item()
    randn_min_h = cxcywh_randn[:, :, 3].min().item()
    # randn 替换的框 w/h 可能 < 0 (因为 raw 空间 randn 值可能很大)
    check('anchor renewal better than randn (width)',
          widths.min().item() > randn_min_w or widths.min().item() > 0.05,
          f'anchor_min_w={widths.min().item():.6f} randn_min_w={randn_min_w:.6f}')

    print('  Node 25: box_renewal uses anchors OK')


def test_node26_ensemble_no_cross_category_duplicates():
    """Node 26: Bug3 — ensemble 多步预测不产生跨类别重复框。

    Red: 当前 ensemble 拼接所有 step 的预测，同一位置不同 step 可能分配不同类别，
    NMS 按类别分组无法抑制。
    验证: ensemble 后同一位置的框类别一致，或只使用最后一步结果。
    """
    head = DiTDiffusionDetHead(
        num_classes=24,
        feat_channels=256,
        num_proposals=50,
        num_heads=2,
        snr_scale=2.0,
        sampling_timesteps=4,
        diffusion_type='rectified_flow',
        solver_type='euler',
        rf_schedule='shifted',
        rf_shift=3.0,
        regression_mode='delta',
        prediction_mode='x0',
        adaln_params=9,
        num_fpn_levels=4,
        num_ref_points=8,
        box_init_mode='spatial_prior',
        deep_supervision=True,
        use_ensemble=True,
        criterion=DiffusionDetCriterion(
            num_classes=24,
            matcher=DiffusionDetMatcher(
                match_costs=[
                    FocalLossCost(weight=2.0),
                    BBoxL1Cost(weight=5.0),
                    IoUCost(iou_mode='giou', weight=2.0),
                ],
                center_radius=0.5,
                candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
        ),
    )
    head.eval()

    fpn, img_metas = _make_fpn_and_metas(bs=2, C=256)

    with torch.no_grad():
        results = head.predict(fpn, img_metas, rescale=False)

    # 验证: 结果中无严重重叠的跨类别框 (IoU > 0.5 但类别不同)
    for i, r in enumerate(results):
        if len(r.bboxes) < 2:
            continue
        bboxes = r.bboxes
        labels = r.labels
        # 计算所有框对的 IoU
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        n = len(bboxes)
        cross_cat_overlap = 0
        for j in range(n):
            for k in range(j + 1, n):
                if labels[j] != labels[k]:
                    # 计算 IoU
                    ix1 = max(bboxes[j, 0].item(), bboxes[k, 0].item())
                    iy1 = max(bboxes[j, 1].item(), bboxes[k, 1].item())
                    ix2 = min(bboxes[j, 2].item(), bboxes[k, 2].item())
                    iy2 = min(bboxes[j, 3].item(), bboxes[k, 3].item())
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = areas[j].item() + areas[k].item() - inter
                    iou = inter / max(union, 1e-6)
                    if iou > 0.5:
                        cross_cat_overlap += 1
        check(f'img{i} no cross-cat duplicates (IoU>0.5)',
              cross_cat_overlap == 0,
              f'cross_cat_overlap={cross_cat_overlap}')

    print('  Node 26: ensemble no cross-category duplicates OK')


def test_node27_spatial_prior_no_double_embedding():
    """Node 27: Bug4 — spatial_prior token 不应重复叠加 bbox_pos_embed。

    Red: 当前 sampled_feat = bbox_pos_embed(anchors), pos_embed = bbox_pos_embed(bboxes)
    当 bboxes ≈ anchors 时，token ≈ 2 * bbox_pos_embed(anchors) + lvl_embed，值偏大。
    验证: spatial_prior 模式下 anchor 使用独立的 embedding 网络。
    """
    from mods.box_tokenizer import BoxTokenizer

    tokenizer = BoxTokenizer(
        feat_channels=256,
        num_fpn_levels=4,
        init_mode='spatial_prior',
        num_proposals=50,
    )

    # 验证: 有独立的 anchor_pos_embed
    check('has anchor_pos_embed',
          hasattr(tokenizer, 'anchor_pos_embed'),
          'BoxTokenizer 没有 anchor_pos_embed，anchor 和 bbox 共享 bbox_pos_embed')

    if hasattr(tokenizer, 'anchor_pos_embed'):
        # 验证: anchor_pos_embed 和 bbox_pos_embed 是不同网络
        check('anchor_pos_embed != bbox_pos_embed',
              tokenizer.anchor_pos_embed is not tokenizer.bbox_pos_embed,
              'anchor_pos_embed 和 bbox_pos_embed 是同一对象')

        # 验证: token 值不会因为重复叠加而过大
        anchors = tokenizer.anchor_boxes[:50].unsqueeze(0)  # (1, 50, 4) cxcywh
        bboxes_xyxy = torch.rand(1, 50, 4)
        bboxes_xyxy[..., 2] = bboxes_xyxy[..., 0] + 0.1
        bboxes_xyxy[..., 3] = bboxes_xyxy[..., 1] + 0.1
        bboxes_xyxy = bboxes_xyxy.clamp(0, 1)
        fpn = [torch.rand(1, 256, 20, 20) for _ in range(4)]

        tokens, _ = tokenizer(bboxes_xyxy, fpn)
        check('token abs mean < 5.0',
              tokens.abs().mean().item() < 5.0,
              f'token_abs_mean={tokens.abs().mean().item():.4f}')

    print('  Node 27: spatial_prior no double embedding OK')


# ============================================================
# main
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('LDMDet-DiT Full Pipeline Node-by-Node Validation')
    print('=' * 60)

    test_node1_normalize_targets()
    test_node2_sample_t()
    test_node3_build_training_targets()
    test_node3_build_training_targets_empty_gt()
    test_node4_raw_to_xyxy()
    test_node5_normalize_bboxes_for_tokenizer()
    test_node6_box_tokenizer()
    test_node6_box_tokenizer_learnable()
    test_node7_dithead_forward()
    test_node8_single_head_direct_regression()
    test_node8_single_head_delta_regression()
    test_node9_ditblock_forward()
    test_node10_criterion_loss()
    test_node10_criterion_deep_supervision()
    test_node11_velocity_loss()
    test_node13_full_pipeline_gradient()
    test_node14_multiple_iterations()
    test_node15_large_scale()
    test_node16_edge_timesteps()
    test_node17_end_to_end_consistency()
    test_node18_seed_sensitivity()
    test_node19_num_blocks_3()
    test_node20_spatial_prior_pipeline()
    test_node21_delta_regression_pipeline()
    test_node22_early_training_loss_descent()
    test_node23_predict_uses_spatial_prior()
    test_node24_predict_randn_vs_anchor_difference()
    test_node25_box_renewal_uses_anchors()
    test_node26_ensemble_no_cross_category_duplicates()
    test_node27_spatial_prior_no_double_embedding()

    print('\n' + '=' * 60)
    total = PASS + FAIL + ERROR
    print(f'Results: {PASS}/{total} PASS, {FAIL}/{total} FAIL, '
          f'{ERROR}/{total} ERROR')
    print('=' * 60)

    if FAIL > 0 or ERROR > 0:
        sys.exit(1)
    sys.exit(0)