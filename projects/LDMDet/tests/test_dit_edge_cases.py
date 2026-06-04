"""
LDMDet-DiT 边界条件 & 鲁棒性测试

覆盖:
  1. box_tokenizer: NaN/Inf/空输入/退化框
  2. deformable_attn: 极端采样位置/梯度流
  3. dit_block: 梯度流/RoPE 正确性/adaln 参数
  4. dit_single_head: velocity_head 无重复/NaN 输入
  5. dit_head: 空 GT/全模式
  6. loss: NaN/Inf GT 框/BBoxL1Cost/RelativeL1Cost

运行方式:
  PYTHONPATH=. python projects/LDMDet/tests/test_dit_edge_cases.py
"""

import sys

import torch
import torch.nn as nn

sys.path.insert(0, 'projects/LDMDet')

from mods.box_tokenizer import BoxTokenizer, bbox_to_reference_points
from mods.deformable_attn import (
    MultiScaleDeformableAttention,
    flatten_fpn_features,
    build_fpn_level_info,
)
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
    """执行 fn, 只要不抛异常就算 PASS"""
    global PASS, FAIL, ERROR
    try:
        fn()
        PASS += 1
        print(f'  [PASS] {name}')
    except Exception as e:
        ERROR += 1
        print(f'  [ERROR] {name} => {type(e).__name__}: {e} {detail}')


# ============================================================
# 1. BoxTokenizer 边界条件
# ============================================================
def test_box_tokenizer_empty():
    """空 bbox 输入"""
    print('\n=== test_box_tokenizer_empty ===')
    for mode in ['zero', 'learnable']:
        tok = BoxTokenizer(feat_channels=256, num_fpn_levels=4, init_mode=mode)
        fpn = [torch.rand(2, 256, 20, 20) for _ in range(4)]
        bboxes = torch.zeros(2, 0, 4)
        check_no_crash(f'{mode}: empty bboxes', lambda: tok(bboxes, fpn))


def test_box_tokenizer_single():
    """单 bbox 输入"""
    print('\n=== test_box_tokenizer_single ===')
    tok = BoxTokenizer(feat_channels=256, num_fpn_levels=4, init_mode='zero')
    fpn = [torch.rand(2, 256, 20, 20) for _ in range(4)]
    bboxes = torch.tensor([[[0.1, 0.2, 0.3, 0.5]]]).expand(2, -1, -1)
    tokens, levels = tok(bboxes, fpn)
    check('tokens shape', tokens.shape == (2, 1, 256))
    check('levels shape', levels.shape == (2, 1))
    check('finite', tokens.isfinite().all().item())


def test_box_tokenizer_nan_bboxes():
    """NaN bbox 输入 — 不应崩溃"""
    print('\n=== test_box_tokenizer_nan_bboxes ===')
    tok = BoxTokenizer(feat_channels=256, num_fpn_levels=4, init_mode='zero')
    fpn = [torch.rand(2, 256, 20, 20) for _ in range(4)]
    bboxes = torch.full((2, 10, 4), float('nan'))
    check_no_crash(
        'NaN bboxes',
        lambda: tok(bboxes, fpn),
    )


def test_box_tokenizer_inf_bboxes():
    """Inf bbox 输入 — 不应崩溃"""
    print('\n=== test_box_tokenizer_inf_bboxes ===')
    tok = BoxTokenizer(feat_channels=256, num_fpn_levels=4, init_mode='zero')
    fpn = [torch.rand(2, 256, 20, 20) for _ in range(4)]
    bboxes = torch.full((2, 10, 4), float('inf'))
    check_no_crash(
        'Inf bboxes',
        lambda: tok(bboxes, fpn),
    )


def test_box_tokenizer_neg_inf_bboxes():
    """-Inf bbox 输入 — 不应崩溃"""
    print('\n=== test_box_tokenizer_neg_inf_bboxes ===')
    tok = BoxTokenizer(feat_channels=256, num_fpn_levels=4, init_mode='zero')
    fpn = [torch.rand(2, 256, 20, 20) for _ in range(4)]
    bboxes = torch.full((2, 10, 4), float('-inf'))
    check_no_crash(
        '-Inf bboxes',
        lambda: tok(bboxes, fpn),
    )


def test_box_tokenizer_degenerate_bboxes():
    """退化框 (x2 <= x1, y2 <= y1)"""
    print('\n=== test_box_tokenizer_degenerate_bboxes ===')
    tok = BoxTokenizer(feat_channels=256, num_fpn_levels=4, init_mode='zero')
    fpn = [torch.rand(2, 256, 20, 20) for _ in range(4)]
    # 零面积框
    bboxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]]).expand(2, 10, -1)
    check_no_crash(
        'zero-area bboxes',
        lambda: tok(bboxes, fpn),
    )
    # 反向框 (x2 < x1)
    bboxes2 = torch.tensor([[[0.5, 0.5, 0.1, 0.1]]]).expand(2, 10, -1)
    check_no_crash(
        'reversed bboxes',
        lambda: tok(bboxes2, fpn),
    )


def test_box_tokenizer_many_bboxes():
    """大量 bbox 输入"""
    print('\n=== test_box_tokenizer_many_bboxes ===')
    tok = BoxTokenizer(feat_channels=256, num_fpn_levels=4, init_mode='zero')
    fpn = [torch.rand(2, 256, 20, 20) for _ in range(4)]
    bboxes = torch.rand(2, 500, 4)
    bboxes[..., 2] = bboxes[..., 0] + torch.rand(2, 500) * 0.3
    bboxes[..., 3] = bboxes[..., 1] + torch.rand(2, 500) * 0.3
    bboxes = bboxes.clamp(0, 1)
    tokens, levels = tok(bboxes, fpn)
    check('tokens shape', tokens.shape == (2, 500, 256))
    check('levels shape', levels.shape == (2, 500))
    check('finite', tokens.isfinite().all().item())


def test_box_tokenizer_fpn_level_assignment():
    """FPN 层级分配逻辑正确性"""
    print('\n=== test_box_tokenizer_fpn_level_assignment ===')
    tok = BoxTokenizer(feat_channels=256, num_fpn_levels=4, init_mode='zero')
    # 所有归一化 bbox 在 [0,1] 范围内, sqrt(wh) 很小, log2 为负, 会被 clamp 到 0
    # 验证各种输入都能正确产生合法层级索引
    test_cases = [
        torch.tensor([[[0.0, 0.0, 0.9, 0.9]]]),   # 大框
        torch.tensor([[[0.4, 0.4, 0.41, 0.41]]]),  # 小框
        torch.tensor([[[0.1, 0.1, 0.3, 0.3]]]),    # 中等框
    ]
    for i, bboxes in enumerate(test_cases):
        levels = tok._assign_fpn_level(bboxes)
        check(f'valid level index {i}', 0 <= levels.item() < 4)
        check(f'integer level {i}', levels.dtype == torch.long)


def test_box_tokenizer_gradient_flow():
    """梯度流验证"""
    print('\n=== test_box_tokenizer_gradient_flow ===')
    tok = BoxTokenizer(feat_channels=256, num_fpn_levels=4, init_mode='zero')
    tok.train()
    fpn = [torch.rand(2, 256, 20, 20, requires_grad=True) for _ in range(4)]
    bboxes = torch.rand(2, 10, 4)
    bboxes[..., 2] = bboxes[..., 0] + 0.1
    bboxes[..., 3] = bboxes[..., 1] + 0.1
    tokens, _ = tok(bboxes, fpn)
    loss = tokens.sum()
    loss.backward()
    check('bbox_pos_embed grad', tok.bbox_pos_embed[0].weight.grad is not None)
    check('level_embed grad', tok.level_embed.weight.grad is not None)


# ============================================================
# 2. deformable_attn 边界条件
# ============================================================
def test_deformable_attn_edge_cases():
    """极端采样位置"""
    print('\n=== test_deformable_attn_edge_cases ===')
    C = 256
    attn = MultiScaleDeformableAttention(
        embed_dim=C, num_heads=8, num_levels=4, num_points=8
    )
    bs, N = 2, 10
    fpn = [torch.rand(bs, C, 20 + l * 5, 20 + l * 5) for l in range(4)]
    flat, shapes, starts = flatten_fpn_features(fpn)

    # 极端参考点: 全 0
    query = torch.rand(bs, N, C)
    ref = torch.zeros(bs, N, 4, 2)
    out = attn(query, ref, flat, shapes, starts)
    check('zero ref: shape', out.shape == (bs, N, C))
    check('zero ref: finite', out.isfinite().all().item())

    # 极端参考点: 全 1
    ref = torch.ones(bs, N, 4, 2)
    out = attn(query, ref, flat, shapes, starts)
    check('ones ref: shape', out.shape == (bs, N, C))
    check('ones ref: finite', out.isfinite().all().item())

    # 极端参考点: 负值
    ref = torch.full((bs, N, 4, 2), -1.0)
    check_no_crash(
        'negative ref',
        lambda: attn(query, ref, flat, shapes, starts),
    )

    # 极端参考点: 超大值
    ref = torch.full((bs, N, 4, 2), 10.0)
    check_no_crash(
        'large ref',
        lambda: attn(query, ref, flat, shapes, starts),
    )


def test_deformable_attn_gradient_flow():
    """梯度流验证"""
    print('\n=== test_deformable_attn_gradient_flow ===')
    C = 256
    attn = MultiScaleDeformableAttention(
        embed_dim=C, num_heads=8, num_levels=4, num_points=8
    )
    attn.train()
    bs, N = 2, 10
    fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
    flat, shapes, starts = flatten_fpn_features(fpn)
    query = torch.rand(bs, N, C, requires_grad=True)
    ref = torch.rand(bs, N, 4, 2)
    out = attn(query, ref, flat, shapes, starts)
    loss = out.sum()
    loss.backward()
    check('query grad', query.grad is not None)
    check('query grad finite', query.grad.isfinite().all().item())


def test_deformable_attn_batch_size_one():
    """batch_size=1"""
    print('\n=== test_deformable_attn_batch_size_one ===')
    C = 256
    attn = MultiScaleDeformableAttention(
        embed_dim=C, num_heads=8, num_levels=4, num_points=8
    )
    bs, N = 1, 5
    fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
    flat, shapes, starts = flatten_fpn_features(fpn)
    query = torch.rand(bs, N, C)
    ref = torch.rand(bs, N, 4, 2)
    out = attn(query, ref, flat, shapes, starts)
    check('shape', out.shape == (bs, N, C))
    check('finite', out.isfinite().all().item())


# ============================================================
# 3. DiTBlock 边界条件
# ============================================================
def test_dit_block_gradient_flow():
    """梯度流验证"""
    print('\n=== test_dit_block_gradient_flow ===')
    C = 256
    block = DiTBlock(
        feat_channels=C, num_heads=8, num_fpn_levels=4,
        num_ref_points=8, dim_feedforward=2048, adaln_params=9,
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
    loss = out.sum()
    loss.backward()
    check('box_tokens grad', box_tokens.grad is not None)
    check('box_tokens grad finite', box_tokens.grad.isfinite().all().item())


def test_dit_block_adaln_params():
    """adaln_params=6 和 =9 都正常工作"""
    print('\n=== test_dit_block_adaln_params ===')
    C = 256
    for params in [6, 9]:
        block = DiTBlock(
            feat_channels=C, num_heads=8, num_fpn_levels=4,
            num_ref_points=8, dim_feedforward=2048, adaln_params=params,
        )
        bs, N = 2, 10
        time_dim = C * 4
        fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
        flat, shapes, starts = flatten_fpn_features(fpn)
        box_tokens = torch.rand(bs, N, C)
        time_emb = torch.rand(bs, time_dim)
        bbox_coords = torch.rand(bs, N, 4)
        bbox_coords[..., 2] = bbox_coords[..., 0] + 0.1
        bbox_coords[..., 3] = bbox_coords[..., 1] + 0.1
        out = block(box_tokens, flat, shapes, starts, time_emb, bbox_coords)
        check(f'adaln={params}: shape', out.shape == (bs, N, C))
        check(f'adaln={params}: finite', out.isfinite().all().item())


def test_dit_block_rope_applied():
    """验证 RoPE 确实被应用（不同位置产生不同输出）

    使用 use_adaln_zero=False 使 alpha 非零，确保 RoPE 效果可见。
    AdaLN-Zero 模式下 alpha 初始化为 0，残差贡献为 0，RoPE 无影响。
    """
    print('\n=== test_dit_block_rope_applied ===')
    C = 256
    block = DiTBlock(
        feat_channels=C, num_heads=8, num_fpn_levels=4,
        num_ref_points=8, dim_feedforward=2048, adaln_params=9,
        use_adaln_zero=False,
    )
    block.eval()
    bs, N = 2, 10
    time_dim = C * 4
    fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
    flat, shapes, starts = flatten_fpn_features(fpn)
    box_tokens = torch.rand(bs, N, C)
    time_emb = torch.rand(bs, time_dim)

    # 每个 token 使用不同的坐标, 使 RoPE 产生不同的旋转
    bbox1 = torch.zeros(bs, N, 4)
    for j in range(N):
        bbox1[:, j, 0] = 0.1 * j
        bbox1[:, j, 1] = 0.1 * j
        bbox1[:, j, 2] = 0.1 * j + 0.05
        bbox1[:, j, 3] = 0.1 * j + 0.05

    bbox2 = torch.zeros(bs, N, 4)
    for j in range(N):
        bbox2[:, j, 0] = 0.5 + 0.1 * j
        bbox2[:, j, 1] = 0.5 + 0.1 * j
        bbox2[:, j, 2] = 0.5 + 0.1 * j + 0.05
        bbox2[:, j, 3] = 0.5 + 0.1 * j + 0.05

    out1 = block(box_tokens, flat, shapes, starts, time_emb, bbox1)
    out2 = block(box_tokens, flat, shapes, starts, time_emb, bbox2)
    max_diff = (out1 - out2).abs().max().item()
    print(f'  RoPE max_diff = {max_diff:.10f}')
    check('RoPE produces different output for different coords',
          max_diff > 1e-6)


def test_dit_block_nan_input():
    """NaN 输入 — 不应崩溃"""
    print('\n=== test_dit_block_nan_input ===')
    C = 256
    block = DiTBlock(
        feat_channels=C, num_heads=8, num_fpn_levels=4,
        num_ref_points=8, dim_feedforward=2048, adaln_params=9,
    )
    bs, N = 2, 10
    time_dim = C * 4
    fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
    flat, shapes, starts = flatten_fpn_features(fpn)
    time_emb = torch.rand(bs, time_dim)
    bbox_coords = torch.rand(bs, N, 4)
    bbox_coords[..., 2] = bbox_coords[..., 0] + 0.1
    bbox_coords[..., 3] = bbox_coords[..., 1] + 0.1

    # NaN in box_tokens
    tokens_nan = torch.full((bs, N, C), float('nan'))
    check_no_crash(
        'NaN box_tokens',
        lambda: block(tokens_nan, flat, shapes, starts, time_emb, bbox_coords),
    )


# ============================================================
# 4. DiTSingleHead 边界条件
# ============================================================
def test_dit_single_head_velocity_no_duplication():
    """velocity_head 不应被重复创建"""
    print('\n=== test_dit_single_head_velocity_no_duplication ===')
    head = DiTSingleHead(
        num_classes=24, feat_channels=256, num_heads=8,
        num_fpn_levels=4, num_ref_points=8,
        prediction_mode='velocity', adaln_params=9, regression_mode='direct',
    )
    # velocity_head 应该在 __init__ 中创建一次
    check('velocity_head exists', head.velocity_head is not None)
    # 验证它是 nn.Sequential
    check('velocity_head is Sequential', isinstance(head.velocity_head, nn.Sequential))
    # 验证它有 7 层 (Linear+LN+ReLU+Linear+LN+ReLU+Linear)
    check('velocity_head has 7 layers', len(head.velocity_head) == 7)
    # 输出维度应为 4
    check('velocity_head output dim', head.velocity_head[-1].out_features == 4)


def test_dit_single_head_all_modes():
    """所有 prediction_mode + regression_mode 组合"""
    print('\n=== test_dit_single_head_all_modes ===')
    C = 256
    time_dim = C * 4
    bs, N = 2, 10
    fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
    flat, shapes, starts = flatten_fpn_features(fpn)
    time_emb = torch.rand(bs, time_dim)
    box_tokens = torch.rand(bs, N, C)
    bbox_coords = torch.rand(bs, N, 4)
    bbox_coords[..., 2] = bbox_coords[..., 0] + 0.1
    bbox_coords[..., 3] = bbox_coords[..., 1] + 0.1

    for pred_mode in ['x0', 'noise', 'velocity']:
        for reg_mode in ['direct', 'delta']:
            for use_obj in [False, True]:
                if pred_mode == 'velocity' and not use_obj:
                    continue  # velocity 模式通常需要 objectness
                head = DiTSingleHead(
                    num_classes=24, feat_channels=C, num_heads=8,
                    num_fpn_levels=4, num_ref_points=8,
                    prediction_mode=pred_mode, adaln_params=9,
                    regression_mode=reg_mode, use_objectness=use_obj,
                )
                cls, bbox, _, obj, vel = head(
                    box_tokens, flat, shapes, starts, time_emb, bbox_coords
                )
                check(f'{pred_mode}/{reg_mode}/obj={use_obj}: cls shape',
                      cls.shape == (bs, N, 24))
                check(f'{pred_mode}/{reg_mode}/obj={use_obj}: bbox shape',
                      bbox.shape == (bs, N, 4))
                check(f'{pred_mode}/{reg_mode}/obj={use_obj}: bbox finite',
                      bbox.isfinite().all().item())
                if pred_mode == 'velocity':
                    check(f'{pred_mode}/{reg_mode}/obj={use_obj}: vel not None',
                          vel is not None)


def test_dit_single_head_nan_input():
    """NaN 输入 — 不应崩溃"""
    print('\n=== test_dit_single_head_nan_input ===')
    C = 256
    head = DiTSingleHead(
        num_classes=24, feat_channels=C, num_heads=8,
        num_fpn_levels=4, num_ref_points=8,
        prediction_mode='x0', adaln_params=9, regression_mode='direct',
    )
    head.eval()
    bs, N = 2, 10
    time_dim = C * 4
    fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
    flat, shapes, starts = flatten_fpn_features(fpn)
    time_emb = torch.rand(bs, time_dim)
    bbox_coords = torch.rand(bs, N, 4)
    bbox_coords[..., 2] = bbox_coords[..., 0] + 0.1
    bbox_coords[..., 3] = bbox_coords[..., 1] + 0.1

    with torch.no_grad():
        tokens_nan = torch.full((bs, N, C), float('nan'))
        check_no_crash(
            'NaN tokens',
            lambda: head(tokens_nan, flat, shapes, starts, time_emb, bbox_coords),
        )


def test_dit_single_head_gradient_flow():
    """梯度流验证"""
    print('\n=== test_dit_single_head_gradient_flow ===')
    C = 256
    head = DiTSingleHead(
        num_classes=24, feat_channels=C, num_heads=8,
        num_fpn_levels=4, num_ref_points=8,
        prediction_mode='x0', adaln_params=9, regression_mode='direct',
    )
    head.train()
    bs, N = 2, 10
    time_dim = C * 4
    fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
    flat, shapes, starts = flatten_fpn_features(fpn)
    box_tokens = torch.rand(bs, N, C, requires_grad=True)
    time_emb = torch.rand(bs, time_dim)
    bbox_coords = torch.rand(bs, N, 4)
    bbox_coords[..., 2] = bbox_coords[..., 0] + 0.1
    bbox_coords[..., 3] = bbox_coords[..., 1] + 0.1
    cls, bbox, _, _, _ = head(
        box_tokens, flat, shapes, starts, time_emb, bbox_coords
    )
    loss = cls.sum() + bbox.sum()
    loss.backward()
    check('box_tokens grad', box_tokens.grad is not None)
    check('box_tokens grad finite', box_tokens.grad.isfinite().all().item())


# ============================================================
# 5. DiTDiffusionDetHead 边界条件
# ============================================================
def test_dit_head_forward_all_modes():
    """所有 prediction_mode 前向"""
    print('\n=== test_dit_head_forward_all_modes ===')
    for pred_mode in ['x0', 'noise', 'velocity']:
        head = DiTDiffusionDetHead(
            num_classes=24, feat_channels=256, num_proposals=50,
            num_heads=2, snr_scale=2.0, sampling_timesteps=2,
            diffusion_type='rectified_flow', solver_type='euler',
            rf_schedule='shifted', rf_shift=3.0,
            regression_mode='direct', prediction_mode=pred_mode,
            adaln_params=9, num_fpn_levels=4, num_ref_points=8,
            box_init_mode='zero', deep_supervision=False,
            criterion=DiffusionDetCriterion(
                num_classes=24,
                matcher=DiffusionDetMatcher(
                    match_costs=[
                        FocalLossCost(weight=2.0),
                        BBoxL1Cost(weight=5.0),
                        IoUCost(iou_mode='giou', weight=2.0),
                    ],
                    center_radius=0.5, candidate_topk=5,
                ),
                loss_cls=FocalLoss(loss_weight=2.0),
                loss_bbox=L1Loss(loss_weight=5.0),
                loss_giou=GIoULoss(loss_weight=2.0),
            ),
        )
        head.eval()
        bs, C = 2, 256
        fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
        img_metas = [
            ImageMeta(img_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])
            for _ in range(bs)
        ]
        x_noisy = torch.randn(bs, 50, 4)
        curr_bboxes = head._raw_to_xyxy(x_noisy, img_metas)
        t_input = torch.full((bs,), 500.0)
        with torch.no_grad():
            all_cls, all_bboxes, _, _, _ = head(
                fpn, curr_bboxes, t_input, img_metas=img_metas
            )
        check(f'{pred_mode}: cls seq shape', all_cls.shape == (1, bs, 50, 24))
        check(f'{pred_mode}: bbox seq shape', all_bboxes.shape == (1, bs, 50, 4))
        check(f'{pred_mode}: bbox finite', all_bboxes.isfinite().all().item())


def test_dit_head_loss_empty_gt():
    """空 GT 框的 loss 计算"""
    print('\n=== test_dit_head_loss_empty_gt ===')
    head = DiTDiffusionDetHead(
        num_classes=24, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        rf_schedule='shifted', rf_shift=3.0,
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=8,
        box_init_mode='zero', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=24,
            matcher=DiffusionDetMatcher(
                match_costs=[
                    FocalLossCost(weight=2.0),
                    BBoxL1Cost(weight=5.0),
                    IoUCost(iou_mode='giou', weight=2.0),
                ],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
        ),
    )
    head.train()
    bs, C = 2, 256
    fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
    img_metas = [
        ImageMeta(img_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])
        for _ in range(bs)
    ]
    # 空 GT
    gt_bboxes = [torch.zeros(0, 4) for _ in range(bs)]
    gt_labels = [torch.zeros(0, dtype=torch.long) for _ in range(bs)]
    check_no_crash(
        'empty GT',
        lambda: head.loss(fpn, img_metas, gt_bboxes, gt_labels),
    )


def test_dit_head_velocity_mode():
    """velocity 模式 loss 计算"""
    print('\n=== test_dit_head_velocity_mode ===')
    head = DiTDiffusionDetHead(
        num_classes=24, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        rf_schedule='shifted', rf_shift=3.0,
        regression_mode='direct', prediction_mode='velocity',
        adaln_params=9, num_fpn_levels=4, num_ref_points=8,
        box_init_mode='zero', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=24,
            matcher=DiffusionDetMatcher(
                match_costs=[
                    FocalLossCost(weight=2.0),
                    BBoxL1Cost(weight=5.0),
                    IoUCost(iou_mode='giou', weight=2.0),
                ],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
        ),
    )
    head.train()
    bs, C = 2, 256
    fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
    img_metas = [
        ImageMeta(img_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])
        for _ in range(bs)
    ]
    gt_bboxes = []
    gt_labels = []
    for i in range(bs):
        M = 5
        bboxes = torch.rand(M, 4)
        bboxes[..., 2] = bboxes[..., 0] + 0.05 + torch.rand(M) * 0.1
        bboxes[..., 3] = bboxes[..., 1] + 0.05 + torch.rand(M) * 0.1
        bboxes = bboxes.clamp(0, 1)
        gt_bboxes.append(bboxes * torch.tensor([1333, 800, 1333, 800]))
        gt_labels.append(torch.randint(0, 24, (M,)))
    losses = head.loss(fpn, img_metas, gt_bboxes, gt_labels)
    check('has loss_cls', 'loss_cls' in losses)
    check('has loss_bbox', 'loss_bbox' in losses)
    check('has loss_giou', 'loss_giou' in losses)
    check('has loss_velocity', 'loss_velocity' in losses)
    for k, v in losses.items():
        check(f'{k} finite', v.isfinite().item(), f'val={v.item()}')


def test_dit_head_ddpm_mode():
    """DDPM 扩散模式"""
    print('\n=== test_dit_head_ddpm_mode ===')
    head = DiTDiffusionDetHead(
        num_classes=24, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='ddpm', solver_type='ddim',
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=8,
        box_init_mode='zero', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=24,
            matcher=DiffusionDetMatcher(
                match_costs=[
                    FocalLossCost(weight=2.0),
                    BBoxL1Cost(weight=5.0),
                    IoUCost(iou_mode='giou', weight=2.0),
                ],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
        ),
    )
    head.eval()
    bs, C = 2, 256
    fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
    img_metas = [
        ImageMeta(img_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])
        for _ in range(bs)
    ]
    with torch.no_grad():
        results = head.predict(fpn, img_metas, rescale=False)
    check('results length', len(results) == bs)
    check('bboxes finite', results[0].bboxes.isfinite().all().item())


def test_dit_head_gradient_flow():
    """端到端梯度流验证"""
    print('\n=== test_dit_head_gradient_flow ===')
    head = DiTDiffusionDetHead(
        num_classes=24, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        rf_schedule='shifted', rf_shift=3.0,
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=8,
        box_init_mode='zero', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=24,
            matcher=DiffusionDetMatcher(
                match_costs=[
                    FocalLossCost(weight=2.0),
                    BBoxL1Cost(weight=5.0),
                    IoUCost(iou_mode='giou', weight=2.0),
                ],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
        ),
    )
    head.train()
    bs, C = 2, 256
    fpn = [torch.rand(bs, C, 20, 20) for _ in range(4)]
    img_metas = [
        ImageMeta(img_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])
        for _ in range(bs)
    ]
    gt_bboxes = []
    gt_labels = []
    for i in range(bs):
        M = 5
        bboxes = torch.rand(M, 4)
        bboxes[..., 2] = bboxes[..., 0] + 0.05 + torch.rand(M) * 0.1
        bboxes[..., 3] = bboxes[..., 1] + 0.05 + torch.rand(M) * 0.1
        bboxes = bboxes.clamp(0, 1)
        gt_bboxes.append(bboxes * torch.tensor([1333, 800, 1333, 800]))
        gt_labels.append(torch.randint(0, 24, (M,)))
    losses = head.loss(fpn, img_metas, gt_bboxes, gt_labels)
    total_loss = sum(losses.values())
    total_loss.backward()

    # 验证关键参数有梯度
    has_grad = False
    for name, param in head.named_parameters():
        if param.grad is not None and param.grad.isfinite().all().item():
            has_grad = True
            break
    check('has valid gradients', has_grad)


# ============================================================
# 6. Loss / Cost 边界条件
# ============================================================
def test_relative_l1_cost_nan_gt():
    """RelativeL1Cost: NaN GT 框不应崩溃 (修复验证)"""
    print('\n=== test_relative_l1_cost_nan_gt ===')
    N, M, C = 50, 5, 24
    pred_logits = torch.randn(N, C)
    pred_bboxes = torch.rand(N, 4)
    pred_bboxes[..., 2] = pred_bboxes[..., 0] + 0.1
    pred_bboxes[..., 3] = pred_bboxes[..., 1] + 0.1
    gt_labels = torch.randint(0, C, (M,))

    cost_fn = RelativeL1Cost(weight=5.0)

    # NaN GT
    gt_bboxes_nan = torch.full((M, 4), float('nan'))
    check_no_crash(
        'NaN GT bboxes',
        lambda: cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes_nan),
    )
    cost = cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes_nan)
    check('NaN GT: cost finite', cost.isfinite().all().item())

    # Inf GT
    gt_bboxes_inf = torch.full((M, 4), float('inf'))
    check_no_crash(
        'Inf GT bboxes',
        lambda: cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes_inf),
    )
    cost = cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes_inf)
    check('Inf GT: cost finite', cost.isfinite().all().item())

    # -Inf GT
    gt_bboxes_neginf = torch.full((M, 4), float('-inf'))
    check_no_crash(
        '-Inf GT bboxes',
        lambda: cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes_neginf),
    )


def test_relative_l1_cost_nan_pred():
    """RelativeL1Cost: NaN pred 框不应崩溃"""
    print('\n=== test_relative_l1_cost_nan_pred ===')
    N, M, C = 50, 5, 24
    pred_logits = torch.randn(N, C)
    gt_labels = torch.randint(0, C, (M,))
    gt_bboxes = torch.rand(M, 4)
    gt_bboxes[..., 2] = gt_bboxes[..., 0] + 0.1
    gt_bboxes[..., 3] = gt_bboxes[..., 1] + 0.1
    cost_fn = RelativeL1Cost(weight=5.0)

    pred_bboxes_nan = torch.full((N, 4), float('nan'))
    check_no_crash(
        'NaN pred bboxes',
        lambda: cost_fn(pred_logits, pred_bboxes_nan, gt_labels, gt_bboxes),
    )
    cost = cost_fn(pred_logits, pred_bboxes_nan, gt_labels, gt_bboxes)
    check('NaN pred: cost finite', cost.isfinite().all().item())


def test_relative_l1_cost_zero_gt():
    """RelativeL1Cost: 零面积 GT 框"""
    print('\n=== test_relative_l1_cost_zero_gt ===')
    N, M, C = 50, 5, 24
    pred_logits = torch.randn(N, C)
    pred_bboxes = torch.rand(N, 4)
    pred_bboxes[..., 2] = pred_bboxes[..., 0] + 0.1
    pred_bboxes[..., 3] = pred_bboxes[..., 1] + 0.1
    gt_labels = torch.randint(0, C, (M,))
    gt_bboxes = torch.zeros(M, 4)  # 零面积
    cost_fn = RelativeL1Cost(weight=5.0)
    check_no_crash(
        'zero GT',
        lambda: cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes),
    )
    cost = cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
    check('zero GT: cost finite', cost.isfinite().all().item())


def test_bbox_l1_cost_nan_gt():
    """BBoxL1Cost: NaN GT 框不应崩溃"""
    print('\n=== test_bbox_l1_cost_nan_gt ===')
    N, M, C = 50, 5, 24
    pred_logits = torch.randn(N, C)
    pred_bboxes = torch.rand(N, 4)
    pred_bboxes[..., 2] = pred_bboxes[..., 0] + 0.1
    pred_bboxes[..., 3] = pred_bboxes[..., 1] + 0.1
    gt_labels = torch.randint(0, C, (M,))
    gt_bboxes_nan = torch.full((M, 4), float('nan'))
    cost_fn = BBoxL1Cost(weight=5.0)
    check_no_crash(
        'NaN GT',
        lambda: cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes_nan),
    )


def test_iou_cost_nan_gt():
    """IoUCost: NaN GT 框不应崩溃"""
    print('\n=== test_iou_cost_nan_gt ===')
    N, M, C = 50, 5, 24
    pred_logits = torch.randn(N, C)
    pred_bboxes = torch.rand(N, 4)
    pred_bboxes[..., 2] = pred_bboxes[..., 0] + 0.1
    pred_bboxes[..., 3] = pred_bboxes[..., 1] + 0.1
    gt_labels = torch.randint(0, C, (M,))
    gt_bboxes_nan = torch.full((M, 4), float('nan'))
    cost_fn = IoUCost(iou_mode='giou', weight=2.0)
    check_no_crash(
        'NaN GT',
        lambda: cost_fn(pred_logits, pred_bboxes, gt_labels, gt_bboxes_nan),
    )


def test_focal_loss_cost_nan_logits():
    """FocalLossCost: NaN logits"""
    print('\n=== test_focal_loss_cost_nan_logits ===')
    N, M, C = 50, 5, 24
    pred_logits_nan = torch.full((N, C), float('nan'))
    pred_bboxes = torch.rand(N, 4)
    pred_bboxes[..., 2] = pred_bboxes[..., 0] + 0.1
    pred_bboxes[..., 3] = pred_bboxes[..., 1] + 0.1
    gt_labels = torch.randint(0, C, (M,))
    gt_bboxes = torch.rand(M, 4)
    gt_bboxes[..., 2] = gt_bboxes[..., 0] + 0.1
    gt_bboxes[..., 3] = gt_bboxes[..., 1] + 0.1
    cost_fn = FocalLossCost(weight=2.0)
    check_no_crash(
        'NaN logits',
        lambda: cost_fn(pred_logits_nan, pred_bboxes, gt_labels, gt_bboxes),
    )


def test_matcher_empty_gt():
    """Matcher: 空 GT"""
    print('\n=== test_matcher_empty_gt ===')
    N, C = 50, 24
    pred_logits = torch.randn(1, N, C)
    pred_bboxes = torch.rand(1, N, 4)
    pred_bboxes[..., 2] = pred_bboxes[..., 0] + 0.1
    pred_bboxes[..., 3] = pred_bboxes[..., 1] + 0.1
    targets = [
        InstanceData(
            labels=torch.zeros(0, dtype=torch.long),
            bboxes=torch.zeros(0, 4),
            img_shape=(800, 1333),
        )
    ]
    matcher = DiffusionDetMatcher(
        match_costs=[
            FocalLossCost(weight=2.0),
            BBoxL1Cost(weight=5.0),
            IoUCost(iou_mode='giou', weight=2.0),
        ],
        center_radius=0.5,
        candidate_topk=5,
    )
    outputs = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_bboxes)
    indices = matcher(outputs, targets)
    check('empty GT: src_idx empty', len(indices[0][0]) == 0)
    check('empty GT: gt_idx empty', len(indices[0][1]) == 0)


def test_criterion_all_bbox_loss_modes():
    """所有 bbox_loss_mode"""
    print('\n=== test_criterion_all_bbox_loss_modes ===')
    N, M, C, bs = 50, 5, 24, 2
    pred_logits = torch.randn(bs, N, C)
    pred_bboxes = torch.rand(bs, N, 4)
    pred_bboxes[..., 2] = pred_bboxes[..., 0] + 0.1
    pred_bboxes[..., 3] = pred_bboxes[..., 1] + 0.1
    targets = []
    for i in range(bs):
        gt_bboxes_i = torch.rand(M, 4)
        gt_bboxes_i[..., 2] = gt_bboxes_i[..., 0] + 0.1
        gt_bboxes_i[..., 3] = gt_bboxes_i[..., 1] + 0.1
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
            RelativeL1Cost(weight=5.0),
            IoUCost(iou_mode='giou', weight=2.0),
        ],
        center_radius=0.5,
        candidate_topk=5,
    )
    for mode in ['l1', 'relative_l1', 'gcd', 'mixed_relative_l1']:
        criterion = DiffusionDetCriterion(
            num_classes=C,
            matcher=matcher,
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            deep_supervision=False,
            bbox_loss_mode=mode,
        )
        outputs = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_bboxes)
        losses = criterion(outputs, targets)
        check(f'{mode}: loss finite',
              losses['loss_cls'].isfinite().item()
              and losses['loss_bbox'].isfinite().item()
              and losses['loss_giou'].isfinite().item())


def test_velocity_loss_nan():
    """FlowMatchingVelocityLoss: NaN 输入"""
    print('\n=== test_velocity_loss_nan ===')
    vel_loss = FlowMatchingVelocityLoss(loss_weight=5.0)
    v_pred = torch.full((50, 4), float('nan'))
    v_target = torch.randn(50, 4)
    check_no_crash(
        'NaN pred',
        lambda: vel_loss(v_pred, v_target),
    )
    v_pred = torch.randn(50, 4)
    v_target = torch.full((50, 4), float('nan'))
    check_no_crash(
        'NaN target',
        lambda: vel_loss(v_pred, v_target),
    )


# ============================================================
# 7. 设备兼容性
# ============================================================
def test_device_compatibility():
    """CPU 设备兼容性"""
    print('\n=== test_device_compatibility ===')
    device = torch.device('cpu')
    C = 256

    # BoxTokenizer
    tok = BoxTokenizer(feat_channels=C, num_fpn_levels=4, init_mode='zero')
    tok = tok.to(device)
    fpn = [torch.rand(2, C, 20, 20, device=device) for _ in range(4)]
    bboxes = torch.rand(2, 10, 4, device=device)
    bboxes[..., 2] = bboxes[..., 0] + 0.1
    bboxes[..., 3] = bboxes[..., 1] + 0.1
    tokens, _ = tok(bboxes, fpn)
    check('box_tokenizer CPU', tokens.device == device)

    # DiTBlock
    block = DiTBlock(
        feat_channels=C, num_heads=8, num_fpn_levels=4,
        num_ref_points=8, dim_feedforward=2048, adaln_params=9,
    ).to(device)
    flat, shapes, starts = flatten_fpn_features(fpn)
    box_tokens = torch.rand(2, 10, C, device=device)
    time_emb = torch.rand(2, C * 4, device=device)
    bbox_coords = torch.rand(2, 10, 4, device=device)
    bbox_coords[..., 2] = bbox_coords[..., 0] + 0.1
    bbox_coords[..., 3] = bbox_coords[..., 1] + 0.1
    out = block(box_tokens, flat, shapes, starts, time_emb, bbox_coords)
    check('dit_block CPU', out.device == device)

    # DeformableAttn
    attn = MultiScaleDeformableAttention(
        embed_dim=C, num_heads=8, num_levels=4, num_points=8,
    ).to(device)
    query = torch.rand(2, 10, C, device=device)
    ref = torch.rand(2, 10, 4, 2, device=device)
    out = attn(query, ref, flat, shapes, starts)
    check('deformable_attn CPU', out.device == device)


# ============================================================
# 8. 确定性测试
# ============================================================
def test_determinism():
    """相同输入产生相同输出"""
    print('\n=== test_determinism ===')
    torch.manual_seed(42)
    C = 256

    tok = BoxTokenizer(feat_channels=C, num_fpn_levels=4, init_mode='zero')
    tok.eval()
    fpn = [torch.rand(2, C, 20, 20) for _ in range(4)]
    bboxes = torch.rand(2, 10, 4)
    bboxes[..., 2] = bboxes[..., 0] + 0.1
    bboxes[..., 3] = bboxes[..., 1] + 0.1

    torch.manual_seed(42)
    tokens1, _ = tok(bboxes, fpn)
    torch.manual_seed(42)
    tokens2, _ = tok(bboxes, fpn)
    check('deterministic output', torch.allclose(tokens1, tokens2, atol=1e-6))


# ============================================================
# 运行入口
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('LDMDet-DiT 边界条件 & 鲁棒性测试')
    print('=' * 60)

    test_box_tokenizer_empty()
    test_box_tokenizer_single()
    test_box_tokenizer_nan_bboxes()
    test_box_tokenizer_inf_bboxes()
    test_box_tokenizer_neg_inf_bboxes()
    test_box_tokenizer_degenerate_bboxes()
    test_box_tokenizer_many_bboxes()
    test_box_tokenizer_fpn_level_assignment()
    test_box_tokenizer_gradient_flow()

    test_deformable_attn_edge_cases()
    test_deformable_attn_gradient_flow()
    test_deformable_attn_batch_size_one()

    test_dit_block_gradient_flow()
    test_dit_block_adaln_params()
    test_dit_block_rope_applied()
    test_dit_block_nan_input()

    test_dit_single_head_velocity_no_duplication()
    test_dit_single_head_all_modes()
    test_dit_single_head_nan_input()
    test_dit_single_head_gradient_flow()

    test_dit_head_forward_all_modes()
    test_dit_head_loss_empty_gt()
    test_dit_head_velocity_mode()
    test_dit_head_ddpm_mode()
    test_dit_head_gradient_flow()

    test_relative_l1_cost_nan_gt()
    test_relative_l1_cost_nan_pred()
    test_relative_l1_cost_zero_gt()
    test_bbox_l1_cost_nan_gt()
    test_iou_cost_nan_gt()
    test_focal_loss_cost_nan_logits()
    test_matcher_empty_gt()
    test_criterion_all_bbox_loss_modes()
    test_velocity_loss_nan()

    test_device_compatibility()
    test_determinism()

    total = PASS + FAIL + ERROR
    print('\n' + '=' * 60)
    print(f'RESULTS: PASS={PASS}  FAIL={FAIL}  ERROR={ERROR}  TOTAL={total}')
    print('=' * 60)

    if FAIL > 0 or ERROR > 0:
        sys.exit(1)