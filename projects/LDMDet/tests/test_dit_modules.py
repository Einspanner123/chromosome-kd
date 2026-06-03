import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, 'projects/LDMDet')

from mods.utils import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy, bbox2roi
from mods.structures import InstanceData, ModelOutput, ImageMeta, DetectionResult
from mods.modules import SinusoidalPositionEmbeddings, cosine_noise_schedule, load_buffer
from mods.box_tokenizer import BoxTokenizer, bbox_to_reference_points, reference_points_with_levels
from mods.deformable_attn import (
    MultiScaleDeformableAttention,
    build_fpn_level_info,
    flatten_fpn_features,
)
from mods.dit_block import DiTBlock
from mods.dit_single_head import DiTSingleHead
from mods.rectified_flow import RectifiedFlow, RFDPMSolverMultistep
from mods.loss import (
    sigmoid_focal_loss,
    FocalLoss,
    GIoULoss,
    L1Loss,
    FocalLossCost,
    BBoxL1Cost,
    RelativeL1Cost,
    IoUCost,
    DiffusionDetMatcher,
    DiffusionDetCriterion,
    FlowMatchingVelocityLoss,
)
from mods.dit_head import DiTDiffusionDetHead

PASS = 0
FAIL = 0


def check(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f'  [PASS] {name}')
    else:
        FAIL += 1
        print(f'  [FAIL] {name} {detail}')


# ============================================================
# 1. utils: bbox_xyxy_to_cxcywh / bbox_cxcywh_to_xyxy
# ============================================================
def test_bbox_conversion():
    print('\n=== test_bbox_conversion ===')
    xyxy = torch.tensor([[10.0, 20.0, 30.0, 60.0]])
    cxcywh = bbox_xyxy_to_cxcywh(xyxy)
    check('cx', torch.isclose(cxcywh[0, 0], torch.tensor(20.0)))
    check('cy', torch.isclose(cxcywh[0, 1], torch.tensor(40.0)))
    check('w', torch.isclose(cxcywh[0, 2], torch.tensor(20.0)))
    check('h', torch.isclose(cxcywh[0, 3], torch.tensor(40.0)))

    back = bbox_cxcywh_to_xyxy(cxcywh)
    check('roundtrip', torch.allclose(xyxy, back, atol=1e-6))

    batch = torch.rand(2, 50, 4)
    batch_cxcywh = bbox_xyxy_to_cxcywh(batch)
    check('batch roundtrip', torch.allclose(batch, bbox_cxcywh_to_xyxy(batch_cxcywh), atol=1e-6))


# ============================================================
# 2. SinusoidalPositionEmbeddings
# ============================================================
def test_sinusoidal_embeddings():
    print('\n=== test_sinusoidal_embeddings ===')
    dim = 128
    emb = SinusoidalPositionEmbeddings(dim)
    t = torch.arange(10, dtype=torch.float32)
    out = emb(t)
    check('output shape', out.shape == (10, dim))
    check('finite', out.isfinite().all().item())
    check('not all zero', (out.abs() > 1e-6).any().item())


# ============================================================
# 3. cosine_noise_schedule
# ============================================================
def test_cosine_schedule():
    print('\n=== test_cosine_schedule ===')
    betas = cosine_noise_schedule(1000)
    check('shape', betas.shape == (1000,))
    check('range', (betas >= 0).all().item() and (betas <= 0.999).all().item())
    check('monotonic trend', betas[-1] > betas[0])


# ============================================================
# 4. BoxTokenizer
# ============================================================
def test_box_tokenizer():
    print('\n=== test_box_tokenizer ===')
    bs, N, C, num_levels = 2, 10, 256, 4
    fpn_feats = [torch.rand(bs, C, 100 + l * 20, 100 + l * 20) for l in range(num_levels)]
    bboxes = torch.rand(bs, N, 4)
    bboxes[:, :, 2] = bboxes[:, :, 0] + torch.rand(bs, N) * 0.3
    bboxes[:, :, 3] = bboxes[:, :, 1] + torch.rand(bs, N) * 0.3
    bboxes = bboxes.clamp(0, 1)

    tok = BoxTokenizer(feat_channels=C, num_fpn_levels=num_levels, init_mode='bilinear')
    tokens, levels = tok(bboxes, fpn_feats)
    check('tokens shape', tokens.shape == (bs, N, C))
    check('levels shape', levels.shape == (bs, N))
    check('levels range', (levels >= 0).all().item() and (levels < num_levels).all().item())
    check('finite', tokens.isfinite().all().item())

    tok_learn = BoxTokenizer(feat_channels=C, num_fpn_levels=num_levels, init_mode='learnable')
    tokens_l, levels_l = tok_learn(bboxes, fpn_feats)
    check('learnable tokens shape', tokens_l.shape == (bs, N, C))
    check('learnable finite', tokens_l.isfinite().all().item())


# ============================================================
# 5. bbox_to_reference_points / reference_points_with_levels
# ============================================================
def test_reference_points():
    print('\n=== test_reference_points ===')
    bboxes = torch.tensor([[[0.1, 0.2, 0.3, 0.6]]])
    ref = bbox_to_reference_points(bboxes)
    check('ref shape', ref.shape == (1, 1, 2))
    check('cx', torch.isclose(ref[0, 0, 0], torch.tensor(0.2)))
    check('cy', torch.isclose(ref[0, 0, 1], torch.tensor(0.4)))

    ref_multi = reference_points_with_levels(ref, 4)
    check('multi shape', ref_multi.shape == (1, 1, 4, 2))
    check('multi same', torch.allclose(ref_multi[0, 0, 0], ref[0, 0]))


# ============================================================
# 6. flatten_fpn_features / build_fpn_level_info
# ============================================================
def test_fpn_utils():
    print('\n=== test_fpn_utils ===')
    bs, C = 2, 256
    fpn_feats = [torch.rand(bs, C, 100, 100), torch.rand(bs, C, 50, 50), torch.rand(bs, C, 25, 25), torch.rand(bs, C, 13, 13)]

    flat, shapes, starts = flatten_fpn_features(fpn_feats)
    total = 100 * 100 + 50 * 50 + 25 * 25 + 13 * 13
    check('flat shape', flat.shape == (bs, total, C))
    check('shapes shape', shapes.shape == (4, 2))
    check('starts shape', starts.shape == (4,))
    check('starts[0]', starts[0].item() == 0)
    check('starts[1]', starts[1].item() == 100 * 100)

    shapes2, starts2 = build_fpn_level_info([(100, 100), (50, 50), (25, 25), (13, 13)], torch.device('cpu'))
    check('build shapes match', torch.equal(shapes, shapes2))
    check('build starts match', torch.equal(starts, starts2))


# ============================================================
# 7. MultiScaleDeformableAttention
# ============================================================
def test_deformable_attn():
    print('\n=== test_deformable_attn ===')
    bs, N, C, num_heads, num_levels, num_points = 2, 10, 256, 8, 4, 8
    attn = MultiScaleDeformableAttention(
        embed_dim=C, num_heads=num_heads, num_levels=num_levels, num_points=num_points
    )

    query = torch.rand(bs, N, C)
    ref_points = torch.rand(bs, N, num_levels, 2)
    fpn_feats = [torch.rand(bs, C, 20 + l * 5, 20 + l * 5) for l in range(num_levels)]
    flat, shapes, starts = flatten_fpn_features(fpn_feats)

    out = attn(query, ref_points, flat, shapes, starts)
    check('output shape', out.shape == (bs, N, C))
    check('finite', out.isfinite().all().item())


# ============================================================
# 8. DiTBlock
# ============================================================
def test_dit_block():
    print('\n=== test_dit_block ===')
    bs, N, C = 2, 10, 256
    num_heads, num_levels, num_points = 8, 4, 8
    time_dim = C * 4

    block = DiTBlock(
        feat_channels=C, num_heads=num_heads, num_fpn_levels=num_levels,
        num_ref_points=num_points, dim_feedforward=2048, adaln_params=9,
    )

    box_tokens = torch.rand(bs, N, C)
    fpn_feats = [torch.rand(bs, C, 20 + l * 5, 20 + l * 5) for l in range(num_levels)]
    flat, shapes, starts = flatten_fpn_features(fpn_feats)
    time_emb = torch.rand(bs, time_dim)
    bbox_coords = torch.rand(bs, N, 4)
    bbox_coords[:, :, 2] = bbox_coords[:, :, 0] + torch.rand(bs, N) * 0.3
    bbox_coords[:, :, 3] = bbox_coords[:, :, 1] + torch.rand(bs, N) * 0.3
    bbox_coords = bbox_coords.clamp(0, 1)

    out = block(box_tokens, flat, shapes, starts, time_emb, bbox_coords)
    check('output shape', out.shape == (bs, N, C))
    check('finite', out.isfinite().all().item())

    block6 = DiTBlock(
        feat_channels=C, num_heads=num_heads, num_fpn_levels=num_levels,
        num_ref_points=num_points, dim_feedforward=2048, adaln_params=6,
    )
    out6 = block6(box_tokens, flat, shapes, starts, time_emb, bbox_coords)
    check('adaln6 output shape', out6.shape == (bs, N, C))
    check('adaln6 finite', out6.isfinite().all().item())


# ============================================================
# 9. DiTSingleHead (direct mode)
# ============================================================
def test_dit_single_head_direct():
    print('\n=== test_dit_single_head_direct ===')
    bs, N, C, num_classes = 2, 10, 256, 24
    num_heads, num_levels, num_points = 8, 4, 8
    time_dim = C * 4

    head = DiTSingleHead(
        num_classes=num_classes, feat_channels=C, num_heads=num_heads,
        num_fpn_levels=num_levels, num_ref_points=num_points,
        prediction_mode='x0', adaln_params=9, regression_mode='direct',
    )

    box_tokens = torch.rand(bs, N, C)
    fpn_feats = [torch.rand(bs, C, 20 + l * 5, 20 + l * 5) for l in range(num_levels)]
    flat, shapes, starts = flatten_fpn_features(fpn_feats)
    time_emb = torch.rand(bs, time_dim)
    bbox_coords = torch.rand(bs, N, 4)
    bbox_coords[:, :, 2] = bbox_coords[:, :, 0] + torch.rand(bs, N) * 0.3
    bbox_coords[:, :, 3] = bbox_coords[:, :, 1] + torch.rand(bs, N) * 0.3
    bbox_coords = bbox_coords.clamp(0, 1)

    cls_logits, pred_bboxes, updated_tokens, obj, vel = head(
        box_tokens, flat, shapes, starts, time_emb, bbox_coords
    )
    check('cls shape', cls_logits.shape == (bs, N, num_classes))
    check('bbox shape', pred_bboxes.shape == (bs, N, 4))
    check('tokens shape', updated_tokens.shape == (bs, N, C))
    check('obj is None', obj is None)
    check('vel is None', vel is None)
    check('bbox in [0,1]', pred_bboxes.min().item() >= 0 and pred_bboxes.max().item() <= 1)
    check('x2>=x1', (pred_bboxes[..., 2] >= pred_bboxes[..., 0] - 1e-6).all().item())
    check('y2>=y1', (pred_bboxes[..., 3] >= pred_bboxes[..., 1] - 1e-6).all().item())
    check('finite', cls_logits.isfinite().all().item() and pred_bboxes.isfinite().all().item())


# ============================================================
# 10. DiTSingleHead (delta mode)
# ============================================================
def test_dit_single_head_delta():
    print('\n=== test_dit_single_head_delta ===')
    bs, N, C, num_classes = 2, 10, 256, 24
    num_heads, num_levels, num_points = 8, 4, 8
    time_dim = C * 4

    head = DiTSingleHead(
        num_classes=num_classes, feat_channels=C, num_heads=num_heads,
        num_fpn_levels=num_levels, num_ref_points=num_points,
        prediction_mode='x0', adaln_params=9, regression_mode='delta',
    )

    box_tokens = torch.rand(bs, N, C)
    fpn_feats = [torch.rand(bs, C, 20 + l * 5, 20 + l * 5) for l in range(num_levels)]
    flat, shapes, starts = flatten_fpn_features(fpn_feats)
    time_emb = torch.rand(bs, time_dim)
    bbox_coords = torch.rand(bs, N, 4)
    bbox_coords[:, :, 2] = bbox_coords[:, :, 0] + 0.05 + torch.rand(bs, N) * 0.3
    bbox_coords[:, :, 3] = bbox_coords[:, :, 1] + 0.05 + torch.rand(bs, N) * 0.3
    bbox_coords = bbox_coords.clamp(0, 1)

    cls_logits, pred_bboxes, _, _, _ = head(
        box_tokens, flat, shapes, starts, time_emb, bbox_coords
    )
    check('cls shape', cls_logits.shape == (bs, N, num_classes))
    check('bbox shape', pred_bboxes.shape == (bs, N, 4))
    check('bbox in [0,1]', pred_bboxes.min().item() >= 0 and pred_bboxes.max().item() <= 1)
    check('finite', pred_bboxes.isfinite().all().item())


# ============================================================
# 11. DiTSingleHead (velocity mode + objectness)
# ============================================================
def test_dit_single_head_velocity():
    print('\n=== test_dit_single_head_velocity ===')
    bs, N, C, num_classes = 2, 10, 256, 24
    num_heads, num_levels, num_points = 8, 4, 8
    time_dim = C * 4

    head = DiTSingleHead(
        num_classes=num_classes, feat_channels=C, num_heads=num_heads,
        num_fpn_levels=num_levels, num_ref_points=num_points,
        prediction_mode='velocity', adaln_params=9, regression_mode='direct',
        use_objectness=True,
    )

    box_tokens = torch.rand(bs, N, C)
    fpn_feats = [torch.rand(bs, C, 20 + l * 5, 20 + l * 5) for l in range(num_levels)]
    flat, shapes, starts = flatten_fpn_features(fpn_feats)
    time_emb = torch.rand(bs, time_dim)
    bbox_coords = torch.rand(bs, N, 4)
    bbox_coords[:, :, 2] = bbox_coords[:, :, 0] + torch.rand(bs, N) * 0.3
    bbox_coords[:, :, 3] = bbox_coords[:, :, 1] + torch.rand(bs, N) * 0.3
    bbox_coords = bbox_coords.clamp(0, 1)

    cls_logits, pred_bboxes, _, obj, vel = head(
        box_tokens, flat, shapes, starts, time_emb, bbox_coords
    )
    check('obj not None', obj is not None)
    check('obj shape', obj.shape == (bs, N, 1))
    check('vel not None', vel is not None)
    check('vel shape', vel.shape == (bs, N, 4))
    check('finite', obj.isfinite().all().item() and vel.isfinite().all().item())


# ============================================================
# 12. RectifiedFlow
# ============================================================
def test_rectified_flow():
    print('\n=== test_rectified_flow ===')
    rf = RectifiedFlow(snr_scale=2.0)

    x0 = torch.randn(8, 4)
    t = torch.rand(8)
    x_noise = torch.randn_like(x0)
    x_t, v = rf.q_sample(x0, x_noise=x_noise, t=t)
    check('x_t shape', x_t.shape == (8, 4))
    check('v shape', v.shape == (8, 4))
    t_view = t.view(-1, 1)
    expected_x_t = (1 - t_view) * x0 + t_view * x_noise
    check('x_t == (1-t)*x0 + t*x_noise', torch.allclose(x_t, expected_x_t, atol=1e-5))
    check('v == x_noise - x0', torch.allclose(v, x_noise - x0, atol=1e-5))

    x0_pred = x0 + torch.randn_like(x0) * 0.1
    x_next = rf.step(x_t, x0_pred, 0.5, 0.25)
    check('step shape', x_next.shape == (8, 4))
    check('step finite', x_next.isfinite().all().item())

    x_noise = torch.randn_like(x0)
    x_t2, v2 = rf.q_sample(x0, x_noise=x_noise, t=t)
    check('q_sample with noise shape', x_t2.shape == (8, 4))


# ============================================================
# 13. RFDPMSolverMultistep
# ============================================================
def test_dpm_solver():
    print('\n=== test_dpm_solver ===')
    solver = RFDPMSolverMultistep(num_steps=4, solver_order=2)
    check('timesteps count', len(solver.timesteps) == 5)
    check('t_start=1', abs(solver.timesteps[0] - 1.0) < 1e-6)
    check('t_end=0', abs(solver.timesteps[-1] - 0.0) < 1e-6)

    x = torch.randn(8, 4)
    x0 = torch.randn(8, 4)
    solver.reset()
    x_next = solver.step(x, x0, solver.timesteps[0], 0)
    check('step shape', x_next.shape == (8, 4))
    check('step finite', x_next.isfinite().all().item())

    solver.reset()
    x_next2 = solver.step(x_next, x0, solver.timesteps[1], 1)
    check('second step finite', x_next2.isfinite().all().item())


# ============================================================
# 14. Loss functions
# ============================================================
def test_loss_functions():
    print('\n=== test_loss_functions ===')
    N, C = 50, 24

    pred_logits = torch.randn(N, C)
    pred_bboxes = torch.rand(N, 4)
    pred_bboxes[:, 2] = pred_bboxes[:, 0] + 0.1
    pred_bboxes[:, 3] = pred_bboxes[:, 1] + 0.1
    gt_labels = torch.randint(0, C, (N,))
    gt_bboxes = torch.rand(N, 4)
    gt_bboxes[:, 2] = gt_bboxes[:, 0] + 0.1
    gt_bboxes[:, 3] = gt_bboxes[:, 1] + 0.1

    fl = FocalLoss(loss_weight=2.0)
    target_onehot = torch.zeros(N, C)
    target_onehot.scatter_(1, gt_labels.unsqueeze(1), 1.0)
    loss_cls = fl(pred_logits, target_onehot)
    check('focal loss scalar', loss_cls.dim() == 0)
    check('focal loss finite', loss_cls.isfinite().item())
    check('focal loss > 0', loss_cls.item() > 0)

    loss_cls_idx = fl(pred_logits, gt_labels)
    check('focal loss idx match', torch.isclose(loss_cls, loss_cls_idx, atol=1e-4))

    l1 = L1Loss(loss_weight=5.0)
    loss_l1 = l1(pred_bboxes, gt_bboxes)
    check('l1 loss scalar', loss_l1.dim() == 0)
    check('l1 loss > 0', loss_l1.item() > 0)

    giou = GIoULoss(loss_weight=2.0)
    loss_giou = giou(pred_bboxes, gt_bboxes)
    check('giou loss scalar', loss_giou.dim() == 0)
    check('giou loss finite', loss_giou.isfinite().item())

    vel_loss = FlowMatchingVelocityLoss(loss_weight=5.0)
    v_pred = torch.randn(N, 4)
    v_target = torch.randn(N, 4)
    loss_vel = vel_loss(v_pred, v_target)
    check('velocity loss scalar', loss_vel.dim() == 0)
    check('velocity loss > 0', loss_vel.item() > 0)


# ============================================================
# 15. Cost functions
# ============================================================
def test_cost_functions():
    print('\n=== test_cost_functions ===')
    N, M, C = 50, 5, 24

    pred_logits = torch.randn(N, C)
    pred_bboxes = torch.rand(N, 4)
    pred_bboxes[:, 2] = pred_bboxes[:, 0] + 0.1
    pred_bboxes[:, 3] = pred_bboxes[:, 1] + 0.1
    gt_labels = torch.randint(0, C, (M,))
    gt_bboxes = torch.rand(M, 4)
    gt_bboxes[:, 2] = gt_bboxes[:, 0] + 0.1
    gt_bboxes[:, 3] = gt_bboxes[:, 1] + 0.1

    fl_cost = FocalLossCost(weight=2.0)
    cost_cls = fl_cost(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
    check('focal cost shape', cost_cls.shape == (N, M))
    check('focal cost finite', cost_cls.isfinite().all().item())

    l1_cost = BBoxL1Cost(weight=5.0)
    cost_l1 = l1_cost(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
    check('l1 cost shape', cost_l1.shape == (N, M))
    check('l1 cost finite', cost_l1.isfinite().all().item())

    rel_cost = RelativeL1Cost(weight=5.0)
    cost_rel = rel_cost(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
    check('rel l1 cost shape', cost_rel.shape == (N, M))
    check('rel l1 cost finite', cost_rel.isfinite().all().item())
    check('rel l1 cost > 0', (cost_rel > 0).all().item())

    iou_cost = IoUCost(iou_mode='giou', weight=2.0)
    cost_iou = iou_cost(pred_logits, pred_bboxes, gt_labels, gt_bboxes)
    check('iou cost shape', cost_iou.shape == (N, M))
    check('iou cost finite', cost_iou.isfinite().all().item())

    same_bboxes = gt_bboxes.clone()
    cost_same = l1_cost(pred_logits[:M], same_bboxes, gt_labels, gt_bboxes)
    check('l1 cost self ~0', cost_same.diagonal().max().item() < 0.01)


# ============================================================
# 16. DiffusionDetMatcher
# ============================================================
def test_matcher():
    print('\n=== test_matcher ===')
    N, M, C = 50, 5, 24
    bs = 2

    pred_logits = torch.randn(bs, N, C)
    pred_bboxes = torch.rand(bs, N, 4)
    pred_bboxes[:, :, 2] = pred_bboxes[:, :, 0] + 0.1
    pred_bboxes[:, :, 3] = pred_bboxes[:, :, 1] + 0.1

    targets = []
    for i in range(bs):
        gt_bboxes_i = torch.rand(M, 4)
        gt_bboxes_i[:, 2] = gt_bboxes_i[:, 0] + 0.1
        gt_bboxes_i[:, 3] = gt_bboxes_i[:, 1] + 0.1
        targets.append(InstanceData(
            labels=torch.randint(0, C, (M,)),
            bboxes=gt_bboxes_i,
            img_shape=(800, 1333),
        ))

    matcher = DiffusionDetMatcher(
        match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
        center_radius=0.5,
        candidate_topk=5,
    )
    outputs = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_bboxes)
    indices = matcher(outputs, targets)
    check('indices length', len(indices) == bs)
    for i, (src_idx, gt_idx) in enumerate(indices):
        check(f'batch {i} src_idx unique', len(src_idx) == len(src_idx.unique()))
        check(f'batch {i} gt_idx in range', (gt_idx < M).all().item())
        check(f'batch {i} matched > 0', len(src_idx) > 0)

    matcher_l1 = DiffusionDetMatcher(
        match_costs=[FocalLossCost(weight=2.0), BBoxL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
        center_radius=2.5,
        candidate_topk=5,
    )
    indices_l1 = matcher_l1(outputs, targets)
    check('l1 matcher indices length', len(indices_l1) == bs)

    targets_empty = [InstanceData(labels=torch.zeros(0, dtype=torch.long), bboxes=torch.zeros(0, 4), img_shape=(800, 1333))]
    outputs_empty = ModelOutput(pred_logits=pred_logits[:1], pred_boxes=pred_bboxes[:1])
    indices_empty = matcher(outputs_empty, targets_empty)
    check('empty gt src_idx', len(indices_empty[0][0]) == 0)
    check('empty gt gt_idx', len(indices_empty[0][1]) == 0)


# ============================================================
# 17. DiffusionDetCriterion (with relative_l1 bbox_loss_mode)
# ============================================================
def test_criterion():
    print('\n=== test_criterion ===')
    N, M, C, bs = 50, 5, 24, 2

    pred_logits = torch.randn(bs, N, C)
    pred_bboxes = torch.rand(bs, N, 4)
    pred_bboxes[:, :, 2] = pred_bboxes[:, :, 0] + 0.1
    pred_bboxes[:, :, 3] = pred_bboxes[:, :, 1] + 0.1

    targets = []
    for i in range(bs):
        gt_bboxes_i = torch.rand(M, 4)
        gt_bboxes_i[:, 2] = gt_bboxes_i[:, 0] + 0.1
        gt_bboxes_i[:, 3] = gt_bboxes_i[:, 1] + 0.1
        targets.append(InstanceData(
            labels=torch.randint(0, C, (M,)),
            bboxes=gt_bboxes_i,
            img_shape=(800, 1333),
        ))

    matcher = DiffusionDetMatcher(
        match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
        center_radius=0.5,
        candidate_topk=5,
    )

    for mode in ['l1', 'relative_l1', 'gcd', 'mixed_relative_l1']:
        criterion = DiffusionDetCriterion(
            num_classes=C, matcher=matcher,
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            deep_supervision=False,
            bbox_loss_mode=mode,
        )
        outputs = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_bboxes)
        losses = criterion(outputs, targets)
        check(f'{mode} has loss_cls', 'loss_cls' in losses)
        check(f'{mode} has loss_bbox', 'loss_bbox' in losses)
        check(f'{mode} has loss_giou', 'loss_giou' in losses)
        check(f'{mode} loss_cls finite', losses['loss_cls'].isfinite().item())
        check(f'{mode} loss_bbox finite', losses['loss_bbox'].isfinite().item())
        check(f'{mode} loss_giou finite', losses['loss_giou'].isfinite().item())

    criterion_ds = DiffusionDetCriterion(
        num_classes=C, matcher=matcher,
        loss_cls=FocalLoss(loss_weight=2.0),
        loss_bbox=L1Loss(loss_weight=5.0),
        loss_giou=GIoULoss(loss_weight=2.0),
        deep_supervision=True,
        bbox_loss_mode='relative_l1',
    )
    aux_outputs = [ModelOutput(pred_logits=pred_logits, pred_boxes=pred_bboxes) for _ in range(3)]
    outputs_ds = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_bboxes, aux_outputs=aux_outputs)
    losses_ds = criterion_ds(outputs_ds, targets)
    check('deep supervision aux losses', any('aux_' in k for k in losses_ds))


# ============================================================
# 18. Coordinate conversion roundtrip (DiTDiffusionDetHead)
# ============================================================
def test_coordinate_roundtrip():
    print('\n=== test_coordinate_roundtrip ===')
    num_classes = 24
    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=256, num_proposals=50,
        num_heads=6, snr_scale=2.0, sampling_timesteps=4,
        diffusion_type='rectified_flow', solver_type='heun',
        rf_schedule='shifted', rf_shift=3.0,
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=8,
        box_init_mode='bilinear',
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )

    img_metas = [ImageMeta(img_shape=(800, 1333), ori_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])]

    gt_bboxes_norm = torch.tensor([[0.1, 0.2, 0.3, 0.5]])
    gt_bboxes_img = gt_bboxes_norm * torch.tensor([1333, 800, 1333, 800])

    raw = head._xyxy_to_raw(gt_bboxes_img.unsqueeze(0), img_metas)
    check('raw shape', raw.shape == (1, 1, 4))
    check('raw finite', raw.isfinite().all().item())

    back_img = head._raw_to_xyxy(raw, img_metas)
    check('roundtrip close', torch.allclose(gt_bboxes_img.unsqueeze(0), back_img, atol=1.0),
          f'max diff: {(gt_bboxes_img.unsqueeze(0) - back_img).abs().max().item()}')

    normed = head._normalize_bboxes_for_tokenizer(gt_bboxes_img.unsqueeze(0), img_metas)
    check('normalize close', torch.allclose(gt_bboxes_norm.unsqueeze(0), normed, atol=1e-4))

    re_img = head._normed_to_img(normed, img_metas)
    check('normed_to_img close', torch.allclose(gt_bboxes_img.unsqueeze(0), re_img, atol=1e-3))


# ============================================================
# 19. DiTDiffusionDetHead forward (training-like)
# ============================================================
def test_dit_head_forward():
    print('\n=== test_dit_head_forward ===')
    bs, C, num_classes = 2, 256, 24
    num_proposals = 50
    num_levels = 4

    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=C, num_proposals=num_proposals,
        num_heads=3, snr_scale=2.0, sampling_timesteps=4,
        diffusion_type='rectified_flow', solver_type='heun',
        rf_schedule='shifted', rf_shift=3.0,
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=num_levels, num_ref_points=8,
        box_init_mode='bilinear', deep_supervision=True,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )

    fpn_features = [torch.rand(bs, C, 20 + l * 5, 20 + l * 5) for l in range(num_levels)]
    img_metas = [ImageMeta(img_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0]) for _ in range(bs)]

    x_noisy = torch.randn(bs, num_proposals, 4)
    curr_bboxes = head._raw_to_xyxy(x_noisy, img_metas)
    t_input = torch.full((bs,), 500.0)

    all_cls, all_bboxes, all_obj, all_vel, all_tokens = head(
        fpn_features, curr_bboxes, t_input, img_metas=img_metas
    )
    check('cls sequence shape', all_cls.shape == (3, bs, num_proposals, num_classes))
    check('bbox sequence shape', all_bboxes.shape == (3, bs, num_proposals, 4))
    check('bbox in [0,1]', all_bboxes.min().item() >= -1e-6 and all_bboxes.max().item() <= 1 + 1e-6)
    check('finite', all_cls.isfinite().all().item() and all_bboxes.isfinite().all().item())


# ============================================================
# 20. DiTDiffusionDetHead loss
# ============================================================
def test_dit_head_loss():
    print('\n=== test_dit_head_loss ===')
    bs, C, num_classes = 2, 256, 24
    num_proposals = 50
    num_levels = 4

    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=C, num_proposals=num_proposals,
        num_heads=3, snr_scale=2.0, sampling_timesteps=4,
        diffusion_type='rectified_flow', solver_type='heun',
        rf_schedule='shifted', rf_shift=3.0,
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=num_levels, num_ref_points=8,
        box_init_mode='bilinear', deep_supervision=True,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )
    head.train()

    fpn_features = [torch.rand(bs, C, 20 + l * 5, 20 + l * 5) for l in range(num_levels)]
    img_metas = [ImageMeta(img_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0]) for _ in range(bs)]
    gt_bboxes = []
    gt_labels = []
    for i in range(bs):
        M = 5
        bboxes = torch.rand(M, 4)
        bboxes[:, 2] = bboxes[:, 0] + 0.05 + torch.rand(M) * 0.1
        bboxes[:, 3] = bboxes[:, 1] + 0.05 + torch.rand(M) * 0.1
        bboxes = bboxes.clamp(0, 1)
        bboxes_img = bboxes * torch.tensor([1333, 800, 1333, 800])
        gt_bboxes.append(bboxes_img)
        gt_labels.append(torch.randint(0, num_classes, (M,)))

    losses = head.loss(fpn_features, img_metas, gt_bboxes, gt_labels)
    check('has loss_cls', 'loss_cls' in losses)
    check('has loss_bbox', 'loss_bbox' in losses)
    check('has loss_giou', 'loss_giou' in losses)
    check('loss_cls finite', losses['loss_cls'].isfinite().item())
    check('loss_bbox finite', losses['loss_bbox'].isfinite().item())
    check('loss_giou finite', losses['loss_giou'].isfinite().item())
    check('loss_cls > 0', losses['loss_cls'].item() > 0)
    check('loss_bbox > 0', losses['loss_bbox'].item() > 0)
    check('has aux losses', any('aux_' in k for k in losses))

    for k, v in losses.items():
        check(f'{k} finite', v.isfinite().item(), f'val={v.item()}')


# ============================================================
# 21. DiTDiffusionDetHead predict
# ============================================================
def test_dit_head_predict():
    print('\n=== test_dit_head_predict ===')
    bs, C, num_classes = 1, 256, 24
    num_proposals = 50
    num_levels = 4

    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=C, num_proposals=num_proposals,
        num_heads=3, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        rf_schedule='shifted', rf_shift=3.0,
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=num_levels, num_ref_points=8,
        box_init_mode='bilinear', deep_supervision=False,
        use_ensemble=True, box_renewal=False,
        use_nms=True, nms_thr=0.5, score_thr=0.05,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )
    head.eval()

    fpn_features = [torch.rand(bs, C, 20 + l * 5, 20 + l * 5) for l in range(num_levels)]
    img_metas = [ImageMeta(img_shape=(800, 1333), ori_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])]

    with torch.no_grad():
        results = head.predict(fpn_features, img_metas, rescale=False)
    check('results length', len(results) == bs)
    check('result type', isinstance(results[0], DetectionResult))
    check('bboxes shape 2d', results[0].bboxes.dim() == 2 and results[0].bboxes.shape[1] == 4)
    check('scores shape 1d', results[0].scores.dim() == 1)
    check('labels shape 1d', results[0].labels.dim() == 1)
    check('bboxes finite', results[0].bboxes.isfinite().all().item())
    check('scores in [0,1]', results[0].scores.min().item() >= 0 and results[0].scores.max().item() <= 1)


# ============================================================
# 22. DiTDiffusionDetHead predict with Heun solver
# ============================================================
def test_dit_head_predict_heun():
    print('\n=== test_dit_head_predict_heun ===')
    bs, C, num_classes = 1, 256, 24
    num_proposals = 50
    num_levels = 4

    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=C, num_proposals=num_proposals,
        num_heads=3, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='heun',
        rf_schedule='shifted', rf_shift=3.0,
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=num_levels, num_ref_points=8,
        box_init_mode='bilinear', deep_supervision=False,
        use_ensemble=True, box_renewal=False,
        use_nms=True, nms_thr=0.5, score_thr=0.05,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )
    head.eval()

    fpn_features = [torch.rand(bs, C, 20 + l * 5, 20 + l * 5) for l in range(num_levels)]
    img_metas = [ImageMeta(img_shape=(800, 1333), ori_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])]

    with torch.no_grad():
        results = head.predict(fpn_features, img_metas, rescale=False)
    check('heun results length', len(results) == bs)
    check('heun bboxes finite', results[0].bboxes.isfinite().all().item())


# ============================================================
# 23. DiTDiffusionDetHead predict with DPM-Solver++
# ============================================================
def test_dit_head_predict_dpm():
    print('\n=== test_dit_head_predict_dpm ===')
    bs, C, num_classes = 1, 256, 24
    num_proposals = 50
    num_levels = 4

    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=C, num_proposals=num_proposals,
        num_heads=3, snr_scale=2.0, sampling_timesteps=4,
        diffusion_type='rectified_flow', solver_type='dpm_solver_pp',
        rf_schedule='shifted', rf_shift=3.0,
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=num_levels, num_ref_points=8,
        box_init_mode='bilinear', deep_supervision=False,
        use_ensemble=True, box_renewal=False,
        use_nms=True, nms_thr=0.5, score_thr=0.05,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )
    head.eval()

    fpn_features = [torch.rand(bs, C, 20 + l * 5, 20 + l * 5) for l in range(num_levels)]
    img_metas = [ImageMeta(img_shape=(800, 1333), ori_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])]

    with torch.no_grad():
        results = head.predict(fpn_features, img_metas, rescale=False)
    check('dpm results length', len(results) == bs)
    check('dpm bboxes finite', results[0].bboxes.isfinite().all().item())


# ============================================================
# 24. RelativeL1Cost vs BBoxL1Cost: 归一化空间中区分度
# ============================================================
def test_cost_sensitivity():
    print('\n=== test_cost_sensitivity ===')
    gt_bboxes = torch.tensor([[0.3, 0.4, 0.5, 0.7]])

    pred_close = torch.tensor([[0.31, 0.41, 0.51, 0.71]])
    pred_far = torch.tensor([[0.6, 0.7, 0.8, 0.9]])

    l1_cost = BBoxL1Cost(weight=5.0)
    rel_cost = RelativeL1Cost(weight=5.0)

    gt_labels = torch.tensor([0])
    dummy_logits = torch.zeros(1, 24)

    cost_l1_close = l1_cost(dummy_logits, pred_close, gt_labels, gt_bboxes)
    cost_l1_far = l1_cost(dummy_logits, pred_far, gt_labels, gt_bboxes)
    cost_rel_close = rel_cost(dummy_logits, pred_close, gt_labels, gt_bboxes)
    cost_rel_far = rel_cost(dummy_logits, pred_far, gt_labels, gt_bboxes)

    l1_ratio = cost_l1_far.item() / max(cost_l1_close.item(), 1e-8)
    rel_ratio = cost_rel_far.item() / max(cost_rel_close.item(), 1e-8)

    check('rel ratio > l1 ratio', rel_ratio > l1_ratio,
          f'l1_ratio={l1_ratio:.2f}, rel_ratio={rel_ratio:.2f}')
    check('rel cost close > 0', cost_rel_close.item() > 0)
    check('rel cost far > close', cost_rel_far.item() > cost_rel_close.item())


# ============================================================
# 25. center_radius impact on matching
# ============================================================
def test_center_radius_impact():
    print('\n=== test_center_radius_impact ===')
    N, M, C = 50, 5, 24
    bs = 1

    pred_logits = torch.randn(bs, N, C)
    pred_bboxes = torch.rand(bs, N, 4)
    pred_bboxes[:, :, 2] = pred_bboxes[:, :, 0] + 0.1
    pred_bboxes[:, :, 3] = pred_bboxes[:, :, 1] + 0.1

    gt_bboxes_i = torch.rand(M, 4)
    gt_bboxes_i[:, 2] = gt_bboxes_i[:, 0] + 0.1
    gt_bboxes_i[:, 3] = gt_bboxes_i[:, 1] + 0.1
    targets = [InstanceData(
        labels=torch.randint(0, C, (M,)),
        bboxes=gt_bboxes_i,
        img_shape=(800, 1333),
    )]

    outputs = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_bboxes)

    for cr in [0.25, 0.5, 1.0, 2.5]:
        matcher = DiffusionDetMatcher(
            match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
            center_radius=cr, candidate_topk=5,
        )
        indices = matcher(outputs, targets)
        src_idx, gt_idx = indices[0]
        n_matched = len(src_idx)
        print(f'    center_radius={cr}: {n_matched} matched proposals')
        check(f'cr={cr} matched > 0', n_matched > 0)
        check(f'cr={cr} matched <= M', n_matched <= M * 5)


# ============================================================
# 26. Direct regression bbox validity
# ============================================================
def test_direct_regression_validity():
    print('\n=== test_direct_regression_validity ===')
    head = DiTSingleHead(
        num_classes=24, feat_channels=256, num_heads=8,
        num_fpn_levels=4, num_ref_points=8,
        prediction_mode='x0', adaln_params=9, regression_mode='direct',
    )

    fc_feature = torch.randn(2, 50, 256)
    bboxes = torch.rand(2, 50, 4)
    bboxes[:, :, 2] = bboxes[:, :, 0] + 0.1
    bboxes[:, :, 3] = bboxes[:, :, 1] + 0.1
    bboxes = bboxes.clamp(0, 1)

    pred = head._predict_bboxes(fc_feature, bboxes)
    check('pred in [0,1]', pred.min().item() >= 0 and pred.max().item() <= 1)
    check('x2 >= x1', (pred[..., 2] >= pred[..., 0] - 1e-6).all().item())
    check('y2 >= y1', (pred[..., 3] >= pred[..., 1] - 1e-6).all().item())
    check('pred finite', pred.isfinite().all().item())

    w = pred[..., 2] - pred[..., 0]
    h = pred[..., 3] - pred[..., 1]
    check('width >= 0', (w >= -1e-6).all().item())
    check('height >= 0', (h >= -1e-6).all().item())


# ============================================================
# 27. Delta regression bbox validity
# ============================================================
def test_delta_regression_validity():
    print('\n=== test_delta_regression_validity ===')
    head = DiTSingleHead(
        num_classes=24, feat_channels=256, num_heads=8,
        num_fpn_levels=4, num_ref_points=8,
        prediction_mode='x0', adaln_params=9, regression_mode='delta',
    )

    fc_feature = torch.randn(2, 50, 256)
    bboxes = torch.rand(2, 50, 4)
    bboxes[:, :, 2] = bboxes[:, :, 0] + 0.05
    bboxes[:, :, 3] = bboxes[:, :, 1] + 0.05
    bboxes = bboxes.clamp(0, 1)

    pred = head._predict_bboxes(fc_feature, bboxes)
    check('delta pred in [0,1]', pred.min().item() >= -1e-3 and pred.max().item() <= 1 + 1e-3)
    check('delta pred finite', pred.isfinite().all().item())


# ============================================================
# 28. Full pipeline: loss -> backward -> grad check
# ============================================================
def test_backward():
    print('\n=== test_backward ===')
    bs, C, num_classes = 1, 256, 24
    num_proposals = 20
    num_levels = 4

    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=C, num_proposals=num_proposals,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        rf_schedule='shifted', rf_shift=3.0,
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=num_levels, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )
    head.train()

    fpn_features = [torch.rand(bs, C, 15 + l * 3, 15 + l * 3) for l in range(num_levels)]
    img_metas = [ImageMeta(img_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0]) for _ in range(bs)]
    gt_bboxes = []
    gt_labels = []
    for i in range(bs):
        M = 3
        bboxes = torch.rand(M, 4)
        bboxes[:, 2] = bboxes[:, 0] + 0.05 + torch.rand(M) * 0.1
        bboxes[:, 3] = bboxes[:, 1] + 0.05 + torch.rand(M) * 0.1
        bboxes = bboxes.clamp(0, 1)
        bboxes_img = bboxes * torch.tensor([1333, 800, 1333, 800])
        gt_bboxes.append(bboxes_img)
        gt_labels.append(torch.randint(0, num_classes, (M,)))

    losses = head.loss(fpn_features, img_metas, gt_bboxes, gt_labels)
    total_loss = sum(v for v in losses.values())
    check('total loss finite', total_loss.isfinite().item())
    check('total loss > 0', total_loss.item() > 0)

    total_loss.backward()

    has_grad = False
    for name, p in head.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    check('has non-zero grad', has_grad)

    grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    check('grad_norm finite', grad_norm.isfinite().item(), f'grad_norm={grad_norm.item():.2f}')


# ============================================================
# 29. Rescale in predict
# ============================================================
def test_predict_rescale():
    print('\n=== test_predict_rescale ===')
    bs, C, num_classes = 1, 256, 24
    num_proposals = 30
    num_levels = 4

    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=C, num_proposals=num_proposals,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        rf_schedule='shifted', rf_shift=3.0,
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=num_levels, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        use_ensemble=True, box_renewal=False,
        use_nms=True, nms_thr=0.5, score_thr=0.05,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )
    head.eval()

    fpn_features = [torch.rand(bs, C, 15 + l * 3, 15 + l * 3) for l in range(num_levels)]
    sf = [0.5, 0.5, 0.5, 0.5]
    img_metas = [ImageMeta(img_shape=(800, 1333), ori_shape=(1600, 2666), scale_factor=sf)]

    with torch.no_grad():
        torch.manual_seed(99)
        results_no_rescale = head.predict(fpn_features, img_metas, rescale=False)
        torch.manual_seed(99)
        results_rescale = head.predict(fpn_features, img_metas, rescale=True)

    check('rescale no error', True)
    n_no = len(results_no_rescale[0].bboxes)
    n_re = len(results_rescale[0].bboxes)
    check('rescale same count', n_no == n_re, f'no_rescale={n_no}, rescale={n_re}')
    if n_no > 0 and n_re > 0 and n_no == n_re:
        sf_tensor = results_no_rescale[0].bboxes.new_tensor([0.5, 0.5, 0.5, 0.5])
        expected_rescaled = results_no_rescale[0].bboxes / sf_tensor
        check('rescale correct', torch.allclose(results_rescale[0].bboxes, expected_rescaled, atol=1.0))


# ============================================================
# 30. DDPM mode forward + predict
# ============================================================
def test_ddpm_mode():
    print('\n=== test_ddpm_mode ===')
    bs, C, num_classes = 1, 256, 24
    num_proposals = 20
    num_levels = 4

    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=C, num_proposals=num_proposals,
        num_heads=2, snr_scale=2.0, timesteps=1000, sampling_timesteps=2,
        diffusion_type='ddpm', solver_type='euler',
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=num_levels, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        use_ensemble=True, box_renewal=False,
        use_nms=True, nms_thr=0.5, score_thr=0.05,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )
    head.eval()

    fpn_features = [torch.rand(bs, C, 15 + l * 3, 15 + l * 3) for l in range(num_levels)]
    img_metas = [ImageMeta(img_shape=(800, 1333), ori_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])]

    with torch.no_grad():
        results = head.predict(fpn_features, img_metas, rescale=False)
    check('ddpm predict results', len(results) == bs)
    check('ddpm bboxes finite', results[0].bboxes.isfinite().all().item())


# ============================================================
# 31. Data structures
# ============================================================
def test_structures():
    print('\n=== test_structures ===')
    inst = InstanceData(
        bboxes=torch.rand(5, 4),
        labels=torch.randint(0, 24, (5,)),
        img_shape=(800, 1333),
    )
    check('InstanceData bboxes shape', inst.bboxes.shape == (5, 4))
    check('InstanceData labels shape', inst.labels.shape == (5,))
    check('InstanceData img_shape', inst.img_shape == (800, 1333))

    model_out = ModelOutput(
        pred_logits=torch.rand(2, 50, 24),
        pred_boxes=torch.rand(2, 50, 4),
    )
    check('ModelOutput logits shape', model_out.pred_logits.shape == (2, 50, 24))
    check('ModelOutput boxes shape', model_out.pred_boxes.shape == (2, 50, 4))
    check('ModelOutput obj None', model_out.pred_objectness is None)
    check('ModelOutput aux None', model_out.aux_outputs is None)

    model_out2 = ModelOutput(
        pred_logits=torch.rand(2, 50, 24),
        pred_boxes=torch.rand(2, 50, 4),
        pred_objectness=torch.rand(2, 50, 1),
        aux_outputs=[
            ModelOutput(pred_logits=torch.rand(2, 50, 24), pred_boxes=torch.rand(2, 50, 4))
            for _ in range(2)
        ],
    )
    check('ModelOutput obj not None', model_out2.pred_objectness is not None)
    check('ModelOutput aux len', len(model_out2.aux_outputs) == 2)

    meta = ImageMeta(img_shape=(800, 1333), ori_shape=(1600, 2666), scale_factor=[0.5, 0.5, 0.5, 0.5])
    check('ImageMeta img_shape', meta.img_shape == (800, 1333))
    check('ImageMeta ori_shape', meta.ori_shape == (1600, 2666))
    check('ImageMeta scale_factor', meta.scale_factor == [0.5, 0.5, 0.5, 0.5])

    meta_min = ImageMeta(img_shape=(800, 1333))
    check('ImageMeta defaults', meta_min.ori_shape is None and meta_min.scale_factor is None)

    det = DetectionResult(
        bboxes=torch.rand(10, 4),
        scores=torch.rand(10),
        labels=torch.randint(0, 24, (10,)),
    )
    check('DetectionResult bboxes shape', det.bboxes.shape == (10, 4))
    check('DetectionResult scores shape', det.scores.shape == (10,))
    check('DetectionResult labels shape', det.labels.shape == (10,))


# ============================================================
# 32. load_buffer
# ============================================================
def test_load_buffer():
    print('\n=== test_load_buffer ===')
    arr = torch.linspace(0, 1, 100)
    steps = torch.tensor([0, 50, 99])
    x_shape = [2, 10, 256]
    result = load_buffer(arr, steps, x_shape)
    check('load_buffer shape', result.shape == (3, 1, 1))
    check('load_buffer values', torch.isclose(result[0, 0, 0], arr[0]) and torch.isclose(result[2, 0, 0], arr[99]))

    steps_single = torch.tensor([25])
    result_single = load_buffer(arr, steps_single, [4, 4])
    check('load_buffer single shape', result_single.shape == (1, 1))
    check('load_buffer single value', torch.isclose(result_single[0, 0], arr[25]))


# ============================================================
# 33. bbox2roi
# ============================================================
def test_bbox2roi():
    print('\n=== test_bbox2roi ===')
    bboxes1 = torch.tensor([[10.0, 20.0, 30.0, 60.0], [40.0, 50.0, 80.0, 90.0]])
    bboxes2 = torch.tensor([[100.0, 200.0, 300.0, 400.0]])
    rois = bbox2roi([bboxes1, bboxes2])
    check('bbox2roi shape', rois.shape == (3, 5))
    check('bbox2roi batch 0 idx', (rois[:2, 0] == 0).all().item())
    check('bbox2roi batch 1 idx', rois[2, 0].item() == 1)
    check('bbox2roi coords match', torch.equal(rois[0, 1:], bboxes1[0]))


# ============================================================
# 34. SingleRoIExtractor
# ============================================================
def test_roi_extractor():
    print('\n=== test_roi_extractor ===')
    from mods.roi_extractor import SingleRoIExtractor

    featmap_strides = [4, 8, 16, 32]
    out_channels = 256
    roi_layer_cfg = dict(type='RoIAlign', output_size=(7, 7), sampling_ratio=0)

    extractor = SingleRoIExtractor(
        roi_layer=roi_layer_cfg,
        out_channels=out_channels,
        featmap_strides=featmap_strides,
    )

    feats = [torch.rand(2, 256, 200 // s, 200 // s) for s in featmap_strides]
    rois = torch.tensor([
        [0, 10.0, 10.0, 50.0, 50.0],
        [0, 30.0, 30.0, 100.0, 100.0],
        [1, 20.0, 20.0, 90.0, 80.0],
    ])

    roi_feats = extractor(feats, rois)
    check('roi_feats shape', roi_feats.shape == (3, 256, 7, 7))
    check('roi_feats finite', roi_feats.isfinite().all().item())

    rois_empty = torch.zeros(0, 5)
    roi_feats_empty = extractor(feats[:1], rois_empty)
    check('empty rois shape', roi_feats_empty.shape == (0, 256, 7, 7))

    lvl = extractor.map_roi_levels(rois, 4)
    check('map_roi_levels shape', lvl.shape == (3,))
    check('map_roi_levels range', (lvl >= 0).all().item() and (lvl < 4).all().item())


# ============================================================
# 35. OT Coupling: Sinkhorn
# ============================================================
def test_ot_sinkhorn():
    print('\n=== test_ot_sinkhorn ===')
    num_classes = 24
    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        ot_coupling=True, ot_matcher='sinkhorn', ot_epsilon=1.0, ot_num_iters=10,
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )

    noise = torch.randn(50, 4)
    gt_diffusion = torch.randn(5, 4)
    gt_labels = torch.randint(0, num_classes, (5,))

    x_start, matched_idx = head._couple_ot(noise, gt_diffusion, gt_labels, torch.device('cpu'))
    check('ot x_start shape', x_start.shape == (50, 4))
    check('ot matched_idx shape', matched_idx.shape == (50,))
    check('ot matched_idx range', (matched_idx >= 0).all().item() and (matched_idx < 5).all().item())
    check('ot x_start finite', x_start.isfinite().all().item())

    transport = head._sinkhorn_transport(torch.cdist(noise, gt_diffusion, p=2))
    check('transport shape', transport.shape == (50, 5))
    check('transport non-negative', (transport >= 0).all().item())
    check('transport finite', transport.isfinite().all().item())


# ============================================================
# 36. OT Coupling: nearest (argmin)
# ============================================================
def test_ot_nearest():
    print('\n=== test_ot_nearest ===')
    num_classes = 24
    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        ot_coupling=True, ot_matcher='nearest',
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )

    noise = torch.randn(50, 4)
    gt_diffusion = torch.randn(5, 4)
    gt_labels = torch.randint(0, num_classes, (5,))

    x_start, matched_idx = head._couple_ot(noise, gt_diffusion, gt_labels, torch.device('cpu'))
    check('nearest x_start shape', x_start.shape == (50, 4))
    check('nearest matched_idx range', (matched_idx >= 0).all().item() and (matched_idx < 5).all().item())

    cost = torch.cdist(noise, gt_diffusion, p=2)
    expected_idx = cost.argmin(dim=1)
    check('nearest matches argmin', torch.equal(matched_idx, expected_idx))


# ============================================================
# 37. OT Coupling: empty GT
# ============================================================
def test_ot_empty_gt():
    print('\n=== test_ot_empty_gt ===')
    num_classes = 24
    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        ot_coupling=True, ot_matcher='sinkhorn',
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )

    noise = torch.randn(50, 4)
    gt_diffusion = torch.zeros(0, 4)
    gt_labels = torch.zeros(0, dtype=torch.long)

    x_start, matched_idx = head._couple_ot(noise, gt_diffusion, gt_labels, torch.device('cpu'))
    check('empty gt x_start == noise', torch.equal(x_start, noise))
    check('empty gt matched_idx zeros', (matched_idx == 0).all().item())


# ============================================================
# 38. Time sampling strategies
# ============================================================
def test_time_sampling():
    print('\n=== test_time_sampling ===')
    num_classes = 24
    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        t_sampling='uniform',
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )

    t_uniform, lp_uniform = head._sample_t(16, torch.device('cpu'))
    check('uniform t shape', t_uniform.shape == (16,))
    check('uniform t > 0', (t_uniform > 0).all().item())
    check('uniform t < 1', (t_uniform < 1).all().item())
    check('uniform no log_probs', lp_uniform is None)

    head_strat = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        t_sampling='stratified', t_sampling_bins=8,
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )

    t_strat, _ = head_strat._sample_t(16, torch.device('cpu'))
    check('stratified t shape', t_strat.shape == (16,))
    check('stratified t > 0', (t_strat > 0).all().item())
    check('stratified t < 1', (t_strat < 1).all().item())

    head_ddpm = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, timesteps=1000, sampling_timesteps=2,
        diffusion_type='ddpm', solver_type='euler',
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )

    t_ddpm, _ = head_ddpm._sample_t(16, torch.device('cpu'))
    check('ddpm t shape', t_ddpm.shape == (16,))
    check('ddpm t long', t_ddpm.dtype == torch.long)
    check('ddpm t range', (t_ddpm >= 0).all().item() and (t_ddpm < 1000).all().item())


# ============================================================
# 39. LSAS time sampling
# ============================================================
def test_lsas_sampling():
    print('\n=== test_lsas_sampling ===')
    num_classes = 24
    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        use_lsas=True, lsas_num_bins=50, lsas_temp=1.0,
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )

    t, log_probs = head._sample_t(16, torch.device('cpu'))
    check('lsas t shape', t.shape == (16,))
    check('lsas t > 0', (t > 0).all().item())
    check('lsas t < 1', (t < 1).all().item())
    check('lsas log_probs not None', log_probs is not None)
    check('lsas log_probs shape', log_probs.shape == (16,))
    check('lsas log_probs finite', log_probs.isfinite().all().item())


# ============================================================
# 40. Shifted RF schedule
# ============================================================
def test_shifted_schedule():
    print('\n=== test_shifted_schedule ===')
    s = 3.0
    t_linear = torch.linspace(0.01, 0.99, 100)
    t_shifted = s * t_linear / (1 + (s - 1) * t_linear)

    check('shifted t range', t_shifted.min().item() > 0 and t_shifted.max().item() < 1)
    check('shifted monotonic', (t_shifted[1:] >= t_shifted[:-1]).all().item())
    check('shifted > linear at low t', t_shifted[0].item() > t_linear[0].item())

    t_at_0 = s * 0.0 / (1 + (s - 1) * 0.0)
    t_at_1 = s * 1.0 / (1 + (s - 1) * 1.0)
    check('shifted t(0)=0', abs(t_at_0) < 1e-6)
    check('shifted t(1)=1', abs(t_at_1 - 1.0) < 1e-6)


# ============================================================
# 41. Scale-aware bbox loss
# ============================================================
def test_scale_aware_loss():
    print('\n=== test_scale_aware_loss ===')
    N, M, C, bs = 50, 5, 24, 2

    pred_logits = torch.randn(bs, N, C)
    pred_bboxes = torch.rand(bs, N, 4)
    pred_bboxes[:, :, 2] = pred_bboxes[:, :, 0] + 0.1
    pred_bboxes[:, :, 3] = pred_bboxes[:, :, 1] + 0.1

    targets = []
    for i in range(bs):
        gt_bboxes_i = torch.rand(M, 4)
        gt_bboxes_i[:, 2] = gt_bboxes_i[:, 0] + 0.1
        gt_bboxes_i[:, 3] = gt_bboxes_i[:, 1] + 0.1
        targets.append(InstanceData(
            labels=torch.randint(0, C, (M,)),
            bboxes=gt_bboxes_i,
            img_shape=(800, 1333),
        ))

    matcher = DiffusionDetMatcher(
        match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
        center_radius=0.5, candidate_topk=5,
    )

    for sa_mode in ['inverse', 'sqrt_inverse', 'log_linear']:
        criterion = DiffusionDetCriterion(
            num_classes=C, matcher=matcher,
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            scale_aware=True, scale_aware_mode=sa_mode,
            bbox_loss_mode='l1',
        )
        outputs = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_bboxes)
        losses = criterion(outputs, targets)
        check(f'sa_{sa_mode} loss_bbox finite', losses['loss_bbox'].isfinite().item())
        check(f'sa_{sa_mode} loss_giou finite', losses['loss_giou'].isfinite().item())


# ============================================================
# 42. Objectness loss
# ============================================================
def test_objectness_loss():
    print('\n=== test_objectness_loss ===')
    N, M, C, bs = 50, 5, 24, 2

    pred_logits = torch.randn(bs, N, C)
    pred_bboxes = torch.rand(bs, N, 4)
    pred_bboxes[:, :, 2] = pred_bboxes[:, :, 0] + 0.1
    pred_bboxes[:, :, 3] = pred_bboxes[:, :, 1] + 0.1
    pred_obj = torch.randn(bs, N, 1)

    targets = []
    for i in range(bs):
        gt_bboxes_i = torch.rand(M, 4)
        gt_bboxes_i[:, 2] = gt_bboxes_i[:, 0] + 0.1
        gt_bboxes_i[:, 3] = gt_bboxes_i[:, 1] + 0.1
        targets.append(InstanceData(
            labels=torch.randint(0, C, (M,)),
            bboxes=gt_bboxes_i,
            img_shape=(800, 1333),
        ))

    matcher = DiffusionDetMatcher(
        match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
        center_radius=0.5, candidate_topk=5,
    )

    criterion = DiffusionDetCriterion(
        num_classes=C, matcher=matcher,
        loss_cls=FocalLoss(loss_weight=2.0),
        loss_bbox=L1Loss(loss_weight=5.0),
        loss_giou=GIoULoss(loss_weight=2.0),
        bbox_loss_mode='relative_l1',
    )
    outputs = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_bboxes, pred_objectness=pred_obj)
    losses = criterion(outputs, targets)
    check('has loss_objectness', 'loss_objectness' in losses)
    check('loss_objectness finite', losses['loss_objectness'].isfinite().item())
    check('loss_objectness > 0', losses['loss_objectness'].item() > 0)


# ============================================================
# 43. apply_deltas numerical stability
# ============================================================
def test_apply_deltas_stability():
    print('\n=== test_apply_deltas_stability ===')
    head = DiTSingleHead(
        num_classes=24, feat_channels=256, num_heads=8,
        num_fpn_levels=4, num_ref_points=8,
        prediction_mode='x0', adaln_params=9, regression_mode='delta',
    )

    bboxes = torch.tensor([[0.4, 0.4, 0.6, 0.6]])
    deltas = torch.zeros(1, 4)
    pred = head.apply_deltas(deltas, bboxes)
    check('zero delta ~ same', torch.allclose(pred, bboxes, atol=1e-4))

    bboxes_tiny = torch.tensor([[0.499, 0.499, 0.501, 0.501]])
    deltas_small = torch.randn(1, 4) * 0.1
    pred_tiny = head.apply_deltas(deltas_small, bboxes_tiny)
    check('tiny bbox finite', pred_tiny.isfinite().all().item())
    check('tiny bbox in [0,1]', pred_tiny.min().item() >= 0 and pred_tiny.max().item() <= 1)

    bboxes_large = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    deltas_large = torch.randn(1, 4) * 2.0
    pred_large = head.apply_deltas(deltas_large, bboxes_large)
    check('large delta finite', pred_large.isfinite().all().item())
    check('large delta in [0,1]', pred_large.min().item() >= 0 and pred_large.max().item() <= 1)


# ============================================================
# 46. Box renewal in predict
# ============================================================
def test_box_renewal():
    print('\n=== test_box_renewal ===')
    bs, C, num_classes = 1, 256, 24
    num_proposals = 30
    num_levels = 4

    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=C, num_proposals=num_proposals,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=num_levels, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        use_ensemble=True, box_renewal=True,
        use_nms=True, nms_thr=0.5, score_thr=0.05,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )
    head.eval()

    fpn_features = [torch.rand(bs, C, 15 + l * 3, 15 + l * 3) for l in range(num_levels)]
    img_metas = [ImageMeta(img_shape=(800, 1333), ori_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])]

    with torch.no_grad():
        results = head.predict(fpn_features, img_metas, rescale=False)
    check('box_renewal results length', len(results) == bs)
    check('box_renewal bboxes finite', results[0].bboxes.isfinite().all().item())


# ============================================================
# 47. Velocity loss in training
# ============================================================
def test_velocity_loss_training():
    print('\n=== test_velocity_loss_training ===')
    bs, C, num_classes = 1, 256, 24
    num_proposals = 20
    num_levels = 4

    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=C, num_proposals=num_proposals,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        regression_mode='direct', prediction_mode='velocity',
        velocity_loss_weight=5.0,
        adaln_params=9, num_fpn_levels=num_levels, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )
    head.train()

    fpn_features = [torch.rand(bs, C, 15 + l * 3, 15 + l * 3) for l in range(num_levels)]
    img_metas = [ImageMeta(img_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0]) for _ in range(bs)]
    gt_bboxes = []
    gt_labels = []
    for i in range(bs):
        M = 3
        bboxes = torch.rand(M, 4)
        bboxes[:, 2] = bboxes[:, 0] + 0.05 + torch.rand(M) * 0.1
        bboxes[:, 3] = bboxes[:, 1] + 0.05 + torch.rand(M) * 0.1
        bboxes = bboxes.clamp(0, 1)
        bboxes_img = bboxes * torch.tensor([1333, 800, 1333, 800])
        gt_bboxes.append(bboxes_img)
        gt_labels.append(torch.randint(0, num_classes, (M,)))

    losses = head.loss(fpn_features, img_metas, gt_bboxes, gt_labels)
    check('velocity has loss_velocity', 'loss_velocity' in losses)
    check('velocity loss finite', losses['loss_velocity'].isfinite().item())
    check('velocity loss > 0', losses['loss_velocity'].item() > 0)


# ============================================================
# 48. OT coupling with KCEC
# ============================================================
# ============================================================
# 49. DDPM q_sample
# ============================================================
def test_ddpm_q_sample():
    print('\n=== test_ddpm_q_sample ===')
    num_classes = 24
    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, timesteps=1000, sampling_timesteps=2,
        diffusion_type='ddpm', solver_type='euler',
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )

    x_start = torch.randn(8, 4)
    t = torch.tensor([0, 250, 500, 750, 999, 0, 500, 999])
    noise = torch.randn_like(x_start)

    x_noisy = head.q_sample(x_start, t, noise=noise)
    check('q_sample shape', x_noisy.shape == (8, 4))
    check('q_sample finite', x_noisy.isfinite().all().item())

    x_noisy_t0 = head.q_sample(x_start, torch.zeros(8, dtype=torch.long), noise=noise)
    check('q_sample t=0 ~ x_start', torch.allclose(x_noisy_t0, x_start, atol=0.05))

    pred_noise = head.predict_noise_from_start(x_noisy, t, x_start)
    check('pred_noise shape', pred_noise.shape == (8, 4))
    check('pred_noise finite', pred_noise.isfinite().all().item())


# ============================================================
# 51. Coordinate conversion edge cases
# ============================================================
def test_coordinate_edge_cases():
    print('\n=== test_coordinate_edge_cases ===')
    num_classes = 24
    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )

    img_metas = [ImageMeta(img_shape=(800, 1333), ori_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])]

    corner_bboxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]) * torch.tensor([1333, 800, 1333, 800])
    raw = head._xyxy_to_raw(corner_bboxes, img_metas)
    check('corner raw finite', raw.isfinite().all().item())

    back = head._raw_to_xyxy(raw, img_metas)
    check('corner roundtrip finite', back.isfinite().all().item())

    tiny_bboxes = torch.tensor([[[0.4, 0.4, 0.4001, 0.4001]]]) * torch.tensor([1333, 800, 1333, 800])
    raw_tiny = head._xyxy_to_raw(tiny_bboxes, img_metas)
    check('tiny raw finite', raw_tiny.isfinite().all().item())
    back_tiny = head._raw_to_xyxy(raw_tiny, img_metas)
    check('tiny roundtrip finite', back_tiny.isfinite().all().item())

    large_raw = torch.tensor([[[5.0, 5.0, 5.0, 5.0]]])
    back_clamp = head._raw_to_xyxy(large_raw, img_metas)
    check('clamped raw finite', back_clamp.isfinite().all().item())


# ============================================================
# 52. Matcher with duplicate GT assignments
# ============================================================
def test_matcher_duplicate():
    print('\n=== test_matcher_duplicate ===')
    N, M, C = 10, 3, 24
    bs = 1

    pred_logits = torch.randn(bs, N, C)
    pred_bboxes = torch.rand(bs, N, 4)
    pred_bboxes[:, :, 2] = pred_bboxes[:, :, 0] + 0.1
    pred_bboxes[:, :, 3] = pred_bboxes[:, :, 1] + 0.1

    gt_bboxes_i = torch.rand(M, 4)
    gt_bboxes_i[:, 2] = gt_bboxes_i[:, 0] + 0.1
    gt_bboxes_i[:, 3] = gt_bboxes_i[:, 1] + 0.1
    targets = [InstanceData(
        labels=torch.randint(0, C, (M,)),
        bboxes=gt_bboxes_i,
        img_shape=(800, 1333),
    )]

    matcher = DiffusionDetMatcher(
        match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
        center_radius=0.5, candidate_topk=3,
    )
    outputs = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_bboxes)
    indices = matcher(outputs, targets)
    src_idx, gt_idx = indices[0]

    check('duplicate src unique', len(src_idx) == len(src_idx.unique()),
          f'src has duplicates: {len(src_idx)} vs {len(src_idx.unique())}')


# ============================================================
# 53. FocalLoss with class indices vs one-hot
# ============================================================
def test_focal_loss_modes():
    print('\n=== test_focal_loss_modes ===')
    N, C = 50, 24
    pred = torch.randn(N, C)
    labels = torch.randint(0, C, (N,))

    fl = FocalLoss(loss_weight=2.0)

    loss_idx = fl(pred, labels)

    one_hot = torch.zeros(N, C)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    loss_oh = fl(pred, one_hot)

    check('idx == one-hot', torch.isclose(loss_idx, loss_oh, atol=1e-4))

    bg_labels = torch.full((N,), C, dtype=torch.long)
    loss_bg = fl(pred, bg_labels)
    check('background loss finite', loss_bg.isfinite().item())
    check('background loss > 0', loss_bg.item() > 0)


# ============================================================
# 54. GIoU loss properties
# ============================================================
def test_giou_loss_properties():
    print('\n=== test_giou_loss_properties ===')
    giou = GIoULoss(loss_weight=1.0)

    same = torch.tensor([[0.1, 0.2, 0.5, 0.6]])
    loss_same = giou(same, same)
    check('giou same ~0', loss_same.item() < 0.01, f'loss_same={loss_same.item():.6f}')

    pred = torch.tensor([[0.1, 0.2, 0.5, 0.6]])
    gt = torch.tensor([[0.3, 0.4, 0.7, 0.8]])
    loss_diff = giou(pred, gt)
    check('giou diff > same', loss_diff.item() > loss_same.item())

    no_overlap = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
    far = torch.tensor([[0.9, 0.9, 1.0, 1.0]])
    loss_no = giou(no_overlap, far)
    check('giou no overlap > 0', loss_no.item() > 0)
    check('giou no overlap finite', loss_no.isfinite().item())


# ============================================================
# 55. DynamicConv
# ============================================================
def test_dynamic_conv():
    print('\n=== test_dynamic_conv ===')
    from mods.modules import DynamicConv

    C = 256
    dc = DynamicConv(feat_channels=C, dynamic_dim=64, dynamic_num=2, pooler_resolution=7)

    proposals = torch.randn(1, 10, C)
    roi_feats = torch.randn(49, 10, C)

    out = dc(proposals, roi_feats)
    check('dynamic_conv shape', out.shape == (1, 10, C))
    check('dynamic_conv finite', out.isfinite().all().item())


# ============================================================
# 56. DiTBlock AdaLN-Zero zero init
# ============================================================
def test_dit_block_zero_init():
    print('\n=== test_dit_block_zero_init ===')
    C = 256
    block = DiTBlock(
        feat_channels=C, num_heads=8, num_fpn_levels=4,
        num_ref_points=8, dim_feedforward=2048, adaln_params=9,
    )

    last_linear = block.adaln_mlp[-1]
    check('adaln weight zero', (last_linear.weight == 0).all().item())
    check('adaln bias zero', (last_linear.bias == 0).all().item())

    bs, N = 2, 10
    box_tokens = torch.rand(bs, N, C)
    fpn_feats = [torch.rand(bs, C, 20 + l * 5, 20 + l * 5) for l in range(4)]
    flat, shapes, starts = flatten_fpn_features(fpn_feats)
    time_emb = torch.rand(bs, C * 4)
    bbox_coords = torch.rand(bs, N, 4)
    bbox_coords[:, :, 2] = bbox_coords[:, :, 0] + 0.1
    bbox_coords[:, :, 3] = bbox_coords[:, :, 1] + 0.1
    bbox_coords = bbox_coords.clamp(0, 1)

    out = block(box_tokens, flat, shapes, starts, time_emb, bbox_coords)
    check('zero init identity', torch.allclose(out, box_tokens, atol=1e-5),
          f'max diff: {(out - box_tokens).abs().max().item():.8f}')


# ============================================================
# 57. DiTSingleHead classification bias init
# ============================================================
def test_cls_bias_init():
    print('\n=== test_cls_bias_init ===')
    num_classes = 24
    prior_prob = 0.01
    expected_bias = -math.log((1 - prior_prob) / prior_prob)

    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )

    for single_head in head.head_series:
        cls_last = single_head.cls_head[-1]
        if cls_last.out_features in [num_classes, num_classes + 1]:
            check('cls bias init', torch.isclose(cls_last.bias.mean(), torch.tensor(expected_bias), atol=0.5),
                  f'bias_mean={cls_last.bias.mean().item():.4f}, expected={expected_bias:.4f}')
            break


# ============================================================
# 58. DiTDiffusionDetHead with OT coupling training
# ============================================================
def test_ot_training():
    print('\n=== test_ot_training ===')
    bs, C, num_classes = 1, 256, 24
    num_proposals = 20
    num_levels = 4

    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=C, num_proposals=num_proposals,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        ot_coupling=True, ot_matcher='sinkhorn', ot_epsilon=1.0, ot_num_iters=5,
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=num_levels, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )
    head.train()

    fpn_features = [torch.rand(bs, C, 15 + l * 3, 15 + l * 3) for l in range(num_levels)]
    img_metas = [ImageMeta(img_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0]) for _ in range(bs)]
    gt_bboxes = []
    gt_labels = []
    for i in range(bs):
        M = 3
        bboxes = torch.rand(M, 4)
        bboxes[:, 2] = bboxes[:, 0] + 0.05 + torch.rand(M) * 0.1
        bboxes[:, 3] = bboxes[:, 1] + 0.05 + torch.rand(M) * 0.1
        bboxes = bboxes.clamp(0, 1)
        bboxes_img = bboxes * torch.tensor([1333, 800, 1333, 800])
        gt_bboxes.append(bboxes_img)
        gt_labels.append(torch.randint(0, num_classes, (M,)))

    losses = head.loss(fpn_features, img_metas, gt_bboxes, gt_labels)
    check('ot training has loss_cls', 'loss_cls' in losses)
    check('ot training loss finite', all(v.isfinite().item() for v in losses.values()))


# ============================================================
# 59. DiTDiffusionDetHead predict with trajectory
# ============================================================
def test_predict_trajectory():
    print('\n=== test_predict_trajectory ===')
    bs, C, num_classes = 1, 256, 24
    num_proposals = 20
    num_levels = 4

    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=C, num_proposals=num_proposals,
        num_heads=2, snr_scale=2.0, sampling_timesteps=3,
        diffusion_type='rectified_flow', solver_type='euler',
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=num_levels, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        use_ensemble=True, box_renewal=False,
        use_nms=True, nms_thr=0.5, score_thr=0.05,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )
    head.eval()

    fpn_features = [torch.rand(bs, C, 15 + l * 3, 15 + l * 3) for l in range(num_levels)]
    img_metas = [ImageMeta(img_shape=(800, 1333), ori_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])]

    with torch.no_grad():
        results, trajectory = head.predict(fpn_features, img_metas, rescale=False, return_trajectory=True)
    check('trajectory results', len(results) == bs)
    check('trajectory length', len(trajectory) == 3)
    for i, (cls, bboxes) in enumerate(trajectory):
        check(f'traj step {i} cls shape', cls.shape[0] == bs)
        check(f'traj step {i} bboxes shape', bboxes.shape[-1] == 4)


# ============================================================
# 60. Power RF schedule
# ============================================================
def test_power_schedule():
    print('\n=== test_power_schedule ===')
    power = 2.0
    t_linear = torch.linspace(0.01, 0.99, 100)
    t_power = t_linear.pow(power)

    check('power t range', t_power.min().item() > 0 and t_power.max().item() < 1)
    check('power monotonic', (t_power[1:] >= t_power[:-1]).all().item())
    check('power < linear', t_power[50].item() < t_linear[50].item())


# ============================================================
# 61. BoxTokenizer._assign_fpn_level
# ============================================================
def test_box_tokenizer_assign_level():
    print('\n=== test_box_tokenizer_assign_level ===')
    tok = BoxTokenizer(feat_channels=256, num_fpn_levels=4, init_mode='bilinear')

    bboxes = torch.rand(2, 20, 4)
    bboxes[:, :, 2] = bboxes[:, :, 0] + torch.rand(2, 20) * 0.3
    bboxes[:, :, 3] = bboxes[:, :, 1] + torch.rand(2, 20) * 0.3
    bboxes = bboxes.clamp(0, 1)

    levels = tok._assign_fpn_level(bboxes, None)
    check('levels shape', levels.shape == (2, 20))
    check('levels range', (levels >= 0).all().item() and (levels < 4).all().item())
    check('levels long', levels.dtype == torch.long)

    tiny_bboxes = torch.tensor([[[0.49, 0.49, 0.51, 0.51]]])
    levels_tiny = tok._assign_fpn_level(tiny_bboxes, None)
    check('tiny bbox level', levels_tiny.item() >= 0)

    large_bboxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    levels_large = tok._assign_fpn_level(large_bboxes, None)
    check('large bbox level', levels_large.item() >= 0)
    check('tiny <= large level', levels_tiny.item() <= levels_large.item(),
          f'tiny={levels_tiny.item()}, large={levels_large.item()}')


# ============================================================
# 62. BoxTokenizer._bilinear_sample
# ============================================================
def test_box_tokenizer_bilinear_sample():
    print('\n=== test_box_tokenizer_bilinear_sample ===')
    bs, N, C, num_levels = 2, 10, 256, 4
    fpn_feats = [torch.rand(bs, C, 20 + l * 5, 20 + l * 5) for l in range(num_levels)]

    bboxes = torch.rand(bs, N, 4)
    bboxes[:, :, 2] = bboxes[:, :, 0] + torch.rand(bs, N) * 0.3
    bboxes[:, :, 3] = bboxes[:, :, 1] + torch.rand(bs, N) * 0.3
    bboxes = bboxes.clamp(0, 1)

    tok = BoxTokenizer(feat_channels=C, num_fpn_levels=num_levels, init_mode='bilinear')
    levels = tok._assign_fpn_level(bboxes, None)
    sampled = tok._bilinear_sample(fpn_feats, bboxes, levels)

    check('sampled shape', sampled.shape == (bs, N, C))
    check('sampled finite', sampled.isfinite().all().item())


# ============================================================
# 63. RectifiedFlow.get_velocity
# ============================================================
def test_rectified_flow_get_velocity():
    print('\n=== test_rectified_flow_get_velocity ===')
    rf = RectifiedFlow(snr_scale=2.0)

    x_t = torch.randn(8, 4)
    x0_pred = torch.randn(8, 4)
    t = torch.tensor([0.5] * 8)

    v = rf.get_velocity(x_t, x0_pred, t)
    check('velocity shape', v.shape == (8, 4))
    check('velocity finite', v.isfinite().all().item())

    t_view = t.view(-1, 1)
    expected_v = (x_t - x0_pred) / t_view
    check('velocity formula', torch.allclose(v, expected_v, atol=1e-5))

    x0_exact = x_t.clone()
    v_zero = rf.get_velocity(x_t, x0_exact, t)
    check('velocity zero when x0==xt', torch.allclose(v_zero, torch.zeros_like(v_zero), atol=1e-5))


# ============================================================
# 64. RectifiedFlow.heun_step
# ============================================================
def test_rectified_flow_heun_step():
    print('\n=== test_rectified_flow_heun_step ===')
    rf = RectifiedFlow(snr_scale=2.0)

    x_t = torch.randn(4, 4)
    x0_pred = torch.randn(4, 4)

    call_count = [0]

    def model_fn(x, t):
        call_count[0] += 1
        return x0_pred + torch.randn_like(x0_pred) * 0.1, None

    x_next = rf.heun_step(x_t, x0_pred, 0.8, 0.4, model_fn)
    check('heun step shape', x_next.shape == (4, 4))
    check('heun step finite', x_next.isfinite().all().item())
    check('heun calls model', call_count[0] == 1)


# ============================================================
# 65. RFDPMSolverMultistep order 3
# ============================================================
def test_dpm_solver_order3():
    print('\n=== test_dpm_solver_order3 ===')
    solver = RFDPMSolverMultistep(num_steps=4, solver_order=3)
    check('order3 timesteps', len(solver.timesteps) == 5)

    x = torch.randn(8, 4)
    x0 = torch.randn(8, 4)
    solver.reset()

    x1 = solver.step(x, x0, solver.timesteps[0], 0)
    check('order3 step1 shape', x1.shape == (8, 4))
    check('order3 step1 finite', x1.isfinite().all().item())

    x0_2 = torch.randn(8, 4)
    x2 = solver.step(x1, x0_2, solver.timesteps[1], 1)
    check('order3 step2 finite', x2.isfinite().all().item())

    x0_3 = torch.randn(8, 4)
    x3 = solver.step(x2, x0_3, solver.timesteps[2], 2)
    check('order3 step3 finite', x3.isfinite().all().item())

    check('order3 history len', len(solver.x0_history) == 3)


# ============================================================
# 66. DiTDiffusionDetHead q_sample (DDPM)
# ============================================================
def test_dit_head_q_sample_ddpm():
    print('\n=== test_dit_head_q_sample_ddpm ===')
    num_classes = 24
    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, timesteps=1000, sampling_timesteps=2,
        diffusion_type='ddpm', solver_type='euler',
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )

    x_start = torch.randn(8, 4)
    t = torch.tensor([0, 100, 500, 999, 0, 100, 500, 999])
    noise = torch.randn_like(x_start)

    x_noisy = head.q_sample(x_start, t, noise=noise)
    check('ddpm q_sample shape', x_noisy.shape == (8, 4))
    check('ddpm q_sample finite', x_noisy.isfinite().all().item())

    x_noisy_t0 = head.q_sample(x_start, torch.zeros(8, dtype=torch.long), noise=noise)
    check('ddpm t=0 ~ x_start', torch.allclose(x_noisy_t0, x_start, atol=0.05))


# ============================================================
# 67. DiTDiffusionDetHead predict_noise_from_start
# ============================================================
def test_dit_head_predict_noise():
    print('\n=== test_dit_head_predict_noise ===')
    num_classes = 24
    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, timesteps=1000, sampling_timesteps=2,
        diffusion_type='ddpm', solver_type='euler',
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )

    x_t = torch.randn(8, 4)
    t = torch.tensor([100, 200, 300, 400, 500, 600, 700, 800])
    x0 = torch.randn(8, 4)

    pred_noise = head.predict_noise_from_start(x_t, t, x0)
    check('pred_noise shape', pred_noise.shape == (8, 4))
    check('pred_noise finite', pred_noise.isfinite().all().item())


# ============================================================
# 68. DiTDiffusionDetHead._get_img_shape / _get_scale_factor
# ============================================================
def test_dit_head_get_img_shape():
    print('\n=== test_dit_head_get_img_shape ===')
    meta_obj = ImageMeta(img_shape=(800, 1333), ori_shape=(1600, 2666), scale_factor=[0.5, 0.5, 0.5, 0.5])
    shape = DiTDiffusionDetHead._get_img_shape(meta_obj)
    check('obj img_shape', shape == (800, 1333))

    meta_dict = {'img_shape': (600, 1000)}
    shape_dict = DiTDiffusionDetHead._get_img_shape(meta_dict)
    check('dict img_shape', shape_dict == (600, 1000))


def test_dit_head_get_scale_factor():
    print('\n=== test_dit_head_get_scale_factor ===')
    meta_obj = ImageMeta(img_shape=(800, 1333), scale_factor=[0.5, 0.5, 0.5, 0.5])
    sf = DiTDiffusionDetHead._get_scale_factor(meta_obj)
    check('obj scale_factor', sf == [0.5, 0.5, 0.5, 0.5])

    meta_dict = {'img_shape': (800, 1333)}
    sf_dict = DiTDiffusionDetHead._get_scale_factor(meta_dict)
    check('dict no scale_factor', sf_dict is None)

    meta_no_sf = ImageMeta(img_shape=(800, 1333))
    sf_none = DiTDiffusionDetHead._get_scale_factor(meta_no_sf)
    check('obj no scale_factor', sf_none is None)


# ============================================================
# 69. sigmoid_focal_loss raw function
# ============================================================
def test_sigmoid_focal_loss_raw():
    print('\n=== test_sigmoid_focal_loss_raw ===')
    inputs = torch.randn(50, 24)
    targets = torch.zeros(50, 24)
    targets.scatter_(1, torch.randint(0, 24, (50, 1)), 1.0)

    loss_none = sigmoid_focal_loss(inputs, targets, reduction='none')
    check('raw shape', loss_none.shape == (50, 24))
    check('raw finite', loss_none.isfinite().all().item())
    check('raw >= 0', (loss_none >= 0).all().item())

    loss_mean = sigmoid_focal_loss(inputs, targets, reduction='mean')
    check('mean scalar', loss_mean.dim() == 0)
    check('mean finite', loss_mean.isfinite().item())

    loss_sum = sigmoid_focal_loss(inputs, targets, reduction='sum')
    check('sum scalar', loss_sum.dim() == 0)
    check('sum > mean', loss_sum.item() > loss_mean.item())

    loss_alpha0 = sigmoid_focal_loss(inputs, targets, alpha=0.0, reduction='sum')
    loss_alpha05 = sigmoid_focal_loss(inputs, targets, alpha=0.5, reduction='sum')
    check('alpha affects loss', not torch.isclose(loss_alpha0, loss_alpha05))


# ============================================================
# 70. ROI rescale
# ============================================================
def test_roi_rescale():
    print('\n=== test_roi_rescale ===')
    from mods.roi_extractor import SingleRoIExtractor

    extractor = SingleRoIExtractor(
        roi_layer=dict(type='RoIAlign', output_size=(7, 7), sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32],
    )

    rois = torch.tensor([
        [0, 10.0, 20.0, 50.0, 60.0],
        [1, 30.0, 40.0, 100.0, 120.0],
    ])
    rescaled = extractor.roi_rescale(rois, 2.0)
    check('rescaled shape', rescaled.shape == rois.shape)
    check('rescaled batch idx same', torch.equal(rescaled[:, 0], rois[:, 0]))

    cx_orig = (rois[:, 1] + rois[:, 3]) / 2
    cy_orig = (rois[:, 2] + rois[:, 4]) / 2
    cx_resc = (rescaled[:, 1] + rescaled[:, 3]) / 2
    cy_resc = (rescaled[:, 2] + rescaled[:, 4]) / 2
    check('center preserved', torch.allclose(cx_orig, cx_resc, atol=1e-4) and torch.allclose(cy_orig, cy_resc, atol=1e-4))

    w_orig = rois[:, 3] - rois[:, 1]
    w_resc = rescaled[:, 3] - rescaled[:, 1]
    check('width scaled', torch.allclose(w_resc, w_orig * 2.0, atol=1e-4))


# ============================================================
# 71. Empty batch forward
# ============================================================
def test_empty_batch_forward():
    print('\n=== test_empty_batch_forward ===')
    bs, C, num_classes = 1, 256, 24
    num_proposals = 20
    num_levels = 4

    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=C, num_proposals=num_proposals,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=num_levels, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )
    head.train()

    fpn_features = [torch.rand(bs, C, 15 + l * 3, 15 + l * 3) for l in range(num_levels)]
    img_metas = [ImageMeta(img_shape=(800, 1333), scale_factor=[1.0, 1.0, 1.0, 1.0])]
    gt_bboxes = [torch.zeros(0, 4)]
    gt_labels = [torch.zeros(0, dtype=torch.long)]

    losses = head.loss(fpn_features, img_metas, gt_bboxes, gt_labels)
    check('empty gt loss finite', all(v.isfinite().item() for v in losses.values()))
    check('empty gt has loss_cls', 'loss_cls' in losses)


# ============================================================
# 72. Zero-width bbox handling
# ============================================================
def test_zero_width_bbox():
    print('\n=== test_zero_width_bbox ===')
    xyxy = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    cxcywh = bbox_xyxy_to_cxcywh(xyxy)
    check('zero w', torch.isclose(cxcywh[0, 2], torch.tensor(0.0), atol=1e-6))
    check('zero h', torch.isclose(cxcywh[0, 3], torch.tensor(0.0), atol=1e-6))

    back = bbox_cxcywh_to_xyxy(cxcywh)
    check('zero roundtrip', torch.allclose(xyxy, back, atol=1e-6))

    rel_cost = RelativeL1Cost(weight=5.0, eps=1e-2)
    pred_bboxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    gt_bboxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    gt_labels = torch.tensor([0])
    dummy_logits = torch.zeros(1, 24)
    cost = rel_cost(dummy_logits, pred_bboxes, gt_labels, gt_bboxes)
    check('zero width cost finite', cost.isfinite().all().item())


# ============================================================
# 73. bbox2roi multi-batch
# ============================================================
def test_bbox2roi_multi_batch():
    print('\n=== test_bbox2roi_multi_batch ===')
    b0 = torch.tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])
    b1 = torch.tensor([[100.0, 200.0, 300.0, 400.0]])
    b2 = torch.zeros(0, 4)

    rois = bbox2roi([b0, b1, b2])
    check('multi batch shape', rois.shape == (3, 5))
    check('batch 0 idx', (rois[:2, 0] == 0).all().item())
    check('batch 1 idx', rois[2, 0].item() == 1)
    check('coords match', torch.equal(rois[0, 1:], b0[0]) and torch.equal(rois[2, 1:], b1[0]))

    rois_with_empty = bbox2roi([b2])
    check('empty batch shape', rois_with_empty.shape == (0, 5))


# ============================================================
# 74. IoUCost with iou mode
# ============================================================
def test_iou_cost_iou_mode():
    print('\n=== test_iou_cost_iou_mode ===')
    pred_bboxes = torch.tensor([[0.1, 0.2, 0.5, 0.6]])
    gt_bboxes = torch.tensor([[0.3, 0.4, 0.7, 0.8]])
    gt_labels = torch.tensor([0])
    dummy_logits = torch.zeros(1, 24)

    iou_cost_giou = IoUCost(iou_mode='giou', weight=2.0)
    c_giou = iou_cost_giou(dummy_logits, pred_bboxes, gt_labels, gt_bboxes)

    iou_cost_iou = IoUCost(iou_mode='iou', weight=2.0)
    c_iou = iou_cost_iou(dummy_logits, pred_bboxes, gt_labels, gt_bboxes)

    check('giou cost shape', c_giou.shape == (1, 1))
    check('iou cost shape', c_iou.shape == (1, 1))
    check('giou cost finite', c_giou.isfinite().all().item())
    check('iou cost finite', c_iou.isfinite().all().item())

    same = torch.tensor([[0.1, 0.2, 0.5, 0.6]])
    c_same = iou_cost_iou(dummy_logits, same, gt_labels, same)
    check('iou self cost ~0', c_same.item() < 0.01, f'cost={c_same.item():.6f}')


# ============================================================
# 75. FocalLoss alpha/gamma sensitivity
# ============================================================
def test_focal_loss_alpha_gamma():
    print('\n=== test_focal_loss_alpha_gamma ===')
    pred = torch.randn(50, 24)
    labels = torch.randint(0, 24, (50,))

    fl_default = FocalLoss(alpha=0.25, gamma=2.0, loss_weight=1.0)
    fl_gamma0 = FocalLoss(alpha=0.25, gamma=0.0, loss_weight=1.0)

    loss_default = fl_default(pred, labels)
    loss_gamma0 = fl_gamma0(pred, labels)

    check('gamma affects loss', not torch.isclose(loss_default, loss_gamma0),
          f'default={loss_default.item():.4f}, gamma0={loss_gamma0.item():.4f}')
    check('both finite', loss_default.isfinite().item() and loss_gamma0.isfinite().item())
    check('both > 0', loss_default.item() > 0 and loss_gamma0.item() > 0)


# ============================================================
# 76. L1Loss reduction modes
# ============================================================
def test_l1_loss_reduction():
    print('\n=== test_l1_loss_reduction ===')
    pred = torch.rand(10, 4)
    target = torch.rand(10, 4)

    l1_sum = L1Loss(reduction='sum', loss_weight=1.0)
    l1_mean = L1Loss(reduction='mean', loss_weight=1.0)

    loss_sum = l1_sum(pred, target)
    loss_mean = l1_mean(pred, target)

    check('sum > mean', loss_sum.item() > loss_mean.item())
    check('both finite', loss_sum.isfinite().item() and loss_mean.isfinite().item())
    check('sum > 0', loss_sum.item() > 0)


# ============================================================
# 77. FlowMatchingVelocityLoss with fg_mask
# ============================================================
def test_velocity_loss_fg_mask():
    print('\n=== test_velocity_loss_fg_mask ===')
    vel_loss = FlowMatchingVelocityLoss(loss_weight=5.0)

    v_pred = torch.randn(50, 4)
    v_target = torch.randn(50, 4)
    fg_mask = torch.zeros(50, dtype=torch.bool)
    fg_mask[:10] = True

    loss_masked = vel_loss(v_pred, v_target, fg_mask=fg_mask)
    check('masked loss finite', loss_masked.isfinite().item())
    check('masked loss > 0', loss_masked.item() > 0)

    loss_unmasked = vel_loss(v_pred, v_target)
    check('unmasked loss finite', loss_unmasked.isfinite().item())

    fg_empty = torch.zeros(50, dtype=torch.bool)
    loss_empty = vel_loss(v_pred, v_target, fg_mask=fg_empty)
    check('empty fg loss == 0', loss_empty.item() == 0.0)


# ============================================================
# 78. Criterion with no positive matches
# ============================================================
def test_criterion_no_positive():
    print('\n=== test_criterion_no_positive ===')
    N, M, C, bs = 10, 5, 24, 1

    pred_logits = torch.randn(bs, N, C)
    pred_bboxes = torch.rand(bs, N, 4)
    pred_bboxes[:, :, 2] = pred_bboxes[:, :, 0] + 0.01
    pred_bboxes[:, :, 3] = pred_bboxes[:, :, 1] + 0.01

    far_gt = torch.tensor([[0.9, 0.9, 0.99, 0.99]] * M)
    targets = [InstanceData(
        labels=torch.randint(0, C, (M,)),
        bboxes=far_gt,
        img_shape=(800, 1333),
    )]

    matcher = DiffusionDetMatcher(
        match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
        center_radius=0.01, candidate_topk=1,
    )
    criterion = DiffusionDetCriterion(
        num_classes=C, matcher=matcher,
        loss_cls=FocalLoss(loss_weight=2.0),
        loss_bbox=L1Loss(loss_weight=5.0),
        loss_giou=GIoULoss(loss_weight=2.0),
        bbox_loss_mode='relative_l1',
    )
    outputs = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_bboxes)
    losses = criterion(outputs, targets)
    check('no pos loss finite', all(v.isfinite().item() for v in losses.values()))


# ============================================================
# 79. Matcher with single GT
# ============================================================
def test_matcher_single_gt():
    print('\n=== test_matcher_single_gt ===')
    N, M, C = 50, 1, 24
    bs = 1

    pred_logits = torch.randn(bs, N, C)
    pred_bboxes = torch.rand(bs, N, 4)
    pred_bboxes[:, :, 2] = pred_bboxes[:, :, 0] + 0.1
    pred_bboxes[:, :, 3] = pred_bboxes[:, :, 1] + 0.1

    gt_bboxes_i = torch.tensor([[0.3, 0.3, 0.5, 0.5]])
    targets = [InstanceData(
        labels=torch.tensor([5]),
        bboxes=gt_bboxes_i,
        img_shape=(800, 1333),
    )]

    matcher = DiffusionDetMatcher(
        match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
        center_radius=0.5, candidate_topk=5,
    )
    outputs = ModelOutput(pred_logits=pred_logits, pred_boxes=pred_bboxes)
    indices = matcher(outputs, targets)
    src_idx, gt_idx = indices[0]

    check('single gt matched > 0', len(src_idx) > 0)
    check('single gt all mapped to 0', (gt_idx == 0).all().item())
    check('single gt src unique', len(src_idx) == len(src_idx.unique()))


# ============================================================
# 80. BoxTokenizer position embedding
# ============================================================
def test_box_tokenizer_pos_embed():
    print('\n=== test_box_tokenizer_pos_embed ===')
    C = 256
    tok = BoxTokenizer(feat_channels=C, num_fpn_levels=4, init_mode='bilinear')

    bboxes = torch.rand(2, 10, 4)
    bboxes[:, :, 2] = bboxes[:, :, 0] + 0.1
    bboxes[:, :, 3] = bboxes[:, :, 1] + 0.1
    bboxes = bboxes.clamp(0, 1)

    pos = tok.bbox_pos_embed(bboxes)
    check('pos_embed shape', pos.shape == (2, 10, C))
    check('pos_embed finite', pos.isfinite().all().item())

    lvl_idx = tok._assign_fpn_level(bboxes, None)
    lvl_emb = tok.level_embed(lvl_idx)
    check('level_embed shape', lvl_emb.shape == (2, 10, C))
    check('level_embed finite', lvl_emb.isfinite().all().item())


# ============================================================
# 81. Deformable attention gradient flow
# ============================================================
def test_deformable_attn_gradient():
    print('\n=== test_deformable_attn_gradient ===')
    bs, N, C, num_heads, num_levels, num_points = 1, 5, 64, 2, 2, 4
    attn = MultiScaleDeformableAttention(
        embed_dim=C, num_heads=num_heads, num_levels=num_levels, num_points=num_points
    )

    query = torch.rand(bs, N, C, requires_grad=True)
    ref_points = torch.rand(bs, N, num_levels, 2)
    fpn_feats = [torch.rand(bs, C, 10 + l * 3, 10 + l * 3, requires_grad=True) for l in range(num_levels)]
    flat, shapes, starts = flatten_fpn_features(fpn_feats)

    out = attn(query, ref_points, flat, shapes, starts)
    loss = out.sum()
    loss.backward()

    check('query grad exists', query.grad is not None)
    check('query grad non-zero', (query.grad.abs() > 0).any().item())
    check('query grad finite', query.grad.isfinite().all().item())


# ============================================================
# 82. DiTBlock adaln=6 modulation
# ============================================================
def test_dit_block_adaln6_modulate():
    print('\n=== test_dit_block_adaln6_modulate ===')
    C = 256
    block = DiTBlock(
        feat_channels=C, num_heads=8, num_fpn_levels=4,
        num_ref_points=8, dim_feedforward=2048, adaln_params=6,
    )

    last_linear = block.adaln_mlp[-1]
    check('adaln6 weight zero', (last_linear.weight == 0).all().item())
    check('adaln6 bias zero', (last_linear.bias == 0).all().item())

    bs, N = 2, 10
    box_tokens = torch.rand(bs, N, C)
    fpn_feats = [torch.rand(bs, C, 20 + l * 5, 20 + l * 5) for l in range(4)]
    flat, shapes, starts = flatten_fpn_features(fpn_feats)
    time_emb = torch.rand(bs, C * 4)
    bbox_coords = torch.rand(bs, N, 4)
    bbox_coords[:, :, 2] = bbox_coords[:, :, 0] + 0.1
    bbox_coords[:, :, 3] = bbox_coords[:, :, 1] + 0.1
    bbox_coords = bbox_coords.clamp(0, 1)

    out = block(box_tokens, flat, shapes, starts, time_emb, bbox_coords)
    check('adaln6 identity', torch.allclose(out, box_tokens, atol=1e-5),
          f'max diff: {(out - box_tokens).abs().max().item():.8f}')


# ============================================================
# 83. Direct regression batch consistency
# ============================================================
def test_direct_regression_batch_consistency():
    print('\n=== test_direct_regression_batch_consistency ===')
    head = DiTSingleHead(
        num_classes=24, feat_channels=256, num_heads=8,
        num_fpn_levels=4, num_ref_points=8,
        prediction_mode='x0', adaln_params=9, regression_mode='direct',
    )

    fc_feature = torch.randn(3, 10, 256)
    bboxes = torch.rand(3, 10, 4)
    bboxes[:, :, 2] = bboxes[:, :, 0] + 0.1
    bboxes[:, :, 3] = bboxes[:, :, 1] + 0.1
    bboxes = bboxes.clamp(0, 1)

    pred = head._predict_bboxes(fc_feature, bboxes)
    check('batch pred shape', pred.shape == (3, 10, 4))
    check('batch pred in [0,1]', pred.min().item() >= 0 and pred.max().item() <= 1)
    check('batch x2>=x1', (pred[..., 2] >= pred[..., 0] - 1e-6).all().item())
    check('batch y2>=y1', (pred[..., 3] >= pred[..., 1] - 1e-6).all().item())


# ============================================================
# 84. Delta regression clamp
# ============================================================
def test_delta_regression_clamp():
    print('\n=== test_delta_regression_clamp ===')
    head = DiTSingleHead(
        num_classes=24, feat_channels=256, num_heads=8,
        num_fpn_levels=4, num_ref_points=8,
        prediction_mode='x0', adaln_params=9, regression_mode='delta',
    )

    bboxes = torch.tensor([[0.4, 0.4, 0.6, 0.6]])
    large_deltas = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    pred = head.apply_deltas(large_deltas, bboxes)
    check('large delta finite', pred.isfinite().all().item())
    check('large delta in [0,1]', pred.min().item() >= 0 and pred.max().item() <= 1)

    neg_deltas = torch.tensor([[-10.0, -10.0, -10.0, -10.0]])
    pred_neg = head.apply_deltas(neg_deltas, bboxes)
    check('neg delta finite', pred_neg.isfinite().all().item())
    check('neg delta in [0,1]', pred_neg.min().item() >= 0 and pred_neg.max().item() <= 1)


# ============================================================
# 85. RF schedule: linear
# ============================================================
def test_rf_schedule_linear():
    print('\n=== test_rf_schedule_linear ===')
    rf = RectifiedFlow(snr_scale=2.0)
    x0 = torch.randn(8, 4)
    x_noise = torch.randn_like(x0)

    t0 = torch.zeros(8)
    x_t0, v0 = rf.q_sample(x0, x_noise=x_noise, t=t0)
    check('linear t=0 == x0', torch.allclose(x_t0, x0, atol=1e-5))

    t1 = torch.ones(8)
    x_t1, v1 = rf.q_sample(x0, x_noise=x_noise, t=t1)
    check('linear t=1 == noise', torch.allclose(x_t1, x_noise, atol=1e-5))

    check('v == noise - x0', torch.allclose(v0, x_noise - x0, atol=1e-5))


# ============================================================
# 86. RF schedule: power
# ============================================================
def test_rf_schedule_power():
    print('\n=== test_rf_schedule_power ===')
    rf = RectifiedFlow(snr_scale=2.0)
    x0 = torch.randn(8, 4)
    x_noise = torch.randn_like(x0)
    t = torch.rand(8)

    x_t, v = rf.q_sample(x0, x_noise=x_noise, t=t)
    t_view = t.view(-1, 1)
    expected = (1 - t_view) * x0 + t_view * x_noise
    check('power schedule formula', torch.allclose(x_t, expected, atol=1e-5))


# ============================================================
# 87. Sinkhorn transport marginal constraints
# ============================================================
def test_sinkhorn_transport_marginals():
    print('\n=== test_sinkhorn_transport_marginals ===')
    num_classes = 24
    head = DiTDiffusionDetHead(
        num_classes=num_classes, feat_channels=256, num_proposals=50,
        num_heads=2, snr_scale=2.0, sampling_timesteps=2,
        diffusion_type='rectified_flow', solver_type='euler',
        ot_coupling=True, ot_matcher='sinkhorn', ot_epsilon=1.0, ot_num_iters=50,
        regression_mode='direct', prediction_mode='x0',
        adaln_params=9, num_fpn_levels=4, num_ref_points=4,
        box_init_mode='bilinear', deep_supervision=False,
        criterion=DiffusionDetCriterion(
            num_classes=num_classes,
            matcher=DiffusionDetMatcher(
                match_costs=[FocalLossCost(weight=2.0), RelativeL1Cost(weight=5.0), IoUCost(iou_mode='giou', weight=2.0)],
                center_radius=0.5, candidate_topk=5,
            ),
            loss_cls=FocalLoss(loss_weight=2.0),
            loss_bbox=L1Loss(loss_weight=5.0),
            loss_giou=GIoULoss(loss_weight=2.0),
            bbox_loss_mode='relative_l1',
        ),
    )

    noise = torch.randn(50, 4)
    gt_diffusion = torch.randn(5, 4)
    cost = torch.cdist(noise, gt_diffusion, p=2)

    transport = head._sinkhorn_transport(cost)
    check('transport shape', transport.shape == (50, 5))
    check('transport non-negative', (transport >= -1e-6).all().item())
    check('transport finite', transport.isfinite().all().item())
    check('transport sum ~1', abs(transport.sum().item() - 1.0) < 0.1,
          f'sum={transport.sum().item():.4f}')


# ============================================================
# 88. DPM-Solver multistep consistency
# ============================================================
def test_dpm_solver_multistep_consistency():
    print('\n=== test_dpm_solver_multistep_consistency ===')
    solver = RFDPMSolverMultistep(num_steps=2, solver_order=2)
    check('2-step timesteps', len(solver.timesteps) == 3)
    check('start=1', abs(solver.timesteps[0] - 1.0) < 1e-6)
    check('end=0', abs(solver.timesteps[-1] - 0.0) < 1e-6)

    x = torch.randn(4, 4)
    x0 = torch.randn(4, 4)
    solver.reset()

    x1 = solver.step(x, x0, solver.timesteps[0], 0)
    check('first step shape', x1.shape == (4, 4))
    check('first step finite', x1.isfinite().all().item())

    x0_2 = torch.randn(4, 4)
    x2 = solver.step(x1, x0_2, solver.timesteps[1], 1)
    check('second step finite', x2.isfinite().all().item())

    solver2 = RFDPMSolverMultistep(num_steps=1, solver_order=1)
    check('1-step timesteps', len(solver2.timesteps) == 2)
    solver2.reset()
    x_single = solver2.step(x, x0, solver2.timesteps[0], 0)
    check('single step finite', x_single.isfinite().all().item())


if __name__ == '__main__':
    torch.manual_seed(42)

    test_bbox_conversion()
    test_sinusoidal_embeddings()
    test_cosine_schedule()
    test_box_tokenizer()
    test_reference_points()
    test_fpn_utils()
    test_deformable_attn()
    test_dit_block()
    test_dit_single_head_direct()
    test_dit_single_head_delta()
    test_dit_single_head_velocity()
    test_rectified_flow()
    test_dpm_solver()
    test_loss_functions()
    test_cost_functions()
    test_matcher()
    test_criterion()
    test_coordinate_roundtrip()
    test_dit_head_forward()
    test_dit_head_loss()
    test_dit_head_predict()
    test_dit_head_predict_heun()
    test_dit_head_predict_dpm()
    test_cost_sensitivity()
    test_center_radius_impact()
    test_direct_regression_validity()
    test_delta_regression_validity()
    test_backward()
    test_predict_rescale()
    test_ddpm_mode()
    test_structures()
    test_load_buffer()
    test_bbox2roi()
    test_roi_extractor()
    test_ot_sinkhorn()
    test_ot_nearest()
    test_ot_empty_gt()
    test_time_sampling()
    test_lsas_sampling()
    test_shifted_schedule()
    test_scale_aware_loss()
    test_objectness_loss()
    test_apply_deltas_stability()
    test_box_renewal()
    test_velocity_loss_training()
    test_ddpm_q_sample()
    test_coordinate_edge_cases()
    test_matcher_duplicate()
    test_focal_loss_modes()
    test_giou_loss_properties()
    test_dynamic_conv()
    test_dit_block_zero_init()
    test_cls_bias_init()
    test_ot_training()
    test_predict_trajectory()
    test_power_schedule()

    # ---- 新增测试: 模块内部方法与边界条件 ----
    test_box_tokenizer_assign_level()
    test_box_tokenizer_bilinear_sample()
    test_rectified_flow_get_velocity()
    test_rectified_flow_heun_step()
    test_dpm_solver_order3()
    test_dit_head_q_sample_ddpm()
    test_dit_head_predict_noise()
    test_dit_head_get_img_shape()
    test_dit_head_get_scale_factor()
    test_sigmoid_focal_loss_raw()
    test_roi_rescale()
    test_empty_batch_forward()
    test_zero_width_bbox()
    test_bbox2roi_multi_batch()
    test_iou_cost_iou_mode()
    test_focal_loss_alpha_gamma()
    test_l1_loss_reduction()
    test_velocity_loss_fg_mask()
    test_criterion_no_positive()
    test_matcher_single_gt()
    test_box_tokenizer_pos_embed()
    test_deformable_attn_gradient()
    test_dit_block_adaln6_modulate()
    test_direct_regression_batch_consistency()
    test_delta_regression_clamp()
    test_rf_schedule_linear()
    test_rf_schedule_power()
    test_sinkhorn_transport_marginals()
    test_dpm_solver_multistep_consistency()

    print(f'\n{"="*60}')
    print(f'Total: {PASS + FAIL} | PASS: {PASS} | FAIL: {FAIL}')
    print(f'{"="*60}')
    if FAIL > 0:
        sys.exit(1)
