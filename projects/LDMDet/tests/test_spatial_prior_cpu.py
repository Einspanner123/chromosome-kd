"""
CPU 模拟训练: 验证 spatial_prior 方案 A 无错误

模拟 3 个训练 step 的完整前向+反向流程:
  1. 构建模型 (spatial_prior 模式)
  2. 生成模拟数据 (含 GT 框)
  3. 前向传播 → loss
  4. 反向传播 → 梯度检查
  5. 多 step 迭代稳定性

运行:
  PYTHONPATH=. python projects/LDMDet/tests/test_spatial_prior_cpu.py
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, 'projects/LDMDet')

from mods.box_tokenizer import BoxTokenizer
from mods.deformable_attn import flatten_fpn_features
from mods.dit_head import DiTDiffusionDetHead
from mods.loss import (
    DiffusionDetMatcher,
    DiffusionDetCriterion,
    FocalLoss,
    L1Loss,
    GIoULoss,
    FocalLossCost,
    RelativeL1Cost,
    IoUCost,
)
from mods.structures import InstanceData, ImageMeta

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


def make_data(
    bs=2, num_proposals=100, num_gt=3, feat_channels=384, img_h=800, img_w=1333
):
    """生成模拟训练数据"""
    device = torch.device('cpu')

    # FPN 特征: P2-P5 (4 层)
    fpn_features = [
        torch.randn(bs, feat_channels, img_h // 2 ** i, img_w // 2 ** i)
        for i in range(4)
    ]

    # GT 框 (xyxy, 图像坐标像素)
    gt_bboxes_list = []
    gt_labels_list = []
    for b in range(bs):
        # 随机生成 GT 框在图像中
        cx = (torch.rand(num_gt) * 0.6 + 0.2) * img_w
        cy = (torch.rand(num_gt) * 0.6 + 0.2) * img_h
        w = torch.rand(num_gt) * 0.3 * img_w + 10
        h = torch.rand(num_gt) * 0.3 * img_h + 10
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        gt_bboxes = torch.stack([x1, y1, x2, y2], dim=1)
        gt_labels = torch.zeros(num_gt, dtype=torch.long)  # class 0
        gt_bboxes_list.append(gt_bboxes)
        gt_labels_list.append(gt_labels)

    # img_metas
    img_metas = []
    for b in range(bs):
        img_metas.append(
            ImageMeta(
                img_shape=(img_h, img_w),
                scale_factor=(1.0, 1.0),
            )
        )

    return fpn_features, gt_bboxes_list, gt_labels_list, img_metas


def build_head(lightweight=False):
    """构建 DiTDiffusionDetHead (spatial_prior 模式)

    Args:
        lightweight: 若为 True, 使用小模型用于 CPU 快速验证
    """
    if lightweight:
        # CPU 轻量级: 避免 OOM
        feat_channels = 64
        num_blocks = 1
        num_heads = 1
        share_heads = True
        dim_feedforward = 256
    else:
        feat_channels = 384
        num_blocks = 3
        num_heads = 3
        share_heads = False
        dim_feedforward = 1536

    # 构建匹配器
    match_costs = [
        FocalLossCost(weight=2.0),
        RelativeL1Cost(weight=5.0),
        IoUCost(iou_mode='giou', weight=2.0),
    ]
    matcher = DiffusionDetMatcher(
        match_costs=match_costs,
        center_radius=5.0,
        candidate_topk=12,
    )
    criterion = DiffusionDetCriterion(
        num_classes=1,
        matcher=matcher,
        loss_cls=FocalLoss(loss_weight=2.0),
        loss_bbox=L1Loss(loss_weight=5.0),
        loss_giou=GIoULoss(loss_weight=2.0),
        bbox_loss_mode='relative_l1',
    )

    head = DiTDiffusionDetHead(
        num_classes=1,
        feat_channels=feat_channels,
        num_proposals=100,
        num_heads=num_heads,
        num_blocks=num_blocks,
        share_heads=share_heads,
        deep_supervision=True,
        prior_prob=0.01,
        snr_scale=2.0,
        sampling_timesteps=6,
        diffusion_type='rectified_flow',
        solver_type='heun',
        rf_schedule='shifted',
        rf_shift=1.0,
        box_renewal=True,
        use_ensemble=True,
        prediction_mode='x0',
        adaln_params=9,
        regression_mode='delta',
        use_adaln_zero=True,
        num_fpn_levels=4,
        num_ref_points=8,
        box_init_mode='spatial_prior',
        ot_coupling=True,
        ot_matcher='sinkhorn',
        single_head=None,
        criterion=criterion,
    )
    return head


def test_spatial_prior_config():
    """测试 1: 验证 spatial_prior 配置正确加载"""
    print('\n=== Test 1: spatial_prior 配置验证 ===')

    head = build_head(lightweight=True)
    check('box_init_mode == spatial_prior',
          head.box_init_mode == 'spatial_prior')
    check('num_proposals == 100', head.num_proposals == 100)

    tokenizer = head.box_tokenizer
    check('tokenizer init_mode == spatial_prior',
          tokenizer.init_mode == 'spatial_prior')
    check('tokenizer has anchor_boxes buffer',
          hasattr(tokenizer, 'anchor_boxes'))
    check('anchor_boxes shape == (100, 4)',
          tokenizer.anchor_boxes.shape == (100, 4))
    check('anchor_boxes cx in [0.05, 0.95]',
          tokenizer.anchor_boxes[:, 0].min() >= 0.04 and
          tokenizer.anchor_boxes[:, 0].max() <= 0.96)
    check('anchor_boxes cy in [0.05, 0.95]',
          tokenizer.anchor_boxes[:, 1].min() >= 0.04 and
          tokenizer.anchor_boxes[:, 1].max() <= 0.96)
    check('anchor_boxes all w=0.1',
          (tokenizer.anchor_boxes[:, 2] - 0.1).abs().max() < 0.01)
    check('anchor_boxes all h=0.1',
          (tokenizer.anchor_boxes[:, 3] - 0.1).abs().max() < 0.01)

    # 验证 anchor 中心点不重复（空间唯一性）
    centers = tokenizer.anchor_boxes[:, :2]
    unique_centers = torch.unique(centers, dim=0)
    check(f'anchor centers unique: {unique_centers.shape[0]}/{centers.shape[0]}',
          unique_centers.shape[0] == centers.shape[0])

    return head


def test_forward_no_nan(head):
    """测试 2: 前向传播无 NaN/Inf"""
    print('\n=== Test 2: 前向传播无 NaN/Inf ===')

    fpn_features, gt_bboxes, gt_labels, img_metas = make_data(
        feat_channels=head.feat_channels, img_h=400, img_w=400
    )

    head.train()
    losses = head.loss(fpn_features, img_metas, gt_bboxes, gt_labels)

    check('losses is dict', isinstance(losses, dict))
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            check(f'  {k}: {v.item():.4f}, no NaN',
                  not v.isnan().any().item())
            check(f'  {k}: {v.item():.4f}, no Inf',
                  not v.isinf().any().item())
            check(f'  {k}: {v.item():.4f}, finite',
                  v.isfinite().all().item())
        print(f'  loss[{k}] = {v}')

    return losses


def test_backward_no_nan(head):
    """测试 3: 反向传播梯度无 NaN/Inf/爆炸"""
    print('\n=== Test 3: 反向传播梯度检查 ===')

    fpn_features, gt_bboxes, gt_labels, img_metas = make_data(
        feat_channels=head.feat_channels, img_h=400, img_w=400
    )

    head.train()
    losses = head.loss(fpn_features, img_metas, gt_bboxes, gt_labels)

    total_loss = sum(
        v for v in losses.values() if isinstance(v, torch.Tensor)
    )
    total_loss.backward()

    max_grad = 0.0
    min_grad = 0.0
    has_nan = False
    has_inf = False
    grad_count = 0

    for name, p in head.named_parameters():
        if p.grad is not None:
            grad_count += 1
            g = p.grad
            if g.isnan().any():
                has_nan = True
            if g.isinf().any():
                has_inf = True
            max_grad = max(max_grad, g.abs().max().item())
            min_grad = min(min_grad, g.abs().min().item())

    check('grad_count > 0', grad_count > 0,
          f'found {grad_count} params with grad')
    check('no NaN in grads', not has_nan)
    check('no Inf in grads', not has_inf)
    check(f'max_grad ({max_grad:.2f}) < 1000', max_grad < 1000)
    check(f'max_grad ({max_grad:.2f}) > 0', max_grad > 0)

    return max_grad


def test_multi_step_stability(head):
    """测试 4: 多步迭代稳定性"""
    print('\n=== Test 4: 多步迭代稳定性 (3 steps) ===')

    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)

    losses_history = []
    grad_norms = []

    for step in range(3):
        fpn_features, gt_bboxes, gt_labels, img_metas = make_data(
            feat_channels=head.feat_channels, img_h=400, img_w=400
        )

        optimizer.zero_grad()
        losses = head.loss(fpn_features, img_metas, gt_bboxes, gt_labels)

        total_loss = sum(
            v for v in losses.values() if isinstance(v, torch.Tensor)
        )
        total_loss.backward()

        # 梯度裁剪
        total_norm = nn.utils.clip_grad_norm_(head.parameters(), 50.0)
        grad_norms.append(total_norm.item())

        optimizer.step()

        total_loss_val = total_loss.item()
        losses_history.append(total_loss_val)

        check(f'step {step}: loss={total_loss_val:.4f}, gn={total_norm.item():.2f}',
              total_loss_val > 0 and total_loss_val < 1e6)
        check(f'step {step}: grad_norm finite',
              total_norm.isfinite().item())

    # 检查 loss 是否在变化 (不是完全不变)
    changes = sum(
        1 for i in range(1, len(losses_history))
        if abs(losses_history[i] - losses_history[i - 1]) > 0.001
    )
    check(f'loss changes across steps: {changes}/{len(losses_history) - 1}',
          changes > 0,
          'loss 应该在迭代中变化')

    return losses_history, grad_norms


def test_inference_no_crash(head):
    """测试 5: 推理流程无崩溃"""
    print('\n=== Test 5: 推理流程验证 ===')

    fpn_features, _, _, img_metas = make_data(
        bs=1, feat_channels=head.feat_channels, img_h=400, img_w=400
    )

    head.eval()
    with torch.no_grad():
        results = head.predict(fpn_features, img_metas)

    check('predict returns list', isinstance(results, list))
    check('predict returns results', len(results) > 0)

    if len(results) > 0 and len(results[0].bboxes) > 0:
        bboxes = results[0].bboxes
        scores = results[0].scores
        # predict 返回的是图像坐标 (rescale=True)，应在 [0, img_w]×[0, img_h] 内
        bbox_ok = (bboxes[:, 0].min() >= -10 and bboxes[:, 2].max() <= 410)
        check(f'bboxes in image coords: x=[{bboxes[:,0].min():.0f},{bboxes[:,2].max():.0f}] y=[{bboxes[:,1].min():.0f},{bboxes[:,3].max():.0f}]',
              bbox_ok)
        check('scores in [0, 1]',
              (scores.min() >= 0 and scores.max() <= 1.0))
        print(f'  predicted {len(bboxes)} boxes, score range=[{scores.min():.4f}, {scores.max():.4f}]')


def main():
    global PASS, FAIL, ERROR
    print('=' * 60)
    print('CPU 模拟训练: spatial_prior 方案 A 验证')
    print('=' * 60)

    # Test 1: 使用轻量级模型
    head = test_spatial_prior_config()

    # Test 2-4
    test_forward_no_nan(head)
    test_backward_no_nan(head)
    test_multi_step_stability(head)

    # Test 5
    test_inference_no_crash(head)

    # Summary
    total = PASS + FAIL + ERROR
    print(f'\n{"=" * 60}')
    print(f'RESULTS: PASS={PASS}, FAIL={FAIL}, ERROR={ERROR}, TOTAL={total}')
    if FAIL == 0 and ERROR == 0:
        print('ALL TESTS PASSED - CPU 模拟训练无错误，可以启动 GPU 训练')
        print('=' * 60)
        return 0
    else:
        print('SOME TESTS FAILED - 请修复后再启动 GPU 训练')
        print('=' * 60)
        return 1


if __name__ == '__main__':
    exit(main())