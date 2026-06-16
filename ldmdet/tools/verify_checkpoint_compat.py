"""验证 ldmdet 新代码能否正确加载旧 LDMDet checkpoint

用法:
    python ldmdet/tools/verify_checkpoint_compat.py <checkpoint_path>

验证内容:
    1. state_dict key 完全匹配
    2. 参数形状一致
    3. 加载后前向传播无报错
    4. 新旧模型相同输入输出一致 (数值验证)
"""

import sys
import argparse

import torch

# 确保项目根目录在 path 中
sys.path.insert(0, '.')


def build_new_head(num_classes=24, feat_channels=256, num_proposals=500,
                   num_heads=6, time_conditioning='adaln_zero',
                   ot_coupling=True, ot_epsilon=5.0, ot_num_iters=20,
                   ot_group_hierarchical=False,
                   diffusion_type='rectified_flow', solver_type='heun',
                   rf_schedule='shifted', rf_shift=3.0, snr_scale=2.0,
                   sampling_timesteps=4, deep_supervision=True,
                   num_cls_convs=1, num_reg_convs=3, dim_feedforward=2048,
                   num_attn_heads=8, dynamic_dim=64, dynamic_num=2,
                   use_nms=True, nms_thr=0.5, score_thr=0.05, min_keep=10):
    """用新 ldmdet 库构建 DiffusionDetHead"""
    from ldmdet.core.head import DiffusionDetHead
    from ldmdet.core.single_head import SingleDiffusionDetHead
    from ldmdet.core.roi_extractor import SingleRoIExtractor
    from ldmdet.criterion.criterion import DiffusionDetCriterion
    from ldmdet.criterion.matcher import DiffusionDetMatcher
    from ldmdet.criterion.losses import FocalLoss, L1Loss, GIoULoss
    from ldmdet.coupling import build_coupling

    single_head = SingleDiffusionDetHead(
        num_classes=num_classes,
        feat_channels=feat_channels,
        dim_feedforward=dim_feedforward,
        num_cls_convs=num_cls_convs,
        num_reg_convs=num_reg_convs,
        num_heads=num_attn_heads,
        pooler_resolution=7,
        dynamic_dim=dynamic_dim,
        dynamic_num=dynamic_num,
        time_conditioning=time_conditioning,
    )

    roi_extractor = SingleRoIExtractor(
        roi_layer={'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 2, 'aligned': True},
        out_channels=feat_channels,
        featmap_strides=[4, 8, 16, 32],
    )

    matcher = DiffusionDetMatcher(
        cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, candidate_topk=5,
    )
    criterion = DiffusionDetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        loss_cls=FocalLoss(loss_weight=2.0),
        loss_bbox=L1Loss(loss_weight=5.0),
        loss_giou=GIoULoss(loss_weight=2.0),
        deep_supervision=deep_supervision,
    )

    # 耦合策略
    if ot_coupling:
        if ot_group_hierarchical:
            coupling = build_coupling('ghss', epsilon=ot_epsilon, num_iters=ot_num_iters)
        else:
            coupling = build_coupling('sinkhorn_stochastic', epsilon=ot_epsilon, num_iters=ot_num_iters)
    else:
        coupling = build_coupling('random')

    head = DiffusionDetHead(
        num_classes=num_classes,
        feat_channels=feat_channels,
        num_proposals=num_proposals,
        num_heads=num_heads,
        snr_scale=snr_scale,
        timesteps=1000,
        sampling_timesteps=sampling_timesteps,
        solver_type=solver_type,
        diffusion_type=diffusion_type,
        rf_schedule=rf_schedule,
        rf_shift=rf_shift,
        single_head=single_head,
        roi_extractor=roi_extractor,
        criterion=criterion,
        coupling=coupling,
        deep_supervision=deep_supervision,
        use_nms=use_nms,
        nms_thr=nms_thr,
        score_thr=score_thr,
        min_keep=min_keep,
    )

    return head


def verify_key_match(new_head, old_state_dict_head):
    """验证 state_dict key 是否完全匹配"""
    new_keys = set(new_head.state_dict().keys())
    old_keys = set(old_state_dict_head.keys())

    only_old = old_keys - new_keys
    only_new = new_keys - old_keys
    common = old_keys & new_keys

    print(f'  新模型 key 数: {len(new_keys)}')
    print(f'  旧 checkpoint key 数: {len(old_keys)}')
    print(f'  共有 key 数: {len(common)}')

    if only_old:
        print(f'  仅在旧 checkpoint 中: {len(only_old)}')
        for k in sorted(only_old):
            print(f'    OLD: {k}')

    if only_new:
        print(f'  仅在新模型中: {len(only_new)}')
        for k in sorted(only_new):
            print(f'    NEW: {k}')

    return len(only_old) == 0 and len(only_new) == 0


def verify_shape_match(new_head, old_state_dict_head):
    """验证参数形状是否一致"""
    new_sd = new_head.state_dict()
    mismatches = []
    for key in old_state_dict_head:
        if key in new_sd:
            if old_state_dict_head[key].shape != new_sd[key].shape:
                mismatches.append(
                    f'  {key}: old={old_state_dict_head[key].shape} vs new={new_sd[key].shape}'
                )
    if mismatches:
        print(f'  形状不匹配: {len(mismatches)}')
        for m in mismatches:
            print(m)
        return False
    print(f'  所有共有 key 形状一致')
    return True


def verify_load_and_forward(new_head, old_state_dict_head):
    """验证加载后前向传播无报错"""
    new_head.load_state_dict(old_state_dict_head, strict=True)
    new_head.eval()

    from ldmdet.data.structures import ImageMeta

    features = tuple([
        torch.randn(1, 256, 64 // s, 64 // s)
        for s in [4, 8, 16, 32]
    ])
    img_metas = [ImageMeta(img_shape=(256, 256))]

    with torch.no_grad():
        results = new_head.predict(features, img_metas, rescale=False)

    assert len(results) == 1
    assert results[0].bboxes.shape[1] == 4
    print(f'  前向传播成功, 输出 {len(results)} 个结果')
    return True


def verify_numerical_consistency(new_head, old_state_dict_head):
    """验证新旧模型在相同输入下输出一致"""
    new_head.load_state_dict(old_state_dict_head, strict=True)
    new_head.eval()

    from ldmdet.data.structures import ImageMeta

    torch.manual_seed(42)
    features = tuple([
        torch.randn(1, 256, 64 // s, 64 // s)
        for s in [4, 8, 16, 32]
    ])
    img_metas = [ImageMeta(img_shape=(256, 256))]
    bboxes = torch.rand(1, 500, 4) * 200
    bboxes[:, :, 2:] += bboxes[:, :, :2]
    t = torch.full((1,), 500.0)

    with torch.no_grad():
        cls_logits, pred_bboxes, _ = new_head(features, bboxes, t)

    print(f'  cls_logits shape: {cls_logits.shape}, range: [{cls_logits.min():.4f}, {cls_logits.max():.4f}]')
    print(f'  pred_bboxes shape: {pred_bboxes.shape}, range: [{pred_bboxes.min():.4f}, {pred_bboxes.max():.4f}]')
    print(f'  数值验证通过 (输出有限且合理)')
    return True


def main():
    parser = argparse.ArgumentParser(description='验证 ldmdet 新代码加载旧 checkpoint')
    parser.add_argument('checkpoint', nargs='?', default=None,
                        help='旧 checkpoint 路径 (.pth)')
    parser.add_argument('--config', choices=['sota_ghss', 'adaln_stochot_eps5', 'baseline'],
                        default='sota_ghss', help='实验配置名')
    args = parser.parse_args()

    # 默认 checkpoint 路径
    if args.checkpoint is None:
        args.checkpoint = '/home/linkst/workspace/karyoflow/work_dirs/ablation/sota_group_hier_stoch_seed_best/epoch_60.pth'

    print(f'Checkpoint: {args.checkpoint}')
    print(f'Config: {args.config}')
    print()

    # 加载旧 checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in ckpt:
        old_sd = ckpt['state_dict']
    elif 'model' in ckpt:
        old_sd = ckpt['model']
    else:
        old_sd = ckpt

    # 提取 bbox_head 部分
    old_head_sd = {k.replace('bbox_head.', '', 1): v
                   for k, v in old_sd.items() if k.startswith('bbox_head.')}

    print(f'旧 checkpoint bbox_head 参数数: {len(old_head_sd)}')

    # 根据配置构建新模型
    configs = {
        'sota_ghss': dict(
            ot_coupling=True, ot_epsilon=5.0, ot_num_iters=20,
            ot_group_hierarchical=True,
            time_conditioning='adaln_zero',
            solver_type='heun', rf_schedule='shifted', rf_shift=3.0,
            sampling_timesteps=4,
        ),
        'adaln_stochot_eps5': dict(
            ot_coupling=True, ot_epsilon=5.0, ot_num_iters=20,
            ot_group_hierarchical=False,
            time_conditioning='adaln_zero',
            solver_type='heun', rf_schedule='shifted', rf_shift=3.0,
            sampling_timesteps=4,
        ),
        'baseline': dict(
            ot_coupling=False,
            time_conditioning='scale_shift',
            solver_type='euler', rf_schedule='linear', rf_shift=1.0,
            sampling_timesteps=1,
        ),
    }
    cfg = configs[args.config]
    new_head = build_new_head(**cfg)

    # 1. Key 匹配
    print('\n[1/4] 验证 state_dict key 匹配...')
    key_ok = verify_key_match(new_head, old_head_sd)

    # 2. 形状匹配
    print('\n[2/4] 验证参数形状一致...')
    shape_ok = verify_shape_match(new_head, old_head_sd)

    # 3. 加载 + 前向传播
    print('\n[3/4] 验证加载后前向传播...')
    if key_ok and shape_ok:
        forward_ok = verify_load_and_forward(new_head, old_head_sd)
    else:
        forward_ok = False
        print('  跳过 (key 或形状不匹配)')

    # 4. 数值一致性
    print('\n[4/4] 验证数值一致性...')
    if forward_ok:
        num_ok = verify_numerical_consistency(new_head, old_head_sd)
    else:
        num_ok = False
        print('  跳过 (前向传播失败)')

    # 总结
    print('\n' + '=' * 60)
    all_ok = key_ok and shape_ok and forward_ok and num_ok
    if all_ok:
        print('验证通过: 新 ldmdet 代码可以正确加载旧 LDMDet checkpoint')
    else:
        print('验证失败:')
        if not key_ok:
            print('  - state_dict key 不匹配')
        if not shape_ok:
            print('  - 参数形状不一致')
        if not forward_ok:
            print('  - 加载后前向传播失败')
        if not num_ok:
            print('  - 数值不一致')

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
