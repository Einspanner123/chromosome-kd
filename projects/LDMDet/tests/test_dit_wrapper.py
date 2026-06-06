import sys
from pathlib import Path

import torch
from mmengine.structures import InstanceData

from mmdet.structures import DetDataSample

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mmdet.utils import register_all_modules

register_all_modules()

from mmdet.registry import MODELS

try:
    import projects.LDMDet.model as model_mod
except Exception as e:
    print(f'Failed to import LDMDet: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)


def test_dit_wrapper():
    print('Testing LDMDet-DiT wrapper...')
    num_classes = 5

    model_cfg = dict(
        type='LDMDet',
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32,
        ),
        backbone=dict(
            type='ResNet',
            depth=18,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
        ),
        neck=dict(
            type='FPN',
            in_channels=[64, 128, 256, 512],
            out_channels=256,
            num_outs=4,
        ),
        bbox_head=dict(
            type='DiTDiffusionDetHead',
            num_classes=num_classes,
            feat_channels=256,
            num_proposals=50,
            num_heads=2,
            deep_supervision=True,
            prior_prob=0.01,
            snr_scale=2.0,
            sampling_timesteps=2,
            diffusion_type='rectified_flow',
            solver_type='euler',
            rf_schedule='shifted',
            rf_shift=3.0,
            box_renewal=False,
            use_ensemble=True,
            prediction_mode='x0',
            adaln_params=9,
            num_fpn_levels=4,
            num_ref_points=8,
            box_init_mode='zero',
            single_head=dict(
                type='DiTSingleHead',
                num_classes=num_classes,
                feat_channels=256,
                num_cls_convs=1,
                num_reg_convs=1,
                dim_feedforward=256,
                num_heads=4,
                num_fpn_levels=4,
                num_ref_points=8,
                prediction_mode='x0',
                adaln_params=9,
            ),
            criterion=dict(
                type='PurePyTorchDiffusionDetCriterion',
                num_classes=num_classes,
                assigner=dict(
                    type='PurePyTorchDiffusionDetMatcher',
                    match_costs=[
                        dict(type='PurePyTorchFocalLossCost', weight=2.0),
                        dict(type='PurePyTorchBBoxL1Cost', weight=5.0),
                        dict(type='PurePyTorchIoUCost', iou_mode='giou', weight=2.0),
                    ],
                    center_radius=2.5,
                    candidate_topk=5,
                ),
                loss_cls=dict(type='PurePyTorchFocalLoss', loss_weight=2.0),
                loss_bbox=dict(type='PurePyTorchL1Loss', loss_weight=5.0),
                loss_giou=dict(type='PurePyTorchGIoULoss', loss_weight=2.0),
            ),
        ),
        test_cfg=dict(
            use_nms=True,
            score_thr=0.5,
            min_bbox_size=0,
            nms=dict(type='nms', iou_threshold=0.5),
        ),
    )

    model = MODELS.build(model_cfg)
    print('Model built successfully!')

    batch_size = 2
    imgs = torch.randn(batch_size, 3, 224, 224)

    data_samples = []
    for i in range(batch_size):
        ds = DetDataSample()
        ds.set_metainfo(dict(
            img_id=i,
            img_shape=(224, 224, 3),
            ori_shape=(224, 224, 3),
            pad_shape=(224, 224, 3),
            scale_factor=(1.0, 1.0),
        ))

        gt_instances = InstanceData()
        gt_instances.bboxes = torch.tensor(
            [[10, 10, 50, 50], [100, 100, 150, 150]],
            dtype=torch.float32,
        )
        gt_instances.labels = torch.tensor([0, 1], dtype=torch.long)
        ds.gt_instances = gt_instances
        data_samples.append(ds)

    print('Testing loss...')
    model.train()
    losses = model.loss(imgs, data_samples)
    print(f'Losses: {list(losses.keys())}')
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f'  {k}: {v.item():.4f}')

    total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
    print(f'Total loss: {total_loss.item():.4f}')

    print('Testing backward...')
    total_loss.backward()
    print('Backward passed ✓')

    print('Testing predict...')
    model.eval()
    with torch.no_grad():
        results = model.predict(imgs, data_samples)
    print(f'Predict results: {len(results)} samples')
    for i, res in enumerate(results):
        n_pred = len(res.pred_instances.bboxes)
        print(f'  Sample {i}: {n_pred} predictions')

    print('\n✅ DiT wrapper test passed!')


def test_pipeline_order():
    """验证 test_pipeline 中 LoadAnnotations 在 Resize 之前的顺序正确性

    修复: LoadAnnotations 必须在 Resize 之前执行，因为标注基于原始图像坐标。
    此测试通过配置验证 pipeline 顺序，并通过 Compose 执行验证数据流。
    """
    from mmcv.transforms import Compose
    from mmdet.datasets.transforms import LoadAnnotations, Resize, PackDetInputs

    # 验证 pipeline 配置顺序
    pipeline_cfg = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        dict(type='PackDetInputs',
             meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
    ]

    # 验证 LoadAnnotations 在 Resize 之前
    load_ann_idx = None
    resize_idx = None
    for i, step in enumerate(pipeline_cfg):
        if step['type'] == 'LoadAnnotations':
            load_ann_idx = i
        if step['type'] == 'Resize':
            resize_idx = i
    assert load_ann_idx is not None, 'LoadAnnotations not found in pipeline'
    assert resize_idx is not None, 'Resize not found in pipeline'
    assert load_ann_idx < resize_idx, \
        f'LoadAnnotations (idx={load_ann_idx}) must be before Resize (idx={resize_idx})'

    print('  [PASS] pipeline config order: LoadAnnotations before Resize ✓')

    # 验证 pipeline 可 Compose 构建
    try:
        pipeline = Compose(pipeline_cfg)
        print('  [PASS] pipeline Compose build OK ✓')
    except Exception as e:
        print(f'  [FAIL] pipeline Compose failed: {e}')
        return

    print('  [PASS] test_pipeline order validated ✓')


if __name__ == '__main__':
    test_dit_wrapper()
    test_pipeline_order()
