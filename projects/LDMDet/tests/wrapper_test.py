import sys
from pathlib import Path

import torch
from mmengine.structures import InstanceData

from mmdet.structures import DetDataSample

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 初始化 mmdet 环境，确保所有模块都已注册
from mmdet.utils import register_all_modules

register_all_modules()

from mmdet.registry import MODELS

# 导入我们的模型包装类，触发注册
try:
    import projects.LDMDet.model as model_mod

    LDMDet = model_mod.LDMDet
except Exception as e:
    print(f'Failed to import LDMDet: {e}')
    import traceback

    traceback.print_exc()


def test_wrapper():
    print('Testing LDMDet wrapper...')
    # 1. 模拟配置
    num_classes = 24
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
            depth=18,  # 使用 ResNet18 加快测试
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
            num_outs=4),
        bbox_head=dict(
            type='PurePyTorchDiffusionDetHead',
            num_classes=num_classes,
            feat_channels=256,
            num_proposals=100,
            num_heads=3,
            single_head=dict(
                type='PurePyTorchSingleDiffusionDetHead',
                num_classes=num_classes,
                num_cls_convs=1,
                num_reg_convs=1,
            ),
            roi_extractor=dict(
                type='PurePyTorchSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32],
            ),
            criterion=dict(
                type='PurePyTorchDiffusionDetCriterion',
                num_classes=num_classes,
                assigner=dict(
                    type='PurePyTorchDiffusionDetMatcher',
                    match_costs=[
                        dict(type='PurePyTorchFocalLossCost', weight=2.0),
                        dict(
                            type='PurePyTorchBBoxL1Cost',
                            weight=5.0,
                            box_format='xyxy'),
                        dict(
                            type='PurePyTorchIoUCost',
                            iou_mode='giou',
                            weight=2.0),
                    ],
                ),
                loss_cls=dict(type='PurePyTorchFocalLoss', loss_weight=2.0),
                loss_bbox=dict(type='PurePyTorchL1Loss', loss_weight=5.0),
                loss_giou=dict(type='PurePyTorchGIoULoss', loss_weight=2.0),
            ),
        ),
    )

    # 2. 构建模型
    model = MODELS.build(model_cfg)
    if torch.cuda.is_available():
        model = model.cuda()

    print('Model built successfully!')

    # 3. 模拟数据
    batch_size = 2
    imgs = torch.randn(batch_size, 3, 224, 224)
    if torch.cuda.is_available():
        imgs = imgs.cuda()

    data_samples = []
    for i in range(batch_size):
        ds = DetDataSample()
        ds.set_metainfo(
            dict(
                img_id=i,
                img_shape=(224, 224, 3),
                ori_shape=(224, 224, 3),
                pad_shape=(224, 224, 3),
                scale_factor=(1.0, 1.0),
            ))

        gt_instances = InstanceData()
        gt_instances.bboxes = torch.tensor(
            [[10, 10, 50, 50], [100, 100, 150, 150]], dtype=torch.float32)
        gt_instances.labels = torch.tensor([0, 1], dtype=torch.long)
        if torch.cuda.is_available():
            gt_instances.bboxes = gt_instances.bboxes.cuda()
            gt_instances.labels = gt_instances.labels.cuda()

        ds.gt_instances = gt_instances
        data_samples.append(ds)

    # 4. 测试 loss
    print('Testing loss...')
    losses = model.loss(imgs, data_samples)
    print(f'Losses: {losses.keys()}')
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f'  {k}: {v.item():.4f}')
        else:
            print(f'  {k}: {v}')

    # 5. 测试 predict
    print('Testing predict...')
    model.eval()
    with torch.no_grad():
        results = model.predict(imgs, data_samples)
    print(f'Predict results: {len(results)} samples')
    for i, res in enumerate(results):
        print(f'  Sample {i}: {len(res.pred_instances.bboxes)} predictions')


if __name__ == '__main__':
    test_wrapper()
