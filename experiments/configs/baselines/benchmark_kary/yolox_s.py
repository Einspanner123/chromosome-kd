"""YOLOX-S baseline config for AutoKary2022 (kary) dataset.

Dataset: amodal chromosome detection, 2048x1408, 24 classes
Splits:  train 848 / valid 118 / test 118 (80/11/11)

Note: kary images are 2048x1408 (vs 24obj's 640x640). The Resize to (1333, 800)
downscales them — this is intentional for baseline consistency with 24obj.
If small-chromosome performance is critical, increase the Resize scale.
"""

_base_ = [
    '../../_base_/datasets/chromo_kary_coco_detection.py',
    '../../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['swanlab.integration.mmengine'],
    allow_failed_imports=False,
)

num_classes = 24
data_root = 'data/AutoKary2022_v1_coco/'
METAINFO = {
    'classes': (
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
        '21', '22', '23', '24',
    ),
}

model = dict(
    type='YOLOX',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(2, 3, 4),
    ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
    ),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=num_classes,
        in_channels=128,
        feat_channels=128,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='IoULoss', reduction='sum', loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0,
        ),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
    ),
    train_cfg=dict(
        assigner=dict(
            type='SimOTAAssigner', center_radius=2.5, candidate_topk=10
        ),
    ),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Mosaic', img_scale=(1333, 800), pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-666, -400),
    ),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='Pad', pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))
    ),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs'),
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                ),
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 1333), (500, 1333), (600, 1333)],
                    keep_ratio=True,
                ),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='Pad', pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
    ),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            data_root=data_root,
            metainfo=METAINFO,
            ann_file='train/_annotations.coco.json',
            data_prefix=dict(img='train/'),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

max_epochs = 200
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.0005, by_epoch=True, begin=0, end=10),
    dict(
        type='CosineAnnealingLR',
        T_max=140,
        eta_min=1e-6,
        begin=10,
        end=max_epochs,
        by_epoch=True,
    ),
]

custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type='EarlyStoppingHook',
        priority=50,
        patience=30,
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        rule='greater',
    ),
]

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-benchmark-kary',
                experiment_name='yolox-s',
                description='Benchmark kary: YOLOX-S | bs=8, 200ep, AdamW',
            ),
        ),
    ],
)

randomness = dict(seed=1, deterministic=False, diff_rank_seed=True)
