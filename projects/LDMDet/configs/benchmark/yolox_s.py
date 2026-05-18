_base_ = [
    '../../../../configs/_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['swanlab.integration.mmengine'],
    allow_failed_imports=False,
)

num_classes = 24
data_root = 'data/Chromosome20240904_NoAug_NoResize_coco/'

METAINFO = {
    'classes': (
        'A1',
        'A2',
        'A3',
        'B4',
        'B5',
        'C10',
        'C11',
        'C12',
        'C6',
        'C7',
        'C8',
        'C9',
        'D13',
        'D14',
        'D15',
        'E16',
        'E17',
        'E18',
        'F19',
        'F20',
        'G21',
        'G22',
        'X',
        'Y',
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
        type='YOLOXCSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(2, 3, 4),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
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

img_scale = (1333, 800)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
    ),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad', pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))
    ),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
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
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
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

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=METAINFO,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox',
)
test_evaluator = val_evaluator

max_epochs = 150
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True
    ),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)

param_scheduler = [
    dict(
        type='QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True,
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0005,
        begin=5,
        T_max=135,
        end=135,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type='ConstantLR', by_epoch=True, factor=1, begin=135, end=max_epochs
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
]

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-benchmark',
                experiment_name='yolox-s',
                description='Benchmark: YOLOX-S R50 | bs=8, 150ep',
            ),
        ),
    ],
)

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)
