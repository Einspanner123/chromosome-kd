# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    '../base/datasets/chromo_coco_detection.py',
    '../base/schedules/schedule_1x.py', '../base/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.DiffusionDet.diffusiondet', 'chromodet.hooks'],
    allow_failed_imports=False)

num_classes = 24
batch_size = 4
num_workers = 4
prefetch_factor = 4

# model settings
model = dict(
    type='DiffusionDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    bbox_head=dict(
        type='DynamicDiffusionDetHead',
        num_classes=num_classes,
        feat_channels=256,
        num_proposals=500,
        num_heads=6,
        deep_supervision=False,
        prior_prob=0.01,
        snr_scale=2.0,
        sampling_timesteps=1,
        ddim_sampling_eta=1.0,
        single_head=dict(
            type='SingleDiffusionDetHead',
            num_classes=num_classes,
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            act_cfg=dict(type='ReLU', inplace=True),
            dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)),
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        # criterion
        criterion=dict(
            type='DiffusionDetCriterion',
            num_classes=num_classes,
            assigner=dict(
                type='DiffusionDetMatcher',
                match_costs=[
                    dict(
                        type='FocalLossCost',
                        alpha=0.25,
                        gamma=2.0,
                        weight=2.0,
                        eps=1e-8),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ],
                center_radius=2.5,
                candidate_topk=5),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                alpha=0.25,
                gamma=2.0,
                reduction='sum',
                loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', reduction='sum', loss_weight=5.0),
            loss_giou=dict(type='GIoULoss', reduction='sum',
                           loss_weight=2.0))),
    test_cfg=dict(
        use_nms=True,
        score_thr=0.5,
        min_bbox_size=0,
        nms=dict(type='nms', iou_threshold=0.5),
    ))

backend = 'pillow'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend=backend),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True,
                backend=backend),
        ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[(400, 1333), (500, 1333), (600, 1333)],
                            keep_ratio=True,
                            backend=backend),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            keep_ratio=True,
                            backend=backend)
                    ]]),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend=backend),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True, backend=backend),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        # indices=[i for i in range(0, 1540, 15)],
        filter_cfg=dict(filter_empty_gt=False, min_size=1e-5),
        pipeline=train_pipeline))

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# randomness = dict(
#     seed=42,                # 控制所有随机操作
#     deterministic=True,     # 使用确定性CUDA计算
#     diff_rank_seed=False,   # 不同GPU使用相同seed
# )

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        priority=50,
        patience=15,
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        rule='greater'),
    dict(type='BackupHook', file='chromodet/model'),
    # dict(  # 新增：权重可视化
    #     type='WeightVizHook',
    #     interval=2,           # 每500 iter 记录一次
    #     with_grads=True,        # 记录梯度
    #     log_conv_images=True,   # 可视化卷积核
    #     # max_kernels=16,         # 每层最多展示64个kernel
    # )
]
log_level = 'INFO'
