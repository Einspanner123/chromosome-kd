_base_ = [
    '../../../configs/_base_/datasets/chromo_coco_detection.py',
    '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'projects.LDMDet.model',
        'projects.LDMDet.hooks',
        'swanlab.integration.mmengine',
    ],
    allow_failed_imports=False,
)

num_classes = 24
batch_size = 4
num_workers = 4
prefetch_factor = 4

model = dict(
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
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4,
    ),
    bbox_head=dict(
        type='DiTDiffusionDetHead',
        num_classes=num_classes,
        feat_channels=256,
        num_proposals=300,  # 减少 Proposal 数量以降低初期匹配压力
        num_heads=6,
        deep_supervision=True,
        prior_prob=0.01,
        snr_scale=2.0,
        sampling_timesteps=4,
        diffusion_type='rectified_flow',
        solver_type='heun',
        rf_schedule='shifted',
        rf_shift=3.0,
        box_renewal=True,
        use_ensemble=True,
        prediction_mode='x0',
        adaln_params=9,
        regression_mode='direct',
        use_adaln_zero=False,  # 关键修改：关闭零初始化，让定位信号在初期就能流过残差路径
        num_fpn_levels=4,
        num_ref_points=8,
        box_init_mode='bilinear',
        single_head=dict(
            type='DiTSingleHead',
            num_classes=num_classes,
            feat_channels=256,
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            num_fpn_levels=4,
            num_ref_points=8,
            prediction_mode='x0',
            adaln_params=9,
            regression_mode='direct',
        ),
        criterion=dict(
            type='PurePyTorchDiffusionDetCriterion',
            num_classes=num_classes,
            bbox_loss_mode='relative_l1',
            assigner=dict(
                type='PurePyTorchDiffusionDetMatcher',
                match_costs=[
                    dict(type='PurePyTorchFocalLossCost', weight=2.0),
                    dict(type='PurePyTorchRelativeL1Cost', weight=5.0),
                    dict(
                        type='PurePyTorchIoUCost', iou_mode='giou', weight=2.0
                    ),
                ],
                # 进一步放宽 center_radius，确保预热期能有更多正样本
                center_radius=4.0,
                candidate_topk=5,
            ),
            loss_cls=dict(type='PurePyTorchFocalLoss', loss_weight=2.0),
            loss_bbox=dict(type='PurePyTorchL1Loss', loss_weight=5.0),
            loss_giou=dict(type='PurePyTorchGIoULoss', loss_weight=2.0),
        ),
    ),
    test_cfg=dict(
        use_nms=True,
        score_thr=0.05,
        min_bbox_size=0,
        nms=dict(type='nms', iou_threshold=0.5),
    ),
)

backend = 'pillow'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend=backend,
    ),
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
                    backend=backend,
                ),
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 1333), (500, 1333), (600, 1333)],
                    keep_ratio=True,
                    backend=backend,
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
                    backend=backend,
                ),
            ],
        ],
    ),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend=backend,
    ),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True, backend=backend),
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
    batch_size=batch_size,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False, min_size=1e-5),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        priority=50,
        patience=30,
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        rule='greater',
    ),
    dict(type='CopyProjectHook', priority='VERY_LOW'),
    dict(
        type='WeightSummaryHook', interval=50, log_norm=True, log_heatmap=True
    ),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.0001, _delete_=True),
    # Fix (chromosome-kd-dit-zero-map): pre-clip grad_norm 报到 1100~3149,
    # max_norm=1.0 太紧导致 update 几乎被切零. Deformable DETR 标准 35.
    clip_grad=dict(max_norm=35.0, norm_type=2),
)

max_epoch = 150
train_cfg = dict(max_epochs=max_epoch, val_interval=1)
# 修复 DDP 错误: 当某些 batch 为空或 contrastive loss 未触发时，防止梯度同步失败
find_unused_parameters = True

param_scheduler = [
    # 极大程度延长预热期：20 Epoch 极其缓慢的上升，给 Deformable Attention 充分时间寻找特征
    dict(type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=20),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        eta_min=0,
        begin=20,
        end=max_epoch,
        by_epoch=True,
    ),
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd',
                experiment_name='ldmdet_dit_r50_shifted3_direct',
                description='DiT: Deformable Cross-Attn + AdaLN-Zero(9) + RF-Shifted3 + Heun + Direct x0 (cx,cy,w,h sigmoid)',
                api_key='Huzvq1fnDeqOwgQo2AMAI',
            ),
        ),
    ],
    name='visualizer',
)

work_dir = 'work_dirs/ldmdet_dit_r50_shifted3_direct'

log_level = 'INFO'
