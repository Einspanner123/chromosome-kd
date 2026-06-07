_base_ = [
    '../../../configs/_base_/datasets/chromo_coco_detection.py',
    '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'projects.LDMDetDiT.model',
        'projects.LDMDetDiT.hooks',
        'swanlab.integration.mmengine',
    ],
    allow_failed_imports=False,
)

# 模型设置
num_classes = 24  # A1-A3, B4-B5, C6-C12, D13-D15, E16-E18, F19-F20, G21-G22, X, Y
feat_channels = 384  # DINOv3-Small 原生维度
max_epoch = 150
batch_size = 4
num_workers = 4
prefetch_factor = 4

model = dict(
    type='PurePyTorchDiffusionDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    # Backbone 切换为 DINOv3-Small
    backbone=dict(
        type='timm_model',
        model_name='vit_small_patch16_dinov3.lvd1689m',
        pretrained=True,
        features_only=True,
        out_indices=(0, 1, 2, 3),  # 对应 ViT 的不同 block 层级
    ),
    neck=dict(
        type='PurePyTorchSimpleFeatureFusion',  # ViT 需要特殊的特征融合逻辑
        in_channels=[384, 384, 384, 384],
        out_channels=feat_channels,
        num_outs=4,
    ),
    bbox_head=dict(
        type='DiTDiffusionDetHead',
        num_classes=num_classes,
        feat_channels=feat_channels,
        num_proposals=100,
        num_heads=3,
        num_blocks=3,  # 每个 head 内 DiTBlock 堆叠数，增加模型深度
        share_heads=False,  # 不使用权重共享，每个 head 独立学习
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
        regression_mode='direct',  # direct模式: sigmoid直接预测(cx,cy,w,h)，不依赖当前框位置，避免delta模式下spatial_prior锚点无法有效偏移
        use_adaln_zero=True,  # DiT 训练核心: 恒等初始化保证稳定收敛
        num_fpn_levels=4,
        num_ref_points=8,
        box_init_mode='spatial_prior',  # DAB-DETR: 均匀分布锚点位置编码，注入空间先验
        ot_coupling=True,  # OT 耦合: 将噪声最优分配到 GT，确保每个 proposal 都有有意义的回归目标
        ot_matcher='sinkhorn',  # Sinkhorn 匹配: 更均匀地分配噪声到各 GT
        single_head=dict(
            type='DiTSingleHead',
            num_classes=num_classes,
            feat_channels=feat_channels,
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=1536,  # 384 * 4
            num_heads=12,  # 384 / 12 = 32
            num_fpn_levels=4,
            num_ref_points=8,
            prediction_mode='x0',
            adaln_params=9,
            regression_mode='direct',
            use_adaln_zero=True,
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
                # 适中的 center_radius，不过于宽松
                center_radius=5.0,
                candidate_topk=12,
            ),
            loss_cls=dict(type='PurePyTorchFocalLoss', loss_weight=5.0),
            loss_bbox=dict(type='PurePyTorchL1Loss', loss_weight=0.5),
            loss_giou=dict(type='PurePyTorchGIoULoss', loss_weight=0.2),
        ),
    ),
    test_cfg=dict(
        use_nms=True,
        score_thr=0.01,
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
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True, backend=backend),
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
    dict(type='CopyProjectHook', priority='VERY_LOW'),
    dict(
        type='WeightSummaryHook', interval=50, log_norm=True, log_heatmap=True
    ),
    dict(
        type='PredictionVisHook',
        num_images=4,
        score_thr=0.01,
    ),
]

# 优化器设置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, weight_decay=0.0001, _delete_=True
    ),
    # deep supervision 7 个 head 梯度叠加，grad_norm 约 250-320，需要适当放宽
    clip_grad=dict(max_norm=50.0, norm_type=2),
)

max_epoch = 150
train_cfg = dict(max_epochs=max_epoch, val_interval=1)
# 修复 DDP 错误: 当某些 batch 为空或 contrastive loss 未触发时，防止梯度同步失败
find_unused_parameters = True

param_scheduler = [
    # 已经开启 AdaLN-Zero，无需极低学习率长预热
    dict(type='LinearLR', start_factor=1.0, by_epoch=True, begin=0, end=2),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        eta_min=1e-6,
        begin=5,
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
                experiment_name='ldmdet_dinov3_small_384_rope_shifted3',
                description='Phase 3: DINOv3-Small (384-Dim) + RoPE (base 10k) + RF-Shifted3 + Heun + Lightweight Tokenizer (No Sample)',
                api_key='Huzvq1fnDeqOwgQo2AMAI',
            ),
        ),
    ],
    name='visualizer',
)

work_dir = 'work_dirs/dit_dinov3_small_384_rope_shifted3'

log_level = 'INFO'
