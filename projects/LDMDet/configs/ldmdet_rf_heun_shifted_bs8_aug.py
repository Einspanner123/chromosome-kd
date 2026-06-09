_base_ = ['./ldmdet_rf_heun_shifted_bs8.py']

# ============================================================================
# 改进的数据增强策略
# ============================================================================
# 相比 baseline 的改进:
#   1. 添加 Rotate 增强覆盖染色体任意方向
#   2. 添加 PhotoMetricDistortion 增强鲁棒性（G-banding 染色深浅差异）
#   3. 双向翻转（水平+垂直）
#   4. 移除 allow_negative_crop=True 避免空标注图像污染训练
#   5. 提高多尺度下限至 600，避免小目标过小
#   6. 添加 RandomAffine 增加几何变换多样性
# ============================================================================

backend = 'pillow'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=None,
        imdecode_backend=backend,
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.7, 1.3),
        border=(0, 0),
        border_val=(114, 114, 114),
    ),
    dict(
        type='Rotate',
        prob=0.5,
        level=None,
        min_mag=0.0,
        max_mag=180.0,
        reversal_prob=0.5,
        img_border_value=128,
    ),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (600, 1333),
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
                    scales=[(500, 1333), (600, 1333), (700, 1333)],
                    keep_ratio=True,
                    backend=backend,
                ),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(480, 600),
                    allow_negative_crop=False,
                ),
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (600, 1333),
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
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=None,
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
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False, min_size=1e-5),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# ============================================================================
# SwanLab & WorkDir
# ============================================================================
experiment_name = 'ldmdet_rf_heun_shifted_bs8_aug_v1'

work_dir = './work_dirs/ldmdet_rf_heun_shifted_bs8_aug_v1'

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd',
                experiment_name=experiment_name,
                description=(
                    'LDMDet RF-Heun shifted3 bs8 | '
                    'Enhanced aug: Rotate180 + PhotoMetricDistortion + '
                    'BiFlip + RandomAffine + NoNegCrop + ScaleMin600'
                ),
            ),
        ),
    ]
)
