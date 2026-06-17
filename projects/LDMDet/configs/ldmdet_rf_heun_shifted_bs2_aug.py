_base_ = ['./ldmdet_rf_heun_shifted_bs2.py']

custom_imports = dict(
    imports=[
        'projects.LDMDet.model',
        'projects.LDMDet.hooks',
        'projects.LDMDet.custom_transforms',
        'swanlab.integration.mmengine',
    ],
    allow_failed_imports=False,
)

# ============================================================================
# 数据增强：同 v3 (bs8_aug) 的渐进式策略
# ============================================================================
# HFlip(p0.5) + Affine(0.9-1.1,无旋转) + MultiScale(480-800) + MinIoUCrop
# 无 Rotate / VerticalFlip / CopyPaste / CLAHE / Sharpness
# ============================================================================

backend = 'pillow'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=None,
        imdecode_backend=backend,
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal']),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.9, 1.1),
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        border=(0, 0),
    ),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (480, 1333), (512, 1333), (544, 1333),
                        (576, 1333), (608, 1333), (640, 1333),
                        (672, 1333), (704, 1333), (736, 1333),
                        (768, 1333), (800, 1333),
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
                    type='MinIoURandomCrop',
                    min_ious=(0.1, 0.3, 0.5, 0.7),
                    bbox_clip_border=True,
                ),
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (480, 1333), (512, 1333), (544, 1333),
                        (576, 1333), (608, 1333), (640, 1333),
                        (672, 1333), (704, 1333), (736, 1333),
                        (768, 1333), (800, 1333),
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
experiment_name = 'ldmdet_rf_heun_shifted_bs2_aug'

work_dir = './work_dirs/ldmdet_rf_heun_shifted_bs2_aug'

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
                    'LDMDet RF-Heun shifted3 bs2 | '
                    'HFlip(p0.5) + Affine(0.9-1.1) + '
                    'MultiScale(480-800) + MinIoUCrop'
                ),
            ),
        ),
    ]
)
