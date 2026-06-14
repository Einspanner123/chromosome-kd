_base_ = ['./ldmdet_rf_heun_shifted_bs8.py']

# 注册自定义 transform（CLAHE, SmallObjectCopyPaste）
# 注意：必须包含 baseline 中的所有 custom_imports，否则模型类不会注册
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
# 改进的数据增强策略 v2
# ============================================================================
# 相比 baseline 的改进:
#   1. Rotate 增强覆盖染色体任意方向（降 prob，限制角度避免 bbox 膨胀）
#   2. 双向翻转（水平+垂直）覆盖染色体多样朝向
#   3. 移除 allow_negative_crop=True 避免空标注图像污染训练
#   4. 提高多尺度下限至 600，保护小目标
#   5. RandomAffine 缩小缩放范围，避免极小目标进一步缩小
#   6. 移除 PhotoMetricDistortion（G-banding 灰度图做色相/饱和度无意义，
#      亮度和对比度扰动可能破坏带纹特征）
#   7. 新增 SmallObjectCopyPaste 选择性复制小目标增加密度
#      （22%小目标+4.2%极小目标，仅复制面积<2500的小目标）
#   8. 新增 MinIoURandomCrop 替代 RandomCrop，保证裁剪后目标可见性
#   9. 新增 CLAHE 增强 G-banding 带纹局部对比度（参考 ChroSegNet 2023）
#  10. 新增 Sharpness 锐化增强带纹清晰度
# ============================================================================

backend = 'pillow'

# 内层 dataset 的 pipeline：加载 + 不需要 mix_results 的增强
# 参考 mmdet 官方 ssj_scp 配置，MultiImageMixDataset 的内层 pipeline 做加载和基础增强，
# 外层 pipeline 只做需要 mix_results 的 CopyPaste + PackDetInputs
train_dataset_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=None,
        imdecode_backend=backend,
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    # RandomAffine: 缩小缩放范围(0.85,1.15)，避免极小目标(4.2% area<500)进一步缩小
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.85, 1.15),
        border=(0, 0),
        border_val=(220, 220, 220),  # 匹配核型图灰白背景色
    ),
    # Rotate: 降 prob 至 0.3，限制角度至 90° 避免小目标 bbox 面积膨胀过大
    dict(
        type='Rotate',
        prob=0.3,
        level=None,
        min_mag=0.0,
        max_mag=90.0,
        reversal_prob=0.5,
        img_border_value=220,  # 匹配核型图灰白背景色
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
                # MinIoURandomCrop: 保证裁剪后目标与 GT 的最小 IoU，
                # 避免裁剪掉小目标，替代原来的 RandomCrop
                dict(
                    type='MinIoURandomCrop',
                    min_ious=(0.1, 0.3, 0.5, 0.7),
                    bbox_clip_border=True,
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
    # CLAHE: 增强 G-banding 带纹局部对比度，参考 ChroSegNet (2023)
    # 在 LAB 色彩空间仅增强 L 通道，保留颜色信息
    dict(
        type='CLAHE',
        prob=0.5,
        clip_limit=2.0,
        tile_grid_size=(8, 8),
    ),
    # Sharpness: 锐化带纹边缘，增强 G-banding 条纹清晰度
    dict(
        type='Sharpness',
        prob=0.3,
        level=None,
        min_mag=0.1,
        max_mag=0.9,
    ),
]

# 外层 pipeline：仅包含需要 mix_results 的 SmallObjectCopyPaste + PackDetInputs
train_pipeline = [
    # SmallObjectCopyPaste: 仅复制小目标(area<2500)增加密度
    # 需配合 MultiImageMixDataset 使用，mix_results 由其自动提供
    dict(
        type='SmallObjectCopyPaste',
        max_num_pasted=8,
        bbox_occluded_thr=10,
        paste_by_box=True,
        area_thr=2500.0,  # 约50x50像素，覆盖78%的目标
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

# 内层 dataset 的 pipeline：加载 + 不需要 mix_results 的增强
# 参考 mmdet 官方 ssj_scp 配置，MultiImageMixDataset 的内层 pipeline 做加载和基础增强，
# 外层 pipeline 只做需要 mix_results 的 CopyPaste + PackDetInputs
train_dataset_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=None,
        imdecode_backend=backend,
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    # RandomAffine: 缩小缩放范围(0.85,1.15)，避免极小目标(4.2% area<500)进一步缩小
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.85, 1.15),
        border=(0, 0),
        border_val=(220, 220, 220),  # 匹配核型图灰白背景色
    ),
    # Rotate: 降 prob 至 0.3，限制角度至 90° 避免小目标 bbox 面积膨胀过大
    dict(
        type='Rotate',
        prob=0.3,
        level=None,
        min_mag=0.0,
        max_mag=90.0,
        reversal_prob=0.5,
        img_border_value=220,  # 匹配核型图灰白背景色
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
                # MinIoURandomCrop: 保证裁剪后目标与 GT 的最小 IoU，
                # 避免裁剪掉小目标，替代原来的 RandomCrop
                dict(
                    type='MinIoURandomCrop',
                    min_ious=(0.1, 0.3, 0.5, 0.7),
                    bbox_clip_border=True,
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
    # CLAHE: 增强 G-banding 带纹局部对比度，参考 ChroSegNet (2023)
    # 在 LAB 色彩空间仅增强 L 通道，保留颜色信息
    dict(
        type='CLAHE',
        prob=0.5,
        clip_limit=2.0,
        tile_grid_size=(8, 8),
    ),
    # Sharpness: 锐化带纹边缘，增强 G-banding 条纹清晰度
    dict(
        type='Sharpness',
        prob=0.3,
        level=None,
        min_mag=0.1,
        max_mag=0.9,
    ),
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=dict(
            type=_base_.train_dataloader.dataset.type,
            data_root=_base_.train_dataloader.dataset.data_root,
            metainfo=_base_.train_dataloader.dataset.metainfo,
            ann_file=_base_.train_dataloader.dataset.ann_file,
            data_prefix=_base_.train_dataloader.dataset.data_prefix,
            filter_cfg=dict(filter_empty_gt=False, min_size=1e-5),
            pipeline=train_dataset_pipeline,
            backend_args=_base_.train_dataloader.dataset.backend_args,
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# ============================================================================
# SwanLab & WorkDir
# ============================================================================
experiment_name = 'ldmdet_rf_heun_shifted_bs8_aug_v2'

work_dir = './work_dirs/ldmdet_rf_heun_shifted_bs8_aug_v2'

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
                    'v2 aug: Rotate90(p0.3) + BiFlip + Affine(0.85-1.15) + '
                    'SmallObjCopyPaste(area<2500,max8) + MinIoUCrop + '
                    'CLAHE(p0.5) + Sharpness(p0.3) + ScaleMin600'
                ),
            ),
        ),
    ]
)
