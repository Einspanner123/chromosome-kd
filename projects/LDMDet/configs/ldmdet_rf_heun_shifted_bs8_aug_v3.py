_base_ = ['./ldmdet_rf_heun_shifted_bs8.py']

# ============================================================================
# 渐进式数据增强 v3 — 仅含生物学合理的增强
# ============================================================================
# 相比 v2 (bs8_aug) 的回退与改进:
#   - 去掉 Rotate(±90°) — 染色体在核型图中固定竖直，旋转产生无效视角
#   - 去掉 VerticalFlip — 染色体 p/q 臂有固定朝向，翻转后不真实
#   - 去掉 SmallObjectCopyPaste — 核型图已经高密度，复制产生不真实重叠
#   - 去掉 CLAHE / Sharpness — 灰度图过度增强会放大噪声
#   - 保留水平翻转 — 核型图中染色体左右对称，生物学合理
#   - 保留 filter_empty_gt=False — 避免丢弃含小目标的图像
#   - 保留 MinIoURandomCrop — 相比 RandomCrop 保护小目标不被裁剪丢失
#   - 保留 RandomAffine(0.9~1.1) — 温和缩放，不旋转不剪切
#   - 降低 multi-scale 下限到 480 — 保持尺度多样性
# ============================================================================

backend = 'pillow'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=None,
        imdecode_backend=backend,
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    # 水平翻转：核型图中染色体左右对称，生物学合理
    dict(type='RandomFlip', prob=0.5, direction=['horizontal']),
    # 温和的随机缩放：范围 0.9~1.1，不旋转，不剪切
    # 放在 multi-scale 之前，与之后的 multi-scale 形成联合尺度扰动
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
                # Branch 1：纯多尺度训练
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
                # Branch 2：缩小 → 裁剪 → 放大（ssj + crop）
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 1333), (500, 1333), (600, 1333)],
                    keep_ratio=True,
                    backend=backend,
                ),
                # MinIoURandomCrop：保证裁剪后目标与 GT 的最小 IoU，
                # 避免裁剪掉小目标，替代原来的 RandomCrop
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
experiment_name = 'ldmdet_rf_heun_shifted_bs8_aug_v3'

work_dir = './work_dirs/ldmdet_rf_heun_shifted_bs8_aug_v3'

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
                    'v3 aug: HFlip(p0.5) + Affine(0.9-1.1) + '
                    'MultiScale(480-800) + MinIoUCrop | '
                    '无 Rotate/VerticalFlip/CopyPaste/CLAHE/Sharpness'
                ),
            ),
        ),
    ]
)
