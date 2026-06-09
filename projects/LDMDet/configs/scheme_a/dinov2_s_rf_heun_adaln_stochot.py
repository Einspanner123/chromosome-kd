"""方案一: DINOv2 ViT-S/14 Backbone + Simple Feature Pyramid + 扩散检测头

架构: DINOv2 ViT-S/14 → Simple FPN(stride 4/8/16/32, out=256) → DiffusionDetHead
核心创新: RF + AdaLN-Zero + Sinkhorn Stochastic OT (保持不变)
Backbone: DINOv2 ViT-S/14 (timm, 22.1M params, 自监督预训练)
Neck: Simple Feature Pyramid (替代传统FPN, 无需ResNet多尺度输出)
Head: DiffusionDetHead (与SOTA完全一致)

关键适配:
  - DINOv2输出单尺度 [B, H/14*W/14, 384], Simple FPN转为4层多尺度
  - RoI Extractor featmap_strides=[4,8,16,32] 与Simple FPN对齐
  - 输入分辨率需为14的倍数 (如 392=28*14, 448=32*14, 518=37*14)

参考:
  - ViTDet (Simple Feature Pyramid): https://arxiv.org/abs/2203.16527
  - DEIMv2 (DINOv3 + STA): https://arxiv.org/abs/2509.20787
  - RF-DETR (DINOv2 + DETR): https://arxiv.org/abs/2511.09554
"""

_base_ = ['../ldmdet_baseline.py']

custom_imports = dict(
    imports=[
        'projects.LDMDet.model',
        'projects.LDMDet.mods.dinov2_backbone',
        'projects.LDMDet.hooks',
        'swanlab.integration.mmengine',
    ],
    allow_failed_imports=False,
)

# ============================================================
# Model: 替换 backbone 和 neck, 保留 SOTA 扩散头
# ============================================================
model = dict(
    # DINOv2 patch_size=14, padding必须对齐14的倍数
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=14,
    ),
    # 替换 backbone: ResNet-50 → DINOv2 ViT-S/14
    backbone=dict(
        type='DINOv2Backbone',
        model_name='dinov2_s',  # vit_small_patch14_reg4_dinov2.lvd142m
        out_channels=256,
        target_strides=[4, 8, 16, 32],
        freeze_layers=-1,  # 不冻结, 全量微调
        use_reg_tokens=True,
        num_reg_tokens=4,
        pretrained=True,
        _delete_=True,
    ),
    # 替换 neck: FPN → None (Simple FPN 已内置于 DINOv2Backbone)
    neck=None,
    # 扩散头: 与 SOTA 完全一致
    bbox_head=dict(
        diffusion_type='rectified_flow',
        solver_type='heun',
        sampling_timesteps=4,
        rf_schedule='shifted',
        rf_shift=3.0,
        snr_scale=2.0,
        use_flash_attn=True,
        # OT Coupling (SOTA)
        ot_coupling=True,
        ot_matcher='sinkhorn',
        ot_epsilon=5.0,
        ot_num_iters=20,
        ot_sample=True,
        # AdaLN-Zero (SOTA)
        single_head=dict(
            time_conditioning='adaln_zero',
        ),
        # RoI Extractor: 适配 Simple FPN 的 4 层输出
        roi_extractor=dict(
            type='PurePyTorchSingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
    ),
)

# ============================================================
# Data: 输入分辨率需为 14 的倍数
# ============================================================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[
            (448, 448),
            (518, 518),
            (560, 560),
            (616, 616),
            (672, 672),
            (728, 728),
        ],
        keep_ratio=False,
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(672, 672), keep_ratio=False),
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

# ============================================================
# Training
# ============================================================
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    prefetch_factor=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False, min_size=1e-5),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# 优化器: ViT 微调需要更小的学习率
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.00004, weight_decay=0.0001, _delete_=True
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

max_epoch = 550
train_cfg = dict(max_epochs=max_epoch)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=10),
    dict(
        type='CosineAnnealingLR',
        T_max=240,
        eta_min=2e-6,
        begin=10,
        end=max_epoch,
        by_epoch=True,
    ),
]

# ============================================================
# Visualization
# ============================================================
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-scheme-a',
                experiment_name='dinov2_s-rf-heun-adaln-stochot',
                description='Scheme A: DINOv2 ViT-S/14 + Simple FPN + DiffusionDetHead (RF+Heun+AdaLN+StochOT)',
            ),
        ),
    ],
)
