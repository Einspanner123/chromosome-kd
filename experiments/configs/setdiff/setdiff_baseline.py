"""SetDiff baseline 配置 — Coupled State Diffusion for Detection

桥接 setdiff 纯 PyTorch 包到 MMDetection 训练管线。
参考 ldmdet_baseline.py 的结构。
"""

_base_ = [
    '../_base_/datasets/chromo_coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'experiments.mmdet_bridge.registry',
        'experiments.mmdet_bridge.detector',
        'experiments.mmdet_bridge.setdiff_detector',
        'experiments.mmdet_bridge.hooks',
        'swanlab.integration.mmengine',
    ],
    allow_failed_imports=False,
)

num_classes = 24
batch_size = 2

# ── Model ──────────────────────────────────────────
model = dict(
    type='SetDiff',
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
        type='SetDiffJointDiffusionHead',
        num_classes=num_classes,
        feat_channels=256,
        num_queries=300,
        num_heads=8,
        num_layers=6,
        dim_feedforward=2048,
        snr_scale=2.0,
        num_sample_steps=4,
        sampler='euler',
    ),
)

# ── Optimizer (AdamW, 参考 ldmdet) ─────────────────
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.00005, weight_decay=0.0001, _delete_=True),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

# ── Schedule (150 epochs, cosine annealing) ─────────
max_epoch = 150
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=5),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        eta_min=0,
        begin=5,
        end=max_epoch,
        by_epoch=True,
    ),
]

# ── Dataloader ─────────────────────────────────────
train_dataloader = dict(batch_size=batch_size)

# ── Checkpoint ─────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_best='coco/bbox_mAP',
        rule='greater',
    ),
)

# ── Custom Hooks (EarlyStopping + Code Backup) ─────
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
]

# ── Visualization (SwanLab: setdiff) ───────────────
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='setdiff',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)

log_level = 'INFO'
