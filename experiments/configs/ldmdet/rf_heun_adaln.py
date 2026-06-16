"""LDMDet RF+Heun+AdaLN 配置 (Random Coupling Baseline)

基础 LDMDet，RF + Heun + Shifted Schedule + AdaLN-Zero。
使用随机耦合 (默认)。
"""

_base_ = [
    '../_base_/default_runtime.py',
    '../../../configs/_base_/datasets/chromo_coco_detection.py',
]

num_classes = 24
batch_size = 4
num_workers = 4

# ── 模型 ──────────────────────────────────────────
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
        # ── Head 主体 ──
        num_classes=num_classes,
        feat_channels=256,
        num_proposals=500,
        num_heads=6,
        deep_supervision=True,
        prior_prob=0.01,
        snr_scale=2.0,
        diffusion_type='rectified_flow',
        solver_type='heun',
        sampling_timesteps=4,
        rf_schedule='shifted',
        rf_shift=3.0,
        # ── 耦合策略 (默认 random) ──
        # coupling=dict(type='random'),
        # ── SingleHead ──
        single_head=dict(
            num_classes=num_classes,
            feat_channels=256,
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            time_conditioning='adaln_zero',
        ),
        # ── RoI Extractor ──
        roi_extractor=dict(
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        # ── Criterion ──
        criterion=dict(
            num_classes=num_classes,
            deep_supervision=True,
            assigner=dict(
                match_costs=[
                    dict(type='PurePyTorchFocalLossCost', weight=2.0),
                    dict(type='PurePyTorchBBoxL1Cost', weight=5.0),
                    dict(type='PurePyTorchIoUCost', iou_mode='giou', weight=2.0),
                ],
                center_radius=2.5,
                candidate_topk=5,
            ),
            loss_cls=dict(type='PurePyTorchFocalLoss', loss_weight=2.0),
            loss_bbox=dict(type='PurePyTorchL1Loss', loss_weight=5.0),
            loss_giou=dict(type='PurePyTorchGIoULoss', loss_weight=2.0),
        ),
    ),
)

# ── 训练 ──────────────────────────────────────────
max_epochs = 150
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=5),
    dict(type='CosineAnnealingLR', T_max=max_epochs, eta_min=0, begin=5, end=max_epochs, by_epoch=True),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
)

# ── Hooks ─────────────────────────────────────────
custom_hooks = [
    dict(type='EarlyStoppingHook', priority=50, patience=30, min_delta=0.001, monitor='coco/bbox_mAP', rule='greater'),
    dict(type='CopyProjectHook', priority='VERY_LOW'),
]

default_hooks = dict(
    checkpoint=dict(
        type='AsyncCheckpointHook', interval=1, max_keep_ckpts=1,
        save_best='coco/bbox_mAP', rule='greater',
    ),
)
