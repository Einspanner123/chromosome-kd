"""DiffuDETR 24obj 配置 — 24 chromosomes object detection

DiffuDETR (ICLR 2026) 复现配置, 移植到 mmdet 框架:
  - 数据集: data/24_chromosomes_object/coco/
  - 24类: A1-A3, B4-B5, C6-C12, D13-D15, E16-E18, F19-F20, G21-G22, X, Y
  - backbone: ResNet-50 + FPN (mmdet 标准)
  - 训练: AdamW lr=5e-5, 50 epochs, batch_size=2
  - val_evaluator: classwise=True (输出 24 per-class AP)
  - SwanLab: project='diffudetr-24obj'

参考:
  - projects/DiffusionDet/configs/diffusiondet_default_24cls.py (配置结构)
  - experiments/configs/setdiff/setdiff_baseline.py (桥接模式)
"""

_base_ = [
    '../../../experiments/configs/_base_/datasets/chromo_coco_detection.py',
    '../../../experiments/configs/_base_/schedules/schedule_1x.py',
    '../../../experiments/configs/_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['projects.diffudetr', 'experiments.mmdet_bridge'],
    allow_failed_imports=False,
)

num_classes = 24
batch_size = 2

# === 数据集 (24 chromosomes object) ===
data_root = 'data/24_chromosomes_object/coco/'

# 24类顺序: A1-A3, B4-B5, C6-C12 (数值序), D13-D15, E16-E18, F19-F20, G21-G22, X, Y
classes = (
    'A1',
    'A2',
    'A3',
    'B4',
    'B5',
    'C6',
    'C7',
    'C8',
    'C9',
    'C10',
    'C11',
    'C12',
    'D13',
    'D14',
    'D15',
    'E16',
    'E17',
    'E18',
    'F19',
    'F20',
    'G21',
    'G22',
    'X',
    'Y',
)

METAINFO = dict(
    classes=classes,
    palette=[
        (220, 20, 60),
        (119, 11, 32),
        (0, 0, 142),
        (0, 0, 230),
        (106, 0, 228),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 70),
        (0, 0, 192),
        (250, 170, 30),
        (100, 170, 30),
        (220, 220, 0),
        (175, 116, 175),
        (250, 0, 30),
        (165, 42, 42),
        (255, 77, 255),
        (0, 226, 252),
        (182, 182, 255),
        (0, 82, 0),
        (120, 166, 157),
        (110, 76, 0),
        (174, 57, 255),
        (199, 100, 0),
        (72, 0, 118),
    ],
)

# 覆盖数据加载器: 使用 24obj 数据集
train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        data_root=data_root,
        metainfo=METAINFO,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
    ),
)
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=METAINFO,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
    ),
)
test_dataloader = val_dataloader

# Val 评估器: 启用 classwise 输出 24 per-class AP (项目硬约束)
val_evaluator = dict(
    ann_file=data_root + 'valid/_annotations.coco.json',
    classwise=True,
)
test_evaluator = val_evaluator

# === Model (DiffuDETR) ===
model = dict(
    type='DiffuDETR',
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
        type='DiffuDETRHead',
        num_classes=num_classes,
        num_queries=300,
        embed_dim=256,
        num_heads=8,
        dim_feedforward=2048,
        num_layers=6,
        num_feature_levels=4,
        timesteps=1000,  # DDPM 总步数
        sampling_timesteps=25,  # DDIM 采样步数
        scale=2.0,  # 扩散空间 [-2, 2]
        box_renewal=True,  # 自适应阈值 box renewal
        use_ensemble=True,  # 多步集成
        use_nms=True,  # NMS 去重
        nms_thr=0.7,  # NMS IoU 阈值
        aux_loss=True,  # 辅助损失 (deep supervision)
        dropout=0.1,
    ),
)

# === Optimizer (AdamW, lr=1e-4, 对齐原仓库) ===
# 修复: lr 5e-5 → 1e-4 (对齐原仓库 base lr)
# 修复: clip_grad max_norm 1.0 → 0.1 (对齐原仓库, 原 1.0 放行 10x 梯度致 NaN 永久传播)
# 修复: 加 backbone lr×0.1 (对齐原仓库 lr_factor_func, backbone 用较小 lr)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.0001, weight_decay=0.0001, _delete_=True
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
)

# === Schedule (50 epochs, cosine annealing) ===
max_epoch = 50
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

# 修复: warmup 从 5 epoch 缩减到 1 epoch, start_factor 从 0.001 提升到 0.1
# 原配置 lr=5e-8 持续 5 epoch 导致模型完全不学习 (mAP=0), warmup 结束后梯度爆炸
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=1),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        eta_min=0,
        begin=1,
        end=max_epoch,
        by_epoch=True,
    ),
]

# === Checkpoint ===
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_best='coco/bbox_mAP',
        rule='greater',
    ),
)

# === Custom Hooks (EarlyStopping + Code Backup) ===
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

# === Visualization (SwanLab: diffudetr-24obj) ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='diffudetr-24obj',
            experiment_name='diffudetr_baseline',
            description='DiffuDETR baseline (扩散检测器) | bs=2, 50ep, DDIM 25步',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)

log_level = 'INFO'
