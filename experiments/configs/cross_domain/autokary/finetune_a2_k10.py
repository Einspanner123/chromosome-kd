"""跨域 Few-shot 微调: LDMDet +AdaLN-Zero (RF+Heun+AdaLN, Random coupling) on AutoKary k=10

源域: 24obj (mAP=0.856)
目标域: AutoKary k=10 (10 images, 455 anns, 24 classes)
checkpoint: work_dirs/a2_rf_heun_adaln_24obj/best_coco_bbox_mAP_epoch_82.pth

Usage:
    python experiments/runners/train.py \\
        experiments/configs/cross_domain/autokary/finetune_a2_k10.py
"""
import glob

_base_ = [
    '../../ldmdet/directions/mainline_ablation_24obj/a2_rf_heun_adaln_24obj.py',
]

# === 覆盖为 AutoKary 数据集 ===
data_root = 'data/AutoKary2022_coco/'

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='train/few_shot_k10.json',
        filter_cfg=dict(filter_empty_gt=False, min_size=1e-5),
    ),
)
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
    ),
)

val_evaluator = dict(
    ann_file=data_root + 'valid/_annotations.coco.json',
    classwise=True,
)
test_evaluator = dict(
    ann_file=data_root + 'test/_annotations.coco.json',
    format_only=False,
    classwise=True,
)

# === Load 24obj pretrained checkpoint ===
ckpt_pattern = 'work_dirs/a2_rf_heun_adaln_24obj/best_coco_bbox_mAP_*.pth'
_ckpts = sorted(glob.glob(ckpt_pattern))
load_from = _ckpts[-1] if _ckpts else ckpt_pattern

# === Finetune optimization ===
optim_wrapper = dict(optimizer=dict(lr=5e-06))

# === Finetune schedule ===
max_epochs = 50
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=5),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs - 5,
        eta_min=0,
        begin=5,
        end=max_epochs,
        by_epoch=True,
    ),
]

# === Early stopping ===
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        priority=50,
        patience=15,
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        rule='greater',
    ),
    dict(type='CopyProjectHook', priority='VERY_LOW'),
]

# === SwanLab ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='cross-domain-autokary',
            experiment_name='finetune_a2_k10',
            description='Few-shot k=10: LDMDet +AdaLN-Zero (Random+AdaLN) 24obj→AutoKary | lr=5e-6, 50ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
