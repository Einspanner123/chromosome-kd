"""跨域 Few-shot 微调: LDMDet A4 (RF+DPM++AdaLN+StochOT) on AutoKary k=5

源域: 24obj
目标域: AutoKary k=5 (5 images, 228 anns, 24 classes)
checkpoint: work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_*.pth

理论对比:
  - A4 (StochOT+DPM++) vs A2 (Random+AdaLN): 验证 StochOT+DPM++ 在少样本下的效果
  - DPM-Solver++ 高阶采样 → 理论预测少样本下稳定性应有优势

Usage:
    python experiments/runners/train.py \\
        experiments/configs/cross_domain/autokary/finetune_a4_k5.py
"""
import glob

_base_ = [
    '../../ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py',
]

# === 覆盖为 AutoKary 数据集 ===
data_root = 'data/AutoKary2022_coco/'

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='train/few_shot_k5.json',
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
ckpt_pattern = 'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_*.pth'
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
            experiment_name='finetune_a4_k5',
            description='Few-shot k=5: LDMDet A4 (StochOT+DPM++) 24obj→AutoKary | lr=5e-6, 50ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
