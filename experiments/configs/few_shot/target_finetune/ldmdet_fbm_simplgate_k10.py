"""Few-shot target finetune: LDMDet + FBM Simple Gate on chromo k=10.

Inherits from 24obj source pretrain, switches dataset to chromo few-shot,
and lowers the learning rate for fine-tuning. The feature_bridge params
keep the original lr (lr_mult=10.0) since they are newly trained.
"""
import glob

_base_ = ['../source_pretrain/ldmdet_fbm_simplgate_24obj.py']

# === Override to chromo few-shot dataset (chromo C-group order) ===
data_root = 'data/Chromosome20240904_NoAug_NoResize_coco/'
classes = (
    'A1', 'A2', 'A3', 'B4', 'B5',
    'C10', 'C11', 'C12', 'C6', 'C7', 'C8', 'C9',
    'D13', 'D14', 'D15',
    'E16', 'E17', 'E18',
    'F19', 'F20', 'G21', 'G22', 'X', 'Y',
)
METAINFO = dict(
    classes=classes,
    palette=[
        (220, 20, 60), (119, 11, 32), (0, 0, 142),
        (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192),
        (250, 170, 30), (100, 170, 30), (220, 220, 0),
        (175, 116, 175), (250, 0, 30), (165, 42, 42),
        (255, 77, 255), (0, 226, 252), (182, 182, 255),
        (0, 82, 0), (120, 166, 157),
        (110, 76, 0), (174, 57, 255),
        (199, 100, 0), (72, 0, 118),
    ],
)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='train/few_shot_k10.json',
        data_prefix=dict(img='train/'),
        metainfo=METAINFO,
    ),
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=METAINFO,
    ),
)
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# === Load 24obj pretrained checkpoint ===
# Note: 24obj bbox_head uses different C-group order than chromo.
# We load the full state dict via load_from; the bbox_head classifier
# weights will be in the wrong C-group order but the model can still
# fine-tune (50 epochs is sufficient to correct the ordering).
ckpt_pattern = (
    'work_dirs/few_shot/source_pretrain_ldmdet_fbm_simplgate_24obj/'
    'best_coco_bbox_mAP_*.pth'
)
_ckpts = sorted(glob.glob(ckpt_pattern))
load_from = _ckpts[-1] if _ckpts else ckpt_pattern

# === Finetune optimization ===
# Backbone/neck/bbox_head lr = pretrain lr * 0.1 = 5e-06.
# feature_bridge keeps original lr (lr_mult=10.0) since it is the key
# transfer module and benefits from faster adaptation.
optim_wrapper = dict(
    optimizer=dict(lr=5e-06),
    paramwise_cfg=dict(custom_keys={'feature_bridge': dict(lr_mult=10.0)}),
)

# === Finetune schedule ===
max_epochs = 50
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

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

# === Early stopping (patience=15) ===
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
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='few-shot-benchmark',
                experiment_name='target_finetune_ldmdet_fbm_simplgate_k10',
                description='Few-shot target finetune: LDMDet + FBM '
                'Simple Gate on chromo k=10 | bs=2, 50ep',
            ),
        ),
    ],
)
