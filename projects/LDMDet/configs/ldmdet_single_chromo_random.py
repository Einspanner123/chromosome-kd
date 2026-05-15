"""LDMDet on single_chromosomes_object — Random coupling baseline.

Inherits the RF + Heun + AdaLN-Zero recipe.
"""

_base_ = ['./ldmdet_flowdet_adaln.py']

# Dataset
dataset_type = 'CocoDataset'
data_root = '/data/linkst/datasets/single_chromosomes_object/'
classes = ('chromosomes',)
METAINFO = {'classes': classes, 'palette': [(220, 20, 60)]}

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
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
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=METAINFO,
        ann_file='annotations/train.json',
        data_prefix=dict(img='JEPG/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=METAINFO,
        ann_file='annotations/val.json',
        data_prefix=dict(img='JEPG/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
)

model = dict(
    bbox_head=dict(
        num_classes=1,
        single_head=dict(num_classes=1),
        criterion=dict(num_classes=1),
        ot_coupling=False,
    ),
)

load_from = None
work_dir = 'work_dirs/ldmdet_single_chromo_random'
