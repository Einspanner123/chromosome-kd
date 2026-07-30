"""Base dataset config for AutoKary2022 (kary) — amodal chromosome dataset.

Dataset: AutoKary2022 v1 (COCO format)
Source:   https://github.com/wangjuncongyu/chromosome-instance-segmentation-dataset
Images:   1084 metaphase images, 2048x1408 RGB
Anns:     48063 amodal bboxes + polygon segmentations
Classes:  24 (category id 1-24; id 0 is a Roboflow supercat placeholder with 0 anns)

Category id -> biological name correspondence (same as 24obj):
    1=RF+Heun, 2=+AdaLN-Zero, 3=+Stoch. Coupling, 4=B4, 5=B5,
    6=C6, 7=C7, 8=C8, 9=C9, 10=C10, 11=C11, 12=C12,
    13=D13, 14=D14, 15=D15,
    16=E16, 17=E17, 18=E18,
    19=F19, 20=F20,
    21=G21, 22=G22,
    23=X, 24=Y

Note: COCO JSON uses string names '1'..'24'. METAINFO must match these names
exactly for mmdet CocoDataset category matching. The biological labels above
are kept as a comment only — use 24obj for 'A1'..'Y' string labels.
"""

dataset_type = 'CocoDataset'
data_root = 'data/AutoKary2022_v1_coco/'
classes = (
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    '21', '22', '23', '24',
)
METAINFO = {
    'classes': (
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
        '21', '22', '23', '24',
    ),
    'palette': [
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
}

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
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=METAINFO,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=METAINFO,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=METAINFO,
        ann_file=data_root + 'test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=data_root + 'test/_annotations.coco.json',
    outfile_prefix='./work_dirs/chromo_kary_coco_detection/test',
)
