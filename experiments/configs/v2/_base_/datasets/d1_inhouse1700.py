"""Provenance-locked in-house cohort: 1190/170/340 group-disjoint split."""

dataset_id = 'D1_INHOUSE1700_V2'
dataset_manifest_sha256 = '48d90fed63ecc107b374a316effc1e5ab0d63b7c4bd9110d33ddd16e1f43146c'
train_annotation_sha256 = 'e62704aa3ce9f7b58170946cca78458ff5331fc4316ba946eace6a1dbc6c8ed1'
val_annotation_sha256 = '9065c7b5fb4df2a9f9c794dfe83c655ddf6695388d2bde562171b208c6d3a5b9'
test_annotation_sha256 = '883696b8e60cc901cfe92b3f009d8c60e7b8cefbb5ba9ce3c742c3720343f08f'
data_root = 'data/ChromosomeSelf1700_coco/'
classes = ('A1','A2','A3','B4','B5','C6','C7','C8','C9','C10','C11','C12',
           'D13','D14','D15','E16','E17','E18','F19','F20','G21','G22','X','Y')
METAINFO = dict(classes=classes)
backend_args = None
scales = [(v, 1333) for v in range(480, 801, 32)]
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, imdecode_backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice', transforms=[
        [dict(type='RandomChoiceResize', scales=scales, keep_ratio=True, backend='pillow')],
        [dict(type='RandomChoiceResize', scales=[(400,1333),(500,1333),(600,1333)], keep_ratio=True, backend='pillow'),
         dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384,600), allow_negative_crop=True),
         dict(type='RandomChoiceResize', scales=scales, keep_ratio=True, backend='pillow')],
    ]),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, imdecode_backend='pillow'),
    dict(type='Resize', scale=(1333,800), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor')),
]
train_dataloader = dict(
    batch_size=2, num_workers=4, prefetch_factor=4,
    persistent_workers=True, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='CocoDataset', data_root=data_root, metainfo=METAINFO,
                 ann_file='train/_annotations.coco.json', data_prefix=dict(img='train/'),
                 filter_cfg=dict(filter_empty_gt=False, min_size=1e-5),
                 pipeline=train_pipeline, backend_args=backend_args),
)
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, pin_memory=True,
    drop_last=False, sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset', data_root=data_root, metainfo=METAINFO,
                 ann_file='valid/_annotations.coco.json', data_prefix=dict(img='valid/'),
                 test_mode=True, pipeline=test_pipeline, backend_args=backend_args),
)
test_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset', data_root=data_root, metainfo=METAINFO,
                 ann_file='test/_annotations.coco.json', data_prefix=dict(img='test/'),
                 test_mode=True, pipeline=test_pipeline, backend_args=backend_args),
)
val_evaluator = dict(type='CocoMetric', ann_file=data_root+'valid/_annotations.coco.json', metric='bbox', format_only=False, classwise=True)
test_evaluator = dict(type='CocoMetric', ann_file=data_root+'test/_annotations.coco.json', metric='bbox', format_only=False, classwise=True)
