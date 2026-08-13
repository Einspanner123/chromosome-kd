"""Provenance-locked in-house cohort: 1190/170/340 group-disjoint split."""

dataset_id = 'D1_INHOUSE1700_V1'
dataset_manifest_sha256 = '57bc9516b11aa642d50c32124777f70abdc2074ee18e3250abf1665940eea43c'
train_annotation_sha256 = '318120afe81184557cd1c13c985c63f5404ae4ce9e97e5cec6e5edb1d6ec534c'
val_annotation_sha256 = '57466f1fe99201b091b65fbbcf687c0bebca86b82fbed86ba48d8b8d68b6dfaf'
test_annotation_sha256 = '52e8868d2f43c609d841d138bd495a410690b399c0904782bc44d5aba7564efd'
data_root = 'data/ChromosomeSelf1700_coco/'
classes = ('A1','A2','A3','B4','B5','C10','C11','C12','C6','C7','C8','C9',
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
