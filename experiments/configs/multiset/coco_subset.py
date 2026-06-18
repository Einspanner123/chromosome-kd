"""COCO 2017 子集验证配置 — 继承 GHSS 基础"""
_base_ = ['../ldmdet/ghss.py']

data_root = 'data/coco/'

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
    ),
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
    ),
)
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')

model = dict(bbox_head=dict(num_classes=80))
