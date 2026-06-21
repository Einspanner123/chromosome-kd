"""Chromosome 24obj — GHSS 耦合，24obj 数据集"""
_base_ = ['../ldmdet/ghss.py']

# 覆盖数据集根路径
data_root = 'data/24_chromosomes_object/coco/'

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
    ),
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
    ),
)
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
