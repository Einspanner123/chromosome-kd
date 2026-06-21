"""ChromoGen Chromosome v2 数据集配置 — 继承 GHSS 基础，替换数据集路径"""
_base_ = ['../ldmdet/ghss.py']

# 覆盖数据集根路径
data_root = 'data/selfmake_chromosome202250604_NoResizeNoAug/'

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
