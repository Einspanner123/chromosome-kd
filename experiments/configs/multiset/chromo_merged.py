"""Chromosome 合并数据集训练 — 原 SOTA 配置 (Sinkhorn stoch ε=5, RF+Heun+AdaLN)"""
_base_ = ['../ldmdet/sinkhorn_stochastic.py']

data_root = 'data/merged_24obj_original/'

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
    ),
)
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
    ),
)
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
