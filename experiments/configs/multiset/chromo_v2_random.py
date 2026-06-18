"""Chromosome v2 — Random baseline"""
_base_ = ['../ldmdet/rf_heun_adaln.py']

data_root = 'data/selfmake_chromosome202250604_NoResizeNoAug/'
train_dataloader = dict(dataset=dict(data_root=data_root, ann_file='train/_annotations.coco.json', data_prefix=dict(img='train/')))
val_dataloader = dict(dataset=dict(data_root=data_root, ann_file='valid/_annotations.coco.json', data_prefix=dict(img='valid/')))
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
