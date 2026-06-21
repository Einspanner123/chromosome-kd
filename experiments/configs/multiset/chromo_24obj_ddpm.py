"""Chromosome 24obj — DDPM baseline"""
_base_ = ['../baselines/diffusiondet_ddpm.py']

data_root = 'data/24_chromosomes_object/coco/'
train_dataloader = dict(dataset=dict(data_root=data_root, ann_file='train/_annotations.coco.json', data_prefix=dict(img='train/')))
val_dataloader = dict(dataset=dict(data_root=data_root, ann_file='valid/_annotations.coco.json', data_prefix=dict(img='valid/')))
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
