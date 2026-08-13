"""Cascade R-CNN R50 baseline on the in-house cohort."""
_base_ = ['../baselines/benchmark/cascade_rcnn_r50.py']
data_root = 'data/ChromosomeSelf1700_coco/'
train_dataloader = dict(dataset=dict(data_root=data_root, ann_file='train/_annotations.coco.json', data_prefix=dict(img='train/')))
val_dataloader = dict(dataset=dict(data_root=data_root, ann_file='valid/_annotations.coco.json', data_prefix=dict(img='valid/')))
test_dataloader = dict(dataset=dict(data_root=data_root, ann_file='test/_annotations.coco.json', data_prefix=dict(img='test/')))
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
test_evaluator = dict(ann_file=data_root + 'test/_annotations.coco.json', outfile_prefix='./work_dirs/self1700/test')
visualizer = dict(type='DetLocalVisualizer', name='visualizer', vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])
