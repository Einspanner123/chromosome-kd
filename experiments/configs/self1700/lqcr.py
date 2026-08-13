"""Paired final-only LQCR child for an in-house KaryoFlow parent."""
import os
_base_ = ['./karyoflow.py']
if 'LQCR_BASE_CHECKPOINT' not in os.environ:
    raise RuntimeError('LQCR_BASE_CHECKPOINT must point to the paired KaryoFlow parent')
load_from = os.path.abspath(os.environ['LQCR_BASE_CHECKPOINT'])
model = dict(bbox_head=dict(quality_score_beta=2.0, quality_calibration_mode='final_only', quality_only_training=True, single_head=dict(predict_iou_quality=True, quality_hidden=128), criterion=dict(quality_loss_weight=0.25, quality_focal_alpha=0.75, quality_focal_gamma=2.0)))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
param_scheduler = [dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=200), dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=12, T_max=12, eta_min=1e-6)]
optim_wrapper = dict(optimizer=dict(lr=1e-3))
data_root = 'data/ChromosomeSelf1700_coco/'
train_dataloader = dict(dataset=dict(data_root=data_root, ann_file='train/_annotations.coco.json', data_prefix=dict(img='train/')))
val_dataloader = dict(dataset=dict(data_root=data_root, ann_file='valid/_annotations.coco.json', data_prefix=dict(img='valid/')))
test_dataloader = dict(dataset=dict(data_root=data_root, ann_file='test/_annotations.coco.json', data_prefix=dict(img='test/')))
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
test_evaluator = dict(ann_file=data_root + 'test/_annotations.coco.json', outfile_prefix='./work_dirs/self1700/test')
visualizer = dict(
    type='DetLocalVisualizer',
    name='visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='KaryoFlow-Self1700',
                resume='allow',
                config=dict(
                    dataset_id='D1_INHOUSE1700_V1',
                    dataset_manifest_sha256=(
                        '57bc9516b11aa642d50c32124777f70abdc2074ee18e3250abf1665940eea43c'
                    ),
                    split='group-disjoint-70-10-20',
                ),
            ),
        ),
    ],
)
