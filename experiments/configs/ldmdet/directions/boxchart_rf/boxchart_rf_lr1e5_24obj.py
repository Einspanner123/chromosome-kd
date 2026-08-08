"""BoxChart-RF learning-rate control: constant 1e-5 for eight epochs.

This run starts from the same a4 checkpoint as the conservative-LR run.  It
isolates whether the 1e-6 effective learning rate is preventing the detector
head from adapting to the new chart geometry.
"""

_base_ = ['./boxchart_rf_24obj.py']

load_from = 'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth'
work_dir = 'work_dirs/boxchart_rf_lr1e5_24obj'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_interval=1)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

# No scheduler: the sole independent variable is the effective learning rate.
param_scheduler = []
