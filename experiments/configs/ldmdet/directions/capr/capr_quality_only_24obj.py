"""CAPR-C2: last-cascade IoU quality calibration on Dataset 2.

This is a deliberately isolated mechanism test. Existing A4 predictions are
unchanged; a 128-hidden-unit branch estimates aligned IoU and inference ranks
by p(class) * q^2, selected from the Phase-0 oracle sweep.
"""

_base_ = [
    '../mainline_ablation_24obj/a4_dpm_pp_24obj.py',
]

model = dict(
    bbox_head=dict(
        quality_score_beta=2.0,
        single_head=dict(
            predict_iou_quality=True,
            quality_hidden=128,
        ),
        criterion=dict(
            quality_loss_weight=0.25,
            quality_focal_alpha=0.75,
            quality_focal_gamma=2.0,
        ),
    ),
)

load_from = 'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth'

# Short mechanism validation. A matched no-quality continuation baseline must
# use the same 12-epoch schedule before attributing a gain to calibration.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=200),
    dict(
        type='CosineAnnealingLR', by_epoch=True, begin=0, end=12,
        T_max=12, eta_min=1e-6),
]
optim_wrapper = dict(optimizer=dict(lr=2.5e-5))

work_dir = 'work_dirs/capr_quality_only_24obj'
