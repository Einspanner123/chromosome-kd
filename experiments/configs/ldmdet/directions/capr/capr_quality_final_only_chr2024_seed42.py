"""CAPR-C2 final-only quality calibration on Dataset 1, A4 seed 42."""

_base_ = ['../../a4_dpm_pp_chr2024.py']

model = dict(
    bbox_head=dict(
        quality_score_beta=2.0,
        quality_calibration_mode='final_only',
        quality_only_training=True,
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

load_from = 'work_dirs/a4_dpm_pp_chr2024_seed42/best_coco_bbox_mAP_epoch_49.pth'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=200),
    dict(
        type='CosineAnnealingLR', by_epoch=True, begin=0, end=12,
        T_max=12, eta_min=1e-6),
]
optim_wrapper = dict(optimizer=dict(lr=1e-3))

visualizer = dict(
    type='DetLocalVisualizer', name='visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
)

work_dir = 'work_dirs/capr_quality_final_only_chr2024_seed42'
