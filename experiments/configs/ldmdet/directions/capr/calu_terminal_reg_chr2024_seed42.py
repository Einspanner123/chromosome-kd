"""CALU terminal-regression gate on Dataset 1, paired with A4 seed 42.

Only the final cascade regressor is updated.  This is a cheap causal gate for
the COCO-aligned localization objective, not the final from-scratch result.
"""

_base_ = ['../../a4_dpm_pp_chr2024.py']

model = dict(
    bbox_head=dict(
        terminal_reg_only_training=True,
        criterion=dict(
            localization_utility_loss_weight=0.5,
            localization_utility_temperature=0.025,
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
optim_wrapper = dict(optimizer=dict(lr=1e-4))

visualizer = dict(
    type='DetLocalVisualizer', name='visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
)

work_dir = 'work_dirs/calu_terminal_reg_chr2024_seed42'
