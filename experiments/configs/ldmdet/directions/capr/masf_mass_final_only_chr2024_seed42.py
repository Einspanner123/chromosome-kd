"""MASF Phase-1A conserved proposal-mass gate on Dataset 1, seed 42."""

_base_ = ['../../a4_dpm_pp_chr2024.py']

model = dict(
    bbox_head=dict(
        mass_score_power=1.0,
        mass_only_training=True,
        single_head=dict(
            predict_set_mass=True,
            mass_hidden=128,
            mass_prior_prob=0.1,
        ),
        criterion=dict(
            mass_loss_weight=0.25,
            mass_conservation_weight=0.01,
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
    type='DetLocalVisualizer',
    name='visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
)

work_dir = 'work_dirs/masf_mass_final_only_chr2024_seed42'
