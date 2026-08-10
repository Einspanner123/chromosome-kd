"""Clean paired Dataset-1 LQCR training on a caller-supplied A4 checkpoint.

The baseline checkpoint is mandatory and is supplied through
``LQCR_BASE_CHECKPOINT`` by ``run_lqcr_after_clean_baseline.py``. The config
inherits the standardized random-coupling baseline, freezes the detector, and
trains only the final localization-quality head.
"""

import os


_base_ = ['../../a4_dpm_pp_random_chr2024.py']

if 'LQCR_BASE_CHECKPOINT' not in os.environ:
    raise RuntimeError('LQCR_BASE_CHECKPOINT must point to the paired clean baseline')

load_from = os.path.abspath(os.environ['LQCR_BASE_CHECKPOINT'])

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
