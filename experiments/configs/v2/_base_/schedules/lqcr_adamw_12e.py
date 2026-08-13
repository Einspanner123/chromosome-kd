"""Optimization policy for a frozen-parent LQCR child."""

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=200),
    dict(type='CosineAnnealingLR', T_max=12, eta_min=1e-6,
         begin=0, end=12, by_epoch=True),
]
