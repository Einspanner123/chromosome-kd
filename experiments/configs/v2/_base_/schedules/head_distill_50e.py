"""Parent-initialized three-head student fine-tuning policy."""

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True,
         begin=0, end=5),
    dict(type='CosineAnnealingLR', T_max=45, eta_min=0,
         begin=5, end=50, by_epoch=True),
]
custom_hooks = [
    dict(type='EarlyStoppingHook', priority=50, patience=15,
         min_delta=0.001, monitor='coco/bbox_mAP', rule='greater'),
]
