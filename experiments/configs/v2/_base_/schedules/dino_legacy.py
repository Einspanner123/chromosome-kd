"""Frozen historical D2 dino optimization policy."""

custom_hooks = [
    dict(
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        patience=30,
        priority=50,
        rule='greater',
        type='EarlyStoppingHook'),
]
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(lr=2.5e-05, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=10, start_factor=0.0005, type='LinearLR'),
    dict(
        T_max=140,
        begin=10,
        by_epoch=True,
        end=150,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
test_cfg = dict(type='TestLoop')
train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=1)
val_cfg = dict(type='ValLoop')
