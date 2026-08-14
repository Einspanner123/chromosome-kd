"""Frozen historical D2 yolox optimization policy."""

custom_hooks = [
    dict(num_last_epochs=15, priority=48, type='YOLOXModeSwitchHook'),
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        type='EMAHook',
        update_buffers=True),
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
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=10, start_factor=0.0005, type='LinearLR'),
    dict(
        T_max=140,
        begin=10,
        by_epoch=True,
        end=200,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
test_cfg = dict(type='TestLoop')
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
val_cfg = dict(type='ValLoop')
