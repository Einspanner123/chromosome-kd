_base_ = ['./ldmdet_rf_heun_shifted_bs2.py']

model = dict(
    bbox_head=dict(
        single_head=dict(
            time_conditioning='adaln_zero',
        ),
        coupling=dict(
            type='ot_flow',
            epsilon=5.0,
            num_iters=20,
            coupling_mode='multinomial',
        ),
    ),
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        priority=50,
        patience=30,
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        rule='greater',
    ),
    dict(type='CopyProjectHook', priority='VERY_LOW'),
]
