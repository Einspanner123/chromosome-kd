_base_ = ['../ldmdet_rf_heun_shifted_bs8.py']

model = dict(
    bbox_head=dict(
        ot_coupling=True,
        ot_matcher='sinkhorn',
        ot_epsilon=1.0,
        ot_num_iters=20,
        ot_sample=True,
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
