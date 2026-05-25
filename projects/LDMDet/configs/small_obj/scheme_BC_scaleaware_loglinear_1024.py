_base_ = ['./scheme_B_scaleaware_loglinear.py']

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1333, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
    ),
]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        scale=(1333, 1024),
        keep_ratio=True,
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
    ),
]

train_dataloader = dict(
    batch_size=1,
    dataset=dict(pipeline=train_pipeline),
)
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

experiment_name = 'smallobj_BC_scaleaware_loglinear_1024'

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-smallobj',
                experiment_name=experiment_name,
                description='SmallObj B+C: scale-aware L1 (log_linear, alpha=0.15) + 1024 resolution | R50+RF+Heun+Sinkhorn bs1',
            ),
        ),
    ]
)
