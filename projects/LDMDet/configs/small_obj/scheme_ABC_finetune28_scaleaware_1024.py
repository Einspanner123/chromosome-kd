_base_ = ['./scheme_AB_finetune28_scaleaware.py']

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
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                        (832, 1333),
                        (864, 1333),
                        (896, 1333),
                        (928, 1333),
                        (960, 1333),
                        (992, 1333),
                        (1024, 1333),
                    ],
                    keep_ratio=True,
                ),
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(512, 1333), (640, 1333), (768, 1333)],
                    keep_ratio=True,
                ),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(480, 800),
                    allow_negative_crop=True,
                ),
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (640, 1333),
                        (704, 1333),
                        (768, 1333),
                        (832, 1333),
                        (896, 1333),
                        (960, 1333),
                        (1024, 1333),
                    ],
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(
    batch_size=1,
    dataset=dict(pipeline=train_pipeline),
)

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

experiment_name = 'smallobj_ABC_finetune28_scaleaware_1024'

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-smallobj',
                experiment_name=experiment_name,
                description='SmallObj A+B+C: finest_scale=28 + scale-aware loss + resize 1333x1024 | R50+RF+Heun+Sinkhorn bs1',
            ),
        ),
    ]
)
