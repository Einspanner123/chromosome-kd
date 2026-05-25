_base_ = ['./scheme_ABC_finetune28_scaleaware_1024.py']

model = dict(
    bbox_head=dict(
        score_thr=0.3,
        nms_thr=0.6,
    ),
)

experiment_name = 'smallobj_ABCD_finetune28_scaleaware_1024_nms'

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-smallobj',
                experiment_name=experiment_name,
                description='SmallObj A+B+C+D: finest_scale=28 + scale-aware loss + resize 1333x1024 + NMS(score_thr=0.3, iou=0.6) | R50+RF+Heun+Sinkhorn bs1',
            ),
        ),
    ]
)
