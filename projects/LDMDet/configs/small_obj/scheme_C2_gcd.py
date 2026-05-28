_base_ = ['../_legacy/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py']

model = dict(
    bbox_head=dict(
        criterion=dict(
            bbox_loss_mode='gcd',
            assigner=dict(
                match_costs=[
                    dict(type='PurePyTorchFocalLossCost', weight=2.0),
                    dict(type='PurePyTorchGCDCost', weight=5.0),
                    dict(
                        type='PurePyTorchIoUCost', iou_mode='giou', weight=2.0
                    ),
                ],
            ),
        ),
    ),
)

experiment_name = 'smallobj_C2_gcd'

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-smallobj',
                experiment_name=experiment_name,
                description='SmallObj C2: GCD loss + GCD cost | Gaussian Combined Distance | R50+RF+Heun+Sinkhorn bs2',
            ),
        ),
    ]
)
