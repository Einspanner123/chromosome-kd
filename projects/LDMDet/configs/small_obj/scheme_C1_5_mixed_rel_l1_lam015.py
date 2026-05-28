_base_ = ['../_legacy/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py']

model = dict(
    bbox_head=dict(
        criterion=dict(
            bbox_loss_mode='mixed_relative_l1',
            mix_lambda=0.15,
            assigner=dict(
                match_costs=[
                    dict(type='PurePyTorchFocalLossCost', weight=2.0),
                    dict(
                        type='PurePyTorchMixedRelativeL1Cost',
                        weight=5.0,
                        mix_lambda=0.15,
                    ),
                    dict(
                        type='PurePyTorchIoUCost', iou_mode='giou', weight=2.0
                    ),
                ],
            ),
        ),
    ),
)

experiment_name = 'smallobj_C1_5_mixed_rel_l1_lam015'

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-smallobj',
                experiment_name=experiment_name,
                description='SmallObj C1.5: mixed relative L1 λ=0.15 | 0.15·|Δ|/s + 0.85·|Δ| | R50+RF+Heun+Sinkhorn bs2',
            ),
        ),
    ]
)
