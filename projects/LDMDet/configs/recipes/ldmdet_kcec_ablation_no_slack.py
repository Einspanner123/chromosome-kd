_base_ = ['./ldmdet_kcec_pure_sota.py']

model = dict(
    bbox_head=dict(
        kcec_slack=0.0,
    )
)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-kcec',
                experiment_name='kcec_ablation_no_slack',
                description='KCEC ablation: slack=0 (hard ploidy constraint, no anomaly flexibility)',
            ),
        ),
    ],
)

work_dir = 'work_dirs/ldmdet_kcec_ablation_no_slack'
