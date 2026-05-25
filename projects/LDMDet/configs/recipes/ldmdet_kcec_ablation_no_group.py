_base_ = ['./ldmdet_kcec_pure_sota.py']

model = dict(
    bbox_head=dict(
        kcec_group_weight=0.0,
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
                experiment_name='kcec_ablation_no_group',
                description='KCEC ablation: group_weight=0 (only morph + quota + structure)',
            ),
        ),
    ],
)

work_dir = 'work_dirs/ldmdet_kcec_ablation_no_group'
