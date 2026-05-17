_base_ = ['../ldmdet_flowdet_adaln.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-ablation',
                experiment_name='02_adaln',
                description='Ablation: RF+Heun+Shifted+AdaLN | seed=1769925607',
            ),
        ),
    ],
)
