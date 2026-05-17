_base_ = ['../recipes/rf_heun_stochot.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-ablation',
                experiment_name='03-stochot',
                description='Ablation: RF+Heun+Shifted+StochOT(eps5) | seed=1769925607',
            ),
        ),
    ],
)
