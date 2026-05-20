_base_ = ['../recipes/rf_heun_adaln_bs8.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-ablation',
                experiment_name='adaln',
                description='Ablation: RF+Heun+Shifted+AdaLN bs=8 | seed=1769925607',
            ),
        ),
    ],
)
