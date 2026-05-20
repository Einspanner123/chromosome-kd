_base_ = ['../recipes/rf_heun_adaln_stochot_eps5.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-ablation',
                experiment_name='adaln-stochot-eps5',
                description='Ablation: RF+Heun+Shifted+AdaLN+StochOT(eps=5) bs=8 | seed=1769925607',
            ),
        ),
    ],
)
