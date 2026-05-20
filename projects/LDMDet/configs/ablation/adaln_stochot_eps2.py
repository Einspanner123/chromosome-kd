_base_ = ['../recipes/rf_heun_adaln_stochot_eps2.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-ablation',
                experiment_name='adaln-stochot-eps2',
                description='Ablation: RF+Heun+Shifted+AdaLN+StochOT(eps=2) bs=8 | seed=1769925607',
            ),
        ),
    ],
)
