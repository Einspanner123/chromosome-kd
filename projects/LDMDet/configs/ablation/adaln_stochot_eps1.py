_base_ = ['../recipes/rf_heun_adaln_stochot_eps1.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-ablation',
                experiment_name='adaln-stochot-eps1',
                description='Ablation: RF+Heun+Shifted+AdaLN+StochOT(eps=1) bs=8 | seed=1769925607',
            ),
        ),
    ],
)
