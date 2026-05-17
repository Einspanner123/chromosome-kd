_base_ = ['../ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-ablation',
                experiment_name='03_adaln_stoch_ot',
                description='Ablation: RF+Heun+Shifted+AdaLN+StochOT(eps5) | seed=1769925607',
            ),
        ),
    ],
)
