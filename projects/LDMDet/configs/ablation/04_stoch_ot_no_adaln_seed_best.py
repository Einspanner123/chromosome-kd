_base_ = ['../ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

model = dict(
    bbox_head=dict(
        single_head=dict(
            time_conditioning='scale_shift',
        ),
    ),
)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-ablation',
                experiment_name='04_stoch_ot_no_adaln',
                description='Ablation: RF+Heun+Shifted+StochOT(eps5) WITHOUT AdaLN | seed=1769925607',
            ),
        ),
    ],
)
