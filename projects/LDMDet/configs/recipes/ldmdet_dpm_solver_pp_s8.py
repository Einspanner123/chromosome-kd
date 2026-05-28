_base_ = ['../_legacy/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py']

model = dict(
    bbox_head=dict(
        solver_type='dpm_solver_pp',
        sampling_timesteps=8,
    )
)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-dpm',
                experiment_name='dpm_solver_pp_order2_s8',
                description='DPM-Solver++ order=2, 8 steps (9 NFE) on SOTA base',
            ),
        ),
    ],
)

work_dir = 'work_dirs/ldmdet_dpm_solver_pp_o2_s8'
