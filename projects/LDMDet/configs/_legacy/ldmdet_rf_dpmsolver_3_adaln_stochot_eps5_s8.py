_base_ = ['./ldmdet_rf_dpmsolver_adaln_stochot_eps5_s8.py']

model = dict(
    bbox_head=dict(
        solver_type='dpm_solver_pp_3',
        sampling_timesteps=8,
    )
)
