"""Strict G1: rectified flow, linear time, scale-shift conditioning."""

_base_ = ['./ldmdet_r50_common.py']

model = dict(bbox_head=dict(
    diffusion_type='rectified_flow', solver_type='dpm_solver_pp',
    sampling_timesteps=4, rf_schedule='linear', rf_shift=1.0,
    ddim_sampling_eta=1.0, coupling=dict(type='random'),
    box_renewal=True, use_ensemble=True,
    single_head=dict(time_conditioning='scale_shift'),
))
