"""Strict G0: DDPM, linear time, scale-shift conditioning."""

_base_ = ['./ldmdet_r50_common.py']

model = dict(bbox_head=dict(
    diffusion_type='ddpm', solver_type='euler', sampling_timesteps=1,
    ddim_sampling_eta=1.0, coupling=dict(type='random'),
    box_renewal=True, use_ensemble=True,
    single_head=dict(time_conditioning='scale_shift'),
))
