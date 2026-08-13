"""Rectified-flow/Heun generation reference."""

_base_ = ['./ldmdet_r50_common.py']

model = dict(bbox_head=dict(
    diffusion_type='rectified_flow', solver_type='heun',
    sampling_timesteps=4, rf_schedule='shifted', rf_shift=3.0,
    ddim_sampling_eta=1.0,
    coupling=dict(type='random'), box_renewal=True, use_ensemble=True,
    single_head=dict(time_conditioning='adaln_zero'),
))
