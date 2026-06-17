"""DiffusionDet DDPM Baseline — 原版 DiffusionDet 复现"""
_base_ = ['rf_heun_adaln.py']

model = dict(bbox_head=dict(
    diffusion_type='ddpm',
    solver_type='ddim',
    sampling_timesteps=1,
    rf_schedule='linear',
    single_head=dict(time_conditioning='scale_shift'),
    use_ensemble=False,
))
