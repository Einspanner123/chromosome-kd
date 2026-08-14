"""Canonical KaryoFlow generation model for new experiments."""

_base_ = ['./ldmdet_r50_common.py']

model = dict(bbox_head=dict(
    diffusion_type='rectified_flow', solver_type='dpm_solver_pp',
    sampling_timesteps=4, rf_schedule='shifted', rf_shift=3.0,
    ddim_sampling_eta=1.0,
    coupling=dict(type='random'),
    # Explicitly preserve the currently trained D1 operating semantics.
    # Renewal-off remains a registered inference ablation, not a silent change.
    box_renewal=True, use_ensemble=True,
    single_head=dict(time_conditioning='adaln_zero'),
))
