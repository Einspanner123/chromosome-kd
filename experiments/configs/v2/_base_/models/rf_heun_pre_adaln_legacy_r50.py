"""Archived RF/Heun detector used by the Dataset 2 A1 checkpoint.

This identity intentionally predates AdaLN-Zero and does not introduce an
explicit proposal coupling block.  It must remain separate from the canonical
v2 RF/Heun reference.
"""

_base_ = ['./ldmdet_r50_common.py']

model = dict(bbox_head=dict(
    diffusion_type='rectified_flow',
    solver_type='heun',
    sampling_timesteps=4,
    rf_schedule='shifted',
    rf_shift=3.0,
    ddim_sampling_eta=1.0,
    box_renewal=True,
    use_ensemble=True,
))
