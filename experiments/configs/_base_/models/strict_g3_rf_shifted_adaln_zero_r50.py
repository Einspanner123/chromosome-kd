"""Strict G3: rectified flow, shifted time, AdaLN-Zero conditioning.

The one-step Euler validation protocol is intentionally identical to G1/G2.
DPM++ is evaluated later as a fixed-checkpoint inference intervention.
"""

_base_ = ['./ldmdet_r50_common.py']

model = dict(
    bbox_head=dict(
        diffusion_type='rectified_flow',
        solver_type='euler',
        sampling_timesteps=1,
        rf_schedule='shifted',
        rf_shift=3.0,
        ddim_sampling_eta=1.0,
        coupling=dict(type='random'),
        box_renewal=True,
        use_ensemble=True,
        single_head=dict(time_conditioning='adaln_zero'),
    )
)
