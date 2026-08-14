"""Archived inference structure for the historical H3 distilled student."""

_base_ = ['./karyoflow_r50.py']

model = dict(bbox_head=dict(
    num_heads=3,
    sampling_timesteps=4,
    coupling=dict(
        _delete_=True,
        type='ot_flow',
        epsilon=5.0,
        num_iters=20,
        coupling_mode='multinomial',
    ),
))
