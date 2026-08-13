"""Archived D2 KaryoFlow structure using the historical OT-flow coupling."""

_base_ = ['./karyoflow_r50.py']

model = dict(bbox_head=dict(coupling=dict(
    _delete_=True,
    type='ot_flow',
    epsilon=5.0,
    num_iters=20,
    coupling_mode='multinomial',
)))
