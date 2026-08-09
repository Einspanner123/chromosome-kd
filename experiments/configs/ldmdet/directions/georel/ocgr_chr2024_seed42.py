"""OCGR full-training gate on Dataset1, seed 42.

Overlap-Conditional Geometric Relations add a zero-initialized relative-box
bias to proposal attention only in cascade heads 4--6.  This is a full
training run, not checkpoint continuation.
"""

_base_ = ['../../a4_dpm_pp_chr2024.py']

model = dict(
    bbox_head=dict(
        geometric_relation_start_head=3,
        single_head=dict(
            geometric_relation_attention=True,
            geometric_relation_hidden=32,
            geometric_relation_proximity_radius=3.0,
        ),
    ),
)

work_dir = 'work_dirs/ocgr_chr2024_seed42'
