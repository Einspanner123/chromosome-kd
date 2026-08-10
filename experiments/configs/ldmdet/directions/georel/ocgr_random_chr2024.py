"""OCGR paired with the clean Dataset1 RF + DPM++ random baseline.

The only model difference from ``a4_dpm_pp_random_chr2024.py`` is the
zero-initialized geometric relation bias in cascade heads 4--6.
"""

_base_ = ['../../a4_dpm_pp_random_chr2024.py']

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
