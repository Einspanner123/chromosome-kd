"""LDMDet + Hard OT

继承自 rf_heun_adaln，仅切换 coupling。
"""

_base_ = ['rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        coupling=dict(type='hard_ot'),
    ),
)
