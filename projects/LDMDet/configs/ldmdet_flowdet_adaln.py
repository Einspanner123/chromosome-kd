"""LDMDet with Rectified Flow + Heun + AdaLN-Zero + random coupling.

This is the strongest random-coupling baseline (0.751 mAP).
"""
_base_ = ['./ldmdet_rf_heun_shifted_bs2.py']

model = dict(
    bbox_head=dict(single_head=dict(time_conditioning='adaln_zero', ), ))
