"""GACS validation on Dataset 1 (Chromosome20240904).

Thresholds were calibrated on seed 42 without renewal, then frozen.  Renewal
occurs only after a sample fails the first-step gate, so it does not change the
calibration statistics and remains compatible with the within-NFE criterion.
"""

_base_ = ['../../a4_dpm_pp_chr2024.py']

model = dict(
    bbox_head=dict(
        sampling_timesteps=2,
        use_ensemble=False,
        box_renewal=True,
        adaptive_stopping=True,
        adaptive_stop_min_steps=1,
        adaptive_stop_geo_threshold=0.005,
        adaptive_stop_cls_threshold=0.013,
        adaptive_stop_score_threshold=0.7,
        adaptive_stop_min_box_scale=0.0,
        adaptive_stop_topk=100,
    )
)

work_dir = 'work_dirs/gacs_chr2024'
