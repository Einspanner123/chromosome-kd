"""Snapshot of the initial Dataset-1 GACS mechanism validation.

This variant intentionally disables box renewal.  It is retained only to
reproduce the first calibration sweep; the production-relevant configuration
is gacs_chr2024.py with renewal enabled.
"""

_base_ = ['../../a4_dpm_pp_chr2024.py']

model = dict(
    bbox_head=dict(
        sampling_timesteps=2,
        use_ensemble=False,
        box_renewal=False,
        adaptive_stopping=True,
        adaptive_stop_min_steps=1,
        adaptive_stop_geo_threshold=0.005,
        adaptive_stop_cls_threshold=0.014,
        adaptive_stop_score_threshold=0.7,
        adaptive_stop_min_box_scale=0.0,
        adaptive_stop_topk=100,
    )
)

work_dir = 'work_dirs/gacs_chr2024_no_renewal'
