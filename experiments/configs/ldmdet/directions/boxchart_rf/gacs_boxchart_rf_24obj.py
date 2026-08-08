"""Geometry-Aware Cascade Stopping (GACS), 1/2-step BoxChart-RF."""

_base_ = ['./boxchart_rf_24obj.py']

model = dict(
    bbox_head=dict(
        sampling_timesteps=2,
        use_ensemble=False,
        box_renewal=False,
        adaptive_stopping=True,
        adaptive_stop_min_steps=1,
        # Calibrated on seed 42, then held fixed for seed 0/1 validation.
        # Classification consistency is the active gate; the geometry cap
        # rejects only extreme non-converged cascade trajectories.
        adaptive_stop_geo_threshold=0.02,
        adaptive_stop_cls_threshold=0.0079,
        adaptive_stop_score_threshold=0.9,
        adaptive_stop_min_box_scale=0.0,
        adaptive_stop_topk=100,
    )
)

load_from = 'work_dirs/boxchart_rf_24obj/best_coco_bbox_mAP_epoch_3.pth'
work_dir = 'work_dirs/gacs_boxchart_rf_24obj'
