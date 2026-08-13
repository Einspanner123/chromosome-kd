"""KaryoFlow child that fits only final-stage localization-quality ranking."""

_base_ = ['./karyoflow_r50.py']

model = dict(bbox_head=dict(
    quality_only_training=True,
    quality_score_beta=2.0,
    quality_calibration_mode='final_only',
    single_head=dict(predict_iou_quality=True, quality_hidden=128),
    criterion=dict(
        quality_loss_weight=0.25,
        quality_focal_alpha=0.75,
        quality_focal_gamma=2.0,
    ),
))
