"""CAPR-C2 strict final-only score calibration ablation.

Uses the already trained epoch-2 quality weights. Solver, Top-K and renewal use
the original A4 class logits; p(class) * q^2 is applied only to detections sent
to post-processing. No retraining is required.
"""

_base_ = ['./capr_quality_only_24obj.py']

model = dict(
    bbox_head=dict(quality_calibration_mode='final_only'),
)

load_from = 'work_dirs/capr_quality_only_24obj/best_coco_bbox_mAP_epoch_2.pth'
work_dir = 'work_dirs/capr_quality_final_only_24obj'
