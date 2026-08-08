"""CAPR-C2 final-only quality training on Dataset 2.

The A4 detector is frozen and only the last-cascade quality MLP is trained.
Unlike the original mechanism run, validation applies quality only to emitted
detection scores; solver, renewal, Top-K and GACS retain raw class logits.
"""

_base_ = ['./capr_quality_only_24obj.py']

model = dict(
    bbox_head=dict(quality_calibration_mode='final_only'),
)

load_from = 'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth'
work_dir = 'work_dirs/capr_quality_final_only_train_24obj'
