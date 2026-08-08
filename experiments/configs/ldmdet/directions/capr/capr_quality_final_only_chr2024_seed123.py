"""CAPR-C2 final-only quality calibration on Dataset 1, A4 seed 123."""

_base_ = ['./capr_quality_final_only_chr2024_seed42.py']

load_from = 'work_dirs/a4_dpm_pp_chr2024_seed123/best_coco_bbox_mAP_epoch_85.pth'
work_dir = 'work_dirs/capr_quality_final_only_chr2024_seed123'
