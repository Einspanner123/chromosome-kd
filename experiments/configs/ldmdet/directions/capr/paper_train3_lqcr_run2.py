"""Final-stage quality branch for Dataset-2 independent detector run 2."""

_base_ = ["../capr_quality_only_24obj.py"]

load_from = "work_dirs/multi_seed/a4_dpm_pp_24obj/seed_789/best_coco_bbox_mAP_epoch_72.pth"
work_dir = "work_dirs/paper_d2_lqcr_trainrun_2"

