"""Final-only test configuration for Dataset-2 detector run 1."""

_base_ = ["./paper_train3_lqcr_run1.py"]

model = dict(bbox_head=dict(quality_calibration_mode="final_only"))
load_from = None

