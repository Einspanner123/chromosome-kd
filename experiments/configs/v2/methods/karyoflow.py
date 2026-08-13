"""Dataset-independent canonical KaryoFlow method."""

_base_ = [
    '../_base_/models/karyoflow_r50.py',
    '../_base_/schedules/detector_adamw_150e.py',
]

method = dict(
    method_id='karyoflow_r50',
    role='main_detector',
    replication_unit='independent_training_seed',
    selection_split='val',
    selection_metric='coco/bbox_mAP',
    test_tuned=False,
)
