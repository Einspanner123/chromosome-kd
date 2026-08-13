"""Canonical D2 KaryoFlow recipe; intentionally uses the new random coupling."""

_base_ = [
    '../../_base_/datasets/d2_taichung.py',
    '../../_base_/models/karyoflow_r50.py',
    '../../_base_/schedules/detector_adamw_150e.py',
    '../../_base_/runtime/default.py',
]
experiment = dict(
    config_id='v2.d2.karyoflow_r50.train', dataset_id='D2',
    method_id='karyoflow_r50', role='main_detector',
    replication_unit='independent_training_seed',
    selection_split='val', selection_metric='coco/bbox_mAP',
    test_tuned=False, tracker_project='KaryoFlow-Dataset2',
)
