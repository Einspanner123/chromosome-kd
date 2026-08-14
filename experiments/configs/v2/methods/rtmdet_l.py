"""Trainable RTMDet-L benchmark with the frozen reference recipe."""

_base_ = [
    '../_base_/models/rtmdet_l.py',
    '../_base_/schedules/rtmdet_150e.py',
]

method = dict(
    method_id='rtmdet_l',
    role='trainable_external_baseline',
    replication_unit='independent_training_seed',
    selection_split='val',
    selection_metric='coco/bbox_mAP',
    test_tuned=False,
    recipe_origin='frozen_historical_benchmark',
)
