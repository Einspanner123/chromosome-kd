"""Trainable YOLOX-S benchmark with the frozen reference recipe."""

_base_ = [
    '../_base_/models/yolox_s.py',
    '../_base_/schedules/yolox_200e.py',
]

method = dict(
    method_id='yolox_s',
    role='trainable_external_baseline',
    replication_unit='independent_training_seed',
    selection_split='val',
    selection_metric='coco/bbox_mAP',
    test_tuned=False,
    recipe_origin='frozen_historical_benchmark',
)
