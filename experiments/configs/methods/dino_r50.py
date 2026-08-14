"""Trainable DINO-R50 benchmark with the frozen reference recipe."""

_base_ = [
    '../_base_/models/dino_r50.py',
    '../_base_/schedules/dino_150e.py',
]

method = dict(
    method_id='dino_r50',
    role='trainable_external_baseline',
    replication_unit='independent_training_seed',
    selection_split='val',
    selection_metric='coco/bbox_mAP',
    test_tuned=False,
    recipe_origin='frozen_historical_benchmark',
)
