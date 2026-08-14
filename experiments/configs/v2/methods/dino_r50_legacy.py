"""Archived DINO-R50 benchmark identity."""

_base_ = [
    '../_base_/models/dino_legacy.py',
    '../_base_/schedules/dino_legacy.py',
]

method = dict(
    method_id='legacy_dino_r50',
    role='archived_baseline',
    replication_unit='fixed_checkpoint',
    selection_split='val',
    selection_metric='coco/bbox_mAP',
    test_tuned=False,
    legacy=True,
)
