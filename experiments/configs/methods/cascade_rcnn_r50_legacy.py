"""Archived Cascade R-CNN R50 benchmark identity."""

_base_ = [
    '../_base_/models/cascade_legacy.py',
    '../_base_/schedules/cascade_legacy.py',
]

method = dict(
    method_id='legacy_cascade_rcnn_r50',
    role='archived_baseline',
    replication_unit='fixed_checkpoint',
    selection_split='val',
    selection_metric='coco/bbox_mAP',
    test_tuned=False,
    legacy=True,
)
