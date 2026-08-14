"""Archived YOLOX-S benchmark identity."""

_base_ = [
    '../_base_/models/yolox_legacy.py',
    '../_base_/schedules/yolox_legacy.py',
]

method = dict(
    method_id='legacy_yolox_s',
    role='archived_baseline',
    replication_unit='fixed_checkpoint',
    selection_split='val',
    selection_metric='coco/bbox_mAP',
    test_tuned=False,
    legacy=True,
)
