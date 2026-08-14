"""Archived RTMDet-L benchmark identity."""

_base_ = [
    '../_base_/models/rtmdet_legacy.py',
    '../_base_/schedules/rtmdet_legacy.py',
]

method = dict(
    method_id='legacy_rtmdet_l',
    role='archived_baseline',
    replication_unit='fixed_checkpoint',
    selection_split='val',
    selection_metric='coco/bbox_mAP',
    test_tuned=False,
    legacy=True,
)
