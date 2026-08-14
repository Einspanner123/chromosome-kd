"""Archived DiffusionDet DDPM benchmark identity."""

_base_ = [
    '../_base_/models/diffusiondet_ddpm_legacy.py',
    '../_base_/schedules/diffusiondet_ddpm_legacy.py',
]

method = dict(
    method_id='legacy_diffusiondet_ddpm_r50',
    role='archived_baseline',
    replication_unit='fixed_checkpoint',
    selection_split='val',
    selection_metric='coco/bbox_mAP',
    test_tuned=False,
    legacy=True,
)
