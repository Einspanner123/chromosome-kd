"""Dataset-independent DDPM/Euler reference method."""

_base_ = [
    '../_base_/models/diffusiondet_ddpm_r50.py',
    '../_base_/schedules/detector_adamw_150e.py',
]

method = dict(
    method_id='diffusiondet_ddpm_r50',
    role='generation_reference',
    replication_unit='independent_training_seed',
    selection_split='val',
    selection_metric='coco/bbox_mAP',
    test_tuned=False,
)
