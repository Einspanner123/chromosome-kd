"""Historical Dataset 2 RF/Heun method before AdaLN-Zero."""

_base_ = [
    '../_base_/models/rf_heun_pre_adaln_legacy_r50.py',
    '../_base_/schedules/detector_adamw_150e.py',
]

method = dict(
    method_id='legacy_rf_heun_pre_adaln_r50',
    role='archived_generation_reference',
    replication_unit='single_training_checkpoint',
    selection_split='val',
    selection_metric='coco/bbox_mAP',
    test_tuned=False,
    legacy=True,
)
