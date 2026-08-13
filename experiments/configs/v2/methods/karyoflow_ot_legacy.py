"""Dataset-independent identity for verified historical D2 OT-flow runs."""

_base_ = [
    '../_base_/models/karyoflow_ot_legacy_r50.py',
    '../_base_/schedules/detector_adamw_150e.py',
]

method = dict(
    method_id='legacy_karyoflow_ot_r50',
    role='archived_generation_method',
    replication_unit='independent_training_seed',
    selection_split='val',
    selection_metric='coco/bbox_mAP',
    test_tuned=False,
    legacy=True,
)
