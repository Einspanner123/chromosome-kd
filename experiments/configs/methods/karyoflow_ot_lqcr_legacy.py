"""Dataset-independent identity for verified historical D2 OT-LQCR runs."""

_base_ = [
    '../_base_/models/karyoflow_ot_lqcr_legacy_r50.py',
    '../_base_/schedules/lqcr_adamw_12e.py',
]

custom_hooks = [
    dict(type='EarlyStoppingHook', priority=50, patience=30,
         min_delta=0.001, monitor='coco/bbox_mAP', rule='greater'),
]

method = dict(
    method_id='legacy_karyoflow_ot_lqcr_r50',
    role='archived_paired_final_stage_intervention',
    replication_unit='paired_quality_head_by_parent',
    parent_method_id='legacy_karyoflow_ot_r50',
    parent_checkpoint_required=True,
    pairing='same_training_seed',
    selection_split='val',
    selection_metric='coco/bbox_mAP',
    test_tuned=False,
    legacy=True,
)
