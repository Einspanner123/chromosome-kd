"""Dataset-independent, parent-matched LQCR child method."""

_base_ = [
    '../_base_/models/karyoflow_lqcr_r50.py',
    '../_base_/schedules/lqcr_adamw_12e.py',
]

method = dict(
    method_id='karyoflow_lqcr_r50',
    role='paired_final_stage_intervention',
    replication_unit='paired_quality_head_by_parent',
    parent_method_id='karyoflow_r50',
    parent_checkpoint_required=True,
    pairing='same_training_seed',
    selection_split='val',
    selection_metric='coco/bbox_mAP',
    test_tuned=False,
)
