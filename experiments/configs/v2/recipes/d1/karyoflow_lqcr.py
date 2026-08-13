"""Canonical D1 parent-matched LQCR training recipe."""

_base_ = [
    '../../_base_/datasets/d1_inhouse1700.py',
    '../../_base_/models/karyoflow_lqcr_r50.py',
    '../../_base_/schedules/lqcr_adamw_12e.py',
    '../../_base_/runtime/default.py',
]
experiment = dict(
    config_id='v2.d1.karyoflow_lqcr_r50.train', dataset_id='D1_INHOUSE1700_V1',
    method_id='karyoflow_lqcr_r50', role='paired_final_stage_intervention',
    replication_unit='paired_quality_head_by_parent',
    parent_method_id='karyoflow_r50', parent_checkpoint_required=True,
    selection_split='val', selection_metric='coco/bbox_mAP',
    test_tuned=False, tracker_project='KaryoFlow-Self1700',
)
