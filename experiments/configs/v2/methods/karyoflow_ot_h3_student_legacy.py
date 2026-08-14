"""Fixed-checkpoint deployment identity for the H3 distilled student."""

_base_ = [
    '../_base_/models/karyoflow_ot_h3_student_legacy_r50.py',
    '../_base_/schedules/detector_adamw_150e.py',
]

method = dict(
    method_id='legacy_karyoflow_ot_h3_student_r50',
    role='archived_deployment_method',
    replication_unit='single_parent_student',
    selection_split='val',
    selection_metric='coco/bbox_mAP',
    test_tuned=False,
    legacy=True,
    inference_only=True,
    parent_method_id='legacy_karyoflow_ot_r50',
)
