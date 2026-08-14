"""H3 student paired with each historical D2 OT parent."""

_base_ = [
    '../_base_/models/karyoflow_ot_h3_distill_r50.py',
    '../_base_/schedules/head_distill_50e.py',
]

method = dict(
    method_id='karyoflow_ot_h3_distill_r50',
    role='paired_head_distillation',
    replication_unit='paired_student_by_parent_training_seed',
    parent_method_id='legacy_karyoflow_ot_r50',
    parent_checkpoint_required=True,
    parent_checkpoint_binding='teacher_checkpoint',
    pairing='same_training_seed',
    selection_split='val',
    selection_metric='coco/bbox_mAP',
    test_tuned=False,
)
