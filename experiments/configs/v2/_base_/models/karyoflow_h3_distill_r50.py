"""Three-head KaryoFlow student trained from a six-head parent."""

_base_ = ['./karyoflow_r50.py']

model = dict(bbox_head=dict(
    num_heads=3,
    use_distillation=True,
    distill_lambda=0.05,
    distill_head_map={0: 0, 1: 2, 2: 5},
    deep_supervision_aux_weight=0.5,
    freeze_backbone=False,
))
