"""G0 of the prespecified generation ablation chain."""

_base_ = [
    '../_base_/models/strict_g0_ddpm_linear_scaleshift_r50.py',
    '../_base_/schedules/detector_adamw_150e.py',
]

method = dict(method_id='strict_g0_ddpm_linear_scaleshift_r50',
              role='strict_generation_ablation',
              replication_unit='independent_training_seed',
              ablation_stage='G0', selection_split='val',
              selection_metric='coco/bbox_mAP', test_tuned=False)
