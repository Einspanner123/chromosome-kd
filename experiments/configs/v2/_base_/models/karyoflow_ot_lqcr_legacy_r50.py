"""Final-only LQCR child of the archived D2 OT-flow detector.

This isolated legacy definition is flattened to keep the public inheritance
depth bounded; it is validated against the parent method by the compatibility
suite and tensor-lineage audit.
"""

_base_ = ['./ldmdet_r50_common.py']

model = dict(bbox_head=dict(
    diffusion_type='rectified_flow', solver_type='dpm_solver_pp',
    sampling_timesteps=4, rf_schedule='shifted', rf_shift=3.0,
    ddim_sampling_eta=1.0,
    coupling=dict(type='ot_flow', epsilon=5.0, num_iters=20,
                  coupling_mode='multinomial'),
    box_renewal=True, use_ensemble=True,
    quality_only_training=True,
    quality_score_beta=2.0,
    quality_calibration_mode='final_only',
    single_head=dict(
        time_conditioning='adaln_zero', predict_iou_quality=True,
        quality_hidden=128),
    criterion=dict(
        quality_loss_weight=0.25, quality_focal_alpha=0.75,
        quality_focal_gamma=2.0),
))
