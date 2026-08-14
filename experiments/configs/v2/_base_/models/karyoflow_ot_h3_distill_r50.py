"""Three-head student paired with the historical OT KaryoFlow parent."""

_base_ = ['./ldmdet_r50_common.py']

model = dict(bbox_head=dict(
    diffusion_type='rectified_flow', solver_type='dpm_solver_pp',
    sampling_timesteps=4, rf_schedule='shifted', rf_shift=3.0,
    ddim_sampling_eta=1.0,
    coupling=dict(
        type='ot_flow', epsilon=5.0, num_iters=20,
        coupling_mode='multinomial'),
    box_renewal=True, use_ensemble=True,
    single_head=dict(time_conditioning='adaln_zero'),
    num_heads=3,
    use_distillation=True,
    distill_lambda=0.05,
    distill_head_map={0: 0, 1: 2, 2: 5},
    deep_supervision_aux_weight=0.5,
    freeze_backbone=False,
))
