"""Credential-free runtime. Tracking is injected by the runner."""

default_scope = 'mmdet'
custom_imports = dict(
    imports=[
        'experiments.mmdet_bridge.registry',
        'experiments.mmdet_bridge.detector',
        'experiments.mmdet_bridge.hooks',
        'experiments.mmdet_bridge.transforms',
    ],
    allow_failed_imports=False,
)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1,
                    save_best='coco/bbox_mAP', rule='greater', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', name='visualizer', vis_backends=vis_backends)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=42, deterministic=False, diff_rank_seed=False)
