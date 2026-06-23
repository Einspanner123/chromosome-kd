"""基础运行时配置 — experiments/configs/_base_/default_runtime.py

自包含版本（已内联合并根目录 configs/_base_/default_runtime.py），
加上 LDMDet bridge 自定义导入。
覆盖 vis_backends 使用独立 SwanLab 项目。
"""

default_scope = 'mmdet'

# ── Hooks ──────────────────────────────────────────
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2, save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

# ── Environment ────────────────────────────────────
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# ── Logging ────────────────────────────────────────
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

# ── Custom Imports (LDMDet bridge) ─────────────────
custom_imports = dict(
    imports=[
        'experiments.mmdet_bridge.registry',
        'experiments.mmdet_bridge.detector',
        'experiments.mmdet_bridge.hooks',
        'swanlab.integration.mmengine',
    ],
    allow_failed_imports=False,
)

# ── Visualization ──────────────────────────────────
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='SwanlabVisBackend',
         init_kwargs=dict(project='ldmdet-ablation',
                          api_key='Huzvq1fnDeqOwgQo2AMAI',
                          resume='allow')),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
