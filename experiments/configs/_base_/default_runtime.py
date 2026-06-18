"""基础运行时配置 — experiments/configs/_base_/default_runtime.py

继承自顶层 configs/_base_/default_runtime.py，加上 LDMDet bridge 自定义导入。
覆盖 vis_backends 使用独立 SwanLab 项目。
"""

_base_ = ['../../../configs/_base_/default_runtime.py']

# 自动注册 ldmdet bridge 模块
custom_imports = dict(
    imports=[
        'experiments.mmdet_bridge.registry',
        'experiments.mmdet_bridge.detector',
        'experiments.mmdet_bridge.hooks',
        'swanlab.integration.mmengine',
    ],
    allow_failed_imports=False,
)

# checkpoint — 仅保留最近 2 个 + 最佳 1 个  
default_hooks = dict(
    checkpoint=dict(max_keep_ckpts=2, save_last=True),
)

# SwanLab — 独立项目
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='SwanlabVisBackend', init_kwargs=dict(project='ldmdet-ablation', api_key='Huzvq1fnDeqOwgQo2AMAI', resume='allow')),
]
visualizer = dict(vis_backends=vis_backends)
