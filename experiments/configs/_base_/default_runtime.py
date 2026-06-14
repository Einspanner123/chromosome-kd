"""基础运行时配置 — experiments/configs/_base_/default_runtime.py

继承自顶层 configs/_base_/default_runtime.py，加上 LDMDet bridge 自定义导入。
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

# 异步 checkpoint
default_hooks = dict(
    checkpoint=dict(
        type='AsyncCheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_best='coco/bbox_mAP',
        rule='greater',
    ),
)
