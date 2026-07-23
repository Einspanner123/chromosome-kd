"""24obj M1 (workstation 专用): 形态感知 RoI 编码器 + BF16 AMP

workstation RTX A5000 仅 24GB, 无法跑 FP32 (A6 实测峰值 37.5GB).
启用 amp_dtype='bfloat16': 激活降至 BF16 (~省 50%), 显存约 20GB, 24GB 可行.
criterion 始终 FP32 (head.py autocast 外), 精度影响可控.
权重/梯度/优化器状态仍 FP32 (autocast 仅影响前向激活).

其余与 m1_morphology_aware_24obj.py 完全一致 (从 A4 微调 30ep, lr=1e-5).
对照基准: A4 (FP32, mAP=0.863). 若 M1+BF16 超过 0.863, 增益明确;
若接近 0.863, 需在本地 A6000 FP32 复现确认 (BF16 可能轻微掉点).

amp_dtype 用字符串 'bfloat16' (head.py 内部 getattr(torch, ...) 转换),
避免配置文件 import torch 触发 mmengine lazy_import 冲突.

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'm1_morphology_aware_ws'
"""

_base_ = ['./m1_morphology_aware_24obj.py']

# === workstation 适配: BF16 AMP (省显存, 24GB A5000 可行) ===
model = dict(
    bbox_head=dict(
        amp_dtype='bfloat16',
    ),
)

# === SwanLab: 区分 workstation 实验 ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-24obj',
            experiment_name='m1_morphology_aware_ws',
            description='24obj M1 (workstation BF16): 形态感知 RoI 编码器 | A5000 24GB, amp=bfloat16, bs=2, lr=1e-5, 30ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
