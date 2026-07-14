"""旧数据集 (Chromosome20240904) StochOT ε=1 消融

用途: ε 扫描 — 验证理论预测 ε*≈1-3
基础: stochot_eps5_old.py, 仅修改 epsilon=1.0
理论: ε 越小 → OT 越精确 (趋向硬 OT), 但训练越不稳定

SwanLab: 项目 'ldmdet-mainline-ablation-old', 实验 'stochot_eps1'
"""
_base_ = ['./stochot_eps5_old.py']

model = dict(
    bbox_head=dict(
        coupling=dict(
            type='ot_flow',
            epsilon=1.0,
            num_iters=20,
            coupling_mode='multinomial',
        ),
    ),
)

# === SwanLab ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-old',
            experiment_name='stochot_eps1',
            description='旧数据集 StochOT ε=1 消融 | bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
