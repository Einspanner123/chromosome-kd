"""旧数据集 (Chromosome20240904) StochOT ε=5 基准配置

用途: 旧数据集主路线 StochOT 基准, 用于多种子训练和 ε 消融对照
基础: ldmdet_rf_heun_adaln_stochot_eps5.py (默认使用旧数据集)
耦合: ot_flow (Sinkhorn Stochastic OT), ε=5, multinomial 采样

SwanLab: 项目 'ldmdet-mainline-ablation-old', 实验 'stochot_eps5'
"""
_base_ = ['../../../ldmdet_rf_heun_adaln_stochot_eps5.py']

# === SwanLab ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-old',
            experiment_name='stochot_eps5',
            description='旧数据集 StochOT ε=5 (RF+Heun+AdaLN+StochOT) | bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
