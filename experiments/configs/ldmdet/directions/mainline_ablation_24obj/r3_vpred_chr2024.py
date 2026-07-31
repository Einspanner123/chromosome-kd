"""Dataset 1 (Chromosome20240904) R3 v-prediction 对照重训

目的: 在 Dataset 1 上补全 §八 v-prediction 对照, 闭合双数据集缺口
设置: 在 Dataset 1 +DPM-Solver++ baseline 基础上, 在 criterion 中启用 v_prediction=True (1/t² 损失重加权)
对照:
  - Dataset 1 +DPM-Solver++ baseline (a4_dpm_pp_chr2024, 3-seed 0.747±0.001) → 验证 R3.2 跨数据集
  - Dataset 2 r3_vpred_24obj (3-seed 0.857±0.0015) → 双数据集对照
预期: v-prediction mAP 低于 x0-prediction (R3.2: shifted schedule 下 x0-prediction 更优)

理论依据: theory_analysis_RF_DPM.md §4 (R3)
实现说明 (同 r3_vpred_24obj.py):
  - 不修改网络输出语义 (仍输出 x0_pred), 通过 1/t² 损失加权模拟 v-prediction 梯度动态
  - L1 loss 下严格等价应为 1/t, 这里用 1/t² 匹配理论 doc §4.3 的梯度放大 claim
  - 对权重做 batch normalization (均值=1), 避免训练崩溃
  - GIoU loss 保持不加权 (不是 t 的简单函数)

3 seeds (42/123/789) 重训, 训练时通过 --seed 覆盖

启动:
  python experiments/runners/train.py experiments/configs/ldmdet/directions/mainline_ablation_24obj/r3_vpred_chr2024.py --seed 42  --gpu-id 1
  python experiments/runners/train.py experiments/configs/ldmdet/directions/mainline_ablation_24obj/r3_vpred_chr2024.py --seed 123 --gpu-id 1
  python experiments/runners/train.py experiments/configs/ldmdet/directions/mainline_ablation_24obj/r3_vpred_chr2024.py --seed 789 --gpu-id 1
"""
_base_ = ['../../a4_dpm_pp_chr2024.py']

# === 启用 v-prediction 等价的 1/t² 损失加权 ===
model = dict(
    bbox_head=dict(
        criterion=dict(
            v_prediction=True,
            v_prediction_t_eps=1e-2,
        ),
    ),
)

# === SwanLab (与 24obj 版本同项目, 便于双数据集对照) ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-r3-vpred',
            experiment_name='r3_vpred_chr2024',
            description='R3 Dataset 1: v-prediction 对照 (1/t² 加权) | 验证 §4.3 梯度放大 claim 跨数据集 | chr2024, bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
