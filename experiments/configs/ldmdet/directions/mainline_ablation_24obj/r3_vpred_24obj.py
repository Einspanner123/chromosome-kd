"""R3 v-prediction 对照重训: 验证理论 doc §4.3 的梯度放大 claim

理论依据: theory_analysis_RF_DPM.md §4 (R3)
目的: 验证 v-prediction 在低维 RF + shifted schedule 下劣于 x0-prediction
设置: 在 A4 baseline (DPM-Solver++ + RF + AdaLN + StochOT eps5) 基础上,
      在 criterion 中启用 v_prediction=True (1/t² 损失重加权)
对照: A4 baseline (x0-prediction) → 验证命题 R3.2 (shifted schedule 下 x0-prediction 更优)
预期: v-prediction mAP 低于 A4 (理论 doc §4.3 预测 t→0 时梯度放大 1/t² 加剧方差)

实现说明:
  - 不修改网络输出语义 (仍输出 x0_pred), 通过 1/t² 损失加权模拟 v-prediction 梯度动态
  - L1 loss 下严格等价应为 1/t, 这里用 1/t² 匹配理论 doc §4.3 的梯度放大 claim
  - 对权重做 batch normalization (均值=1), 避免训练崩溃
  - GIoU loss 保持不加权 (不是 t 的简单函数)

3 seeds (42/123/789) 重训, 训练时通过 --cfg-options 覆盖 experiment_name 和 --seed
"""
_base_ = ['./a4_dpm_pp_24obj.py']

# === 启用 v-prediction 等价的 1/t² 损失加权 ===
model = dict(
    bbox_head=dict(
        criterion=dict(
            v_prediction=True,
            v_prediction_t_eps=1e-2,
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
            project='ldmdet-r3-vpred',
            experiment_name='r3_vpred',
            description='R3: v-prediction 对照 (1/t² 加权) | 验证 §4.3 梯度放大 claim | bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
