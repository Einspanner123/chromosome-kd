"""SC-RF (Self-Conditioned Rectified Flow) 24obj 实验

目的: 在 A4 完整 SOTA (RF+AdaLN+StochOT+DPM-Solver++) 基础上,
      验证自条件化机制能否通过迭代精炼提升检测性能

理论: docs/paper/proposals/SC-RF_Self-Conditioned_Rectified_Flow.md
  - 模型条件化于自身上一步 x0 预测, 实现迭代精炼 (残差学习视角)
  - 训练时 50% 概率启用自条件化 (no-grad 前向获取 x0_pred_prev)
  - 50% 使用零输入 (fallback 路径, 保证模型在无 x0_prev 时也能工作)
  - 零初始化 x0_prev_proj, 初始时 SC-RF 等价于标准 RF (平滑过渡)

组件 (在 A4 基础上叠加):
  + use_self_conditioning=True (bbox_head + single_head)
  + self_conditioning_prob=0.5 (50% 训练切换)
  - 其余: RF+AdaLN+StochOT eps5+DPM-Solver++ 4步 保持不变

对照: A4 DPM-Solver++ (0.8620 mAP) → 验证 SC-RF 的增益
SwanLab: 项目 'ldmdet-breakthrough', 实验 'sc_rf_24obj'
"""
_base_ = ['../mainline_ablation_24obj/a4_dpm_pp_24obj.py']

# === SC-RF: 自条件化 ===
model = dict(
    bbox_head=dict(
        use_self_conditioning=True,
        self_conditioning_prob=0.5,
        single_head=dict(
            use_self_conditioning=True,
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
            project='ldmdet-breakthrough',
            experiment_name='sc_rf_24obj',
            description='SC-RF: 自条件化 RF (基于 A4 DPM-Solver++ SOTA) | bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
