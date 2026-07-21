"""24obj 主路线消融实验 A6: 方向 C — Step-aware Embedding

目的: 让 cascade head 感知当前 DPM-Solver++ 的 step 编号, 测试 step-conditional
      行为是否能改善少步推理精度。
组件:
  + use_step_aware=True (在 DiffusionDetHead 中新增 step_mlp + step_proj)
  + step_proj 零初始化, 确保加载 A4 预训练权重时行为不变 (step_emb≡0)
  + solver_type='dpm_solver_pp', sampling_timesteps=4 (与 A4 一致)
  + num_solver_steps=4 (匹配 sampling_timesteps)

理论:
  DPM-Solver++ 的多步推理中, 早期 step (t大) 框粗定位, 后期 step (t小) 框精细调整。
  当前 cascade head 对所有 step 用同一组参数, 无法区分粗/细定位需求。
  step-aware embedding 让 head 知道"我在第几步", 学到 step-conditional 行为。

训练:
  - loss() 随机采样 step_idx ∈ [0, num_solver_steps-1], 与 t 独立
  - 让模型见到所有 step 模式, 学到 step-conditional 行为
  - 从 A4 checkpoint 初始化 (step_proj=0, 不破坏预训练)

推理:
  - predict() 在每个 solver step 前设置 _current_step_tensor = step_idx
  - 与训练时分布一致

风险:
  - 可能落入 AdaLN-Zero 零贡献陷阱 (step_proj 始终接近 0)
  - 需监控 step_proj.weight 的 L2 norm 是否随训练增长

对照: A4 (DPM-Solver++ 4步, 无 step-aware) → 验证 step-aware 的增益

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'a6_step_aware'
"""
_base_ = ['./a4_dpm_pp_24obj.py']

# === 方向 C: 启用 step-aware embedding ===
model = dict(
    bbox_head=dict(
        use_step_aware=True,
        num_solver_steps=4,  # 匹配 sampling_timesteps=4
    ),
)

# === SwanLab ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-24obj',
            experiment_name='a6_step_aware',
            description='24obj 方向 C: Step-aware Embedding (DPM-Solver++ 4步 + step_idx 融合) | bs=8, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
