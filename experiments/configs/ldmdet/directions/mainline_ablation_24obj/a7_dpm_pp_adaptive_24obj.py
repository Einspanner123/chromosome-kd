"""24obj 主路线消融实验 A7: 方向 D — 自适应阶次 DPM-Solver++ (推理时改动, 无需重训练)

目的: 验证 R1 数据支持的自适应阶次假设 — 早期 step (t大) 曲率高用 3 阶, 后期 step (t小)
      曲率低用 2 阶, 在保持精度的同时减少高阶校正项的无效计算。
组件:
  + solver_type='dpm_solver_pp_adaptive' (RFDPMSolverAdaptive)
  + adaptive_solver_mode='static' (前 num_3rd_steps 步用 3 阶, 其余用 2 阶)
  + adaptive_num_3rd_steps=2 (前 2 步用 3 阶, 后 2 步用 2 阶)
  + sampling_timesteps=4 (与 A4 一致)

理论 (R1 数据支持):
  R1 诊断显示 η_str (||D1||/||x0||) 单调递减:
    step1=3.43 → step2=2.45 → step3=1.68
  即早期 step (t大) 曲率最高, 后期 step (t小) 曲率最低。
  原始方向 D 假设 ("t 小时曲率最大, 需高阶") 被证伪, 重构为相反假设。

  自适应策略:
    - static: 前 2 步用 3 阶 (计算 D2 校正), 后 2 步用 2 阶 (跳过 D2)
    - eta_threshold: 在线计算 ||D2||/||x0||, 超阈值才用 3 阶

无需重训练:
  - 基于 A4 (DPM-Solver++ 2阶) checkpoint 直接推理
  - RFDPMSolverAdaptive 在 solver 层面调整, 不影响网络权重
  - 仅推理时改动, 训练流程不变

对照:
  - A4 (DPM-Solver++ 2阶 4步) → 验证自适应 vs 固定 2 阶
  - A4_3 (DPM-Solver++ 3阶 4步) → 验证自适应 vs 固定 3 阶

SwanLab: 项目 'ldmdet-inference', 实验 'a7_adaptive_static'
"""
_base_ = ['./a4_dpm_pp_24obj.py']

# === 方向 D: 自适应阶次 DPM-Solver++ ===
# 注意: 仅修改 solver_type, 不修改训练相关参数
# 推理时使用 A4 checkpoint, 不需要重训练
model = dict(
    bbox_head=dict(
        solver_type='dpm_solver_pp_adaptive',
        adaptive_solver_mode='static',
        adaptive_num_3rd_steps=2,
        adaptive_eta_3rd_threshold=0.5,  # 仅 eta_threshold 模式生效
    ),
)

# === SwanLab (推理独立项目) ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-inference',
            experiment_name='a7_adaptive_static',
            description='24obj 方向 D: 自适应阶次 DPM-Solver++ (static, 前2步3阶+后2步2阶) | 推理时改动',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
