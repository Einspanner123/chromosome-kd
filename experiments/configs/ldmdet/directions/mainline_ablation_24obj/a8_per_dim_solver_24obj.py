"""24obj 方向 A Phase 2: per-dim 阶数分配 DPM-Solver++

(推理时改动, 无需重训练)

目的: 验证 per-dim eta_str 诊断支持的假设 — h 维度曲率最小可用 1 阶 (Euler),
      cx/cy/w 维度曲率较大用 2 阶 (DPM-Solver++), 在保持精度的同时探索
      检测专用 solver 设计。

组件:
  + solver_type='dpm_solver_pp_per_dim' (RFDPMSolverPerDim)
  + 默认 euler_dims=(3,) (h 维度用 1 阶), dpm_dims=(0,1,2) (cx,cy,w 用 2 阶)
  + sampling_timesteps=4 (与 +DPM-Solver++ 一致)

理论 (方向 A Phase 1 诊断):
  per-dim eta_str 诊断显示:
    h 维度 eta_str = 4-11 (曲率最小)
    cx,cy 维度 eta_str = 17-50 (曲率最大)
    w 维度 eta_str 介于二者之间, 与 cx/cy 接近
  即 h 维度的轨迹最接近直线, 1 阶 (Euler) 应足够;
  cx/cy/w 维度曲率较大, 需要 2 阶 (DPM-Solver++) 校正。

无需重训练:
  - 基于 +DPM-Solver++ (DPM-Solver++ 2阶) checkpoint 直接推理
  - RFDPMSolverPerDim 在 solver 层面调整, 不影响网络权重
  - 仅推理时改动, 训练流程不变

对照:
  - +DPM-Solver++ (DPM-Solver++ 2阶 4步, 全维度 2 阶) → 验证 per-dim vs 全 2 阶
  - 自适应阶次 DPM-Solver++ (全维度但 per-step 降阶) → 方向 D 的 per-step 降阶

风险:
  - per-dim 阶数分配可能破坏 bbox 4 维度的耦合性 (位置 cx,cy 和尺度 w,h 在物理上相关)

SwanLab: 项目 'ldmdet-inference', 实验 'a8_per_dim_solver'
"""
_base_ = ['./a4_dpm_pp_24obj.py']

# === 方向 A Phase 2: per-dim 阶数分配 DPM-Solver++ ===
# 注意: 仅修改 solver_type, 不修改训练相关参数
# 推理时使用 +DPM-Solver++ checkpoint, 不需要重训练
model = dict(
    bbox_head=dict(
        solver_type='dpm_solver_pp_per_dim',
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
            experiment_name='a8_per_dim_solver',
            description='24obj 方向 A Phase 2: per-dim solver (h=1阶, cxcy/w=2阶) | 推理时改动',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
