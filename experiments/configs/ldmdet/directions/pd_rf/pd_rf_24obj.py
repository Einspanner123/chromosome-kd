"""PD-RF (Direct Knowledge Distillation for RF) 24obj 实验

目的: 将 A4 (4步 DPM-Solver++) 蒸馏到 1步 Euler, 实现推理加速
      4步 (55ms) → 1步 (14ms), 目标 mAP ≥ A1 baseline + 0.005 (≥0.861)

理论: docs/paper/proposals/PD-RF_Progressive_Distillation.md
  - 直接蒸馏 (非渐进式): 4步教师 → 1步学生, 一步到位
  - 教师使用 DPM-Solver++ 4步推理, 学生使用 Euler 1步
  - 蒸馏损失: MSE(student_x0_raw, teacher_x0_raw.detach()) (raw 空间)
  - 共享噪声 x_raw_shared: 教师和学生使用同一初始噪声, 保证 proposal 对应
  - 教师推理关闭 box_renewal (保持 proposal 对应)
  - 教师参数冻结, 仅学生参与梯度更新
  - 蒸馏定位为 box 正则化信号 (非梯度对齐, 见理论 2.3)

组件 (在 A4 架构基础上):
  学生: solver_type='euler', sampling_timesteps=1 (1步 Euler)
  教师: solver_type='dpm_solver_pp', sampling_timesteps=4 (4步 DPM-Solver++)
        权重从 A4 best checkpoint 加载, 自动冻结
  + use_distillation=True, distill_lambda=1.0
  - RF + AdaLN + StochOT eps5 架构保持不变

对照: A4 (4步 DPM-Solver++, mAP=0.862) → 验证蒸馏后 1步能否逼近 4步质量
SwanLab: 项目 'ldmdet-breakthrough', 实验 'pd_rf_24obj'
"""
_base_ = ['../mainline_ablation_24obj/a4_dpm_pp_24obj.py']

# === PD-RF: 学生配置 (1步 Euler) + 教师配置 (4步 DPM-Solver++) ===
model = dict(
    bbox_head=dict(
        # 学生: 1步 Euler (从 A4 的 dpm_solver_pp 4步 改为 euler 1步)
        solver_type='euler',
        sampling_timesteps=1,
        # 启用蒸馏
        use_distillation=True,
        distill_lambda=1.0,
    ),
    # 教师: 4步 DPM-Solver++, 权重从 A4 best checkpoint 加载
    teacher_cfg=dict(
        solver_type='dpm_solver_pp',
        sampling_timesteps=4,
        checkpoint='work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth',
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
            experiment_name='pd_rf_24obj',
            description='PD-RF: 直接蒸馏 4步→1步 (A4 teacher → Euler 1步 student) | bs=8, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
