"""PD-RF (Direct Knowledge Distillation for RF) 24obj 实验

目的: 将 A4 (4步 DPM-Solver++) 蒸馏到 1步 Euler, 实现推理加速
      4步 (55ms) → 1步 (14ms), 目标 mAP ≥ A1 baseline + 0.005 (≥0.861)

理论: docs/paper/proposals/PD-RF_Progressive_Distillation.md
  - 直接蒸馏 (非渐进式): 4步教师 → 1步学生, 一步到位
  - 教师使用 DPM-Solver++ 4步推理, 学生使用 Euler 1步
  - 蒸馏损失: MSE(student_x0_raw, teacher_x0_raw.detach()) (raw 空间)
    + KL(student_cls || teacher_cls) (分类蒸馏, 修复分类头训练不足)
  - 共享噪声 x_raw_shared: 教师和学生使用同一初始噪声, 保证 proposal 对应
  - 教师推理关闭 box_renewal (保持 proposal 对应)
  - 教师参数冻结, 仅学生参与梯度更新
  - 蒸馏定位为 box 正则化信号 (非梯度对齐, 见理论 2.3)

修复 (v2):
  - load_from A4 checkpoint: 学生从 A4 权重初始化, 避免从零学习
  - cascade_detach=False: 允许蒸馏梯度训练整个级联 (原 True 仅训练末 head)
  - 增加分类蒸馏: KL散度损失, 补充分类头在 t=1.0 的训练

组件 (在 A4 架构基础上):
  学生: solver_type='euler', sampling_timesteps=1 (1步 Euler)
        从 A4 checkpoint 加载权重, cascade_detach=False
  教师: solver_type='dpm_solver_pp', sampling_timesteps=4 (4步 DPM-Solver++)
        权重从 A4 best checkpoint 加载, 自动冻结
  + use_distillation=True, distill_lambda=1.0
  - RF + AdaLN + StochOT eps5 架构保持不变

对照: A4 (4步 DPM-Solver++, mAP=0.862) → 验证蒸馏后 1步能否逼近 4步质量
SwanLab: 项目 'ldmdet-breakthrough', 实验 'pd_rf_24obj_v2'
"""
_base_ = ['../mainline_ablation_24obj/a4_dpm_pp_24obj.py']

# === 学生从 A4 checkpoint 初始化 (避免从零学习, 初始 gap 可控) ===
load_from = 'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth'

# === PD-RF: 学生配置 (1步 Euler) + 教师配置 (4步 DPM-Solver++) ===
model = dict(
    bbox_head=dict(
        # 学生: 1步 Euler (从 A4 的 dpm_solver_pp 4步 改为 euler 1步)
        solver_type='euler',
        sampling_timesteps=1,
        # 启用蒸馏
        use_distillation=True,
        distill_lambda=1.0,
        # cascade_detach=False: 允许蒸馏梯度训练整个级联
        # (原 True 导致 Head 1-5 无蒸馏梯度, 推理时级联失效)
        cascade_detach=False,
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
            experiment_name='pd_rf_24obj_v2',
            description='PD-RF v2: load_from A4 + cascade_detach=False + cls蒸馏 | bs=8, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
