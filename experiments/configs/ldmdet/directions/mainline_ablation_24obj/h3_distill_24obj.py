"""Head Distillation v2: 少 Head (H=3) 蒸馏多 Head (H=6)

目的: 通过 headwise feature 蒸馏, 将 +DPM-Solver++ Teacher (H=6) 的知识压缩到
      Student (H=3), 实现 NFE 24→12 (2× 加速) 同时保持精度

机制:
  L_total = L_det(student) + λ · L_distill
  L_distill = (1/K) Σ_k MSE(student_fc_k, teacher_fc_{map(k)}.detach())

v2 六项修正 (见 docs/research/proposals/REFLOW_HEAD_DISTILL_IMPL_PLAN.md §2.2):
  1. 冻结 Student backbone (freeze_backbone=True)
  2. 蒸馏 fc_feature (非 pred_bboxes)
  3. head 映射 {0→0, 1→2, 2→5} (输入/中间/main 对齐)
  4. λ=0.05 保守起步
  5. coupling_mode='argmax' (确定性, 保证 proposal 对齐)
  6. distill 仅作用于 main head, aux 权重降至 0.5

Teacher: +DPM-Solver++ (mAP=0.863, H=6), 冻结, 仅 forward
Student: H=3, 从 Teacher head 0/2/5 初始化 (非随机)

SwanLab: 项目 'ldmdet-head-distill', 实验 'h3_distill'
"""
_base_ = ['./a4_dpm_pp_24obj.py']

# === Student 配置: H=3 + 蒸馏参数 ===
model = dict(
    bbox_head=dict(
        num_heads=3,                    # 6 → 3 (减半, NFE 24→12)
        use_distillation=True,          # 启用蒸馏
        distill_lambda=0.05,            # v2: 保守起步
        distill_head_map={0: 0, 1: 2, 2: 5},  # 输入/中间/main 对齐
        deep_supervision_aux_weight=0.5,      # v2: aux 权重降低
        freeze_backbone=True,           # v2: 冻结 backbone
        coupling=dict(
            coupling_mode='argmax',     # v2: 确定性 coupling
        ),
    ),
    # Teacher: +DPM-Solver++ checkpoint (auto-construct teacher config from student)
    teacher_checkpoint='work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth',
)

# === SwanLab ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-head-distill',
            experiment_name='h3_distill',
            description='Head Distillation v2: H=3 Student 蒸馏 H=6 Teacher (+DPM-Solver++) | λ=0.05, freeze backbone | bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
