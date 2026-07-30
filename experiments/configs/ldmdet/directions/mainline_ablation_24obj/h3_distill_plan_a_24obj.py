"""Head Distillation 方案A: 解冻 backbone + 从 +DPM-Solver++ 加载 backbone/neck 权重

目的: 修复 v2 (freeze_backbone=True) 训练不理想 (best mAP 0.711, 远低于判据 0.84) 的问题

根因诊断 (2026-07-25):
  v2 冻结 Student backbone 在 ImageNet 预训练状态, 而 Teacher head 权重是在
  +DPM-Solver++ (染色体训练) backbone 特征上学习的 → 特征分布不匹配, 模型学习缓慢

方案A 修正:
  1. freeze_backbone=False — 解冻 backbone, 允许其适应 H=3 蒸馏任务
  2. 从 +DPM-Solver++ checkpoint 加载 backbone/neck 权重 (通过 detector.init_weights,
     **不用 load_from** — load_from 会覆盖 init_student_from_teacher 的 head 映射)
  3. lr 5e-5 → 1e-5 (微调场景, 起点 ~最优)
  4. max_epoch 150 → 50 (微调收敛快)
  5. warmup 5ep + cosine 50ep

保留 v2 其余修正 (已验证正确):
  - 蒸馏 fc_feature (非 pred_bboxes)
  - head 映射 {0→0, 1→2, 2→5} (输入/中间/main 对齐)
  - λ=0.05 保守起步
  - coupling_mode='argmax' (确定性, 保证 proposal 对齐)
  - aux 权重 0.5

Teacher: +DPM-Solver++ (mAP=0.863, H=6), 冻结, 仅 forward
Student: H=3, 从 Teacher head 0/2/5 初始化 + backbone/neck 从 +DPM-Solver++ 加载

SwanLab: 项目 'ldmdet-head-distill', 实验 'h3_distill_plan_a'
work_dir: work_dirs/h3_distill_plan_a_24obj/ (新目录, 不覆盖 v2)
"""
_base_ = ['./a4_dpm_pp_24obj.py']

# === Student 配置: H=3 + 蒸馏参数 (方案A: freeze=False) ===
model = dict(
    bbox_head=dict(
        num_heads=3,                    # 6 → 3 (减半, NFE 24→12)
        use_distillation=True,          # 启用蒸馏
        distill_lambda=0.05,            # v2: 保守起步
        distill_head_map={0: 0, 1: 2, 2: 5},  # 输入/中间/main 对齐
        deep_supervision_aux_weight=0.5,      # v2: aux 权重降低
        freeze_backbone=False,          # 方案A: 解冻 backbone (v2 为 True)
        coupling=dict(
            coupling_mode='argmax',     # v2: 确定性 coupling
        ),
    ),
    # Teacher: +DPM-Solver++ checkpoint (detector.__init__ 构建 Teacher, init_weights 加载 backbone)
    teacher_checkpoint='work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth',
)

# ⚠ 方案A 不使用 load_from — 会在 init_weights 后覆盖全部 state_dict,
#   破坏 init_student_from_teacher 的 head 映射 (+DPM-Solver++ head 1/2 错误覆盖 Student head 1/2)
#   backbone/neck 由 detector.init_weights._load_backbone_from_checkpoint 加载

# === 微调训练计划 (50 epoch, lr=1e-5) ===
max_epoch = 50
train_cfg = dict(max_epochs=max_epoch)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.00001, weight_decay=0.0001, _delete_=True
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=5),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        eta_min=0,
        begin=5,
        end=max_epoch,
        by_epoch=True,
    ),
]

# === batch_size=2 (与 v2 一致, ross A6000 48GB) ===
train_dataloader = dict(batch_size=2)

# === SwanLab ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-head-distill',
            experiment_name='h3_distill_plan_a',
            description='Head Distillation 方案A: H=3 蒸馏 +DPM-Solver++, 解冻 backbone + 从 +DPM-Solver++ 加载 backbone/neck | lr=1e-5, 50ep, bs=2',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
