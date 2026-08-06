"""24obj AAC Phase 2: Anderson-Accelerated Cascade 端到端重训

目的: Phase 1 微调显示增益 (ΔmAP > 0.002) 后, 从 RF+Heun 阶段重训,
      AAC 贯穿整个训练, 验证端到端统计显著性。

与 Phase 1 (aac_24obj.py) 的区别:
  - 起点: RF+Heun (a1) 而非 +DPM-Solver++ (a4)
  - β: 1.0 (全幅 Anderson) 而非 0.5 (阻尼)
  - lr: 5e-5 (baseline) 而非 1e-5 (微调)
  - epoch: 150 (完整训练) 而非 50 (微调)
  - 无 load_from (从头训) 而非 从 checkpoint 微调

机制 (详见 docs/research/proposals/AAC_DESIGN.md §4.3 Phase 2):
  - AAC(m=2, β=1.0) 贯穿 150 epoch, 让 head 从训练初期适应 Anderson 混合
  - β=1.0 全幅: 训练充分时不需阻尼, 直接用 Anderson 全校外推
  - 与 baseline (a1_rf_heun) 唯一区别: use_aac=True

对照: a1_rf_heun_24obj (无 AAC) → 隔离 AAC 端到端增益
判据 (§10.2): 3-seed mAP 0.863~0.866 ± 0.003

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'aac_e2e_m2_beta10'
"""

_base_ = ['./a1_rf_heun_24obj.py']

# === AAC: Anderson-Accelerated Cascade (Phase 2 端到端) ===
# β=1.0 (全幅, 训练充分无需阻尼), m=2 (有限内存 AA(2))
model = dict(
    bbox_head=dict(
        use_aac=True,
        aac_mem_depth=2,           # m=2 (AA(2) 超线性收敛)
        aac_beta=1.0,              # 全幅 Anderson (Phase 2, 训练充分)
        aac_lambda=1e-6,           # Tikhonov 正则化 (数值稳定)
        aac_stop_grad_history=True,  # Phantom gradient: 历史不反传
        aac_gamma_norm_clip=10.0,  # γ 范数裁剪 (防止爆炸)
    ),
)

# === 端到端训练计划 (与 baseline 对齐: lr=5e-5, 150ep) ===
# a1_rf_heun_24obj 的 base (ldmdet_rf_heun_shifted_bs2.py) 已设 lr=5e-5, max_epoch=150,
# 此处显式声明以隔离 AAC 变量 (确保唯一差异是 use_aac)
max_epoch = 150
train_cfg = dict(max_epochs=max_epoch)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.00005, weight_decay=0.0001, _delete_=True
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        eta_min=0,
        begin=5,
        end=max_epoch,
        by_epoch=True,
    ),
]

# === EarlyStopping (与 multiseed 配置对齐, patience=30) ===
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        priority=50,
        patience=30,
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        rule='greater',
    ),
    dict(type='CopyProjectHook', priority='VERY_LOW'),
]

# === SwanLab ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-24obj',
            experiment_name='aac_e2e_m2_beta10',
            description='24obj AAC Phase 2 端到端: Anderson(m=2, β=1.0) 贯穿训练 | 从 RF+Heun 重训 150ep | bs=2, lr=5e-5',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
