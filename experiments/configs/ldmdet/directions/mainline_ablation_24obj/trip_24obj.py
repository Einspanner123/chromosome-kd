"""24obj TRIP: Tikhonov/MAP 正则化回归目标 (保守方向 Phase 1)

目的: 将 RF x0-prediction 的训练回归目标由 GT x_0 替换为
      贝叶斯 MAP 收缩估计 x_tilde(t) = (1-s(t))*x_0 + s(t)*mu_p^c,
      缓解大 t 段 (噪声端) SNR 退化导致的过拟合.

机制 (详见 docs/research/proposals/FEASIBLE_TRIP.md §7):
  - 仅改 criterion box target (从 GT 替换为 Tikhonov/MAP 收缩估计)
  - 收缩因子 s(t) = (t²/σ_p²) / ((1-t)² + t²/σ_p²), MAP 主方案 λ=t²
  - 边界: s(0)=0 (目标=GT, 零正则), s(1)=1 (目标=mu_p, 完全收缩)
  - 不改架构, 不改推理流程, 不改 coupling/solver
  - 类条件先验 (mu_p^c, sigma_bar_sq^c) 离线估计, 在线向量化查询

前置步骤 (必须先运行, 先验文件已生成):
    python ldmdet/tools/estimate_class_priors.py \\
        --ann-file data/24_chromosomes_object/coco/train/_annotations.coco.json \\
        --output data/class_priors_24obj.pkl \\
        --num-classes 24

与已证伪方向的严格区分 (§5):
  - vs SeesawLoss/CBS: TRIP 由类几何统计驱动 (非类频率)
  - vs scale_aware_loss: TRIP 不引入尺度先验, 收缩因子由 MAP 导出
  - vs ReFlow: TRIP 独立 (互斥 mode), 不依赖 +DPM-Solver++ 推理 coupling

训练:
  - 从 +DPM-Solver++ best checkpoint 微调 (load_from)
  - 50 epoch, lr=1e-5 (与 AAC/LVD 微调配置对齐)
  - 5 epoch linear warmup + cosine annealing

对照: +DPM-Solver++ (DPM-Solver++, 无目标收缩) → 验证 SNR 退化段正则化效果
预期 (R2 确认): mAP +0.001~+0.005; Y AP +0.002~+0.010; η_str 下降 10~25%
判据 (Phase 1): val/mAP ≥ 0.863 (不掉点) + trip_s_fg_mean 合理分布

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'trip_map'
work_dir: work_dirs/trip_24obj/
"""

_base_ = ['./a4_dpm_pp_24obj.py']

# === TRIP: Tikhonov/MAP 正则化回归目标 ===
model = dict(
    bbox_head=dict(
        criterion=dict(
            box_target_mode='trip',               # box target = MAP 收缩估计
            class_priors='data/class_priors_24obj.pkl',  # 离线估
            trip_lambda_mode='map',               # R1 反馈 2: 'map' (λ=t², 主) / 'morozov' (备选)
            trip_tau=1.0,                         # Morozov 偏差原理 τ (仅 morozov 模式生效)
        ),
    ),
)

# === 从 +DPM-Solver++ checkpoint 微调 ===
# TRIP 无新可学习参数 (仅改 target), load_from 直接加载 +DPM-Solver++ 全部权重
load_from = 'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth'

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

# === SwanLab ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-24obj',
            experiment_name='trip_map',
            description='24obj TRIP Phase 1: Tikhonov/MAP 收缩目标 (λ=t², 类条件先验) | 从 +DPM-Solver++ 微调 50ep | bs=2, lr=1e-5',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
