"""24obj LVD-RF: Lyapunov Velocity Direction Regularization (保守方向 Phase 1)

目的: 在训练损失中增加 Lyapunov 方向余弦正则项 (默认 sin² 形式),
      利用 d=4 低维优势以零额外前向传播计算方向余弦,
      通过 Lyapunov 稳定性条件约束速度场方向, 间接降低 η_str 并减少
      DPM-Solver++ 截断误差.

机制 (详见 docs/research/proposals/FEASIBLE_LVD_RF.md §7):
  - 在 loss() 中新增 _compute_lvd_loss(), 复用训练步已有量 (x_t, x_hat_0, x_0)
  - L_LVD = E[sin²(α)], α 为 v_θ 与 (x_t - x_0) 的夹角
  - 由定理 2.4, 1/t 在余弦中抵消, 数值稳定 (避免 t→0 发散)
  - 仅末级 cascade head 预测参与正则 (与 TFR/VCR 一致)
  - 自适应切换: cos_sim_mean > 0.99 持续 1000 iter → sin2 切换为 sqrt

与已证伪方向的严格区分 (§5):
  - vs TFR (幅度正则): LVD-RF 是方向正则 (尺度不变, 余弦)
  - vs VCR (跨时间速度一致): LVD-RF 单步, GT 锚定
  - vs scale_aware_loss: LVD-RF 不引入尺度先验, 训推一致

训练:
  - 从 +DPM-Solver++ best checkpoint 微调 (load_from)
  - 50 epoch, lr=1e-5 (与 AAC 微调配置对齐)
  - 5 epoch linear warmup + cosine annealing
  - mmengine load_from 默认 strict=False: +DPM-Solver++ 权重加载, LVD 无新参数 (纯损失项)

对照: +DPM-Solver++ (DPM-Solver++, 无方向正则) → 验证 Lyapunov 方向约束效果
预期 (R2 确认): mAP +0.002~+0.007; mAP_75 +0.001~+0.004; Y AP +0.002~+0.010
判据 (Phase 1): cos_sim_mean > 0.90 且 η_str 较 baseline 下降 > 20% 且 val/mAP ≥ 0.863

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'lvd_rf_sin2'
work_dir: work_dirs/lvd_rf_24obj/
"""

_base_ = ['./a4_dpm_pp_24obj.py']

# === LVD-RF: Lyapunov Velocity Direction Regularization ===
model = dict(
    bbox_head=dict(
        use_lvd=True,
        lvd_lambda=0.1,             # 默认正则化系数 (经验初值, 需消融)
        lvd_eps=1e-6,               # 数值稳定常数
        lvd_t_threshold=0.05,      # t < 0.05 跳过 (||x_t-x_0||→0 余弦不稳定)
        lvd_form='sin2',           # R1 K3: 默认 sin² (梯度比 1-cos 强 2×)
    ),
)

# === 从 +DPM-Solver++ checkpoint 微调 ===
# LVD-RF 无新可学习参数 (纯损失项), load_from 直接加载 +DPM-Solver++ 全部权重
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
            experiment_name='lvd_rf_sin2',
            description='24obj LVD-RF Phase 1: Lyapunov Velocity Direction Regularization (sin², λ=0.1) | 从 +DPM-Solver++ 微调 50ep | bs=2, lr=1e-5',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
