"""24obj AAC: Anderson-Accelerated Cascade (中等激进方向 Phase 1 微调)

目的: 将 6 级 cascade head 的顺序更新 (Picard / Anderson m=0) 替换为
      Anderson 加速 (m=2), 利用历史残差加速横向 (固定 t) 收敛。
      理论上将收敛阶从线性 O(ρ^k) 提升至超线性 O(φ^{-k}), φ≈1.618。

机制 (详见 docs/research/proposals/AAC_DESIGN.md):
  - 形式化 cascade head 为不动点迭代: x_{k+1} = G_k(x_k)
  - Anderson 加速: x_{k+1} = x_k + β [f_k - (ΔX + ΔF) γ]
    其中 f_k = G_k(x_k) - x_k (残差), γ 由跨 proposal 全局最小二乘求得
  - m=2 (有限内存), β=0.5 (阻尼, 训练初期稳定)
  - 仅对 box 做混合 (cls_logits 不构成不动点), 末级 head 不混合
  - 历史不跨 solver step 累积 (每个 forward 开始时 reset)

与已证伪方向的严格区分:
  - vs Cascade Head Count E2E (mAP 0.684): AAC 不改 H=6, 仅改 head 间混合
  - vs Adaptive Step: AAC 在横向 (固定 t), 不在纵向 (t 方向)
  - vs Head Early-Exit: AAC 不退出任何 head, 全部 6 级保留
  - vs d=4 维度退化: 全局 γ 跨 N=500 proposal, 系统矩阵 [2000, 2] 满秩

训练:
  - 从 +DPM-Solver++ best checkpoint 微调 (load_from)
  - 50 epoch, lr=1e-5 (与 Head Distillation 微调配置对齐)
  - 5 epoch linear warmup + cosine annealing
  - mmengine load_from 默认 strict=False: +DPM-Solver++ 权重加载, AAC 无新参数 (纯算法)

对照: +DPM-Solver++ (DPM-Solver++, 无 Anderson 加速) → 验证 AAC 的横向收敛加速效果

预期: mAP 0.865~0.868 (Δ=+0.002~0.005, 接近 DINO 0.868)
判据: AAC-m2 vs AAC-m0 的 ΔmAP > 0.002 (超 noise), 则 Phase 2 启动

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'aac_m2_beta05'
"""

_base_ = ['./a4_dpm_pp_24obj.py']

# === AAC: Anderson-Accelerated Cascade ===
model = dict(
    bbox_head=dict(
        use_aac=True,
        aac_mem_depth=2,           # m=2 (有限内存, AA(2) 超线性收敛)
        aac_beta=0.5,              # 阻尼 β=0.5 (训练初期稳定, Henderson-Varadhan 2019)
        aac_lambda=1e-6,           # Tikhonov 正则化 (数值稳定)
        aac_stop_grad_history=True,  # Phantom gradient: 历史不反传
        aac_gamma_norm_clip=10.0,  # γ 范数裁剪 (防止爆炸)
    ),
)

# === 从 +DPM-Solver++ checkpoint 微调 ===
# AAC 无新可学习参数 (纯算法模块), load_from 直接加载 +DPM-Solver++ 全部权重
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
            experiment_name='aac_m2_beta05',
            description='24obj AAC Phase 1: Anderson-Accelerated Cascade (m=2, β=0.5) | 从 +DPM-Solver++ 微调 50ep | bs=2, lr=1e-5',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
