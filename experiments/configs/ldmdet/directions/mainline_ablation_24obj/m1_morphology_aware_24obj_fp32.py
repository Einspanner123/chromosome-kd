"""24obj M1 (FP32 复现): 形态感知 RoI 编码器 — 移除 BF16 混淆 + 加速 fuse 学习

BF16 实验结论 (workstation A5000):
  - M1+BF16 = 0.818 vs A4+BF16 = 0.825 (Δ=-0.007, noise 范围负面)
  - BF16 本身掉点 -0.038 (A4 FP32 0.863 → BF16 0.825)
  - fuse 权重均匀 (h_conv/v_conv ratio=1.01, std=0.0000) — 未学到方向性

FP32 复现目的:
  1. 移除 BF16 混淆, 公平评估 M1 真实效果
  2. lr 从 1e-5 → 2e-5 (2×), warmup 3ep → 1ep, 让 fuse 更快学习
  3. 若 FP32 下 h_conv/v_conv 仍均匀, 确认是设计问题而非精度/lr 问题

对照基准: A4 (FP32, mAP=0.863)
显存: FP32 峰值 ~37.5GB (A6000 49GB 可行, A5000 24GB 不可行)

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'm1_morphology_aware_fp32'
"""

_base_ = ['./m1_morphology_aware_24obj.py']

# === FP32 复现: 加速 fuse 学习 ===
# lr 2× 提升 (1e-5 → 2e-5), warmup 缩短 (3ep → 1ep)
# 理由: BF16 实验中 fuse norm 30ep 仅增长到 0.238, h_conv/v_conv 完全未分化
# 更高 lr + 更短 warmup 让 fuse 更快获得梯度信号, 帮助方向卷积分化
max_epoch = 30
train_cfg = dict(max_epochs=max_epoch)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.00002, weight_decay=0.0001, _delete_=True
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=1),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        eta_min=0,
        begin=1,
        end=max_epoch,
        by_epoch=True,
    ),
]

# === SwanLab: 区分 FP32 复现实验 ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-24obj',
            experiment_name='m1_morphology_aware_fp32',
            description='24obj M1 (FP32 复现): 形态感知 RoI 编码器 | A6000 FP32, lr=2e-5, 1ep warmup, 30ep | 移除 BF16 混淆',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
