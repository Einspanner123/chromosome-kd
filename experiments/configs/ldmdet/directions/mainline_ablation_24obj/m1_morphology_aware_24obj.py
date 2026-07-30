"""24obj M1: 形态感知 RoI 编码器 (Morphology-Aware RoI Encoder)

目的: 用方向解耦卷积增强 7×7 RoI 的染色体形态学特征, 提升细粒度类别判别
组件:
  + shape_attention = MorphologyAwareRoIEncoder (填充现有 hook)
    - h_conv (7,1): 沿臂长方向扫描, 捕获臂长比 (p/q)
    - v_conv (1,7): 沿着丝粒方向扫描, 捕获着丝粒位置
    - fuse 零初始化: 初始 morph_emb≡0, 加载 +DPM-Solver++ 预训练时行为不变
  - 其余与 +DPM-Solver++ 完全一致 (DPM-Solver++ 4步, RF, AdaLN-Zero, StochOT eps5)

D1 消融验证 (2026-07-22):
  抹平 7×7 空间信息 → mAP 0.863→0.009 (Δ=-0.854 灾难性崩溃)
  → 现有 DynamicConv 已有效提取空间信息, M1 应增强而非重建
  → 零初始化残差设计 (roi_features + morph_emb) 不破坏已验证的空间通路

训练:
  - 从 +DPM-Solver++ best checkpoint 微调 (load_from)
  - 30 epoch, lr=1e-5 (低于 from-scratch 的 5e-5)
  - 3 epoch linear warmup + cosine annealing
  - mmengine load_from 默认 strict=False, M1 新参数 (h_conv/v_conv/fuse/norm)
    不在 +DPM-Solver++ checkpoint 中, 保持初始化值 (fuse=0 → 恒等)

重点观察:
  - C 组 (C6-C12, 亚中着丝粒, 臂长比是主要区分依据)
  - G/Y 组 (尺寸相近, mAP 0.789/0.781, 需形态区分)
  - 整体 mAP 是否超越 +DPM-Solver++ baseline (0.863)

对照: +DPM-Solver++ (DPM-Solver++, 无形态感知) → 验证 M1 形态增强的增益

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'm1_morphology_aware'
"""

_base_ = ['./a4_dpm_pp_24obj.py']

# === M1: 形态感知 RoI 编码器 ===
model = dict(
    bbox_head=dict(
        single_head=dict(
            shape_attention=dict(
                type='PurePyTorchMorphologyAwareRoIEncoder',
                channels=256,
                reduction=4,
                pooler_resolution=7,
            ),
        ),
    ),
)

# === 从 +DPM-Solver++ checkpoint 微调 ===
# mmengine load_from 默认 strict=False: +DPM-Solver++ 权重加载, M1 新参数保持初始化 (fuse=0)
load_from = 'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth'

# === 微调训练计划 (30 epoch, 低 lr) ===
max_epoch = 30
train_cfg = dict(max_epochs=max_epoch)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.00001, weight_decay=0.0001, _delete_=True
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=3),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        eta_min=0,
        begin=3,
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
            experiment_name='m1_morphology_aware',
            description='24obj M1: 形态感知 RoI 编码器 (零初始化残差, 从 +DPM-Solver++ 微调 30ep) | bs=2, lr=1e-5',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
