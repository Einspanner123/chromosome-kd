"""LDMDet + Generative Transfer Phase 1 (方向六: 生成模型感知迁移)

E6.3b frozen-enhanced: E6.3 增强 FBM 架构 + UNet 全冻结 (对照实验)

目的: 隔离 FBM 架构增强 vs UNet 部分解冻 两个变量的贡献
- E6.2: 旧 FBM (标量 alpha, 零初始化) + 全冻结  → alpha→0, mAP=0.737
- E6.3: 新 FBM (per-channel gate, gate_init=0.1) + 部分解冻  → 进行中
- E6.3b: 新 FBM (per-channel gate, gate_init=0.1) + 全冻结  → 本实验

通过 E6.3 vs E6.3b 对比, 可判断:
- 若 E6.3b mAP > E6.2: FBM 架构增强有效
- 若 E6.3 mAP > E6.3b: UNet 部分解冻有额外增益

显存预估: ~11 GB (与 E6.2 一致, UNet 全冻结用 no_grad)
适合在 16GB GPU (A4000) 上运行。
"""

_base_ = [
    '../../../projects/LDMDet/configs/recipes/rf_heun_adaln_stochot_eps5.py'
]

custom_imports = dict(
    imports=[
        'projects.LDMDet.model',
        'projects.LDMDet.hooks',
        'projects.LDMDet.custom_transforms',
        'projects.LDMDet.async_checkpoint_hook',
        'ldmdet.feature_bridge',
        'ldmdet.diagnostics',
        'swanlab.integration.mmengine',
    ],
    allow_failed_imports=False,
)

# ============================================================
# ChromoGen 权重路径
# ============================================================
chromogen_checkpoint = 'work_dirs/chromogen_phase1/final_model.pt'
chromogen_vae_path = 'work_dirs/chromogen_phase1/vae'

# ============================================================
# 模型扩展: 增强 FBM + 全冻结 ChromoGen (对照 E6.3)
# ============================================================
model = dict(
    # Feature Bridge Module (与 E6.3 相同的增强架构)
    feature_bridge=dict(
        type='FeatureBridgeModule',
        cg_channels=[320, 640, 1280, 1280],
        ld_channels=256,
        gate_init=0.1,  # 与 E6.3 一致
        num_groups=32,
    ),
    # ChromoGen UNet 特征提取器 (全冻结, 与 E6.2 一致)
    chromogen_extractor=dict(
        type='ChromoGenFeatureExtractor',
        unet_cfg=dict(
            type='ChromoGenUNet',
            sample_size=96,
            in_channels=4,
            out_channels=4,
            block_out_channels=(320, 640, 1280, 1280),
            attention_head_dim=8,
            cross_attention_dim=768,
            layers_per_block=2,
            gradient_checkpointing=True,
            freeze_last_n_blocks=0,  # E6.3b: 全冻结 (对照 E6.3 的部分解冻)
        ),
        config=dict(
            type='UNetFeatureConfig',
            extract_down1=True,
            extract_down2=True,
            extract_down3=True,
            extract_mid=True,
        ),
        checkpoint=chromogen_checkpoint,
    ),
    # ChromoGen VAE encoder (冻结)
    chromogen_vae=dict(
        type='ChromoGenVAE',
        vae_model=chromogen_vae_path,
        scaling_factor=0.18215,
        freeze=True,
        input_mean=[123.675, 116.28, 103.53],
        input_std=[58.395, 57.12, 57.375],
    ),
)

# ============================================================
# 数据加载
# ============================================================
train_dataloader = dict(batch_size=2, num_workers=4)
val_dataloader = dict(batch_size=1)

# ============================================================
# 优化器: 仅 FBM 加速 (无解冻 UNet 参数)
# ============================================================
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001, _delete_=True),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'feature_bridge': dict(lr_mult=10.0),
        },
    ),
)

# ============================================================
# 自定义 Hooks: FBM 诊断
# ============================================================
custom_hooks = [
    dict(
        type='FeatureBridgeDiagnosticsHook',
        interval=50,
        log_gate=True,
        log_norms=True,
        log_proj=True,
        log_gate_grad=True,
        log_unet_grad=False,  # 全冻结, 无 UNet 梯度可记录
    ),
]

# ============================================================
# SwanLab 实验追踪
# ============================================================
vis_backends = [
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-ablation',
            experiment_name='gen_transfer_phase1_e6_3b_frozen_enhanced',
            description=(
                '方向六 Phase 1 E6.3b: 增强 FBM (与 E6.3 相同) + '
                'UNet 全冻结 (与 E6.2 相同), 隔离 FBM 架构增强效果'
            ),
        ),
    ),
]
visualizer = dict(vis_backends=vis_backends)
