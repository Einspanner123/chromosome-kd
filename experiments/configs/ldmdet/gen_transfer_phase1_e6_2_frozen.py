"""LDMDet + Generative Transfer Phase 1 (方向六: 生成模型感知迁移)

E6.2 frozen: FBM alpha 可学习, ChromoGen 冻结, 联合训练 150 epochs

继承 SOTA baseline (rf_heun_adaln_stochot_eps5), 集成:
- ChromoGenVAE: 冻结的 VAE 编码器 (image → latent)
- ChromoGenUNet + ChromoGenFeatureExtractor: 冻结的 UNet 特征提取
- FeatureBridgeModule: 可学习的特征桥接 (alpha 零初始化, 投影层可学习)

Phase 1 目标: 验证 ChromoGen 生成特征能否提升 LDMDet 检测性能。
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
# 模型扩展: FBM + ChromoGen 特征提取
# ============================================================
model = dict(
    # Feature Bridge Module (零初始化 alpha, 投影层可学习)
    feature_bridge=dict(
        type='FeatureBridgeModule',
        cg_channels=[320, 640, 1280, 1280],  # ChromoGen UNet down1/down2/down3/mid
        ld_channels=256,  # LDMDet FPN 输出通道
    ),
    # ChromoGen UNet 特征提取器 (冻结)
    chromogen_extractor=dict(
        type='ChromoGenFeatureExtractor',
        unet_cfg=dict(
            type='ChromoGenUNet',
            sample_size=96,  # 768/8
            in_channels=4,
            out_channels=4,
            block_out_channels=(320, 640, 1280, 1280),
            attention_head_dim=8,
            cross_attention_dim=768,
            layers_per_block=2,
            gradient_checkpointing=True,
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
    # ChromoGen VAE encoder (冻结, 用于 image → latent)
    chromogen_vae=dict(
        type='ChromoGenVAE',
        vae_model=chromogen_vae_path,
        scaling_factor=0.18215,
        freeze=True,
        # LDMDet DetDataPreprocessor 使用 ImageNet 归一化
        input_mean=[123.675, 116.28, 103.53],
        input_std=[58.395, 57.12, 57.375],
    ),
)

# ============================================================
# 数据加载: 降低 batch_size 以容纳 ChromoGen UNet 前向显存开销
# ============================================================
train_dataloader = dict(batch_size=2, num_workers=4)
val_dataloader = dict(batch_size=1)

# ============================================================
# 优化器: FBM 参数使用 10x 学习率
# ============================================================
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001, _delete_=True),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'feature_bridge': dict(lr_mult=10.0),  # FBM 加速收敛
        },
    ),
)

# ============================================================
# SwanLab 实验追踪
# ============================================================
vis_backends = [
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-ablation',
            experiment_name='gen_transfer_phase1_e6_2_frozen',
            description='方向六 Phase 1 E6.2: FBM alpha 可学习 + ChromoGen 冻结, 基于 SOTA stochot_eps5',
        ),
    ),
]
visualizer = dict(vis_backends=vis_backends)
