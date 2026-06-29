"""LDMDet + Generative Transfer Phase 1 (方向六: 生成模型感知迁移)

E6.3 enhanced: 针对 E6.2 alpha→0 失效问题的增强方案

E6.2 问题诊断:
- alpha 学到 ~1e-3 (接近 0), FBM 实际失效
- best mAP=0.737 < baseline 0.753
- 原因: 标量 alpha + 零初始化 + 全冻结 → 模型忽略 ChromoGen 特征

E6.3 改进:
1. FBM 架构增强:
   - per-channel sigmoid gate (替代标量 alpha)
   - gate_init=0.1 (替代零初始化, 提供非零梯度信号)
   - 投影层加 GroupNorm + GELU (增强特征对齐)
   - residual gating: fused = ld*(1-g) + g*cg_proj
2. ChromoGen UNet 部分解冻:
   - freeze_last_n_blocks=1 (解冻 down_blocks[-1] + mid_block)
   - 让特征适应检测任务
3. 诊断插桩:
   - FeatureBridgeDiagnosticsHook 监控 gate/范数/梯度
   - 便于定位哪个模块在贡献/拖累

继承 SOTA baseline (rf_heun_adaln_stochot_eps5)。
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
        'ldmdet.diagnostics',  # 注册 FeatureBridgeDiagnosticsHook
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
# 模型扩展: 增强 FBM + 部分解冻 ChromoGen
# ============================================================
model = dict(
    # Feature Bridge Module (E6.3 增强: per-channel gate + GroupNorm + GELU)
    feature_bridge=dict(
        type='FeatureBridgeModule',
        cg_channels=[320, 640, 1280, 1280],  # ChromoGen UNet down1/down2/down3/mid
        ld_channels=256,  # LDMDet FPN 输出通道
        gate_init=0.1,  # 初始 10% 融合权重 (非零, 提供梯度信号)
        num_groups=32,  # GroupNorm 分组数
    ),
    # ChromoGen UNet 特征提取器 (部分解冻)
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
            gradient_checkpointing=True,  # 节省显存
            freeze_last_n_blocks=1,  # E6.3: 解冻 down_blocks[-1] + mid_block
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
# 数据加载: 降低 batch_size 以容纳 ChromoGen UNet 前向 + 反向显存
# ============================================================
train_dataloader = dict(batch_size=2, num_workers=4)
val_dataloader = dict(batch_size=1)

# ============================================================
# 优化器: 分组学习率
# - FBM (gate + proj): 10x lr (加速收敛)
# - 解冻的 UNet block: 0.1x lr (预训练参数, 小 lr 微调)
# - 其他: baseline lr
# ============================================================
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001, _delete_=True),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            # FBM 参数加速收敛
            'feature_bridge': dict(lr_mult=10.0),
            # 解冻的 UNet down_blocks[3] (DownBlock2D, 1280ch)
            'chromogen_extractor.unet.unet.down_blocks.3': dict(lr_mult=0.1),
            # 解冻的 UNet mid_block
            'chromogen_extractor.unet.unet.mid_block': dict(lr_mult=0.1),
        },
    ),
)

# ============================================================
# 自定义 Hooks: FBM 诊断
# ============================================================
custom_hooks = [
    dict(
        type='FeatureBridgeDiagnosticsHook',
        interval=50,  # 每 50 iter 记录一次
        log_gate=True,
        log_norms=True,
        log_proj=True,
        log_gate_grad=True,
        log_unet_grad=True,
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
            experiment_name='gen_transfer_phase1_e6_3_enhanced',
            description=(
                '方向六 Phase 1 E6.3: 增强 FBM (per-channel gate + GroupNorm) + '
                'UNet 部分解冻 (last block + mid), 基于 SOTA stochot_eps5'
            ),
        ),
    ),
]
visualizer = dict(vis_backends=vis_backends)
