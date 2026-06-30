"""LDMDet + Generative Transfer Phase 1 (方向六: 生成模型感知迁移)

E6.4 cross-attn: Cross-Attention 融合方案

E6.3 问题:
- Gate 保持在初始值 0.1 附近, 模型没有主动调整 CG 贡献度
- Simple residual gating 下模型要么全盘接受要么全盘拒绝 CG 特征
- mAP=0.703 < E6.2=0.737 < baseline=0.753

E6.4 改进: Cross-Attention 融合
1. Level 1/2 (H/8, H/16): 简单 zero-init 标量 gate (无 sigmoid, 梯度不消失)
2. Level 3 (H/32): Cross-Attention (LD=Q, CG down3+mid=K/V)
   - 模型在空间位置级别选择性查询 CG 信息
   - Gamma zero-init (类似 AdaLN-Zero), 保证 baseline 不被破坏
   - K/V 源拼接 down3+mid, 提供更丰富上下文
3. ChromoGen UNet 部分解冻 (freeze_last_n_blocks=1, 与 E6.3 一致)

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
# 模型扩展: Cross-Attention FBM + 部分解冻 ChromoGen
# ============================================================
model = dict(
    # Cross-Attention Feature Bridge Module
    feature_bridge=dict(
        type='CrossAttnFeatureBridgeModule',
        cg_channels=[320, 640, 1280, 1280],
        ld_channels=256,
        num_heads=4,  # 256/4=64 dim per head
        num_groups=32,
    ),
    # ChromoGen UNet 特征提取器 (部分解冻, 与 E6.3 一致)
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
            freeze_last_n_blocks=1,  # 解冻 down_blocks[-1] + mid_block
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
# 优化器: 分组学习率
# ============================================================
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001, _delete_=True),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            # FBM 参数 (cross-attn + gate) 加速收敛
            'feature_bridge': dict(lr_mult=10.0),
            # 解冻的 UNet block 小 lr 微调
            'chromogen_extractor.unet.unet.down_blocks.3': dict(lr_mult=0.1),
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
        interval=50,
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
            experiment_name='gen_transfer_phase1_e6_4_crossattn',
            description=(
                '方向六 Phase 1 E6.4: Cross-Attention FBM (Level 3 LD=Q, CG=K/V) + '
                'UNet 部分解冻, zero-init gamma 保证 baseline 不被破坏'
            ),
        ),
    ),
]
visualizer = dict(vis_backends=vis_backends)
