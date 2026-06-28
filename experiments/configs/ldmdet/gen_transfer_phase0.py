"""LDMDet + Generative Transfer Phase 0 (方向六: 生成模型感知迁移)

继承自 rf_heun_adaln baseline, 集成 FeatureBridgeModule (FBM) 和 ChromoGenFeatureExtractor。

注意: 此配置为代码框架, 实际运行需要:
1. ChromoGen Phase1 预训练权重 (work_dirs/chromogen_phase1/checkpoint_epoch_100.pt)
2. diffusers 库 (pip install diffusers)
3. sd-vae-ft-mse 权重缓存

Phase 0 目标: 验证 FBM 零初始化不破坏 baseline, ChromoGen 特征能正确提取和融合。

开关控制:
- feature_bridge: None=禁用, dict=启用 FBM
- chromogen_extractor: None=禁用, dict=启用特征提取
- chromogen_vae: None=禁用, dict=启用 VAE 编码

消融实验:
- baseline (无 FBM): 与 rf_heun_adaln 完全一致
- FBM 零初始化: alpha=0, 验证无回归
- FBM 微调: alpha 可学习, 联合训练
"""

_base_ = ['rf_heun_adaln.py']

custom_imports = dict(
    imports=[
        'projects.LDMDet.model',
        'projects.LDMDet.hooks',
        'swanlab.integration.mmengine',
    ],
    allow_failed_imports=False,
)

# ============================================================
# ChromoGen 权重路径 (需要预训练)
# ============================================================
chromogen_checkpoint = None  # 'work_dirs/chromogen_phase1/checkpoint_epoch_100.pt'

# ============================================================
# 模型扩展: 添加 FeatureBridgeModule 和 ChromoGenFeatureExtractor
# ============================================================
model = dict(
    # 方向六: Feature Bridge Module (零初始化, 训练初始不破坏 baseline)
    feature_bridge=dict(
        type='FeatureBridgeModule',
        cg_channels=[320, 640, 1280, 1280],  # ChromoGen UNet down1/down2/down3/mid
        ld_channels=256,  # LDMDet FPN 输出通道
    ),
    # 方向六: ChromoGen UNet 特征提取器 (冻结)
    chromogen_extractor=dict(
        type='ChromoGenFeatureExtractor',
        config=dict(
            type='UNetFeatureConfig',
            extract_down1=True,
            extract_down2=True,
            extract_down3=True,
            extract_mid=True,
        ),
    ),
    # 方向六: ChromoGen VAE encoder (冻结, 用于 image → latent)
    # chromogen_vae=dict(
    #     type='AutoencoderKL',
    #     pretrained_path='stabilityai/sd-vae-ft-mse',
    # ),
)

# ============================================================
# 训练配置
# ============================================================
max_epochs = 150

# 方向六: FBM 参数使用更高学习率 (alpha 从 0 开始, 需要快速激活)
# 其他参数继承 baseline 的 AdamW (lr=5e-5)
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    # 方向六: 自定义参数组 (FBM 投影层和 alpha 用 10x 学习率)
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
            experiment_name='gen_transfer_phase0',
            description='方向六 Phase 0: FBM 零初始化验证 + ChromoGen 特征融合',
        ),
    ),
]
visualizer = dict(vis_backends=vis_backends)
