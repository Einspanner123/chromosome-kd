_base_ = ['./ldmdet_baseline.py']

# 基础训练设置
train_dataloader = dict(
    batch_size=2,
    persistent_workers=True,  # 已经在 baseline 中设置，这里显式确认
)

# 模型配置：Rectified Flow + Heun Solver + Shifted Schedule
model = dict(
    bbox_head=dict(
        diffusion_type='rectified_flow',
        solver_type='heun',  # 使用二阶 Heun 求解器提高采样精度
        sampling_timesteps=4,  # 采样步数
        rf_schedule='shifted',  # 启用非线性 Shifted Schedule
        rf_shift=3.0,  # 针对染色体目标的采样密度增强
        snr_scale=2.0,
    )
)

# 1. 效率优化：启用 AMP (自动混合精度)
# 这将大幅提升训练速度并减少显存占用，从而抵消确定性计算带来的开销
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00005, weight_decay=0.0001),
)

# 2. 稳定性优化：引入 EMA (指数移动平均)
# 通过平滑模型权重，缓解小 batch 和随机采样带来的性能波动
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_interval=1,
        priority=49,
    )
]

# 3. 复现性优化：固定天选随机种子
# deterministic=False 保证了在享受 CuDNN 加速的同时，数据流和加噪序列是固定的
randomness = dict(seed=1769925607, deterministic=False)

# 训练策略
max_epoch = 150
train_cfg = dict(max_epochs=max_epoch)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=5),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        eta_min=0,
        begin=5,
        end=max_epoch,
        by_epoch=True,
    ),
]
