_base_ = ["./ldmdet_baseline.py"]

# 进阶实验：Rectified Flow + Heun Solver (二阶采样) + Logit Coupling + Shifted Schedule
model = dict(
    bbox_head=dict(
        diffusion_type="rectified_flow",
        solver_type="heun",  # 使用二阶 Heun 求解器
        sampling_timesteps=4,  # 采样步数 (实际推理次数 = sampling_timesteps * 2)
        rf_schedule="shifted",  # 启用非线性 Shifted Schedule
        rf_shift=3.0,  # 数据端采样密度增强
        snr_scale=2.0,
    )
)

# 优化器配置
optim_wrapper = dict(optimizer=dict(type="AdamW", lr=0.00005, weight_decay=0.0001))

# 训练配置
max_epoch = 150
train_cfg = dict(max_epochs=max_epoch)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=True, begin=0, end=5),
    dict(
        type="CosineAnnealingLR",
        T_max=max_epoch,
        eta_min=0,
        begin=5,
        end=max_epoch,
        by_epoch=True,
    ),
]
