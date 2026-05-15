_base_ = ['./ldmdet_baseline.py']

# 启用 Rectified Flow 且训练/推理同步使用 Shifted Schedule
model = dict(
    bbox_head=dict(
        diffusion_type='rectified_flow',  # 启用 Rectified Flow
        sampling_timesteps=4,  # 推理步数
        rf_schedule='shifted',  # 启用 Shifted Schedule
        rf_shift=3.0,  # 增加数据端采样密度 (参考 SD3/Flux)
        snr_scale=2.0,
    ))

# --- 针对 Rectified Flow 的优化器和学习率调整 ---
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.00005,  # RF 通常可以使用略大的学习率
        weight_decay=0.0001,
    ))

# 训练周期配置
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
