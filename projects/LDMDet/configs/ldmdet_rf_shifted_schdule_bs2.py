_base_ = ["./ldmdet_baseline.py"]
train_dataloader = dict(batch_size=2)
# 覆盖模型配置以使用 Rectified Flow
model = dict(
    bbox_head=dict(
        diffusion_type="rectified_flow",  # 启用 Rectified Flow
        sampling_timesteps=4,  # RF 推理步数，通常 1-4 步效果就很好
        rf_schedule="shifted",  # 使用 shifted schedule
        rf_shift=3.0,  # 增加数据端采样密度 (参考 SD3/Flux)
        snr_scale=2.0,
    )
)

# --- 针对 Rectified Flow 的优化器和学习率调整 ---
# RF 的直线路径学习通常比弯曲的 DDPM 更稳定，可以使用略大的学习率
# 同时由于 RF 的损失函数更直接（速度预测），收敛速度往往更快

optim_wrapper = dict(
    optimizer=dict(
        type="AdamW",
        lr=0.00005,  # 从 0.000025 翻倍，RF 学习率耐受度更高
        weight_decay=0.0001,
    )
)

# 学习率调度器：延长训练周期或使用余弦退火
max_epoch = 150  # RF 通常不需要 baseline 那么长的 200 epoch，但可以更精细地训练
train_cfg = dict(max_epochs=max_epoch)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=True, begin=0, end=5),
    dict(
        type="CosineAnnealingLR",  # 使用余弦退火，更适合 RF 的平滑收敛
        T_max=max_epoch,
        eta_min=0,
        begin=5,
        end=max_epoch,
        by_epoch=True,
    ),
]
