_base_ = ['./ldmdet_rf_heun_shifted.py']

# 分布式训练配置 (单机双卡 2-GPU DDP)
# 遵循 Linear Scaling Rule: 当总 Batch Size 翻倍时，学习率也应相应调整

# 1. 自动缩放学习率 (MMEngine 提供自动缩放功能)
# 如果总 batch_size 从 4 变为 8，LR 会自动从 0.00005 变为 0.0001
optim_wrapper = dict(optimizer=dict(lr=0.00005  # 基础学习率，开启自动缩放后会根据卡数调整
                                    ))
auto_scale_lr = dict(enable=True, base_batch_size=4)

# 2. 数据加载配置
# 在 DDP 模式下，每张卡会分担 batch_size
# 总 batch_size = batch_size_per_gpu * num_gpus
train_dataloader = dict(
    batch_size=4,  # 每张显卡的 batch_size，保持与单卡一致以维持显存占用稳定
    num_workers=4,
)

# 3. 运行环境配置
# 启用分布式训练相关的 Hook
default_hooks = dict(
    # 确保每个进程的随机种子不同，避免过拟合
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),  # 分布式采样种子 Hook
)

# 4. 显存优化 (可选)
# 如果双卡显存压力大，可以开启 FP16 混合精度训练
# optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
