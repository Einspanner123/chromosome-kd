_base_ = ['./ldmdet_baseline.py']

# 覆盖模型配置以使用 Rectified Flow
model = dict(
    bbox_head=dict(
        diffusion_type='rectified_flow',  # 启用 Rectified Flow
        sampling_timesteps=4,  # RF 推理步数，通常 1-4 步效果就很好
        rf_schedule='shifted',  # 使用 shifted schedule
        rf_shift=3.0,  # 增加数据端采样密度 (参考 SD3/Flux)
        snr_scale=2.0,
    ))

# --- 针对 Rectified Flow 的优化器和学习率调整 ---
# RF 的直线路径学习通常比弯曲的 DDPM 更稳定，可以使用略大的学习率
# 同时由于 RF 的损失函数更直接（速度预测），收敛速度往往更快
