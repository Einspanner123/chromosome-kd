_base_ = ['./ldmdet_baseline.py']

# 覆盖模型配置以使用 Rectified Flow
model = dict(
    bbox_head=dict(
        diffusion_type='rectified_flow',  # 启用 Rectified Flow
        sampling_timesteps=4,  # RF 推理步数，通常 1-4 步效果就很好
        snr_scale=2.0,
    )
)

# 可以在此处针对 RF 调整优化策略
# 例如，由于 RF 推理更快，可以适当增加训练时的迭代次数或调整学习率
# 但为了对比公平，这里暂时保持 baseline 的 schedule
