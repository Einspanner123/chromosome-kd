_base_ = ['./ldmdet_rf_heun_shifted_bs2.py']

# Override: 原始 DiffusionDet (DDPM, 1-step), 保持 AdamW + CosineAnnealingLR 训练配方对齐
model = dict(
    bbox_head=dict(
        diffusion_type='ddpm',
        sampling_timesteps=1,
    )
)
