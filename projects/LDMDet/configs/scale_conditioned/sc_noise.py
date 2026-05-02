_base_ = ["../ldmdet_flowdet_adaln.py"]

# 尺度条件噪声: 小染色体 → 更小的噪声 → 更紧致的流路径
# noise_scale = (relative_size)^power
# Denver 组相对尺寸: A=1.0, B=0.75, C=0.50, D=0.30, E=0.20, F=0.12, G=0.08, Sex=0.40
model = dict(
    bbox_head=dict(
        scale_conditioned_noise=True,
        scale_noise_power=0.5,
    ),
)
