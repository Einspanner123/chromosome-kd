"""方向4: Velocity-Guided Adaptive Renewal (VGAR)

box_renewal × RF 速度场耦合:
- renewal 不再用纯随机噪声, 而是保留部分 v_θ 预测的 x0 方向
- alpha 随时间步自适应: 早期更随机 (探索), 后期更确定 (利用)
- 理论: v_θ 的 x0_pred 编码了"框应该去哪里", 纯随机 renewal 浪费了这个信息

数学公式:
    alpha(t) = 0.2 + 0.6 * sigmoid(5 * (0.5 - t))
    x_renewed = alpha(t) * x0_pred + (1 - alpha(t)) * randn

基线: A4 DPM-Solver++ (mAP=0.863)

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'a4_vgar'
"""

_base_ = ['./a4_dpm_pp_24obj.py']

# === 启用 VGAR ===
model = dict(
    bbox_head=dict(
        velocity_guided_renewal=True,  # 方向4: 速度场引导的 box renewal
    ),
)

# === SwanLab ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-24obj',
            experiment_name='a4_vgar',
            description='24obj 方向4: VGAR 速度场引导 renewal (vs A4 纯随机) | bs=8, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
