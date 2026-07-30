"""方向2: Per-Head Time Reparameterization (PHTR)

时间条件×级联头层级耦合:
- 每个级联头有可学习的 time_scale 和 time_shift
- 打破 6 头共享 time_emb 的限制, 使不同头关注不同时间区间
- 初始化为 identity (scale=1, shift=0), 不破坏预训练兼容性

根因: AdaLN-Zero 时间条件注入对 6 级联头零贡献 (RF+Heun 无 AdaLN = +AdaLN-Zero 有 AdaLN)。
原因之一是所有头共享同一个 time_emb, 缺乏头层级特化。PHTR 通过仿射变换
使每个头获得特有的时间编码: time_emb_i = time_emb * scale_i + shift_i。

基线: +DPM-Solver++ (mAP=0.863)
对照: +DPM-Solver++ (无 PHTR) → 验证 PHTR 对时间条件耦合的增益

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'a4_phtr'
"""

_base_ = ['./a4_dpm_pp_24obj.py']

# === 方向2: 启用 PHTR (Per-Head Time Reparameterization) ===
model = dict(
    bbox_head=dict(
        use_time_reparam=True,  # 启用 PHTR: 每头独立 time_scale/time_shift
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
            experiment_name='a4_phtr',
            description='24obj 方向2 PHTR: Per-Head Time Reparam | bs=8, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
