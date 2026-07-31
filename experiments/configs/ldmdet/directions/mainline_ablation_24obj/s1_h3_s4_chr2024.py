"""Dataset 1 (Chromosome20240904) S1 cascade head × solver step 解耦: H=3 S=4 (12 NFE)

目的: 在 Dataset 1 上补全 §七 Cascade × Solver 解耦 h3_s4, 闭合双数据集缺口
设置: H=3, S=4 → 12 NFE (DPM-Solver++ 1 NFE/step)
对照:
  - Dataset 1 +DPM-Solver++ baseline (a4_dpm_pp_chr2024, H=6 S=4, 24 NFE, 3-seed 0.747±0.001)
  - Dataset 2 s1_h3_s4_24obj (H=3 S=4, best 0.860@ep59) → 双数据集对照
预期: 若 H×S 可交换性 (命题 S1.3) 跨数据集成立, Dataset 1 h3_s4 mAP 应与 baseline 在 noise 范围内

理论依据: theory_analysis_RF_DPM.md §2 (S1)
基线: a4_dpm_pp_chr2024.py (Dataset 1 +DPM-Solver++)
变化: 仅 num_heads 6 → 3 (减半)

启动:
  python experiments/runners/train.py experiments/configs/ldmdet/directions/mainline_ablation_24obj/s1_h3_s4_chr2024.py --seed 42 --gpu-id 1
"""
_base_ = ['../../a4_dpm_pp_chr2024.py']

# === H=3 S=4 (减半 cascade head, 保持 solver step) ===
model = dict(
    bbox_head=dict(
        num_heads=3,  # 6 → 3 (减半)
        sampling_timesteps=4,  # 保持 S=4
    ),
)

# === SwanLab (与 24obj 版本同项目, 便于双数据集对照) ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-s1-cascade-decouple',
            experiment_name='s1_h3_s4_chr2024',
            description='S1 Dataset 1: H=3 S=4 (12 NFE) | 验证 H×S 可交换性跨数据集 | chr2024, bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
