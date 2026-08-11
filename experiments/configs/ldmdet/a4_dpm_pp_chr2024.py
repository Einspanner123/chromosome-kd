"""Dataset 1 (Chromosome20240904) 主路线消融 +DPM-Solver++

目的: 在 Dataset 1 上补全主消融链第 4 环, 将 Heun 替换为 DPM-Solver++
组件:
  + solver_type='dpm_solver_pp' (DPM-Solver++ 高阶 ODE 求解器)
  + sampling_timesteps=4 (与 Stoch baseline Heun 4步一致, 公平对比采样器)
  - RF + AdaLN + StochOT eps5 保持不变 (继承自 ldmdet_rf_heun_adaln_stochot_eps5)

对照: Dataset 1 RF+Heun+AdaLN+Stoch (3-seed 0.747) → 验证 DPM-Solver++ vs Heun
关联: Dataset 2 +DPM-Solver++ (a4_dpm_pp_24obj.py, 3-seed 0.859, seed42 best 0.863)

SwanLab: 项目 'ldmdet-ablation' (default_runtime 默认), 实验 'a4_dpm_pp_chr2024_seed{seed}'
         (train.py 自动生成 experiment_name = config_name + '_seed' + seed)

3-seed 启动:
  python experiments/runners/train.py experiments/configs/ldmdet/a4_dpm_pp_chr2024.py --seed 42  --gpu-id 0
  python experiments/runners/train.py experiments/configs/ldmdet/a4_dpm_pp_chr2024.py --seed 123 --gpu-id 0
  python experiments/runners/train.py experiments/configs/ldmdet/a4_dpm_pp_chr2024.py --seed 789 --gpu-id 0
"""
_base_ = ['./ldmdet_rf_heun_adaln_stochot_eps5.py']

# === 替换采样器为 DPM-Solver++ ===
model = dict(
    bbox_head=dict(
        solver_type='dpm_solver_pp',
        sampling_timesteps=4,
    ),
)

# === Val 评估器: 启用 classwise 输出 24 个 per-class AP (项目硬约束) ===
data_root = 'data/Chromosome20240904_NoAug_NoResize_coco/'
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox',
    classwise=True,
    format_only=False,
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/_annotations.coco.json',
    metric='bbox',
    classwise=True,
    format_only=False,
)
