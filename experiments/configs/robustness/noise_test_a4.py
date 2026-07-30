"""Robustness 实验 §4.8: 在扰动 (噪声) 标注上评估 +DPM-Solver++ (DPM-Solver++) checkpoint

源域 / 训练域: Dataset 2 (24obj, 5000 imgs, 24 classes)
测试集: Dataset 2 test (1000 imgs, 45980 anns), 但 ann_file 替换为
        experiments/runners/robustness_noise.py 生成的扰动 JSON。
图片本身不动, 仅 GT 标注被扰动 (bbox jitter + class flip)。

通过环境变量 NOISE_ANN_FILE 指定扰动 JSON 路径, 复用同一份 config 跑 9 个噪声级别。

Usage:
    # 1. 先生成扰动 JSON (3x3 = 9 个 + 1 个 clean)
    python experiments/runners/robustness_noise.py \\
        --src data/24_chromosomes_object/coco/test/_annotations.coco.json \\
        --out-dir work_dirs/robustness_noise/perturbed \\
        --grid --seed 42

    # 2. 对每个扰动 JSON 跑推理 (替换 NOISE_ANN_FILE)
    NOISE_ANN_FILE=work_dirs/robustness_noise/perturbed/clean.json \\
        python experiments/runners/test.py \\
        experiments/configs/robustness/noise_test_a4.py \\
        --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \\
        --dataset test --seed 42

checkpoint: work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth
            (+DPM-Solver++ = +Stoch. Coupling + IO3; DPM-Solver++ 4-step; mAP=0.863 on clean test)
"""
import os

_base_ = ['../ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py']

# 通过环境变量注入扰动 JSON 路径 (默认回退到 clean test set)
_noise_ann = os.environ.get(
    'NOISE_ANN_FILE',
    'data/24_chromosomes_object/coco/test/_annotations.coco.json',
)
if not os.path.exists(_noise_ann):
    raise FileNotFoundError(
        f'[noise_test_a4] NOISE_ANN_FILE 不存在: {_noise_ann}. '
        f'请先运行 experiments/runners/robustness_noise.py 生成扰动 JSON, '
        f'或设置 NOISE_ANN_FILE 指向已存在的 JSON。'
    )
# mmengine BaseDataset._join_prefix() 会把 data_root + ann_file 拼接起来,
# 除非 ann_file 是绝对路径。这里强制转绝对路径, 避免被错误拼接。
_noise_ann = os.path.abspath(_noise_ann)

# 测试图片路径保持原 test/ 目录 (只换标注)
_data_root = 'data/24_chromosomes_object/coco/'

# 显式覆盖 test_dataloader 与 test_evaluator
# (+Stoch. Coupling base config 中 test_dataloader = val_dataloader, 故需重建)
test_dataloader = dict(
    dataset=dict(
        data_root=_data_root,
        ann_file=_noise_ann,
        data_prefix=dict(img='test/'),
    ),
)
test_evaluator = dict(
    ann_file=_noise_ann,
    format_only=False,
    classwise=True,
)

# SwanLab: 独立项目, 实验名标注噪声级别
_noise_tag = os.path.splitext(os.path.basename(_noise_ann))[0]
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-robustness-noise',
            experiment_name=f'a4_noise_{_noise_tag}',
            description=f'Robustness §4.8: +DPM-Solver++ on noisy 24obj test ({_noise_tag})',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
