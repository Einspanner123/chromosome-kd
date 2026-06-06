_base_ = ['./ldmdet_rf_heun_adaln_stochot_eps5.py']

custom_imports = dict(
    imports=[
        'projects.LDMDet.model',
        'projects.LDMDet.hooks',
        'swanlab.integration.mmengine',
    ],
    allow_failed_imports=False,
)

model = dict(
    bbox_head=dict(
        solver_type='dpm_solver_pp',
        sampling_timesteps=8,
    )
)

# 暴力覆盖所有旧数据集的残留路径
test_dataloader = dict(
    dataset=dict(
        data_root='data/24_chromosomes_object/coco/',
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/')
    )
)
test_evaluator = dict(
    ann_file='data/24_chromosomes_object/coco/valid/_annotations.coco.json'
)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-benchmark-24obj',
                experiment_name='ldmdet-rf-adaln-stochot-eps5-dpmsolver-s8',
                description='Benchmark 24obj: DPM-Solver++ 8-step Inference',
            ),
        ),
    ],
)
