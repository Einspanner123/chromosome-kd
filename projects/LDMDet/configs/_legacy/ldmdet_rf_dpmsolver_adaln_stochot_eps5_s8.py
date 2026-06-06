_base_ = ['./ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py']

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

# 确保测试集指向旧数据集的 valid
test_dataloader = dict(
    dataset=dict(
        data_root='data/Chromosome20240904_NoAug_NoResize_coco/',
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/')
    )
)
test_evaluator = dict(
    ann_file='data/Chromosome20240904_NoAug_NoResize_coco/valid/_annotations.coco.json'
)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
)
