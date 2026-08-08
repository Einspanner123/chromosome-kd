"""BoxChart-RF Phase 1 on the 24-class chromosome dataset.

This first ablation changes only the RF state parameterization.  The detector
head and physical-space detection losses remain identical to a4.
"""

_base_ = ['../mainline_ablation_24obj/a4_dpm_pp_24obj.py']


BOX_CHART_MEAN = [
    1.0017207107598445,
    -0.5855985603122745,
    0.9710205793066153,
    -0.5488219027034528,
]

BOX_CHART_COVARIANCE = [
    [0.41058209142280205, 0.285167011455603, 0.03741102578524599, -0.019413705541656834],
    [0.285167011455603, 0.7838589280490502, -0.022180730155047335, 0.012369192545681603],
    [0.03741102578524599, -0.022180730155047335, 0.4519470489895308, 0.3310360789559165],
    [-0.019413705541656834, 0.012369192545681603, 0.3310360789559165, 0.8343814636002196],
]

model = dict(
    bbox_head=dict(
        box_parameterization='gap_ilr',
        box_chart_eps=1e-6,
        box_chart_mean=BOX_CHART_MEAN,
        box_chart_covariance=BOX_CHART_COVARIANCE,
    )
)

# The backbone and existing detection heads are useful initializers.  The new
# chart buffers are absent in a4 and are initialized from the constants above.
load_from = 'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth'
work_dir = 'work_dirs/boxchart_rf_24obj'

# Short mechanism-validation run.  A full schedule is justified only if the
# AP75/AP_S and low-step gates pass.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=1)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, begin=0, end=1, by_epoch=True),
    dict(
        type='CosineAnnealingLR', T_max=29, begin=1, end=30,
        eta_min=1e-7, by_epoch=True),
]

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
