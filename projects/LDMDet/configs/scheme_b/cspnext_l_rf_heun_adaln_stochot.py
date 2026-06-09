"""方案二: RTMDet CSPNeXt-L Backbone + CSPNeXtPAFPN + 扩散检测头

架构: CSPNeXt-L → CSPNeXtPAFPN(out=256, stride 8/16/32) → DiffusionDetHead
核心创新: RF + AdaLN-Zero + Sinkhorn Stochastic OT (保持不变)
Backbone: CSPNeXt-L (大核5×5 DWConv, ImageNet预训练, 52M params)
Neck: CSPNeXtPAFPN (与backbone同构CSP块, 3层输出)
Head: DiffusionDetHead (与SOTA完全一致)

关键适配:
  - PAFPN输出3层 [256,256,256] stride [8,16,32], 无stride=4层
  - RoI Extractor featmap_strides=[8,16,32] (3层)
  - CSPNeXt使用ImageNet预训练权重 (mmdet提供)

对比基线:
  - 原SOTA: ResNet-50 → FPN(4层) → DiffusionDetHead
  - 本方案: CSPNeXt-L → PAFPN(3层) → DiffusionDetHead
  - 验证: 更强backbone+neck是否提升扩散检测性能
"""

_base_ = ['../ldmdet_baseline.py']

custom_imports = dict(
    imports=[
        'projects.LDMDet.model',
        'projects.LDMDet.hooks',
        'swanlab.integration.mmengine',
    ],
    allow_failed_imports=False,
)

# ============================================================
# Model: 替换 backbone 和 neck, 保留 SOTA 扩散头
# ============================================================
model = dict(
    # 替换 backbone: ResNet-50 → CSPNeXt-L
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1,
        widen_factor=1,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth',
        ),
        _delete_=True,
    ),
    # 替换 neck: FPN → CSPNeXtPAFPN
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True),
        _delete_=True,
    ),
    # 扩散头: 与 SOTA 完全一致, 仅适配 RoI strides
    bbox_head=dict(
        diffusion_type='rectified_flow',
        solver_type='heun',
        sampling_timesteps=4,
        rf_schedule='shifted',
        rf_shift=3.0,
        snr_scale=2.0,
        use_flash_attn=True,
        # OT Coupling (SOTA)
        ot_coupling=True,
        ot_matcher='sinkhorn',
        ot_epsilon=5.0,
        ot_num_iters=20,
        ot_sample=True,
        # AdaLN-Zero (SOTA)
        single_head=dict(
            time_conditioning='adaln_zero',
        ),
        # RoI Extractor: 适配 PAFPN 3层输出 (stride 8/16/32)
        roi_extractor=dict(
            type='PurePyTorchSingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[8, 16, 32],
        ),
    ),
)

# ============================================================
# Training
# ============================================================
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    prefetch_factor=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False, min_size=1e-5),
    ),
)

# 优化器
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.0001, weight_decay=0.0001, _delete_=True
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

max_epoch = 150
train_cfg = dict(max_epochs=max_epoch)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=5),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        eta_min=1e-6,
        begin=5,
        end=max_epoch,
        by_epoch=True,
    ),
]

# ============================================================
# Visualization
# ============================================================
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-scheme-b',
                experiment_name='cspnext-l-rf-heun-adaln-stochot',
                description='Scheme B: CSPNeXt-L + PAFPN + DiffusionDetHead (RF+Heun+AdaLN+StochOT)',
            ),
        ),
    ],
)
