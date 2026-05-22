# projects/LDMDet/configs/ldmdet_convnextv2_mae.py

_base_ = ['./recipes/rf_heun_adaln_stochot_eps5.py']

# ==============================================================================
# 开关配置 (Switch Configuration)
# ==============================================================================
# 设置为 'local' 使用本地 mods/convnextv2.py
# 设置为 'timm' 使用 timm 库 (推荐)
# 设置为 False 则维持原有的 ResNet-50 骨干网络
BACKBONE_TYPE = 'timm'
# ==============================================================================

if BACKBONE_TYPE == 'local':
    model = dict(
        backbone=dict(
            _delete_=True,
            type='ConvNeXtV2',  # model.py 现在支持直接识别这个 type
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
            drop_path_rate=0.1,
            init_cfg=dict(
                type='Pretrained',
                # 使用一个更稳定的链接或本地路径，防止 403 Forbidden
                checkpoint='https://download.pytorch.org/models/convnext_tiny-983f1562.pth',
                prefix='backbone.',
            ),
        ),
        neck=dict(
            _delete_=True,
            type='FPN',
            in_channels=[96, 192, 384, 768],
            out_channels=256,
            num_outs=4,
        ),
    )
elif BACKBONE_TYPE == 'timm':
    model = dict(
        backbone=dict(
            _delete_=True,
            type='timm',
            model_name='convnextv2_tiny.fcmae_ft_in1k',  # timm 中的对应模型名
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        ),
        neck=dict(
            _delete_=True,
            type='FPN',
            in_channels=[96, 192, 384, 768],
            out_channels=256,
            num_outs=4,
        ),
    )

# 针对 ConvNeXt 优化训练参数（可选建议）
if BACKBONE_TYPE:
    # 针对 24GB 显存服务器优化 (Batch Size = 8)
    train_dataloader = dict(batch_size=8, num_workers=4)
    # 随 Batch Size 扩大等比例调整学习率 (1 -> 8: 5e-5 -> 4e-4, 这里取 2e-4 比较稳健)
    optim_wrapper = dict(optimizer=dict(lr=2e-4, weight_decay=0.05))

# ==============================================================================
# 性能优化 (Performance Optimizations)
# ==============================================================================
model = dict(
    bbox_head=dict(
        # 优化1: 开启 FlashAttention 加速 Transformer 计算
        use_flash_attn=True,
    )
)

# 优化2: 开启 Torch.compile 编译加速 (要求 PyTorch 2.0+)
# 注意：在某些复杂的 MMEngine 数据预处理场景下可能会有兼容性问题，若报错请设为 False
compile = False

# ==============================================================================
# SwanLab 实验配置 (SwanLab Experiment Configuration)
# ==============================================================================
# 在此处指定具体的实验名称
experiment_name = f'ldmdet_{BACKBONE_TYPE}_tiny_mae_bs8_lr2e-4_shifted3_adaln'

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd',
                experiment_name=experiment_name,
                api_key='Huzvq1fnDeqOwgQo2AMAI',
            ),
        ),
    ]
)
# ==============================================================================
