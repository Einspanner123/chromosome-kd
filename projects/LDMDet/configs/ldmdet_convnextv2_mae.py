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
    # 极低 batch_size 以防止在共享 GPU 时 OOM
    train_dataloader = dict(batch_size=1, num_workers=2)
    # 通常 ConvNeXt 适合较小的学习率和更强的权重衰减
    optim_wrapper = dict(optimizer=dict(lr=5e-5, weight_decay=0.05))
