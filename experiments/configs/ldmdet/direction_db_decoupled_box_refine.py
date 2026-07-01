"""方向 D+B 组合: DecoupledHead (双分支) + BoxRefineNet (框细化残差)

D (BoxRefineNet): head 末端加零初始化残差, 精化高 IoU 定位
B (DecoupledHead): cls/reg 各自独立 self_attn + FFN, 解耦分类与定位

二者位于不同维度 (末端残差 vs 内部双分支), 理论可叠加 (192 extra keys)。
单独: D +0.002, B +0.004; 理论叠加 +0.006 (不确定)。

控制变量: 仅改 single_head (type + box_refine), 其余与 baseline 一致。
"""

_base_ = ['./rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        single_head=dict(
            num_classes=24,
            feat_channels=256,
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            time_conditioning='adaln_zero',
            # B: 解耦双分支 (cls/reg 各自 self_attn + FFN)
            type='PurePyTorchDecoupledSingleHead',
            # D: 框细化残差 (零初始化, 初始时不改变 baseline)
            box_refine=dict(feat_channels=256, type='BoxRefineNet'),
        ),
    ),
)
