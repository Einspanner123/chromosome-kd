"""方向 D: 扩散过程改进 (D1 框细化网络)

基于 rf_heun_adaln baseline, 仅修改:
  - D1: single_head 增加 box_refine (框细化残差头)
    * 在 reg_head 输出后加 BoxRefineNet (MLP → 4 维残差偏移)
    * 零初始化最后一层, 初始时不改变 baseline
    * 训练时学习高 IoU 精度的细粒度偏移

预期收益: mAP75 +0.01~0.03, P25 IoU 从 0.86 提升到 0.89+
"""

_base_ = ['./rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        single_head=dict(
            type='PurePyTorchSingleDiffusionDetHead',
            num_classes={{_base_.model.bbox_head.single_head.num_classes}},
            feat_channels=256,
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            time_conditioning='adaln_zero',
            # D1: 框细化网络
            box_refine=dict(
                type='BoxRefineNet',
                feat_channels=256,
            ),
        ),
    ),
)
