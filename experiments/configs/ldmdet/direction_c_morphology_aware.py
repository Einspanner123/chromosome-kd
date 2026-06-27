"""方向 C: 形态感知分类 (C1 ShapeAttention + C2 ContrastiveLoss)

基于 rf_heun_adaln baseline, 仅修改:
  - C1: single_head 增加 shape_attention (RoI 特征上的水平/垂直方向卷积)
    * 水平 (7,1): 捕获臂长
    * 垂直 (1,7): 捕获着丝粒位置
    * RoI 内部无重叠, 不受框重叠干扰
  - C2: ContrastiveLoss 作为辅助损失 (拉大形态相似类距离)
    * 注: C2 需修改 criterion 集成, 本配置先启用 C1

预期收益: G22/G21/Y 类 AP +0.02~0.05, mAP50 +0.01~0.02
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
            # C1: 局部形状注意力
            shape_attention=dict(
                type='ShapeAttention',
                channels=256,
                reduction=4,
            ),
        ),
    ),
)
