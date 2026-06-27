"""方向 B: 解耦分类/定位双分支头 (B1 双分支 + B2 独立 LayerNorm)

基于 rf_heun_adaln baseline, 仅修改:
  - single_head: SingleDiffusionDetHead → DecoupledSingleHead
    * cls 分支: 独立的 self_attn + FFN + LayerNorms
    * reg 分支: 独立的 self_attn + FFN + LayerNorms
    * 共享: inst_interact, time_mlp/adaln_mlp, cls_head/reg_head

预期收益: 解开分类与定位的特征耦合, mAP +0.01~0.02
"""

_base_ = ['./rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        # B1+B2: 解耦双分支头
        single_head=dict(
            _delete_=True,
            type='PurePyTorchDecoupledSingleHead',
            num_classes={{_base_.model.bbox_head.single_head.num_classes}},
            feat_channels=256,
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            time_conditioning='adaln_zero',
        ),
    ),
)
