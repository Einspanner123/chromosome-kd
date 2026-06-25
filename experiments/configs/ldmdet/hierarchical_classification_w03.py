"""LDMDet + Hierarchical Classification E5.1-tune: loss_hier=0.3

调参实验: 降低分层分类辅助损失权重 (1.0→0.3), 减少对主分类头的干扰.
E5.1 原始实验中 loss_hier=1.0 导致 AP75/AP50 略低于 baseline, 末尾过拟合.

对比:
- E5.1 (hierarchical_classification): loss_hier_weight=1.0 → mAP=0.745, 过拟合
- E5.1-tune (本配置): loss_hier_weight=0.3 → 预期 AP75 回升, mAP 提升
"""

_base_ = ['hierarchical_classification.py']

model = dict(
    bbox_head=dict(
        loss_hier_weight=0.3,
    ),
)
