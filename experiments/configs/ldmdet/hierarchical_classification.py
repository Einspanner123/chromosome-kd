"""LDMDet + Hierarchical Classification (方向五: 组条件化分层分类)

继承自 rf_heun_adaln, 启用分层分类辅助头.
与 rf_heun_adaln.py (baseline) 差异: hierarchical_head.

实验目标: 验证分层分类对染色体组内细粒度区分的提升
- 假设: 先分组 (A-G + Sex) 再组内分类, 降低类别混淆
- 假设: 组条件化使同组框的生成更聚焦 (同组框形态相似)
- 对比: rf_heun_adaln.py (baseline, 扁平分类)

开关控制:
- hierarchical_head: None=不启用, dict=启用分层分类辅助头
  作为辅助损失 (loss_hier) 加入训练, 不改变主分类头

消融实验:
- 仅分层分类: hierarchical_head 启用 (本配置默认)
- 组条件化 SingleHead: 需替换 single_head 为 GroupConditionedSingleHead (实验性)

注意:
  方向五的 GroupConditionedSingleHead 需修改 head.forward 传递 group_labels,
  当前作为实验性选项, 默认不启用. 主流程通过 hierarchical_head 辅助损失验证方向有效性.
"""

_base_ = ['rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        # 方向五: 分层分类辅助头 (先分组再组内分类)
        hierarchical_head=dict(
            type='HierarchicalClsHead',
            feat_channels=256,
            num_classes=24,
            num_groups=8,           # A, B, C, D, E, F, G, Sex
            num_cls_convs=1,
        ),
    ),
)

# 方向五诊断: 覆盖 custom_hooks, 追加诊断 Hook
custom_hooks = [
    # baseline hooks
    dict(type='EarlyStoppingHook', priority=50, patience=30, min_delta=0.001, monitor='coco/bbox_mAP', rule='greater'),
    dict(type='CopyProjectHook', priority='VERY_LOW'),
    # 诊断 hooks
    dict(
        type='TrainingDiagnosticsHook',
        priority='LOW',
        weight_grad_interval=100,
        activation_interval=500,
        log_weights=True,
        log_grads=True,
        log_activations=True,
        log_numerical_health=True,
        log_loss_breakdown=True,
        module_prefixes=dict(
            backbone='backbone',
            neck='neck',
            bbox_head_head_series='head',
            bbox_head_time_mlp='time_mlp',
            bbox_head_criterion='criterion',
        ),
        activation_layers=[],
        diagnostics_callback=None,
    ),
    dict(type='HierarchicalDiagInjector', priority='NORMAL', interval=100),
]
