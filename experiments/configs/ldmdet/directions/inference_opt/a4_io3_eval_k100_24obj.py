"""A4 + IO3 推理评估 — K=100 (在 A4 DPM-Solver++ checkpoint 上直接评估, 不训练)

目的: 探索 A4 (DPM-Solver++) 叠加激进 Top-K 剪枝 K=100 的边界
方法: 加载 a4_dpm_pp checkpoint, 推理时启用 topk_pruning K=100
理论: A4 每步 1 次模型调用, 叠加 K=100 后续步 Self-Attn 加速 25x
风险: A3+K100 训练时 mAP=0.823 (-0.035), 但 A4 基线更高 (0.863), 推理评估可能不同

组合卖点: 极限速度探索, 验证 DPM-Solver++ 对剪枝的鲁棒性
"""
_base_ = ['../mainline_ablation_24obj/a4_dpm_pp_24obj.py']

# 显式覆盖 test_dataloader/test_evaluator
# (base chromo_coco_detection.py 中 test_evaluator 使用旧 data_root 拼接绝对路径,
#  多层继承下 test_evaluator = val_evaluator 可能未正确覆盖 ann_file)
data_root = 'data/24_chromosomes_object/coco/'
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
    )
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox',
    classwise=True,
    _delete_=True,
)

# IO3: Top-K 剪枝 K=100 (在 A4 DPM-Solver++ 基础上, 激进探索)
model = dict(
    bbox_head=dict(
        topk_pruning_enabled=True,
        topk_k=100,
        topk_pruning_step=0,
    ),
)
