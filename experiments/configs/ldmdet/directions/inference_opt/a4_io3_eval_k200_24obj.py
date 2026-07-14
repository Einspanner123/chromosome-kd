"""A4 + IO3 推理评估 — K=200 (在 A4 DPM-Solver++ checkpoint 上直接评估, 不训练)

目的: 测试 A4 (DPM-Solver++) 叠加 Top-K 剪枝 K=200 的 mAP 与速度
方法: 加载 a4_dpm_pp checkpoint, 推理时启用 topk_pruning K=200
理论: A4 每步 1 次模型调用, 叠加 K=200 后续步 Self-Attn 加速 6.25x
预期: mAP 下降 <0.005, 延迟低于 A4+K300

组合卖点: 更激进的速度优化, 精度可控
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

# IO3: Top-K 剪枝 K=200 (在 A4 DPM-Solver++ 基础上)
model = dict(
    bbox_head=dict(
        topk_pruning_enabled=True,
        topk_k=200,
        topk_pruning_step=0,
    ),
)
