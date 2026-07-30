"""+DPM-Solver++ + IO3 推理评估 — K=300 (在 +DPM-Solver++ checkpoint 上直接评估, 不训练)

目的: 测试 +DPM-Solver++ (DPM-Solver++) 叠加 Top-K 剪枝 K=300 的 mAP 与速度
方法: 加载 a4_dpm_pp checkpoint, 推理时启用 topk_pruning K=300
理论: +DPM-Solver++ 每步 1 次模型调用 (vs Heun 2 次), 叠加 K=300 后续步 Self-Attn 加速 2.8x
预期: mAP 下降 <0.003, 延迟显著低于 +DPM-Solver++ 单独 (75ms)

组合卖点: 精度+速度双收益的进一步加速
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

# IO3: Top-K 剪枝 K=300 (在 +DPM-Solver++ 基础上)
model = dict(
    bbox_head=dict(
        topk_pruning_enabled=True,
        topk_k=300,
        topk_pruning_step=0,
    ),
)
