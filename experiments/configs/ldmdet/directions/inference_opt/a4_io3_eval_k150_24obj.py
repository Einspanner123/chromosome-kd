"""+DPM-Solver++ + IO3 推理评估 — K=150 (边界探索, 在 +DPM-Solver++ checkpoint 上直接评估)

目的: 探索 renewal OFF 开始退化的 K 阈值 (100 < K_threshold < 200)
方法: 加载 a4_dpm_pp checkpoint, 推理时启用 topk_pruning K=150
组合卖点: 确认 box_renewal 推理关闭的安全边界 (K≥200 已验证安全, K=100 已验证退化)
"""
_base_ = ['../mainline_ablation_24obj/a4_dpm_pp_24obj.py']

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

# IO3: Top-K 剪枝 K=150 (边界探索)
model = dict(
    bbox_head=dict(
        topk_pruning_enabled=True,
        topk_k=150,
        topk_pruning_step=0,
    ),
)
