"""+DPM-Solver++ + IO3 Top-K K=100 在 24obj test set 上的评估配置

用途: test set 评估 (1000张测试图), 用于 val/test 一致性核对
继承: a4_test_eval_24obj.py (已指向 test 目录) + topk_pruning K=100
checkpoint: work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth
"""
_base_ = ['../mainline_ablation_24obj/a4_test_eval_24obj.py']

# IO3: Top-K 剪枝 K=100 (在 +DPM-Solver++ 基础上)
model = dict(
    bbox_head=dict(
        topk_pruning_enabled=True,
        topk_k=100,
        topk_pruning_step=0,
    ),
)
