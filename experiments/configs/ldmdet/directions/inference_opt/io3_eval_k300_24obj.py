"""IO3 推理评估 — K=300 (在 SOTA checkpoint 上直接评估, 不训练)"""
_base_ = ['./io3_eval_k200_24obj.py']

model = dict(
    bbox_head=dict(
        topk_k=300,
    ),
)
