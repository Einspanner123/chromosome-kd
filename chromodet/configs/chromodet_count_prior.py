_base_ = ["./chromodet_baseline.py"]

# HyperParam
use_count_prior = True

# model settings
model = dict(
    bbox_head=dict(
        type="ChromoDetDynamicHead",
        criterion=dict(
            type="ChromoDetCriterion",
            use_count_prior=use_count_prior,
        ),  # 长度先验
    ),
)

train_dataloader = dict(batch_size=8)  # better than 4
