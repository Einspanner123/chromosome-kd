_base_ = ["./chromodet_baseline.py"]

# HyperParam
use_length_ordering = True

# model settings
model = dict(
    bbox_head=dict(
        type="ChromoDetDynamicHead",
        use_length_prior=use_length_ordering,
        single_head=dict(use_length_prior=use_length_ordering),
        # criterion
        criterion=dict(
            type="ChromoDetCriterion",
            use_length_ordering=use_length_ordering,
        ),  # 长度先验
    ),
)

train_dataloader = dict(batch_size=4)
