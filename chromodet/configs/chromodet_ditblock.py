_base_ = ['./chromodet_baseline.py']

use_dit = True
# model settings
model = dict(
    bbox_head=dict(
        type='ChromoDetDynamicHead',
        single_head=dict(
            type='ChromoDetSingleHead',
            use_dit=use_dit),
    )
)

train_dataloader=dict(
    batch_size = 4
)