"""AsyncCheckpointHook max_keep_ckpts 测试 — 4 epoch"""
_base_ = ['../ldmdet/rf_heun_adaln.py']

data_root = 'data/24_chromosomes_object/coco/'
train_dataloader = dict(
    dataset=dict(data_root=data_root, ann_file='train/_annotations.coco.json', data_prefix=dict(img='train/')),
)
val_dataloader = dict(
    dataset=dict(data_root=data_root, ann_file='valid/_annotations.coco.json', data_prefix=dict(img='valid/')),
)
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')

max_epochs = 4
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=4),
]

# 使用 AsyncCheckpointHook（已从 rf_heun_adaln 继承）
# 覆盖验证 max_keep_ckpts=2 是否能清理旧 ckpt
custom_hooks = [
    dict(type='CopyProjectHook', priority='VERY_LOW'),
]
