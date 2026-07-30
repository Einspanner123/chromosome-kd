"""+DPM-Solver++ 在旧数据集 Chromosome20240904 test set 上的评估配置

用途: test set 评估 (220张测试图)
注意: +DPM-Solver++ checkpoint 在 24obj 上训练, 此配置用于跨数据集 test 评估
      如需旧数据集上训练的 checkpoint 评估, 请使用对应的旧数据集 checkpoint
"""
_base_ = ['./a4_dpm_pp_24obj.py']

data_root = 'data/Chromosome20240904_NoAug_NoResize_coco/'

# 覆盖所有数据加载器指向旧数据集
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
    )
)

val_evaluator = dict(
    ann_file=data_root + 'valid/_annotations.coco.json',
    _delete_=True,
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/_annotations.coco.json',
    metric='bbox',
    classwise=True,
    _delete_=True,
)
