"""+DPM-Solver++ 在 24obj test set 上的评估配置

用途: test set 评估 (1000张测试图)
继承: a4_dpm_pp_24obj.py (仅覆盖 test_dataloader/test_evaluator 指向 test 目录)
"""
_base_ = ['./a4_dpm_pp_24obj.py']

data_root = 'data/24_chromosomes_object/coco/'

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
    )
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/_annotations.coco.json',
    metric='bbox',
    classwise=True,
    _delete_=True,
)
