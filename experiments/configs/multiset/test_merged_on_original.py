"""测试合并模型在 Original test 上的性能"""
_base_ = ['../ldmdet/sinkhorn_stochastic.py']

data_root = 'data/Chromosome20240904_NoAug_NoResize_coco/'

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
    ),
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/_annotations.coco.json',
    metric='bbox',
    format_only=False,
)
