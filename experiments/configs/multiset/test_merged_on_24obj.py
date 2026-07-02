"""测试合并模型在 Original 和 24obj 独立 test 上的性能"""
_base_ = ['../ldmdet/sinkhorn_stochastic.py']

# 在 Original test 上评估时，取消注释下行并注释 24obj 部分
# ====== Original test ======
# data_root = 'data/Chromosome20240904_NoAug_NoResize_coco/'
# ann_file = 'test/_annotations.coco.json'

# ====== 24obj test ======
data_root = 'data/24_chromosomes_object/coco/'
ann_file = 'test/_annotations.coco.json'

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=dict(img='test/'),
        test_mode=True,
    ),
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + ann_file,
    metric='bbox',
    format_only=False,
)
