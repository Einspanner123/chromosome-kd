"""RTMDet-L 在 24obj test set 上的评估配置

用途: test set 评估 (1000张测试图), 用于 val/test 一致性核对
继承: rtmdet_l.py (仅覆盖 test_dataloader/test_evaluator 指向 test 目录)
checkpoint: work_dirs/baselines/rtmdet_l_24obj/epoch_85.pth (best val mAP=0.863)
"""
_base_ = ['./rtmdet_l.py']

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
