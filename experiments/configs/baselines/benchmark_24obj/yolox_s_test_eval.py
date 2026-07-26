"""YOLOX-S 在 24obj test set 上的评估配置

用途: test set 评估 (1000张测试图), 用于 val/test 一致性核对
继承: yolox_s.py
  - 覆盖 test_dataloader/test_evaluator 指向 test 目录
  - 清空 custom_hooks (移除 EMAHook/YOLOXModeSwitchHook/EarlyStoppingHook,
    这些是训练专用 hook; EMA 权重已保存在 best checkpoint 的 state_dict 中,
    无需 EMAHook 即可加载)
checkpoint: work_dirs/baselines/yolox_s/best_coco_bbox_mAP_epoch_200.pth
"""
_base_ = ['./yolox_s.py']

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

# 清空训练专用 hooks (EMAHook 的 after_load_checkpoint 在 test 模式下会因
# ema_model 未初始化而报错; best checkpoint 已含 EMA 权重, 直接加载即可)
custom_hooks = []
