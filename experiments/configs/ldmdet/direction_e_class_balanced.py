"""方向 E: 数据驱动改进 — 类别平衡采样 (Y 类过采样)

基于 rf_heun_adaln baseline, 仅修改:
  - E: train_dataloader.dataset 包装为 ClassBalancedDataset
    * 对少数类 (Y=580, 1:3.85 不平衡) 过采样
    * 使用 mmdet 的 ClassBalancedDataset 包装器

数据集分析:
  - Y 类样本最少 (580), 最多类 2232, 比例 1:3.85
  - 每图框数均值 43.2 (固定核型 46 条)
  - E1/E2/E3 (mixup/cutmix/copy-paste/类别单独训练) 不可行 (破坏核型)
  - E4 (对比学习) 已并入方向 C (C2)

预期收益: Y 类 AP +0.01~0.03 (有限, 因每图都含各类)
"""

_base_ = ['./rf_heun_adaln.py']

# E: 用 ClassBalancedDataset 包装训练集, 对少数类过采样
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        dataset=dict(
            type='CocoDataset',
            data_root='data/Chromosome20240904_NoAug_NoResize_coco/',
            metainfo={{_base_.train_dataloader.dataset.metainfo}},
            ann_file='train/_annotations.coco.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline={{_base_.train_dataloader.dataset.pipeline}},
            backend_args=None,
        ),
        oversample_thr=0.5,  # 频率低于均值 * 0.5 的类别过采样
    ),
)
