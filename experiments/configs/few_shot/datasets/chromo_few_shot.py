"""Chromo few-shot 数据集配置

从 Chromosome20240904 训练集中采样 k-shot 子集
使用预生成的 few_shot_k{k}.json 标注文件
"""

# 数据根目录
data_root = 'data/Chromosome20240904_NoAug_NoResize_coco/'

# 24 类染色体 (chromo 顺序: C10,C11,C12 在 C6 之前)
classes = (
    'A1', 'A2', 'A3', 'B4', 'B5',
    'C10', 'C11', 'C12', 'C6', 'C7', 'C8', 'C9',
    'D13', 'D14', 'D15',
    'E16', 'E17', 'E18',
    'F19', 'F20', 'G21', 'G22', 'X', 'Y',
)

# k-shot 值 (由具体配置覆盖)
# few_shot_k = 5  或  10

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(type='Resize', scale=(480, 1333), keep_ratio=True),
                dict(type='RandomCrop', crop_type='relative_range', crop_size=(0.3, 0.3), allow_negative_crop=True, recompute_bbox=True),
                dict(type='Resize', scale=(480, 1333), keep_ratio=True),
            ],
            [
                dict(type='Resize', scale=(640, 1333), keep_ratio=True),
                dict(type='RandomCrop', crop_type='relative_range', crop_size=(0.3, 0.3), allow_negative_crop=True, recompute_bbox=True),
                dict(type='Resize', scale=(640, 1333), keep_ratio=True),
            ],
            [
                dict(type='Resize', scale=(800, 1333), keep_ratio=True),
                dict(type='RandomCrop', crop_type='relative_range', crop_size=(0.3, 0.3), allow_negative_crop=True, recompute_bbox=True),
                dict(type='Resize', scale=(800, 1333), keep_ratio=True),
            ],
            [
                dict(type='Resize', scale=(480, 1333), keep_ratio=True),
            ],
            [
                dict(type='Resize', scale=(640, 1333), keep_ratio=True),
            ],
            [
                dict(type='Resize', scale=(800, 1333), keep_ratio=True),
            ],
        ],
    ),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs'),
]


def make_few_shot_dataloader(k=5, batch_size=2):
    """生成 k-shot 数据加载器配置"""
    ann_file = f'train/few_shot_k{k}.json'
    return dict(
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
            type='CocoDataset',
            data_root=data_root,
            ann_file=ann_file,
            data_prefix=dict(img='train/'),
            filter_empty_gt=False,
            min_size=1e-5,
            pipeline=train_pipeline,
            metainfo=dict(classes=classes),
        ),
    )


def make_val_dataloader(batch_size=1):
    """生成验证数据加载器配置 (完整验证集)"""
    return dict(
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='CocoDataset',
            data_root=data_root,
            ann_file='valid/_annotations.coco.json',
            data_prefix=dict(img='valid/'),
            test_mode=True,
            pipeline=test_pipeline,
            metainfo=dict(classes=classes),
        ),
    )


# 默认使用 k=5
train_dataloader = make_few_shot_dataloader(k=5, batch_size=2)
val_dataloader = make_val_dataloader(batch_size=1)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox',
    format_only=False,
)
