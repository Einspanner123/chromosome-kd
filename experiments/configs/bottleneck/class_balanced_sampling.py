"""瓶颈消融实验 — 类别平衡采样 (ClassBalancedDataset)

实验目标: 缓解 Y 类样本不足 (202 vs ~900) 导致的分类瓶颈
依据: 诊断显示 Y 类 GT 仅 202 个, 不平衡比 4.6.
      Y 同时是分类最差 (AP=0.624) 和定位最差 (IoU=0.847) 的类.
假设: 若类别平衡采样提升 Y 的 AP, 说明数据不平衡是 Y 类瓶颈;
      若无提升, 说明 Y 类困难是形态相似性导致 (Y→E18 混淆 5.4%).

对比: rf_heun_adaln.py (baseline, DefaultSampler)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

train_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.5,  # 少数类过采样阈值
        dataset=dict(
            type='CocoDataset',
            data_root='data/Chromosome20240904_NoAug_NoResize_coco/',
            metainfo=dict(
                classes=(
                    'A1', 'A2', 'A3', 'B4', 'B5', 'C10', 'C11', 'C12',
                    'C6', 'C7', 'C8', 'C9', 'D13', 'D14', 'D15',
                    'E16', 'E17', 'E18', 'F19', 'F20', 'G21', 'G22', 'X', 'Y',
                ),
            ),
            ann_file='train/_annotations.coco.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline={{_base_.train_pipeline}},
        ),
    ),
)
