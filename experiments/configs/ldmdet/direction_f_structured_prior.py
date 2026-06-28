"""方向 F: 扩散框架创新 (F1 结构化噪声先验 + F3 数量约束)

基于 rf_heun_adaln baseline, 修改:
  - F1: structured_prior — 用训练集框分布 GMM 代替标准高斯噪声
    * 从训练集统计框的 cxcywh 分布, 拟合 GMM
    * 训练/推理时从 GMM 采样初始框 (而非 randn)
    * 降低"从纯噪声找 GT"的学习难度
  - F3: num_proposals 500 → 100 (覆盖 46 条 + 冗余)
    * 减少背景 proposals 浪费

前置: 需先运行 scripts/compute_box_stats.py 生成 work_dirs/box_stats.pt

预期收益: mAP +0.01~0.03, 推理速度提升
"""

_base_ = ['./rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        # F3: 减少 proposals 数量 (覆盖 46 条 + 冗余)
        num_proposals=100,
        # F1: 结构化噪声先验
        structured_prior=dict(
            type='StructuredPrior',
            stats_file='work_dirs/box_stats.pt',
        ),
    ),
)
