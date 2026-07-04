"""方向 F': 仅 F1 结构化噪声先验 (控制变量, 排除 F3 干扰)

F (F1+F3 同时改) 失败: mAP=0.574 (-0.171), 根因是 num_proposals 500→100 过激。
F' 仅保留 F1 (structured_prior), 保持 num_proposals=500, 单独验证 GMM 噪声先验的效果。

基于 rf_heun_adaln baseline, 修改:
  - F1: structured_prior — 用训练集框分布 GMM 代替标准高斯噪声
    * 从训练集统计框的 cxcywh 分布, 拟合 GMM
    * 训练/推理时从 GMM 采样初始框 (而非 randn)
    * 降低"从纯噪声找 GT"的学习难度
  - 保持 num_proposals=500 (与 baseline 一致, 控制变量)

前置: 需先运行 scripts/compute_box_stats.py 生成 work_dirs/box_stats.pt

预期收益: mAP +0.01~0.03 (若 structured_prior 有效)
"""

_base_ = ['../nonlinear_trajectory/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        # F1: 结构化噪声先验 (仅此一项, 不改 num_proposals)
        structured_prior=dict(
            type='StructuredPrior',
            stats_file='work_dirs/box_stats.pt',
        ),
    ),
)
