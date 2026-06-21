"""LDMDet + Unbalanced GHSS (方向一: 非平衡最优传输耦合)

继承自 rf_heun_adaln, 切换 coupling 为 unbalanced_ghss.
与 ghss.py (baseline) 唯一差异: coupling type + lambda 参数.

实验目标: 验证非平衡 OT 在 GT 分布不均匀场景下的收益
- 假设: 边缘松弛提升稀疏图 recall, 改善组间均衡
- 对比: ghss.py (mAP=0.753 baseline)
- 消融: lambda_row, lambda_col 网格搜索
"""

_base_ = ['rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        coupling=dict(
            type='unbalanced_ghss',
            epsilon=5.0,          # 与 baseline 一致
            num_iters=20,         # 与 baseline 一致
            lambda_row=1.0,       # 行边缘松弛 (推荐起点)
            lambda_col=1.0,       # 列边缘松弛 (推荐起点)
        ),
    ),
)
