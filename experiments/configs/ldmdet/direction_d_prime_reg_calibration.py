"""方向 D': 回归偏移校正 (D'2 训练时 reg bias 正则化)

基于 rf_heun_adaln baseline, 仅修改:
  - D'2: reg bias 正则化
    * 在 loss 中加 L2 正则项: reg_bias_weight * (dw_mean^2 + dh_mean^2)
    * 约束 reg_head 输出的 dw/dh 均值接近 0, 消除系统性尺度收缩
    * 白盒依据: 所有 ckpt 的 dw/dh 均显著偏负 (-1.2 ~ -1.8), 跨数据集一致

白盒发现 5 (依据):
- baseline: dw=-1.819, dh=-1.633 (per_dim_mean)
- 24obj: dw=-1.310, dh=-1.270 (跨数据集方向一致 → 架构固有)
- dx, dy 接近 0, 仅尺度预测系统性偏小

注: D'1 (后处理校准) 因校准因子推导不确定 (-1.8 含背景 proposal) 暂不实施,
    先用 D'2 (训练时正则) 让模型自适应学习消除偏移。

预期收益: mAP +0.005~0.015 (通过减少尺度预测偏移提升定位精度)
"""

_base_ = ['./rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        # D'2: reg bias 正则化权重
        # baseline 下 dw_mean^2+dh_mean^2 ≈ 5.98, weight=0.05 → loss ≈ 0.30
        # 随训练进行 dw/dh 趋近 0, loss 自动衰减
        reg_bias_weight=0.05,
    ),
)
