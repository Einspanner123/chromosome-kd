"""方向 G1: LAMFPN 局部注意力特征金字塔 (单阶段轻量版)

基于 rf_heun_adaln baseline, 仅替换 neck:
  - G1: LAMFPN 替换标准 FPN
    * top-down 路径在 P3, P4 层级用 LAMModule (softmax 注意力) 替换简单相加
    * 输出层应用 DualAttention (通道+空间双重增强, 带跳过机制)
    * 接口完全兼容, num_outs=4 不变, RoIExtractor 无需修改

瓶颈对应:
  - 分类误差 43.4% (同组形态相似类混淆) → DualAttention 增强判别特征
  - 特征融合质量差 (FPN 简单相加) → LAMModule 自适应融合

预期收益: mAP +0.005~0.015, 训练耗时 +10~15%
"""

_base_ = ['./rf_heun_adaln.py']

model = dict(
    neck=dict(
        _delete_=True,
        type='LAMFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4,
        # G1 关键参数
        apply_lam_levels=(1, 2),         # 在 P3, P4 融合时应用 LAM
        attention_type='dual',           # 启用 DualAttention (通道+空间)
        use_cross_layer_attention=False, # G3 独立实验, 默认关闭
        # 效率优化
        use_dw_conv=True,                # 深度可分离卷积减少参数
        channel_reduction=4,            # LAM 通道压缩比
        use_modern_norm=True,            # GroupNorm (对小 batch 更稳)
        use_modern_act='silu',           # SiLU 激活
        use_checkpoint=False,            # 训练显存充裕时不开启 (A6000 49GB)
        # 跳过注意力阈值 (0 = 不跳过, 始终应用)
        skip_attention_thresh=0.0,
    ),
)
