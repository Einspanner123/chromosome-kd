"""方向 G3: LAMFPN + CrossLayerAttention (跨层注意力)

基于 G1, 额外启用 CrossLayerAttention:
  - 每个输出层级融合其他所有层级的加权特征 (sigmoid 权重)
  - 跨层语义补充: P2 的细节 + P5 的语义

预期收益: mAP +0.003~0.008 (在 G1 基础上, 边际递减)
"""

_base_ = ['./direction_g_lamfpn.py']

model = dict(
    neck=dict(
        use_cross_layer_attention=True,  # 启用跨层注意力
    ),
)
