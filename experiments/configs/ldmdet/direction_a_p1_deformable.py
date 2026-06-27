"""方向 A: 多尺度特征增强 (A1 P1 层 + A2 DeformableRoIAlign)

基于 rf_heun_adaln baseline, 仅修改:
  - neck: FPN → FPNWithP1 (增加 stride 2 的 P1 层)
  - roi_extractor: SingleRoIExtractor → DeformableRoIExtractor (可学习空间偏移)
  - featmap_strides: [4, 8, 16, 32] → [2, 4, 8, 16, 32] (新增 P1 对应 stride 2)

预期收益: mAP_s (小目标) +0.03~0.05
"""

_base_ = ['./rf_heun_adaln.py']

# A1: neck 改为 FPNWithP1 (输出 P1-P5 共 5 层)
model = dict(
    neck=dict(
        _delete_=True,
        type='FPNWithP1',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4,  # FPN 原有 P2-P5 (P1 由 FPNWithP1 自动添加)
        start_level=0,
    ),
    bbox_head=dict(
        # A2: roi_extractor 改为 DeformableRoIExtractor
        roi_extractor=dict(
            _delete_=True,
            type='PurePyTorchDeformableRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[2, 4, 8, 16, 32],  # 新增 P1 (stride 2)
            deform_groups=1,
        ),
    ),
)
