"""+DPM-Solver++ + BF16 AMP 评估配置 (诊断用)

用于 BF16 掉点诊断: +DPM-Solver++ (FP32, mAP=0.863) vs +DPM-Solver+++BF16.
仅修改 amp_dtype='bfloat16', 其余与 +DPM-Solver++ 完全一致.
"""

_base_ = ['./a4_dpm_pp_24obj.py']

model = dict(
    bbox_head=dict(
        amp_dtype='bfloat16',
    ),
)
