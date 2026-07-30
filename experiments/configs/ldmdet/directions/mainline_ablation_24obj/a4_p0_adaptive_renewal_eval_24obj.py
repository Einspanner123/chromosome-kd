"""+DPM-Solver++ + P0 自适应阈值 box_renewal 评估配置

用途: 在 +DPM-Solver++ checkpoint 上验证 P0 自适应阈值效果
基础: a4_dpm_pp_24obj.py, 仅添加 adaptive_renewal_threshold=True
对比: 与 a4_dpm_pp_24obj.py (固定阈值 score_thr=0.05) 对比

P0 设计:
  - adaptive_renewal_threshold=True: 阈值随时间步递减
  - 早期 (t_curr→1.0): threshold≈0.9 (积极淘汰低分框)
  - 后期 (t_curr→0.0): threshold=0.05 (保守保留, 兜底)
  - adaptive_renewal_scale=0.9: 线性递减系数

SwanLab: 项目 'ldmdet-inference', 实验 'a4_p0_adaptive_{solver}_{steps}step'
"""
_base_ = ['./a4_dpm_pp_24obj.py']

# === P0: 启用自适应阈值 box_renewal ===
model = dict(
    bbox_head=dict(
        adaptive_renewal_threshold=True,
        adaptive_renewal_scale=0.9,
    ),
)
