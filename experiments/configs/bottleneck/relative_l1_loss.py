"""瓶颈消融实验 — 相对 L1 损失

实验目标: 验证相对 L1 损失 (按框尺寸归一化) 是否改善定位
假设: 相对 L1 对大/小框更公平, 可能提升高 IoU 阈值下的 AP.

对比: rf_heun_adaln.py (baseline, 绝对 L1 loss)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        criterion=dict(
            bbox_loss_mode='relative_l1',  # 相对 L1 损失
            bbox_loss_eps=1e-2,
        ),
    ),
)
