"""LDMDet + Count-Prior Constrained Diffusion (方向二: 计数先验约束)

继承自 rf_heun_adaln, 启用路径 B (拉格朗日约束 NMS) + 路径 C (计数分支).
与 rf_heun_adaln.py (baseline) 差异: counting_branch + use_count_constraint.

实验目标: 验证计数先验约束对染色体计数准确性的提升
- 假设: 路径 B+C 联合可将 |N_pred - c| 从 5-10 降到 1-2
- 对比: rf_heun_adaln.py (baseline)
- 路径 B: 推理时二分搜索置信度阈值, 使检测数 ≈ 预测 count
- 路径 C: 训练时从 FPN 特征预测 count, 用交叉熵监督

开关控制:
- counting_branch: None=不启用路径 C, dict=启用路径 C
- use_count_constraint: True=启用路径 B (推理时计数约束 NMS)
- default_target_count: 无 counting_branch 时的默认目标计数 (46)
- count_loss_weight: 计数分支损失权重 (默认 1.0)

消融实验:
- 仅路径 B: 注释 counting_branch, 保留 use_count_constraint=True
- 仅路径 C: 注释 use_count_constraint, 保留 counting_branch
- B+C 联合: 两者都启用 (推荐, 本配置默认)
"""

_base_ = ['rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        # 路径 C: 计数分支 (从 FPN 特征预测染色体总数)
        counting_branch=dict(
            type='CountingBranch',
            feat_channels=256,
            num_levels=4,
            count_min=44,
            count_max=48,
        ),
        # 路径 B: 推理时计数约束 NMS
        use_count_constraint=True,
        default_target_count=46,
        count_constraint_iou_threshold=0.5,
        count_constraint_min_keep=10,
        # 计数分支损失权重
        count_loss_weight=1.0,
    ),
)
