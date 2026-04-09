_base_ = ["./ldmdet_flowdet_adaln.py"]

# 论文核心配置: AdaLN-Zero + OT 训练耦合 + Objectness
#
# 这是 FlowDet 的理论核心:
# 1. AdaLN-Zero: 更好的时间条件化 (+0.3 mAP over baseline)
# 2. OT 训练耦合: 最优 (noise, GT) 配对 → 更直的传输路径
# 3. Objectness: 显式前景/背景判别
#
# Matcher 仍用 SimOTA (推理时不变，OT 只在训练的 noising 阶段生效)

model = dict(
    bbox_head=dict(
        ot_coupling=True,
        single_head=dict(
            use_objectness=True,
        ),
        criterion=dict(
            loss_objectness_weight=1.0,
        ),
    ),
)
