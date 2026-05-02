# 仅示例：把大核限制在高层特征，并做温和注入+归一化
_base_ = ["./chromodet_baseline.py"]

use_large_kernel = True
use_large_kernel_levels = [2, 3]  # 依你FPN层顺序，通常是 P4/P5
lk_alpha_init = 0.1
lk_norm_groups = 32

model = dict(
    bbox_head=dict(
        type="ChromoDetDynamicHead",
        single_head=dict(
            type="ChromoDetSingleHead",
            use_large_kernel=use_large_kernel,
            use_large_kernel_levels=use_large_kernel_levels,
            lk_alpha_init=lk_alpha_init,
            lk_norm_groups=lk_norm_groups,
        ),
    )
)

train_dataloader = dict(batch_size=2)
# 训练侧建议（在你的runner/optim里做）：base_lr *= 0.5, weight_decay=0.05（仅lk_block）
# 以及 grad_clip=1.0，训练轮次+20~30%
