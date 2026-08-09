"""Matched continuation control for the CALU Dataset-1 gate.

The final regressor receives the existing L1/GIoU objective for the same 12
epochs, but CALU is disabled.  Batch size one with two-step accumulation fits
the A4000 while preserving the effective batch size of the CALU run.
"""

_base_ = ['./calu_terminal_reg_chr2024_seed42.py']

model = dict(
    bbox_head=dict(
        criterion=dict(localization_utility_loss_weight=0.0),
    ),
)

train_dataloader = dict(batch_size=1)
optim_wrapper = dict(accumulative_counts=2)

work_dir = 'work_dirs/terminal_reg_control_chr2024_seed42'
