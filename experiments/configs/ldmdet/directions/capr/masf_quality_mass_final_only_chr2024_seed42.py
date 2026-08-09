"""MASF Phase-1A2 COCO-utility mass allocation, Dataset 1 seed 42."""

_base_ = ['./masf_mass_final_only_chr2024_seed42.py']

model = dict(
    bbox_head=dict(
        criterion=dict(
            mass_target_mode='coco_utility_softmax',
            mass_target_temperature=0.1,
        ),
    ),
)

work_dir = 'work_dirs/masf_quality_mass_final_only_chr2024_seed42'
