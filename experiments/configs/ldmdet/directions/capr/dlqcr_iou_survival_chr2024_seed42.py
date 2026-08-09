"""D-LQCR gate on Dataset 1, paired with the A4 seed-42 detector.

The quality branch predicts the monotone survival probabilities
P(IoU >= tau | feature) at the ten COCO thresholds. Final ranking uses their
mean, with no solver, renewal, class, or box-coordinate feedback.
"""

_base_ = ['./capr_quality_final_only_chr2024_seed42.py']

model = dict(
    bbox_head=dict(
        quality_score_beta=1.0,
        single_head=dict(
            quality_thresholds=(
                0.50, 0.55, 0.60, 0.65, 0.70,
                0.75, 0.80, 0.85, 0.90, 0.95,
            ),
        ),
    ),
)

work_dir = 'work_dirs/dlqcr_iou_survival_chr2024_seed42'
