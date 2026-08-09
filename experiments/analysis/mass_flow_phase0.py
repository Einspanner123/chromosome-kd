#!/usr/bin/env python3
"""Measure proposal-mass redundancy before implementing Mass-Aware Set Flow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from torchvision.ops import batched_nms, box_iou

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.analysis.benchmark_inference import load_model, load_val_samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-images', type=int, default=100)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--score-threshold', type=float, default=0.05)
    parser.add_argument('--match-iou', type=float, default=0.5)
    parser.add_argument('--nms-iou', type=float, default=0.5)
    parser.add_argument(
        '--apply-ranking-gates',
        action='store_true',
        help='Apply final-only quality/mass gates to the emitted scores.',
    )
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def mean(values):
    return sum(values) / max(len(values), 1)


def summarize(records):
    return {key: mean(values) for key, values in records.items()}


@torch.no_grad()
def main():
    args = parse_args()
    device = f'cuda:{args.gpu_id}'
    model, cfg = load_model(args.config, args.checkpoint, device)
    samples = load_val_samples(cfg, args.num_images)
    per_step = defaultdict(lambda: defaultdict(list))
    final_records = defaultdict(list)

    for sample in samples:
        image = sample['img'].to(device)
        gt_boxes = sample['gt_boxes'].to(device)
        gt_labels = sample['gt_labels'].to(device)
        features = model.backbone(image)
        features = model.neck(features)
        _, trajectory = model.bbox_head.predict(
            features,
            [sample['img_meta']],
            rescale=False,
            return_trajectory=True,
        )

        num_gt = int(gt_boxes.shape[0])
        final_step_index = len(trajectory) - 1
        for step_index, (class_logits, boxes) in enumerate(trajectory):
            if args.apply_ranking_gates and step_index == final_step_index:
                class_logits = model.bbox_head._quality_ranking_logits(
                    class_logits
                )
                class_logits = model.bbox_head._mass_ranking_logits(
                    class_logits
                )
            probabilities = class_logits[0].sigmoid()
            scores, labels = probabilities.max(dim=-1)
            step = per_step[step_index]
            step['score_mass'].append(float(scores.sum()))
            step['mass_per_gt'].append(float(scores.sum()) / max(num_gt, 1))
            for threshold in (0.05, 0.10, 0.20, 0.50):
                step[f'count_score_ge_{threshold:.2f}'].append(
                    float((scores >= threshold).sum())
                )

        class_logits, boxes = trajectory[-1]
        if args.apply_ranking_gates:
            class_logits = model.bbox_head._quality_ranking_logits(class_logits)
            class_logits = model.bbox_head._mass_ranking_logits(class_logits)
        boxes = boxes[0]
        probabilities = class_logits[0].sigmoid()
        scores, labels = probabilities.max(dim=-1)
        selected = scores >= args.score_threshold
        selected_boxes = boxes[selected]
        selected_scores = scores[selected]
        selected_labels = labels[selected]
        selected_count = int(selected.sum())

        if selected_count:
            keep = batched_nms(
                selected_boxes,
                selected_scores,
                selected_labels,
                args.nms_iou,
            )
            nms_count = int(keep.numel())
        else:
            nms_count = 0

        matched_proposals = 0
        covered_gt = 0
        correct_match_counts = torch.zeros(num_gt, device=device)
        if selected_count and num_gt:
            pairwise_iou = box_iou(selected_boxes, gt_boxes)
            same_class = selected_labels[:, None] == gt_labels[None, :]
            valid = (pairwise_iou >= args.match_iou) & same_class
            matched_proposals = int(valid.any(dim=1).sum())
            covered_gt = int(valid.any(dim=0).sum())
            correct_match_counts = valid.sum(dim=0).float()

        duplicate_surplus = int(
            (correct_match_counts - 1).clamp(min=0).sum().item()
        )
        final_records['num_gt'].append(float(num_gt))
        final_records['selected_count'].append(float(selected_count))
        final_records['nms_count'].append(float(nms_count))
        final_records['nms_removed'].append(float(selected_count - nms_count))
        final_records['nms_removed_fraction'].append(
            (selected_count - nms_count) / max(selected_count, 1)
        )
        final_records['matched_proposals'].append(float(matched_proposals))
        final_records['covered_gt'].append(float(covered_gt))
        final_records['gt_coverage'].append(covered_gt / max(num_gt, 1))
        final_records['duplicate_surplus'].append(float(duplicate_surplus))
        final_records['duplicates_per_gt'].append(
            duplicate_surplus / max(num_gt, 1)
        )

    report = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'num_images': len(samples),
        'score_threshold': args.score_threshold,
        'match_iou': args.match_iou,
        'nms_iou': args.nms_iou,
        'apply_ranking_gates': args.apply_ranking_gates,
        'per_step': {
            str(index): summarize(records)
            for index, records in sorted(per_step.items())
        },
        'final': summarize(final_records),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
