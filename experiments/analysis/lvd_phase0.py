#!/usr/bin/env python3
"""Measure whether RF box trajectories have useful direction-error headroom."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def direction_stats(predicted, target):
    eps = 1e-8
    pred_norm = predicted.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    valid = (pred_norm > eps) & (target_norm > eps)
    cosine = (
        (predicted[valid] * target[valid]).sum(dim=-1)
        / (pred_norm[valid] * target_norm[valid]).clamp_min(eps)
    ).clamp(-1, 1)
    return cosine, pred_norm[valid], target_norm[valid]


def summarize(values):
    tensor = torch.cat(values).float().cpu()
    return {
        'count': int(tensor.numel()),
        'mean': float(tensor.mean()),
        'std': float(tensor.std(unbiased=False)),
        'p05': float(torch.quantile(tensor, 0.05)),
        'p50': float(torch.quantile(tensor, 0.50)),
        'p95': float(torch.quantile(tensor, 0.95)),
        'negative_fraction': float((tensor < 0).float().mean()),
        'below_0.90_fraction': float((tensor < 0.90).float().mean()),
        'sin2_mean': float((1 - tensor.square()).mean()),
    }


@torch.no_grad()
def main():
    args = parse_args()
    device = f'cuda:{args.gpu_id}'
    torch.manual_seed(args.seed)
    model, cfg = load_model(args.config, args.checkpoint, device)
    samples = load_val_samples(cfg, args.num_images)
    head = model.bbox_head
    time_grid = [pair[0] for pair in head._sampler.build_time_pairs(
        torch.device(device)
    )]
    records = {
        str(time): defaultdict(list) for time in time_grid
    }

    for sample in samples:
        image = sample['img'].to(device)
        gt_boxes = sample['gt_boxes'].to(device)
        gt_labels = sample['gt_labels'].to(device)
        if gt_boxes.numel() == 0:
            continue
        img_metas = [sample['img_meta']]
        features = model.neck(model.backbone(image))
        targets = head._normalize_targets(
            [gt_boxes], [gt_labels], img_metas, 1
        )
        gt_raw = head._sampler.normalized_xyxy_to_raw(targets[0].bboxes)
        noise = torch.randn(head.num_proposals, 4, device=device)
        x_start, coupling_indices = head._couple_single_image(
            noise, gt_raw, gt_labels, torch.device(device)
        )

        for time in time_grid:
            t = torch.tensor([time], device=device)
            x_t = (1 - time) * x_start + time * noise
            curr_bboxes = head._sampler.raw_to_xyxy(
                x_t.unsqueeze(0), img_metas
            )
            t_input = t * head.timesteps
            all_logits, all_boxes, _ = head(
                features, curr_bboxes, t_input, img_metas
            )
            norm_boxes = head._normalize_pred_bboxes(all_boxes, img_metas)
            outputs = head._build_outputs(all_logits, norm_boxes)
            indices = head.criterion.matcher(outputs, targets)
            foreground, matched_gt = indices[0]
            if not foreground.any():
                continue

            pred_raw = head._sampler.normalized_xyxy_to_raw(
                outputs.pred_boxes
            )[0]
            predicted_direction = x_t - pred_raw
            dynamic_direction = x_t - gt_raw[matched_gt.clamp(
                min=0, max=gt_raw.shape[0] - 1
            )]
            coupling_direction = x_t - gt_raw[coupling_indices]

            dyn_cos, pred_norm, target_norm = direction_stats(
                predicted_direction[foreground],
                dynamic_direction[foreground],
            )
            coupling_cos, _, _ = direction_stats(
                predicted_direction[foreground],
                coupling_direction[foreground],
            )
            key = str(time)
            records[key]['dynamic_cosine'].append(dyn_cos.cpu())
            records[key]['coupling_cosine'].append(coupling_cos.cpu())
            records[key]['predicted_norm'].append(pred_norm.cpu())
            records[key]['target_norm'].append(target_norm.cpu())
            records[key]['foreground_count'].append(
                torch.tensor([foreground.sum().item()])
            )

    per_time = {}
    for time, time_records in records.items():
        per_time[time] = {
            'dynamic_match': summarize(time_records['dynamic_cosine']),
            'original_coupling': summarize(time_records['coupling_cosine']),
            'predicted_norm_mean': float(torch.cat(
                time_records['predicted_norm']
            ).mean()),
            'target_norm_mean': float(torch.cat(
                time_records['target_norm']
            ).mean()),
            'foreground_per_image': float(torch.cat(
                time_records['foreground_count']
            ).float().mean()),
        }

    report = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'num_images': len(samples),
        'seed': args.seed,
        'time_grid': time_grid,
        'per_time': per_time,
        'gate': {
            'criterion': 'dynamic-match mean cosine < 0.95 at any solver time',
            'passed': any(
                item['dynamic_match']['mean'] < 0.95
                for item in per_time.values()
            ),
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
