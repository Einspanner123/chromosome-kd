#!/usr/bin/env python3
"""Fit and export BoxChart moments from a COCO training annotation file."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ldmdet.diffusion.box_chart import ValidBoxChart


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--eps', type=float, default=1e-6)
    return parser.parse_args()


def nested_list(tensor):
    return tensor.detach().cpu().tolist()


def main():
    args = parse_args()
    with open(args.annotations, encoding='utf-8') as f:
        coco = json.load(f)
    image_sizes = {
        int(image['id']): (float(image['width']), float(image['height']))
        for image in coco['images']
    }

    boxes = []
    rejected = 0
    for ann in coco['annotations']:
        x, y, width, height = map(float, ann['bbox'])
        image_width, image_height = image_sizes[int(ann['image_id'])]
        box = [
            x / image_width,
            y / image_height,
            (x + width) / image_width,
            (y + height) / image_height,
        ]
        if width <= 0 or height <= 0:
            rejected += 1
            continue
        boxes.append(box)
    boxes = torch.tensor(boxes, dtype=torch.float64)

    chart = ValidBoxChart(eps=args.eps).double()
    mean, covariance = chart.fit(boxes)
    latent = chart.encode(boxes)
    reconstructed = chart.decode(latent)
    latent_mean = latent.mean(dim=0)
    centered = latent - latent_mean
    latent_covariance = centered.T @ centered / latent.shape[0]
    widths = reconstructed[:, 2] - reconstructed[:, 0]
    heights = reconstructed[:, 3] - reconstructed[:, 1]

    report = {
        'annotations': args.annotations,
        'num_boxes': int(boxes.shape[0]),
        'num_rejected': rejected,
        'eps': args.eps,
        'chart_mean': nested_list(mean),
        'chart_covariance': nested_list(covariance),
        'whitening': nested_list(chart.whitening),
        'coloring': nested_list(chart.coloring),
        'whitened_mean': nested_list(latent_mean),
        'whitened_covariance': nested_list(latent_covariance),
        'roundtrip_abs_error': {
            'mean': float((reconstructed - boxes).abs().mean()),
            'max': float((reconstructed - boxes).abs().max()),
        },
        'decoded_validity': {
            'invalid_count': int((
                (reconstructed[:, 0] < 0) |
                (reconstructed[:, 1] < 0) |
                (reconstructed[:, 2] > 1) |
                (reconstructed[:, 3] > 1) |
                (widths <= 0) | (heights <= 0)
            ).sum()),
            'min_width': float(widths.min()),
            'min_height': float(heights.min()),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
