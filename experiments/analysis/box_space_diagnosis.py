#!/usr/bin/env python3
"""Zero-training diagnosis for the current RF bounding-box state space.

The script intentionally uses monkey patches local to the process, so it does
not alter the detector implementation.  It measures clamp saturation,
zero/near-zero decoded boxes, FPN level switching, and proposal success
conditioned on an initially degenerate width or height.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-images', type=int, default=100)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--renewal', choices=('config', 'on', 'off'), default='off')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def tensor_boxes(boxes):
    return boxes.tensor if hasattr(boxes, 'tensor') else boxes


def paired_max_iou(boxes, gt):
    if boxes.numel() == 0 or gt.numel() == 0:
        return boxes.new_zeros((boxes.shape[0],))
    area_b = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    area_g = (gt[:, 2] - gt[:, 0]).clamp(min=0) * (
        gt[:, 3] - gt[:, 1]).clamp(min=0)
    lt = torch.maximum(boxes[:, None, :2], gt[None, :, :2])
    rb = torch.minimum(boxes[:, None, 2:], gt[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_b[:, None] + area_g[None, :] - inter
    return (inter / union.clamp(min=1e-8)).max(dim=1).values


def describe(values):
    if not values:
        return {'mean': None, 'std': None, 'n': 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'n': int(arr.size),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)
    cfg.val_dataloader = dict(cfg.val_dataloader)
    cfg.val_dataloader.update(
        batch_size=1, num_workers=0, persistent_workers=False)
    if args.renewal != 'config':
        cfg.model.bbox_head.box_renewal = args.renewal == 'on'
    cfg.load_from = args.checkpoint
    cfg.work_dir = '/tmp/box_space_diagnosis'
    cfg.gpu_id = args.gpu_id
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='DetLocalVisualizer', vis_backends=cfg.vis_backends,
        name='visualizer')

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    model.eval()
    head = model.bbox_head
    snr = float(head._sampler.snr_scale)
    uses_chart = head._sampler.box_chart is not None

    scalar = defaultdict(list)
    per_step = defaultdict(lambda: defaultdict(list))
    level_switch = defaultdict(list)
    conditional = defaultdict(list)
    raw_moments = defaultdict(list)
    n_processed = 0

    original_forward = head._forward_at_t
    captures = []

    def capture_forward(features, x_raw, t, img_metas):
        out = original_forward(features, x_raw, t, img_metas)
        captures.append({
            't': float(t),
            'x_raw': x_raw.detach().clone(),
            'cls_logits': out[0].detach().clone(),
            'pred_bboxes': out[1].detach().clone(),
            'x0_raw': out[2].detach().clone(),
            'img_metas': img_metas,
        })
        return out

    head._forward_at_t = capture_forward

    try:
        with torch.no_grad():
            for data in runner.val_dataloader:
                if n_processed >= args.num_images:
                    break
                captures.clear()
                data = model.data_preprocessor(data)
                samples = data['data_samples']
                _ = model.predict(data['inputs'], samples)
                if not captures:
                    continue

                levels = []
                for step, entry in enumerate(captures):
                    raw = entry['x_raw'][0]
                    decoded = head._sampler.raw_to_xyxy(
                        entry['x_raw'], entry['img_metas'])[0]
                    pred = entry['pred_bboxes'][0]
                    x0 = entry['x0_raw'][0]
                    meta = entry['img_metas'][0]
                    shape = meta.img_shape if hasattr(meta, 'img_shape') else meta['img_shape']
                    img_h, img_w = shape[:2]

                    decoded_w = (decoded[:, 2] - decoded[:, 0]) / img_w
                    decoded_h = (decoded[:, 3] - decoded[:, 1]) / img_h
                    pred_w = (pred[:, 2] - pred[:, 0]) / img_w
                    pred_h = (pred[:, 3] - pred[:, 1]) / img_h
                    low_sat = raw <= -snr
                    high_sat = raw >= snr
                    degenerate = (decoded_w <= 0) | (decoded_h <= 0)

                    per_step[step]['t'].append(entry['t'])
                    per_step[step]['raw_any_clamp'].append(
                        float((low_sat | high_sat).any(dim=-1).float().mean()))
                    if not uses_chart:
                        per_step[step]['raw_w_low_clamp'].append(float(low_sat[:, 2].float().mean()))
                        per_step[step]['raw_h_low_clamp'].append(float(low_sat[:, 3].float().mean()))
                    per_step[step]['state_degenerate'].append(float(degenerate.float().mean()))
                    per_step[step]['decoded_zero_size'].append(
                        float(((decoded_w == 0) | (decoded_h == 0)).float().mean()))
                    per_step[step]['decoded_near_zero_size'].append(
                        float(((decoded_w < 1e-3) | (decoded_h < 1e-3)).float().mean()))
                    per_step[step]['pred_invalid_size'].append(
                        float(((pred_w <= 0) | (pred_h <= 0)).float().mean()))
                    per_step[step]['pred_near_zero_size'].append(
                        float(((pred_w < 1e-3) | (pred_h < 1e-3)).float().mean()))
                    for dim, name in enumerate(('cx', 'cy', 'w', 'h')):
                        raw_moments[f'step{step}_xraw_{name}'].extend(
                            raw[:, dim].float().cpu().tolist())
                        raw_moments[f'step{step}_x0_{name}'].extend(
                            x0[:, dim].float().cpu().tolist())

                    scale = torch.sqrt(
                        (decoded[:, 2] - decoded[:, 0]).clamp(min=0) *
                        (decoded[:, 3] - decoded[:, 1]).clamp(min=0))
                    lvl = torch.floor(torch.log2(
                        (scale / head.roi_extractor.finest_scale).clamp(min=1e-6)))
                    lvl = lvl.clamp(0, len(head.roi_extractor.roi_layers) - 1).long()
                    levels.append(lvl)

                for step in range(1, len(levels)):
                    if levels[step].shape == levels[step - 1].shape:
                        level_switch[step].append(float(
                            (levels[step] != levels[step - 1]).float().mean()))

                initial_entry = captures[0]
                initial_boxes = head._sampler.raw_to_xyxy(
                    initial_entry['x_raw'], initial_entry['img_metas'])[0]
                initial_deg = (
                    (initial_boxes[:, 2] - initial_boxes[:, 0] <= 0) |
                    (initial_boxes[:, 3] - initial_boxes[:, 1] <= 0))
                final_boxes = captures[-1]['pred_bboxes'][0]
                gt = tensor_boxes(samples[0].gt_instances.bboxes).to(final_boxes.device)
                best_iou = paired_max_iou(final_boxes, gt)
                final_score = torch.sigmoid(captures[-1]['cls_logits'][0]).max(dim=-1).values
                for label, mask in (('initial_degenerate', initial_deg),
                                    ('initial_healthy', ~initial_deg)):
                    if mask.any():
                        conditional[f'{label}_iou'].extend(best_iou[mask].cpu().tolist())
                        conditional[f'{label}_score'].extend(final_score[mask].cpu().tolist())
                        conditional[f'{label}_iou50'].extend(
                            (best_iou[mask] >= 0.5).float().cpu().tolist())

                scalar['initial_degenerate_count'].append(float(initial_deg.sum()))
                scalar['initial_degenerate_ratio'].append(float(initial_deg.float().mean()))
                n_processed += 1
                if n_processed % 10 == 0:
                    print(f'[{n_processed}/{args.num_images}]', flush=True)
    finally:
        head._forward_at_t = original_forward

    result = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'seed': args.seed,
        'renewal': bool(head.box_renewal),
        'box_parameterization': head.box_parameterization,
        'snr_scale': snr,
        'n_images': n_processed,
        'summary': {k: describe(v) for k, v in sorted(scalar.items())},
        'per_step': {
            str(step): {k: describe(v) for k, v in sorted(stats.items())}
            for step, stats in sorted(per_step.items())
        },
        'fpn_level_switch': {
            str(step): describe(v) for step, v in sorted(level_switch.items())
        },
        'conditional_final_quality': {
            k: describe(v) for k, v in sorted(conditional.items())
        },
        'raw_moments': {
            k: describe(v) for k, v in sorted(raw_moments.items())
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
