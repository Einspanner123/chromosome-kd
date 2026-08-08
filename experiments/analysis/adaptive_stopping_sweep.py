#!/usr/bin/env python3
"""Evaluate GACS thresholds with deterministic noise and actual NFE counts."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def evaluate(args, geo_threshold):
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS, METRICS

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = Config.fromfile(args.config)
    head_cfg = cfg.model.bbox_head
    head_cfg.adaptive_stop_geo_threshold = geo_threshold
    head_cfg.adaptive_stop_cls_threshold = args.cls_threshold
    head_cfg.adaptive_stop_score_threshold = args.score_threshold
    head_cfg.adaptive_stop_min_box_scale = args.min_box_scale
    head_cfg.adaptive_stop_topk = args.topk

    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    evaluator = METRICS.build(dict(
        type='CocoMetric', ann_file=args.ann, metric='bbox',
        classwise=False, format_only=False))
    evaluator.dataset_meta = dataset.metainfo

    latencies, steps, geo, cls, scores, box_scales = [], [], [], [], [], []
    for i in range(len(dataset)):
        data = dataset[i]
        data['inputs'] = data['inputs'].unsqueeze(0).to(args.device)
        if not isinstance(data['data_samples'], list):
            data['data_samples'] = [data['data_samples']]
        if 'cuda' in args.device:
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = model.test_step(data)
        if 'cuda' in args.device:
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)

        stat = model.bbox_head._adaptive_stop_stats
        steps.append(stat['actual_steps'])
        geo.extend(stat['geo_residual'])
        cls.extend(stat['cls_residual'])
        scores.extend(stat['mean_topk_score'])
        box_scales.extend(stat['lower_box_scale'])

        eval_samples = []
        for j, result in enumerate(out if isinstance(out, list) else [out]):
            pred = result.pred_instances
            eval_samples.append(dict(
                pred_instances=dict(
                    bboxes=pred.bboxes.cpu(), scores=pred.scores.cpu(),
                    labels=pred.labels.cpu()),
                img_id=getattr(result, 'img_id', i + j),
                ori_shape=getattr(result, 'ori_shape', (1, 1))))
        evaluator.process({}, eval_samples)
        if (i + 1) % 100 == 0:
            print(f'  threshold={geo_threshold:g}: {i + 1}/{len(dataset)}')

    metrics = evaluator.evaluate(len(dataset))
    timed = np.asarray(latencies[10:] if len(latencies) > 10 else latencies)
    step_array = np.asarray(steps)
    return dict(
        geo_threshold=geo_threshold,
        cls_threshold=args.cls_threshold,
        score_threshold=args.score_threshold,
        min_box_scale=args.min_box_scale,
        mAP=float(metrics['coco/bbox_mAP']),
        AP50=float(metrics['coco/bbox_mAP_50']),
        AP75=float(metrics['coco/bbox_mAP_75']),
        APs=float(metrics['coco/bbox_mAP_s']),
        APm=float(metrics['coco/bbox_mAP_m']),
        APl=float(metrics['coco/bbox_mAP_l']),
        early_exit_rate=float(np.mean(step_array == 1)),
        mean_steps=float(step_array.mean()),
        avg_latency_ms=float(timed.mean() * 1000),
        p99_latency_ms=float(np.percentile(timed, 99) * 1000),
        geo_quantiles=np.quantile(geo, [0, .1, .25, .5, .75, .9, 1]).tolist(),
        cls_quantiles=np.quantile(cls, [0, .1, .25, .5, .75, .9, 1]).tolist(),
        score_quantiles=np.quantile(scores, [0, .1, .25, .5, .75, .9, 1]).tolist(),
        box_scale_quantiles=np.quantile(
            box_scales, [0, .1, .25, .5, .75, .9, 1]).tolist(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann', required=True)
    parser.add_argument('--geo-thresholds', default='0,0.02,0.05,0.1,100')
    parser.add_argument('--cls-threshold', type=float, default=1.0)
    parser.add_argument('--score-threshold', type=float, default=0.0)
    parser.add_argument('--min-box-scale', type=float, default=0.0)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    results = []
    for threshold in [float(x) for x in args.geo_thresholds.split(',')]:
        result = evaluate(args, threshold)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
    with open(args.output, 'w') as handle:
        json.dump(dict(config=args.config, checkpoint=args.checkpoint,
                       seed=args.seed, results=results), handle, indent=2)


if __name__ == '__main__':
    main()
