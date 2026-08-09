#!/usr/bin/env python3
"""Phase-0 test for conditional geometric information in cascade attention."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_meta(sample):
    from ldmdet.data.structures import ImageMeta
    meta = sample.metainfo
    return ImageMeta(
        img_shape=meta['img_shape'], pad_shape=meta.get('pad_shape'),
        ori_shape=meta.get('ori_shape'), scale_factor=meta.get('scale_factor'),
        img_id=meta.get('img_id'))


def pairwise_geometry(boxes):
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1e-4)
    ctr = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    delta = ctr[:, None, :] - ctr[None, :, :]
    scale = torch.sqrt(wh[:, None, :] * wh[None, :, :]).clamp(min=1e-4)
    normalized_delta = delta.abs() / scale
    log_ratio = (wh[:, None, :] / wh[None, :, :]).log().abs()
    lt = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    rb = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    inter_wh = (rb - lt).clamp(min=0)
    intersection = inter_wh[..., 0] * inter_wh[..., 1]
    area = wh[:, 0] * wh[:, 1]
    union = area[:, None] + area[None, :] - intersection
    iou = intersection / union.clamp(min=1e-6)
    distance = torch.sqrt(normalized_delta.square().sum(-1) + 1e-8)
    area_ratio = (area[:, None] / area[None, :]).log().abs()
    return torch.cat((
        normalized_delta, log_ratio,
        iou.unsqueeze(-1), distance.unsqueeze(-1), area_ratio.unsqueeze(-1),
    ), dim=-1)


def auc_score(labels, scores):
    labels = labels.astype(bool)
    positives = scores[labels]
    negatives = scores[~labels]
    if not len(positives) or not len(negatives):
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties.
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = ranks[labels].sum()
    return float((rank_sum - len(positives) * (len(positives) + 1) / 2)
                 / (len(positives) * len(negatives)))


def ridge_score(train_x, train_y, test_x, alpha=10.0):
    mean = train_x.mean(0, keepdims=True)
    std = train_x.std(0, keepdims=True)
    std[std < 1e-6] = 1.0
    train = (train_x - mean) / std
    test = (test_x - mean) / std
    train = np.concatenate((train, np.ones((len(train), 1))), 1)
    test = np.concatenate((test, np.ones((len(test), 1))), 1)
    penalty = np.eye(train.shape[1]) * alpha
    penalty[-1, -1] = 0
    weight = np.linalg.solve(train.T @ train + penalty, train.T @ train_y)
    return test @ weight


def grouped_cv(attention, geometry, labels, groups, folds=5, alpha=10.0):
    features = {
        'attention_only': attention[:, None],
        'geometry_only': geometry,
        'attention_plus_geometry': np.concatenate((attention[:, None], geometry), 1),
    }
    predictions = {name: np.zeros(len(labels), dtype=np.float64)
                   for name in features}
    unique = np.unique(groups)
    target = labels.astype(np.float64)
    for fold in range(folds):
        test = np.isin(groups, unique[fold::folds])
        train = ~test
        for name, values in features.items():
            predictions[name][test] = ridge_score(
                values[train], target[train], values[test], alpha)
    result = {name: {'auc': auc_score(labels, values)}
              for name, values in predictions.items()}
    result['raw_attention'] = {'auc': auc_score(labels, attention)}
    result['incremental'] = {
        'auc_gain_vs_attention': float(
            result['attention_plus_geometry']['auc']
            - result['attention_only']['auc']),
    }
    return result


def run(args):
    from ldmdet.data.structures import ModelOutput
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS
    from torchvision.ops import box_iou

    set_seed(args.seed)
    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    detector_head = model.bbox_head
    head = detector_head.head_series[-1]
    head.use_sdpa = False
    capture = {'attention': [], 'head': []}

    def attention_hook(_module, _inputs, output):
        if output[1] is not None:
            capture['attention'].append(output[1].detach())

    def head_hook(_module, inputs, output):
        capture['head'].append((
            inputs[1].detach(), output[0].detach(), output[1].detach()))

    handles = [
        head.self_attn.register_forward_hook(attention_hook),
        head.register_forward_hook(head_hook),
    ]
    records = {key: [] for key in ('attention', 'geometry', 'label', 'group', 'overlap')}
    rng = np.random.default_rng(args.seed)
    limit = min(len(dataset), args.max_images) if args.max_images else len(dataset)
    try:
        for image_index in range(limit):
            capture['attention'].clear()
            capture['head'].clear()
            data = dataset[image_index]
            data['inputs'] = data['inputs'].unsqueeze(0).to(args.device)
            if not isinstance(data['data_samples'], list):
                data['data_samples'] = [data['data_samples']]
            sample = data['data_samples'][0]
            model.test_step(data)
            if not capture['attention'] or not capture['head']:
                continue
            attention = capture['attention'][-1][0]
            input_boxes, logits, pred_boxes = capture['head'][-1]
            input_boxes = input_boxes[0]
            meta = build_meta(sample)
            gt_boxes = sample.gt_instances.bboxes
            if hasattr(gt_boxes, 'tensor'):
                gt_boxes = gt_boxes.tensor
            gt_boxes = gt_boxes.to(args.device)
            gt_labels = sample.gt_instances.labels.to(args.device)
            targets = detector_head._normalize_targets(
                [gt_boxes], [gt_labels], [meta], 1)
            norm_boxes = detector_head._normalize_pred_bboxes(
                pred_boxes.unsqueeze(0), [meta])[-1]
            foreground, matched = detector_head.criterion.matcher(
                ModelOutput(pred_logits=logits, pred_boxes=norm_boxes), targets)[0]
            indices = foreground.nonzero(as_tuple=False).flatten()
            if len(indices) < 3:
                continue
            assigned = matched[indices]
            selected_output = pred_boxes[0, indices]
            aligned = box_iou(selected_output, gt_boxes)[
                torch.arange(len(indices), device=indices.device), assigned]
            valid = aligned >= 0.30
            indices = indices[valid]
            assigned = assigned[valid]
            if len(indices) < 3:
                continue
            selected_boxes = input_boxes[indices]
            geometry = pairwise_geometry(selected_boxes)
            selected_attention = attention[indices][:, indices].clamp(min=1e-12).log()
            same = assigned[:, None] == assigned[None, :]
            diagonal = torch.eye(len(indices), dtype=torch.bool, device=indices.device)
            positive = (same & ~diagonal).nonzero(as_tuple=False)
            negative = (~same).nonzero(as_tuple=False)
            if not len(positive) or not len(negative):
                continue
            count = min(len(positive), len(negative), args.max_pairs_per_class)
            pos_choice = rng.choice(len(positive), count, replace=False)
            neg_choice = rng.choice(len(negative), count, replace=False)
            pairs = torch.cat((positive[pos_choice], negative[neg_choice]), 0)
            labels = np.concatenate((np.ones(count, dtype=bool),
                                     np.zeros(count, dtype=bool)))

            if len(gt_boxes) > 1:
                overlap_matrix = box_iou(gt_boxes, gt_boxes)
                overlap_matrix.fill_diagonal_(0)
                gt_overlap = overlap_matrix.max(1).values
            else:
                gt_overlap = gt_boxes.new_zeros(len(gt_boxes))
            pair_overlap = torch.maximum(
                gt_overlap[assigned[pairs[:, 0]]],
                gt_overlap[assigned[pairs[:, 1]]])
            records['attention'].append(
                selected_attention[pairs[:, 0], pairs[:, 1]].cpu().numpy())
            records['geometry'].append(
                geometry[pairs[:, 0], pairs[:, 1]].cpu().numpy())
            records['label'].append(labels)
            records['group'].append(np.full(2 * count, image_index))
            records['overlap'].append(pair_overlap.cpu().numpy())
            if (image_index + 1) % 50 == 0:
                print(f'{image_index + 1}/{limit}', flush=True)
    finally:
        for handle in handles:
            handle.remove()

    arrays = {key: np.concatenate(values) for key, values in records.items()}
    overall = grouped_cv(
        arrays['attention'], arrays['geometry'], arrays['label'], arrays['group'],
        args.folds, args.alpha)
    overlap_mask = arrays['overlap'] >= 0.20
    overlap = None
    if overlap_mask.sum() >= 200 and len(np.unique(arrays['group'][overlap_mask])) >= args.folds:
        overlap = grouped_cv(
            arrays['attention'][overlap_mask], arrays['geometry'][overlap_mask],
            arrays['label'][overlap_mask], arrays['group'][overlap_mask],
            args.folds, args.alpha)
    augmented_auc = overall['attention_plus_geometry']['auc']
    gain = overall['incremental']['auc_gain_vs_attention']
    return {
        'config': args.config, 'checkpoint': args.checkpoint, 'seed': args.seed,
        'num_images': limit, 'num_pairs': int(len(arrays['label'])),
        'positive_fraction': float(arrays['label'].mean()),
        'overall': overall, 'overlap_ge_0.20': overlap,
        'gate': {
            'required_augmented_auc': 0.70,
            'required_auc_gain_vs_attention': 0.10,
            'pass': bool(augmented_auc >= 0.70 and gain >= 0.10),
        },
        'definition': (
            'Balanced directed proposal pairs from the final solver call and '
            'last cascade head; positives share a dynamic-matcher GT, and all '
            'proposals have aligned output IoU >= 0.30.'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-images', type=int, default=200)
    parser.add_argument('--max-pairs-per-class', type=int, default=100)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=10.0)
    args = parser.parse_args()
    result = run(args)
    with open(args.output, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
