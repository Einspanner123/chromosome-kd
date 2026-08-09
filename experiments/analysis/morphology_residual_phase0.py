#!/usr/bin/env python3
"""Phase-0 conditional-information test for an axis-aware RoI branch.

The diagnostic asks a narrow question: after conditioning on the final
cascade feature, do explicit spatial moments and principal-axis profiles still
predict held-out endpoint box residuals?  Images, rather than proposals, form
the cross-validation groups to prevent within-image leakage.
"""

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


def image_meta(sample):
    from ldmdet.data.structures import ImageMeta

    meta = sample.metainfo
    return ImageMeta(
        img_shape=meta['img_shape'], pad_shape=meta.get('pad_shape'),
        ori_shape=meta.get('ori_shape'), scale_factor=meta.get('scale_factor'),
        img_id=meta.get('img_id'))


def morphology_descriptors(roi_features, num_bins=7, sigma=0.30):
    """Return sign-invariant moment/profile descriptors for 7x7 RoIs."""
    features = roi_features.float()
    n, _, height, width = features.shape
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=features.device),
        torch.linspace(-1, 1, width, device=features.device), indexing='ij')
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    flat = features.flatten(2)
    energy = flat.square().mean(1)
    probability = (energy + 1e-6) / (energy.sum(1, keepdim=True) + 1e-6)
    mu_x = (probability * x).sum(1, keepdim=True)
    mu_y = (probability * y).sum(1, keepdim=True)
    dx = x - mu_x
    dy = y - mu_y
    cov_xx = (probability * dx.square()).sum(1, keepdim=True)
    cov_yy = (probability * dy.square()).sum(1, keepdim=True)
    cov_xy = (probability * dx * dy).sum(1, keepdim=True)
    cos2 = cov_xx - cov_yy
    sin2 = 2 * cov_xy
    anisotropy = torch.sqrt(cos2.square() + sin2.square() + 1e-8)
    eccentricity = anisotropy / (cov_xx + cov_yy + 1e-6)
    theta = 0.5 * torch.atan2(sin2, cos2)
    axis_x, axis_y = theta.cos(), theta.sin()
    axial = axis_x * dx + axis_y * dy
    transverse = -axis_y * dx + axis_x * dy

    centers = torch.linspace(-1, 1, num_bins, device=features.device)

    def profiles(coordinate):
        weights = torch.exp(
            -0.5 * ((coordinate.unsqueeze(-1) - centers) / sigma).square())
        weights = weights / (weights.sum(1, keepdim=True) + 1e-6)
        saliency = (probability.unsqueeze(-1) * weights).sum(1)
        feature_mean = flat.mean(1)
        feature_rms = flat.square().mean(1).sqrt()
        mean_profile = (feature_mean.unsqueeze(-1) * weights).sum(1)
        rms_profile = (feature_rms.unsqueeze(-1) * weights).sum(1)
        # Principal-axis sign is unidentifiable.  Pairing a profile with its
        # reversal by sum and absolute difference preserves morphology while
        # making the descriptor exactly invariant to e -> -e.
        values = torch.cat((saliency, mean_profile, rms_profile), dim=1)
        reverse = values.flip(1)
        return torch.cat((values + reverse, (values - reverse).abs()), dim=1)

    moments = torch.cat((
        mu_x, mu_y, cov_xx, cov_yy, cov_xy,
        eccentricity, cos2, sin2,
    ), dim=1)
    descriptor = torch.cat((moments, profiles(axial), profiles(transverse)), 1)
    assert descriptor.shape[0] == n
    return torch.nan_to_num(descriptor)


def box_residuals(pred_boxes, target_boxes):
    pred_wh = (pred_boxes[:, 2:] - pred_boxes[:, :2]).clamp(min=1e-4)
    target_wh = (target_boxes[:, 2:] - target_boxes[:, :2]).clamp(min=1e-4)
    pred_ctr = (pred_boxes[:, :2] + pred_boxes[:, 2:]) * 0.5
    target_ctr = (target_boxes[:, :2] + target_boxes[:, 2:]) * 0.5
    return torch.cat((
        (target_ctr - pred_ctr) / pred_wh,
        torch.log(target_wh / pred_wh),
    ), dim=1)


def apply_residuals(boxes, residuals):
    wh = np.maximum(boxes[:, 2:] - boxes[:, :2], 1e-6)
    ctr = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    new_ctr = ctr + residuals[:, :2] * wh
    new_wh = np.exp(np.clip(residuals[:, 2:], -2.0, 2.0)) * wh
    return np.concatenate((new_ctr - 0.5 * new_wh, new_ctr + 0.5 * new_wh), 1)


def aligned_iou(boxes, targets):
    lt = np.maximum(boxes[:, :2], targets[:, :2])
    rb = np.minimum(boxes[:, 2:], targets[:, 2:])
    inter_wh = np.maximum(rb - lt, 0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    box_wh = np.maximum(boxes[:, 2:] - boxes[:, :2], 0)
    target_wh = np.maximum(targets[:, 2:] - targets[:, :2], 0)
    union = (box_wh[:, 0] * box_wh[:, 1]
             + target_wh[:, 0] * target_wh[:, 1] - inter)
    return inter / np.maximum(union, 1e-8)


def ridge_predict(train_x, train_y, test_x, alpha=10.0):
    mean = train_x.mean(0, keepdims=True)
    std = train_x.std(0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (train_x - mean) / std
    x_test = (test_x - mean) / std
    x_train = np.concatenate((x_train, np.ones((len(x_train), 1))), 1)
    x_test = np.concatenate((x_test, np.ones((len(x_test), 1))), 1)
    gram = x_train.T @ x_train
    penalty = np.eye(gram.shape[0]) * alpha
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(gram + penalty, x_train.T @ train_y)
    return x_test @ weights


def summarize_predictions(pred, target, boxes, target_boxes):
    corrected = apply_residuals(boxes, pred)
    before_iou = aligned_iou(boxes, target_boxes)
    after_iou = aligned_iou(corrected, target_boxes)
    result = {
        'residual_mse': float(np.mean((pred - target) ** 2)),
        'mean_iou_before': float(before_iou.mean()),
        'mean_iou_after': float(after_iou.mean()),
        'mean_iou_delta': float((after_iou - before_iou).mean()),
    }
    for threshold in (0.75, 0.85, 0.90, 0.95):
        result[f'pass_{threshold:.2f}_before'] = float((before_iou >= threshold).mean())
        result[f'pass_{threshold:.2f}_after'] = float((after_iou >= threshold).mean())
    return result


def grouped_cross_validation(fc, morph, target, boxes, target_boxes, groups,
                             folds=5, alpha=10.0):
    unique_groups = np.unique(groups)
    predictions = {
        'zero': np.zeros_like(target),
        'fc_only': np.zeros_like(target),
        'fc_plus_morphology': np.zeros_like(target),
    }
    for fold in range(folds):
        test_groups = unique_groups[fold::folds]
        test_mask = np.isin(groups, test_groups)
        train_mask = ~test_mask
        predictions['fc_only'][test_mask] = ridge_predict(
            fc[train_mask], target[train_mask], fc[test_mask], alpha)
        combined = np.concatenate((fc, morph), 1)
        predictions['fc_plus_morphology'][test_mask] = ridge_predict(
            combined[train_mask], target[train_mask], combined[test_mask], alpha)
    summaries = {
        name: summarize_predictions(value, target, boxes, target_boxes)
        for name, value in predictions.items()
    }
    base = summaries['fc_only']['residual_mse']
    augmented = summaries['fc_plus_morphology']['residual_mse']
    summaries['incremental'] = {
        'mse_reduction_fraction_vs_fc': float((base - augmented) / max(base, 1e-12)),
        'mean_iou_gain_vs_fc': float(
            summaries['fc_plus_morphology']['mean_iou_after']
            - summaries['fc_only']['mean_iou_after']),
    }
    return summaries


def run(args):
    from ldmdet.data.structures import ModelOutput
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS

    set_seed(args.seed)
    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    detector_head = model.bbox_head
    final_head = detector_head.head_series[-1]
    capture = {'roi': [], 'output': []}

    def dynamic_hook(_module, inputs, _output):
        roi = inputs[1].detach()
        n, channels = roi.shape[1], roi.shape[2]
        capture['roi'].append(
            roi.permute(1, 2, 0).reshape(n, channels, 7, 7))

    def head_hook(_module, _inputs, output):
        capture['output'].append(tuple(value.detach() for value in output[:3]))

    handles = [
        final_head.inst_interact.register_forward_hook(dynamic_hook),
        final_head.register_forward_hook(head_hook),
    ]
    collected = {key: [] for key in (
        'fc', 'morph', 'target', 'boxes', 'target_boxes', 'groups',
        'area', 'overlap')}
    limit = min(len(dataset), args.max_images) if args.max_images else len(dataset)
    try:
        for image_index in range(limit):
            capture['roi'].clear()
            capture['output'].clear()
            data = dataset[image_index]
            data['inputs'] = data['inputs'].unsqueeze(0).to(args.device)
            if not isinstance(data['data_samples'], list):
                data['data_samples'] = [data['data_samples']]
            sample = data['data_samples'][0]
            model.test_step(data)
            if not capture['roi'] or not capture['output']:
                continue
            roi = capture['roi'][-1]
            logits, pred_boxes, fc = capture['output'][-1]
            meta = image_meta(sample)
            gt_boxes = sample.gt_instances.bboxes
            if hasattr(gt_boxes, 'tensor'):
                gt_boxes = gt_boxes.tensor
            gt_boxes = gt_boxes.to(args.device)
            gt_labels = sample.gt_instances.labels.to(args.device)
            targets = detector_head._normalize_targets(
                [gt_boxes], [gt_labels], [meta], 1)
            norm_boxes = detector_head._normalize_pred_bboxes(
                pred_boxes.unsqueeze(0), [meta])[-1]
            indices = detector_head.criterion.matcher(
                ModelOutput(pred_logits=logits, pred_boxes=norm_boxes), targets)[0]
            foreground, matched_indices = indices
            if not foreground.any():
                continue
            matched = matched_indices[foreground]
            selected_gt = gt_boxes[matched]
            selected_boxes = pred_boxes[0, foreground]
            lt = torch.maximum(selected_boxes[:, :2], selected_gt[:, :2])
            rb = torch.minimum(selected_boxes[:, 2:], selected_gt[:, 2:])
            inter_wh = (rb - lt).clamp(min=0)
            intersection = inter_wh[:, 0] * inter_wh[:, 1]
            pred_wh = (selected_boxes[:, 2:] - selected_boxes[:, :2]).clamp(min=0)
            gt_wh = (selected_gt[:, 2:] - selected_gt[:, :2]).clamp(min=0)
            union = (pred_wh[:, 0] * pred_wh[:, 1]
                     + gt_wh[:, 0] * gt_wh[:, 1] - intersection)
            aligned = intersection / union.clamp(min=1e-6)
            endpoint = ((aligned >= 0.50)
                        & (pred_wh[:, 0] >= 1.0)
                        & (pred_wh[:, 1] >= 1.0))
            if not endpoint.any():
                continue
            selected_gt = selected_gt[endpoint]
            selected_boxes = selected_boxes[endpoint]
            selected_roi = roi[foreground][endpoint]
            selected_fc = fc.reshape(-1, fc.shape[-1])[foreground][endpoint]
            matched = matched[endpoint]
            residual = box_residuals(selected_boxes, selected_gt)

            # GT overlap is stored for stratified interpretation.
            if len(gt_boxes) > 1:
                from torchvision.ops import box_iou
                overlap_matrix = box_iou(gt_boxes, gt_boxes)
                overlap_matrix.fill_diagonal_(0)
                gt_overlap = overlap_matrix.max(1).values
            else:
                gt_overlap = gt_boxes.new_zeros(len(gt_boxes))
            gt_area = ((gt_boxes[:, 2] - gt_boxes[:, 0])
                       * (gt_boxes[:, 3] - gt_boxes[:, 1]))

            collected['fc'].append(selected_fc.cpu().float().numpy())
            collected['morph'].append(
                morphology_descriptors(selected_roi).cpu().numpy())
            collected['target'].append(residual.cpu().numpy())
            collected['boxes'].append(selected_boxes.cpu().numpy())
            collected['target_boxes'].append(selected_gt.cpu().numpy())
            count = int(endpoint.sum())
            collected['groups'].append(np.full(count, image_index))
            collected['area'].append(gt_area[matched].cpu().numpy())
            collected['overlap'].append(gt_overlap[matched].cpu().numpy())
            if (image_index + 1) % 25 == 0:
                print(f'{image_index + 1}/{limit}', flush=True)
    finally:
        for handle in handles:
            handle.remove()

    arrays = {key: np.concatenate(values, axis=0)
              for key, values in collected.items()}
    overall = grouped_cross_validation(
        arrays['fc'], arrays['morph'], arrays['target'], arrays['boxes'],
        arrays['target_boxes'], arrays['groups'], args.folds, args.alpha)
    strata = {}
    masks = {
        'small': arrays['area'] < 32 ** 2,
        'medium_or_large': arrays['area'] >= 32 ** 2,
        'overlap_ge_0.20': arrays['overlap'] >= 0.20,
        'overlap_lt_0.20': arrays['overlap'] < 0.20,
    }
    # Reuse predictions would be ideal for strata, but per-stratum grouped CV
    # tests the stronger claim that morphology is independently useful there.
    for name, mask in masks.items():
        if mask.sum() < 100 or len(np.unique(arrays['groups'][mask])) < args.folds:
            continue
        strata[name] = grouped_cross_validation(
            arrays['fc'][mask], arrays['morph'][mask], arrays['target'][mask],
            arrays['boxes'][mask], arrays['target_boxes'][mask],
            arrays['groups'][mask], args.folds, args.alpha)

    return {
        'config': args.config, 'checkpoint': args.checkpoint,
        'seed': args.seed, 'num_images': limit,
        'num_matched_proposals': int(len(arrays['target'])),
        'fc_dim': int(arrays['fc'].shape[1]),
        'morphology_dim': int(arrays['morph'].shape[1]),
        'ridge_alpha': args.alpha, 'folds': args.folds,
        'overall': overall, 'strata': strata,
        'gate': {
            'required_mse_reduction_fraction': 0.05,
            'required_mean_iou_gain': 0.002,
            'pass': bool(
                overall['incremental']['mse_reduction_fraction_vs_fc'] >= 0.05
                and overall['incremental']['mean_iou_gain_vs_fc'] >= 0.002),
        },
        'interpretation': (
            'A pass only establishes conditional endpoint information. It does '
            'not establish mAP gain; the branch must still be trained from the '
            'same initialization and compared against a structural placebo.'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-images', type=int, default=100)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=10.0)
    args = parser.parse_args()
    result = run(args)
    with open(args.output, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({
        'num_matched_proposals': result['num_matched_proposals'],
        'overall': result['overall'], 'gate': result['gate'],
    }, indent=2))


if __name__ == '__main__':
    main()
