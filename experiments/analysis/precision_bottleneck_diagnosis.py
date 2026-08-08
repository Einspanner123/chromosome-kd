#!/usr/bin/env python3
"""Decompose LDMDet accuracy bottlenecks on a COCO-format validation set.

The script performs one deterministic inference pass and reports:
1. AP at every IoU threshold from 0.50 to 0.95;
2. classification, localization, and joint oracle upper bounds;
3. greedy error decomposition at IoU 0.50/0.75/0.90;
4. recall stratified by GT overlap and object area;
5. AP for every solver-step x cascade-head intermediate prediction.

Oracle definitions are deliberately explicit and conservative enough to be
reproducible.  They are upper-bound diagnostics, not deployable algorithms.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.asarray([x, y, x + w, y + h], dtype=np.float64)


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(max(0.0, x2 - x1)),
            float(max(0.0, y2 - y1))]


def pairwise_iou(boxes1, boxes2):
    boxes1 = np.asarray(boxes1, dtype=np.float64).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float64).reshape(-1, 4)
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = np.maximum(boxes1[:, 2] - boxes1[:, 0], 0.0) * np.maximum(
        boxes1[:, 3] - boxes1[:, 1], 0.0)
    area2 = np.maximum(boxes2[:, 2] - boxes2[:, 0], 0.0) * np.maximum(
        boxes2[:, 3] - boxes2[:, 1], 0.0)
    return inter / np.maximum(area1[:, None] + area2[None, :] - inter, 1e-12)


def coco_metrics(coco_gt, predictions):
    from pycocotools.cocoeval import COCOeval

    if not predictions:
        return {'mAP': 0.0, 'AP50': 0.0, 'AP75': 0.0, 'APs': 0.0,
                'APm': 0.0, 'APl': 0.0, 'AP_by_iou': {}}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
        evaluator.params.maxDets = [100, 300, 1000]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval['precision'][:, :, :, 0, -1]
    by_iou = {}
    for index, threshold in enumerate(evaluator.params.iouThrs):
        values = precision[index]
        values = values[values > -1]
        by_iou[f'{threshold:.2f}'] = float(values.mean()) if values.size else 0.0
    stats = evaluator.stats
    return {
        'mAP': float(stats[0]), 'AP50': float(stats[1]),
        'AP75': float(stats[2]), 'APs': float(stats[3]),
        'APm': float(stats[4]), 'APl': float(stats[5]),
        'AP_by_iou': by_iou,
    }


def subset_coco(coco_gt, image_ids):
    from pycocotools.coco import COCO

    selected = set(image_ids)
    dataset = copy.deepcopy(coco_gt.dataset)
    dataset['images'] = [x for x in dataset['images'] if x['id'] in selected]
    dataset['annotations'] = [
        x for x in dataset['annotations'] if x['image_id'] in selected]
    result = COCO()
    result.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        result.createIndex()
    return result


def gt_by_image(coco_gt):
    grouped = defaultdict(list)
    for ann in coco_gt.dataset['annotations']:
        if ann.get('iscrowd', 0):
            continue
        item = dict(ann)
        item['xyxy'] = xywh_to_xyxy(ann['bbox'])
        item['area'] = float(ann.get('area', ann['bbox'][2] * ann['bbox'][3]))
        grouped[ann['image_id']].append(item)
    return grouped


def pred_by_image(predictions):
    grouped = defaultdict(list)
    for pred in predictions:
        item = dict(pred)
        item['xyxy'] = xywh_to_xyxy(pred['bbox'])
        grouped[pred['image_id']].append(item)
    for values in grouped.values():
        values.sort(key=lambda x: x['score'], reverse=True)
    return grouped


def build_oracles(predictions, grouped_gt, min_iou=0.10):
    variants = {'classification': [], 'localization': [], 'joint': []}
    for pred in predictions:
        gts = grouped_gt.get(pred['image_id'], [])
        base = dict(pred)
        if not gts:
            for values in variants.values():
                values.append(dict(base))
            continue
        ious = pairwise_iou([xywh_to_xyxy(pred['bbox'])],
                            [gt['xyxy'] for gt in gts])[0]
        best_any = int(np.argmax(ious))
        same = [i for i, gt in enumerate(gts)
                if gt['category_id'] == pred['category_id']]
        best_same = max(same, key=lambda i: ious[i]) if same else None

        cls_pred = dict(base)
        if ious[best_any] >= min_iou:
            cls_pred['category_id'] = int(gts[best_any]['category_id'])
        variants['classification'].append(cls_pred)

        loc_pred = dict(base)
        if best_same is not None and ious[best_same] >= min_iou:
            loc_pred['bbox'] = list(map(float, gts[best_same]['bbox']))
        variants['localization'].append(loc_pred)

        joint_pred = dict(base)
        if ious[best_any] >= min_iou:
            joint_pred['category_id'] = int(gts[best_any]['category_id'])
            joint_pred['bbox'] = list(map(float, gts[best_any]['bbox']))
        variants['joint'].append(joint_pred)
    return variants


def build_quality_oracles(predictions, grouped_gt, betas=(0.25, 0.5, 1.0, 2.0)):
    """Re-rank fixed detections using their true same-class localization IoU.

    This isolates an ideal IoU-quality estimator: classes and coordinates stay
    fixed. Multiplicative fusion mirrors ``p(class) * q(IoU) ** beta``.
    """
    variants = {f'quality_beta_{beta:g}': [] for beta in betas}
    variants['quality_only'] = []
    for pred in predictions:
        gts = grouped_gt.get(pred['image_id'], [])
        same = [gt for gt in gts
                if gt['category_id'] == pred['category_id']]
        quality = 0.0
        if same:
            quality = float(pairwise_iou(
                [xywh_to_xyxy(pred['bbox'])],
                [gt['xyxy'] for gt in same])[0].max())
        quality = min(max(quality, 0.0), 1.0)
        for beta in betas:
            item = dict(pred)
            item['score'] = float(pred['score']) * quality ** beta
            variants[f'quality_beta_{beta:g}'].append(item)
        item = dict(pred)
        item['score'] = quality
        variants['quality_only'].append(item)
    return variants


def greedy_error_counts(predictions, grouped_gt, threshold):
    grouped_pred = pred_by_image(predictions)
    counts = defaultdict(int)
    for image_id, gts in grouped_gt.items():
        preds = grouped_pred.get(image_id, [])
        matched = set()
        gt_boxes = [gt['xyxy'] for gt in gts]
        for pred in preds:
            if not gts:
                counts['background'] += 1
                continue
            ious = pairwise_iou([pred['xyxy']], gt_boxes)[0]
            same = [i for i, gt in enumerate(gts)
                    if gt['category_id'] == pred['category_id']]
            same_unmatched = [i for i in same if i not in matched]
            best_same = max(same_unmatched, key=lambda i: ious[i]) \
                if same_unmatched else None
            if best_same is not None and ious[best_same] >= threshold:
                matched.add(best_same)
                counts['correct'] += 1
                continue
            best_any = int(np.argmax(ious))
            if ious[best_any] >= threshold:
                if best_any in matched:
                    counts['duplicate'] += 1
                else:
                    counts['classification'] += 1
                continue
            best_same_all = max(same, key=lambda i: ious[i]) if same else None
            if best_same_all is not None and ious[best_same_all] >= 0.10:
                counts['localization'] += 1
            else:
                counts['background'] += 1
        counts['missed_gt'] += len(gts) - len(matched)
    return dict(counts)


def annotate_gt_strata(grouped_gt):
    for gts in grouped_gt.values():
        boxes = [gt['xyxy'] for gt in gts]
        matrix = pairwise_iou(boxes, boxes)
        if len(gts):
            np.fill_diagonal(matrix, 0.0)
        for index, gt in enumerate(gts):
            overlap = float(matrix[index].max()) if len(gts) > 1 else 0.0
            gt['max_other_iou'] = overlap
            if overlap < 0.01:
                gt['overlap_stratum'] = 'isolated_<0.01'
            elif overlap < 0.20:
                gt['overlap_stratum'] = 'near_0.01-0.20'
            else:
                gt['overlap_stratum'] = 'overlap_>=0.20'
            area = gt['area']
            gt['area_stratum'] = (
                'small' if area < 32 ** 2 else
                'medium' if area < 96 ** 2 else 'large')


def stratified_recall(predictions, grouped_gt):
    grouped_pred = pred_by_image(predictions)
    accum = defaultdict(lambda: defaultdict(list))
    confusion = defaultdict(lambda: defaultdict(int))
    for image_id, gts in grouped_gt.items():
        preds = grouped_pred.get(image_id, [])
        pred_boxes = [pred['xyxy'] for pred in preds]
        for gt in gts:
            ious = pairwise_iou([gt['xyxy']], pred_boxes)[0] \
                if pred_boxes else np.zeros(0)
            best_any = float(ious.max()) if ious.size else 0.0
            same = [i for i, pred in enumerate(preds)
                    if pred['category_id'] == gt['category_id']]
            best_same = max((float(ious[i]) for i in same), default=0.0)
            for key in (gt['overlap_stratum'], gt['area_stratum']):
                accum[key]['best_iou_any'].append(best_any)
                accum[key]['best_iou_same_class'].append(best_same)
            if best_any >= 0.50:
                pred_index = int(np.argmax(ious))
                confusion[int(gt['category_id'])][
                    int(preds[pred_index]['category_id'])] += 1
    result = {}
    for key, values in accum.items():
        any_iou = np.asarray(values['best_iou_any'])
        same_iou = np.asarray(values['best_iou_same_class'])
        result[key] = {
            'num_gt': int(len(any_iou)),
            'localization_recall@0.50': float(np.mean(any_iou >= 0.50)),
            'detection_recall@0.50': float(np.mean(same_iou >= 0.50)),
            'detection_recall@0.75': float(np.mean(same_iou >= 0.75)),
            'detection_recall@0.90': float(np.mean(same_iou >= 0.90)),
            'mean_best_same_class_iou': float(same_iou.mean()),
        }
    return result, {str(k): dict(v) for k, v in confusion.items()}


def result_to_coco(result, image_id, cat_ids):
    records = []
    boxes = result.bboxes.detach().cpu().numpy()
    scores = result.scores.detach().cpu().numpy()
    labels = result.labels.detach().cpu().numpy()
    for box, score, label in zip(boxes, scores, labels):
        records.append({
            'image_id': int(image_id),
            'category_id': int(cat_ids[int(label)]),
            'bbox': xyxy_to_xywh(box),
            'score': float(score),
        })
    return records


def build_image_meta(sample):
    from ldmdet.data.structures import ImageMeta

    meta = sample.metainfo
    return ImageMeta(
        img_shape=meta['img_shape'], pad_shape=meta.get('pad_shape'),
        ori_shape=meta.get('ori_shape'), scale_factor=meta.get('scale_factor'),
        img_id=meta.get('img_id'))


def run(args):
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS
    from pycocotools.coco import COCO

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    coco_gt = COCO(args.ann)
    # CocoDataset builds labels from ``cat_ids`` returned by COCO, whose order
    # is not necessarily the order of ``metainfo['classes']``.  Reconstructing
    # this mapping from class names silently corrupts labels when (for example)
    # C10 is listed before C6 in a config.  Reuse the dataset's authoritative
    # label-to-category mapping, exactly as CocoMetric does.
    cat_ids = list(dataset.cat_ids)
    valid_cat_ids = {int(cat['id']) for cat in coco_gt.dataset['categories']}
    if len(cat_ids) != len(dataset.metainfo['classes']):
        raise RuntimeError('CocoDataset cat_ids/classes length mismatch')
    if not set(map(int, cat_ids)).issubset(valid_cat_ids):
        raise RuntimeError('Validation dataset and --ann categories differ')

    traces = [[] for _ in model.bbox_head.head_series]
    handles = []
    for head_index, head in enumerate(model.bbox_head.head_series):
        def hook(_module, _inputs, output, index=head_index):
            traces[index].append((output[0].detach(), output[1].detach()))
        handles.append(head.register_forward_hook(hook))

    final_predictions = []
    stage_predictions = defaultdict(list)
    image_ids = []
    limit = min(len(dataset), args.max_images) if args.max_images else len(dataset)
    try:
        for index in range(limit):
            for values in traces:
                values.clear()
            data = dataset[index]
            data['inputs'] = data['inputs'].unsqueeze(0).to(args.device)
            if not isinstance(data['data_samples'], list):
                data['data_samples'] = [data['data_samples']]
            sample = data['data_samples'][0]
            image_id = int(sample.metainfo.get('img_id', index))
            image_ids.append(image_id)
            output = model.test_step(data)[0]
            final_predictions.extend(result_to_coco(
                output.pred_instances, image_id, cat_ids))

            meta = build_image_meta(sample)
            max_steps = max(len(values) for values in traces)
            for step_index in range(max_steps):
                for head_index, values in enumerate(traces):
                    if step_index >= len(values):
                        continue
                    cls_logits, pred_bboxes = values[step_index]
                    result = model.bbox_head._sampler.post_process(
                        [(cls_logits, pred_bboxes)], [meta], rescale=True)[0]
                    key = f'step{step_index + 1}_head{head_index + 1}'
                    stage_predictions[key].extend(
                        result_to_coco(result, image_id, cat_ids))
            if (index + 1) % 50 == 0:
                print(f'{index + 1}/{limit}')
    finally:
        for handle in handles:
            handle.remove()

    if limit < len(dataset):
        coco_gt = subset_coco(coco_gt, image_ids)
    grouped_gt = gt_by_image(coco_gt)
    annotate_gt_strata(grouped_gt)
    baseline = coco_metrics(coco_gt, final_predictions)
    oracle_predictions = build_oracles(final_predictions, grouped_gt)
    oracle_predictions.update(
        build_quality_oracles(final_predictions, grouped_gt))
    oracles = {name: coco_metrics(coco_gt, values)
               for name, values in oracle_predictions.items()}
    errors = {f'{threshold:.2f}': greedy_error_counts(
        final_predictions, grouped_gt, threshold)
        for threshold in (0.50, 0.75, 0.90)}
    strata, confusion = stratified_recall(final_predictions, grouped_gt)
    stages = {key: coco_metrics(coco_gt, values)
              for key, values in sorted(stage_predictions.items())}

    return {
        'config': args.config, 'checkpoint': args.checkpoint,
        'annotation': args.ann, 'seed': args.seed,
        'num_images': limit, 'baseline': baseline,
        'oracles': oracles, 'errors': errors,
        'stratified_recall': strata,
        'confusion_at_best_iou_0.50': confusion,
        'stage_metrics': stages,
        'definitions': {
            'classification_oracle': 'best-overlap GT class if IoU >= 0.10',
            'localization_oracle': 'same-class best GT box if IoU >= 0.10',
            'joint_oracle': 'best-overlap GT class and box if IoU >= 0.10',
            'quality_oracle': (
                'fixed boxes/classes, score re-ranked by true same-class IoU; '
                'multiplicative beta grid and quality-only ranking'),
            'overlap_strata': '<0.01, 0.01-0.20, >=0.20 max IoU with another GT',
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-images', type=int, default=0)
    args = parser.parse_args()
    result = run(args)
    with open(args.output, 'w') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps({
        'baseline': result['baseline'], 'oracles': result['oracles'],
        'stratified_recall': result['stratified_recall']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
