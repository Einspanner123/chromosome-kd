#!/usr/bin/env python3
"""LDMDet 检测能力瓶颈诊断脚本

对训练好的 LDMDet 模型进行全方位瓶颈分析, 定位检测能力的瓶颈所在.

诊断维度:
1. AP-IoU 曲线: 定位精度瓶颈 (在哪个 IoU 阈值开始下降)
2. 混淆矩阵: 分类瓶颈 (哪些类别互相混淆)
3. 尺度分析: 尺度瓶颈 (小/中/大目标 AP 差异)
4. 密度分析: 拥挤度瓶颈 (每图目标数 vs AP)
5. 提案利用率: 容量瓶颈 (500 proposals 的利用情况)
6. 误差分解: TIDE 风格误差归因 (Cls/Loc/Dupe/Bkg/Miss)
7. 置信度校准: 分数校准瓶颈 (置信度与准确率的关系)
8. 边界框回归质量: 框回归瓶颈 (L1/GIoU 误差分布)

Usage:
    python experiments/analysis/bottleneck_diagnosis.py \
        experiments/configs/ldmdet/rf_heun_adaln.py \
        work_dirs/ldmdet_rf_heun_adaln/best_coco_bbox_mAP_epoch_102.pth \
        --ann data/Chromosome20240904_NoAug_NoResize_coco/valid/_annotations.coco.json \
        --output experiments/analysis/bottleneck_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def box_iou_matrix(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """计算两组框的 IoU 矩阵 [N, M]"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-8)


def preds_to_tensors(preds: list[dict], cat_id_to_idx: dict | None = None):
    """将 COCO DT 格式的预测列表转为 xyxy 框/分数/标签张量.

    每个 pred dict: {image_id, category_id (int), bbox (xywh), score (float)}
    返回: (pred_boxes[N,4] xyxy, pred_scores[N], pred_labels[N], pred_cat_ids[N])
    """
    if not preds:
        return (torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long), [])

    boxes_xyxy = []
    scores = []
    labels = []
    cat_ids = []
    for p in preds:
        x, y, w, h = p['bbox']
        boxes_xyxy.append([x, y, x + w, y + h])
        scores.append(p['score'])
        cat_ids.append(p['category_id'])
        if cat_id_to_idx is not None:
            labels.append(cat_id_to_idx[p['category_id']])

    pred_boxes = torch.tensor(boxes_xyxy).float()
    pred_scores = torch.tensor(scores).float()
    pred_labels = (torch.tensor(labels).long()
                   if cat_id_to_idx is not None and labels else torch.zeros(0, dtype=torch.long))
    return pred_boxes, pred_scores, pred_labels, cat_ids


def box_giou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """计算 GIoU [N]"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-8)
    enc_lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    enc_rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[..., 0] * enc_wh[..., 1]
    return iou - (enc_area - union) / enc_area.clamp(min=1e-8)


# ──────────────────────────────────────────────
# 1. AP-IoU 曲线分析
# ──────────────────────────────────────────────

def compute_ap_at_iou(
    coco_gt: COCO,
    coco_dt: list,
    iou_thr: float,
) -> tuple[float, dict]:
    """计算指定 IoU 阈值下的 AP"""
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.iouThrs = np.array([iou_thr])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    ap = coco_eval.stats[0]  # AP at specified IoU

    # 提取 per-class AP
    per_class = {}
    precision = coco_eval.eval['precision']  # [T, R, K, A, M]
    for cat_id, cat_name in coco_gt.cats.items():
        cat_idx = coco_eval.params.catIds.index(cat_id)
        ap_cat = precision[0, :, cat_idx, 0, -1]
        ap_cat = ap_cat[ap_cat > -1]
        per_class[cat_name['name']] = float(ap_cat.mean()) if len(ap_cat) > 0 else 0.0

    return float(ap), per_class


def analyze_ap_iou_curve(coco_gt, coco_dt) -> dict:
    """1. AP-IoU 曲线: 定位精度瓶颈"""
    print('\n[1/8] AP-IoU 曲线分析...')
    iou_thrs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    aps = []
    for thr in iou_thrs:
        ap, _ = compute_ap_at_iou(coco_gt, coco_dt, thr)
        aps.append(ap)
        print(f'  IoU={thr:.2f}: AP={ap:.4f}')

    # 找到 AP 下降最快的区间
    drops = [(iou_thrs[i+1] - iou_thrs[i], aps[i] - aps[i+1]) for i in range(len(aps)-1)]
    max_drop_idx = max(range(len(drops)), key=lambda i: drops[i][1])
    max_drop_interval = (iou_thrs[max_drop_idx], iou_thrs[max_drop_idx + 1])
    max_drop = drops[max_drop_idx][1]

    return {
        'iou_thrs': iou_thrs,
        'aps': aps,
        'ap50': aps[0],
        'ap75': aps[iou_thrs.index(0.75)],
        'ap90': aps[iou_thrs.index(0.9)] if 0.9 in iou_thrs else None,
        'max_drop_interval': max_drop_interval,
        'max_drop': max_drop,
        'analysis': (
            f'AP 从 IoU=0.5 的 {aps[0]:.4f} 下降到 IoU=0.95 的 {aps[-1]:.4f}, '
            f'最大下降区间在 IoU=[{max_drop_interval[0]:.2f}, {max_drop_interval[1]:.2f}], '
            f'下降 {max_drop:.4f}. '
            + ('定位精度是主要瓶颈.' if max_drop > 0.1 else '定位精度不是主要瓶颈.')
        ),
    }


# ──────────────────────────────────────────────
# 2. 混淆矩阵分析
# ──────────────────────────────────────────────

def analyze_confusion_matrix(
    coco_gt: COCO,
    predictions: list[dict],
    iou_thr: float = 0.5,
    score_thr: float = 0.3,
) -> dict:
    """2. 混淆矩阵: 分类瓶颈"""
    print('\n[2/8] 混淆矩阵分析...')
    cat_ids = coco_gt.getCatIds()
    cat_id_to_name = {c['id']: c['name'] for c in coco_gt.cats.values()}
    cat_names = [cat_id_to_name[cid] for cid in cat_ids]
    num_classes = len(cat_ids)
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

    # 混淆矩阵: [num_classes+1, num_classes+1], 最后一行/列是 background
    confusion = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    img_ids = coco_gt.getImgIds()
    for img_id in img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(cat_id_to_idx[ann['category_id']])

        gt_boxes = torch.tensor(gt_boxes).float() if gt_boxes else torch.zeros(0, 4)
        gt_labels = torch.tensor(gt_labels).long() if gt_labels else torch.zeros(0, dtype=torch.long)

        preds = [p for p in predictions if p['image_id'] == img_id]
        if not preds:
            for l in gt_labels:
                confusion[l, num_classes] += 1  # GT missed
            continue

        # 汇总该图所有预测 (COCO DT 格式: bbox=xywh, category_id=单 int)
        pred_boxes_list = []
        pred_scores_list = []
        pred_labels_list = []
        for p in preds:
            x, y, w, h = p['bbox']
            pred_boxes_list.append([x, y, x + w, y + h])  # xywh -> xyxy
            pred_scores_list.append(p['score'])
            pred_labels_list.append(cat_id_to_idx[p['category_id']])

        pred_boxes = torch.tensor(pred_boxes_list).float()
        pred_scores = torch.tensor(pred_scores_list).float()
        pred_labels = torch.tensor(pred_labels_list).long()

        keep = pred_scores > score_thr
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        if len(pred_boxes) == 0:
            continue

        if len(gt_boxes) == 0:
            for l in pred_labels:
                confusion[num_classes, l] += 1  # Background predicted as class
            continue

        ious = box_iou_matrix(pred_boxes, gt_boxes)
        matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)

        sort_inds = torch.argsort(pred_scores, descending=True)
        for idx in sort_inds:
            max_iou, max_gt = ious[idx].max(0)
            if max_iou >= iou_thr and not matched_gt[max_gt]:
                confusion[gt_labels[max_gt], pred_labels[idx]] += 1
                matched_gt[max_gt] = True
            else:
                confusion[num_classes, pred_labels[idx]] += 1  # FP

        for i, matched in enumerate(matched_gt):
            if not matched:
                confusion[gt_labels[i], num_classes] += 1  # Missed

    # 计算每类的分类准确率
    per_class_acc = {}
    top_confusions = []
    for i in range(num_classes):
        tp = confusion[i, i]
        total = confusion[i, :].sum()
        acc = tp / max(total, 1)
        per_class_acc[cat_names[i]] = float(acc)

        # 找到最常混淆的类别
        if total > 0:
            row = confusion[i, :].copy()
            row[i] = 0  # 排除对角线
            for j in range(num_classes):
                if row[j] > 0:
                    top_confusions.append({
                        'gt_class': cat_names[i],
                        'pred_class': cat_names[j],
                        'count': int(row[j]),
                        'ratio': float(row[j] / total),
                    })

    top_confusions.sort(key=lambda x: x['count'], reverse=True)
    top_confusions = top_confusions[:15]

    # 找出分类最差的类别
    worst_classes = sorted(per_class_acc.items(), key=lambda x: x[1])[:5]

    return {
        'confusion_matrix': confusion.tolist(),
        'cat_names': cat_names,
        'per_class_accuracy': per_class_acc,
        'top_confusions': top_confusions,
        'worst_classes': worst_classes,
        'analysis': (
            f'分类最差的 5 个类别: {", ".join(f"{c}({a:.3f})" for c, a in worst_classes)}. '
            + ('存在显著类别混淆, 分类是瓶颈.' if (top_confusions and top_confusions[0]['count'] > 10)
               else '类别混淆不严重, 分类不是主要瓶颈.')
        ),
    }


# ──────────────────────────────────────────────
# 3. 尺度分析
# ──────────────────────────────────────────────

def analyze_scale_distribution(
    coco_gt: COCO,
    predictions: list[dict],
    iou_thr: float = 0.5,
) -> dict:
    """3. 尺度分析: 小/中/大目标 AP 差异"""
    print('\n[3/8] 尺度分析...')

    # 收集所有 GT 框的面积
    img_ids = coco_gt.getImgIds()
    gt_areas = []
    gt_per_img = defaultdict(list)
    for img_id in img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        for ann in anns:
            area = ann['area']
            gt_areas.append(area)
            gt_per_img[img_id].append(area)

    gt_areas = np.array(gt_areas)
    area_median = float(np.median(gt_areas))
    area_25 = float(np.percentile(gt_areas, 25))
    area_75 = float(np.percentile(gt_areas, 75))

    # 按面积分三组: 小 (< 25th), 中 (25-75), 大 (> 75th)
    small_thr = area_25
    large_thr = area_75

    # 计算每组的检测率
    scale_stats = {'small': {'total': 0, 'detected': 0},
                   'medium': {'total': 0, 'detected': 0},
                   'large': {'total': 0, 'detected': 0}}

    pred_by_img = defaultdict(list)
    for p in predictions:
        pred_by_img[p['image_id']].append(p)

    for img_id in img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = []
        gt_areas_list = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
            gt_areas_list.append(ann['area'])

        if not gt_boxes:
            continue

        gt_boxes = torch.tensor(gt_boxes).float()
        preds = pred_by_img.get(img_id, [])
        if preds:
            pred_boxes, pred_scores, _, pred_cat_ids = preds_to_tensors(preds)
            keep = pred_scores > 0.3
            pred_boxes = pred_boxes[keep]
            pred_cat_ids = [pred_cat_ids[i] for i in range(len(keep)) if keep[i]]

            if len(pred_boxes) > 0:
                ious = box_iou_matrix(pred_boxes, gt_boxes)
                for j, ann in enumerate(anns):
                    max_iou = ious[:, j].max().item() if len(pred_boxes) > 0 else 0
                    # 检查类别是否匹配
                    if max_iou >= iou_thr:
                        best_pred = ious[:, j].argmax().item()
                        if best_pred < len(pred_cat_ids):
                            pred_cat = pred_cat_ids[best_pred]
                            if pred_cat == ann['category_id']:
                                detected = True
                            else:
                                detected = False
                        else:
                            detected = False
                    else:
                        detected = False

                    area = ann['area']
                    if area < small_thr:
                        scale = 'small'
                    elif area > large_thr:
                        scale = 'large'
                    else:
                        scale = 'medium'
                    scale_stats[scale]['total'] += 1
                    if detected:
                        scale_stats[scale]['detected'] += 1
        else:
            for ann in anns:
                area = ann['area']
                if area < small_thr:
                    scale = 'small'
                elif area > large_thr:
                    scale = 'large'
                else:
                    scale = 'medium'
                scale_stats[scale]['total'] += 1

    for scale in scale_stats:
        total = scale_stats[scale]['total']
        detected = scale_stats[scale]['detected']
        scale_stats[scale]['recall'] = detected / max(total, 1)

    return {
        'area_median': area_median,
        'area_25th': area_25,
        'area_75th': area_75,
        'scale_stats': scale_stats,
        'analysis': (
            f'小目标 recall={scale_stats["small"]["recall"]:.4f}, '
            f'中目标 recall={scale_stats["medium"]["recall"]:.4f}, '
            f'大目标 recall={scale_stats["large"]["recall"]:.4f}. '
            + ('尺度差异是瓶颈.' if max(scale_stats[s]['recall'] for s in scale_stats) -
               min(scale_stats[s]['recall'] for s in scale_stats) > 0.1
               else '尺度差异不是主要瓶颈.')
        ),
    }


# ──────────────────────────────────────────────
# 4. 密度分析
# ──────────────────────────────────────────────

def analyze_density(coco_gt: COCO, predictions: list[dict], iou_thr: float = 0.5) -> dict:
    """4. 密度分析: 每图目标数 vs 检测性能"""
    print('\n[4/8] 密度分析...')
    img_ids = coco_gt.getImgIds()

    # 按目标数分组
    density_bins = [(0, 20), (20, 35), (35, 45), (45, 55), (55, 100)]
    density_stats = {f'{lo}-{hi}': {'total_gt': 0, 'detected': 0, 'num_imgs': 0}
                     for lo, hi in density_bins}

    pred_by_img = defaultdict(list)
    for p in predictions:
        pred_by_img[p['image_id']].append(p)

    for img_id in img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        n_gt = len(anns)

        # 找到对应密度组
        bin_key = None
        for lo, hi in density_bins:
            if lo <= n_gt < hi:
                bin_key = f'{lo}-{hi}'
                break
        if bin_key is None:
            continue

        density_stats[bin_key]['num_imgs'] += 1
        density_stats[bin_key]['total_gt'] += n_gt

        gt_boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])

        if not gt_boxes:
            continue

        gt_boxes = torch.tensor(gt_boxes).float()
        preds = pred_by_img.get(img_id, [])
        if preds:
            pred_boxes, pred_scores, _, pred_cat_ids = preds_to_tensors(preds)
            keep = pred_scores > 0.3
            pred_boxes = pred_boxes[keep]
            pred_cat_ids = [pred_cat_ids[i] for i in range(len(keep)) if keep[i]]

            if len(pred_boxes) > 0:
                ious = box_iou_matrix(pred_boxes, gt_boxes)
                for j, ann in enumerate(anns):
                    max_iou, best_pred = ious[:, j].max(0)
                    if max_iou >= iou_thr and best_pred < len(pred_cat_ids):
                        if pred_cat_ids[best_pred] == ann['category_id']:
                            density_stats[bin_key]['detected'] += 1

    for key in density_stats:
        total = density_stats[key]['total_gt']
        detected = density_stats[key]['detected']
        density_stats[key]['recall'] = detected / max(total, 1)

    return {
        'density_stats': density_stats,
        'analysis': (
            '各密度组 recall: ' +
            ', '.join(f'{k}={v["recall"]:.4f}' for k, v in density_stats.items()) +
            '. ' + ('高密度图像检测困难, 密度是瓶颈.'
                    if any(density_stats[k]['recall'] < 0.7 for k in density_stats if density_stats[k]['total_gt'] > 0)
                    else '密度对检测影响不大.')
        ),
    }


# ──────────────────────────────────────────────
# 5. 误差分解 (TIDE 风格)
# ──────────────────────────────────────────────

def analyze_error_decomposition(
    coco_gt: COCO,
    predictions: list[dict],
    iou_thr: float = 0.5,
    score_thr: float = 0.1,
) -> dict:
    """5. TIDE 风格误差分解"""
    print('\n[5/8] 误差分解 (TIDE 风格)...')
    stats = {'TP': 0, 'Cls': 0, 'Loc': 0, 'Both': 0, 'Dupe': 0, 'Bkg': 0, 'Miss': 0}
    total_gt = 0

    pred_by_img = defaultdict(list)
    for p in predictions:
        pred_by_img[p['image_id']].append(p)

    cat_id_to_name = {c['id']: c['name'] for c in coco_gt.cats.values()}

    img_ids = coco_gt.getImgIds()
    for img_id in img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(ann['category_id'])

        gt_boxes = torch.tensor(gt_boxes).float() if gt_boxes else torch.zeros(0, 4)
        gt_labels = torch.tensor(gt_labels).long() if gt_labels else torch.zeros(0, dtype=torch.long)
        total_gt += len(gt_boxes)

        preds = pred_by_img.get(img_id, [])
        if preds:
            pred_boxes, pred_scores, _, pred_cat_ids = preds_to_tensors(preds)
            pred_labels = torch.tensor(pred_cat_ids).long()
            keep = pred_scores > score_thr
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]
        else:
            pred_boxes = torch.zeros(0, 4)
            pred_scores = torch.zeros(0)
            pred_labels = torch.zeros(0, dtype=torch.long)

        if len(pred_boxes) == 0:
            stats['Miss'] += len(gt_boxes)
            continue
        if len(gt_boxes) == 0:
            stats['Bkg'] += len(pred_boxes)
            continue

        ious = box_iou_matrix(pred_boxes, gt_boxes)
        matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
        sort_inds = torch.argsort(pred_scores, descending=True)

        for idx in sort_inds:
            max_iou, max_gt = ious[idx].max(0)
            if max_iou >= iou_thr:
                if pred_labels[idx] == gt_labels[max_gt]:
                    if not matched_gt[max_gt]:
                        stats['TP'] += 1
                        matched_gt[max_gt] = True
                    else:
                        stats['Dupe'] += 1
                else:
                    stats['Cls'] += 1
            elif max_iou >= 0.1:
                if pred_labels[idx] == gt_labels[max_gt]:
                    stats['Loc'] += 1
                else:
                    stats['Both'] += 1
            else:
                stats['Bkg'] += 1

        stats['Miss'] += (~matched_gt).sum().item()

    total_errors = sum(v for k, v in stats.items() if k != 'TP')
    error_dist = {k: v / max(total_errors, 1) for k, v in stats.items() if k != 'TP'}

    # 确定主要误差类型
    main_error = max(error_dist.items(), key=lambda x: x[1])

    return {
        'stats': stats,
        'total_gt': total_gt,
        'error_distribution': error_dist,
        'main_error_type': main_error[0],
        'main_error_ratio': main_error[1],
        'analysis': (
            f'总 GT: {total_gt}, TP: {stats["TP"]}. '
            f'主要误差类型: {main_error[0]} ({main_error[1]:.1%}). '
            + ('定位误差是主要瓶颈.' if main_error[0] in ('Loc', 'Both') else
               '分类误差是主要瓶颈.' if main_error[0] == 'Cls' else
               '漏检是主要瓶颈.' if main_error[0] == 'Miss' else
               '重复检测是主要瓶颈.' if main_error[0] == 'Dupe' else
               '背景误检是主要瓶颈.')
        ),
    }


# ──────────────────────────────────────────────
# 6. 置信度校准分析
# ──────────────────────────────────────────────

def analyze_confidence_calibration(
    coco_gt: COCO,
    predictions: list[dict],
    iou_thr: float = 0.5,
    num_bins: int = 10,
) -> dict:
    """6. 置信度校准: 预测分数与实际准确率的关系"""
    print('\n[6/8] 置信度校准分析...')
    bins = np.linspace(0, 1, num_bins + 1)
    bin_accs = []
    bin_counts = []
    bin_confs = []

    pred_by_img = defaultdict(list)
    for p in predictions:
        pred_by_img[p['image_id']].append(p)

    for img_id in coco_gt.getImgIds():
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(ann['category_id'])

        gt_boxes = torch.tensor(gt_boxes).float() if gt_boxes else torch.zeros(0, 4)
        gt_labels = torch.tensor(gt_labels).long() if gt_labels else torch.zeros(0, dtype=torch.long)

        preds = pred_by_img.get(img_id, [])
        if not preds or len(gt_boxes) == 0:
            continue

        pred_boxes, pred_scores, _, pred_cat_ids = preds_to_tensors(preds)
        pred_labels = torch.tensor(pred_cat_ids).long()

        if len(pred_boxes) == 0:
            continue

        ious = box_iou_matrix(pred_boxes, gt_boxes)
        for i in range(len(pred_boxes)):
            score = pred_scores[i].item()
            bin_idx = min(int(score * num_bins), num_bins - 1)
            max_iou, max_gt = ious[i].max(0)
            is_tp = (max_iou >= iou_thr and pred_labels[i] == gt_labels[max_gt])

            while len(bin_accs) <= bin_idx:
                bin_accs.append(0)
                bin_counts.append(0)
                bin_confs.append(0.0)
            bin_accs[bin_idx] += int(is_tp)
            bin_counts[bin_idx] += 1
            bin_confs[bin_idx] += score

    calibration = []
    for i in range(num_bins):
        if bin_counts[i] > 0:
            acc = bin_accs[i] / bin_counts[i]
            conf = bin_confs[i] / bin_counts[i]
            calibration.append({
                'bin': f'{bins[i]:.1f}-{bins[i+1]:.1f}',
                'accuracy': acc,
                'confidence': conf,
                'count': bin_counts[i],
                'gap': conf - acc,
            })

    # 计算 ECE (Expected Calibration Error)
    total = sum(bin_counts)
    ece = sum(abs(c['gap']) * c['count'] / max(total, 1) for c in calibration)

    return {
        'calibration': calibration,
        'ECE': ece,
        'analysis': (
            f'ECE={ece:.4f}. ' +
            ('模型置信度校准较差, 存在过度自信或自信不足.' if ece > 0.1 else
             '模型置信度校准良好.')
        ),
    }


# ──────────────────────────────────────────────
# 7. 边界框回归质量分析
# ──────────────────────────────────────────────

def analyze_bbox_quality(
    coco_gt: COCO,
    predictions: list[dict],
    iou_thr: float = 0.5,
) -> dict:
    """7. 边界框回归质量: TP 框的 L1/GIoU 误差分布"""
    print('\n[7/8] 边界框回归质量分析...')
    l1_errors = []
    giou_values = []
    iou_values = []

    pred_by_img = defaultdict(list)
    for p in predictions:
        pred_by_img[p['image_id']].append(p)

    for img_id in coco_gt.getImgIds():
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(ann['category_id'])

        gt_boxes = torch.tensor(gt_boxes).float() if gt_boxes else torch.zeros(0, 4)
        gt_labels = torch.tensor(gt_labels).long() if gt_labels else torch.zeros(0, dtype=torch.long)

        preds = pred_by_img.get(img_id, [])
        if not preds or len(gt_boxes) == 0:
            continue

        pred_boxes, pred_scores, _, pred_cat_ids = preds_to_tensors(preds)
        pred_labels = torch.tensor(pred_cat_ids).long()

        if len(pred_boxes) == 0:
            continue

        ious = box_iou_matrix(pred_boxes, gt_boxes)
        for i in range(len(pred_boxes)):
            max_iou, max_gt = ious[i].max(0)
            if max_iou >= iou_thr and pred_labels[i] == gt_labels[max_gt]:
                # TP: 计算回归质量
                pb = pred_boxes[i]
                gb = gt_boxes[max_gt]
                l1 = (pb - gb).abs().sum().item() / 4.0  # 平均每维 L1
                giou = box_giou(pb.unsqueeze(0), gb.unsqueeze(0)).item()
                l1_errors.append(l1)
                giou_values.append(giou)
                iou_values.append(max_iou.item())

    if not l1_errors:
        return {'analysis': '无 TP 框, 无法分析回归质量.'}

    l1_errors = np.array(l1_errors)
    giou_values = np.array(giou_values)
    iou_values = np.array(iou_values)

    return {
        'l1_mean': float(l1_errors.mean()),
        'l1_std': float(l1_errors.std()),
        'l1_median': float(np.median(l1_errors)),
        'l1_p90': float(np.percentile(l1_errors, 90)),
        'giou_mean': float(giou_values.mean()),
        'giou_std': float(giou_values.std()),
        'iou_mean': float(iou_values.mean()),
        'iou_std': float(iou_values.std()),
        'iou_median': float(np.median(iou_values)),
        'iou_p10': float(np.percentile(iou_values, 10)),
        'num_tp': len(l1_errors),
        'analysis': (
            f'TP 框 IoU 均值={iou_values.mean():.4f}, 中位数={np.median(iou_values):.4f}, '
            f'P10={np.percentile(iou_values, 10):.4f}. '
            f'L1 均值={l1_errors.mean():.4f}, GIoU 均值={giou_values.mean():.4f}. '
            + ('框回归精度不足, 高 IoU 阈值下会大量丢失 TP.' if np.percentile(iou_values, 10) < 0.6
               else '框回归精度尚可.')
        ),
    }


# ──────────────────────────────────────────────
# 8. 提案利用率分析
# ──────────────────────────────────────────────

def analyze_proposal_utilization(
    coco_gt: COCO,
    predictions: list[dict],
    score_thr: float = 0.05,
) -> dict:
    """8. 提案利用率: 每图预测框数 vs GT 数"""
    print('\n[8/8] 提案利用率分析...')
    num_proposals = 500  # 默认配置

    pred_counts = []
    pred_counts_high = []  # 高置信度预测
    gt_counts = []
    nms_suppression_rates = []

    pred_by_img = defaultdict(list)
    for p in predictions:
        pred_by_img[p['image_id']].append(p)

    for img_id in coco_gt.getImgIds():
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        n_gt = len(coco_gt.loadAnns(ann_ids))
        gt_counts.append(n_gt)

        preds = pred_by_img.get(img_id, [])
        if preds:
            _, pred_scores_t, _, _ = preds_to_tensors(preds)
            scores = pred_scores_t.numpy()
            pred_counts.append(len(scores))
            pred_counts_high.append(int((scores > 0.5).sum()))
            # NMS 后的框数 / 原始提案数 = 利用率
            nms_suppression_rates.append(1.0 - len(scores) / num_proposals)
        else:
            pred_counts.append(0)
            pred_counts_high.append(0)
            nms_suppression_rates.append(1.0)

    pred_counts = np.array(pred_counts)
    gt_counts = np.array(gt_counts)

    return {
        'num_proposals_config': num_proposals,
        'avg_pred_count': float(pred_counts.mean()),
        'avg_pred_count_high': float(np.mean(pred_counts_high)),
        'avg_gt_count': float(gt_counts.mean()),
        'pred_gt_ratio': float(pred_counts.mean() / max(gt_counts.mean(), 1)),
        'nms_suppression_rate': float(np.mean(nms_suppression_rates)),
        'pred_count_std': float(pred_counts.std()),
        'analysis': (
            f'平均每图预测 {pred_counts.mean():.1f} 框 (高置信度 {np.mean(pred_counts_high):.1f}), '
            f'平均 GT {gt_counts.mean():.1f} 个, '
            f'预测/GT 比={pred_counts.mean() / max(gt_counts.mean(), 1):.2f}. '
            f'NMS 抑制率={np.mean(nms_suppression_rates):.2%}. '
            + ('提案利用率低, 大量提案被 NMS 抑制.' if np.mean(nms_suppression_rates) > 0.8
               else '提案利用率合理.')
        ),
    }


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def run_inference(config_path: str, checkpoint_path: str, ann_file: str, device: str = 'cuda:0'):
    """运行推理, 返回 COCO 格式的预测结果"""
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS

    cfg = Config.fromfile(config_path)
    model = init_detector(cfg, checkpoint_path, device=device)

    # 构建验证集
    val_dataset_cfg = cfg.val_dataloader.dataset
    dataset = DATASETS.build(val_dataset_cfg)

    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            data['inputs'] = data['inputs'].unsqueeze(0).to(device)
            if not isinstance(data['data_samples'], list):
                data['data_samples'] = [data['data_samples']]

            out = model.test_step(data)
            for r in (out if isinstance(out, list) else [out]):
                pi = r.pred_instances
                img_id = getattr(r, 'img_id', i)
                for j in range(len(pi.bboxes)):
                    bbox = pi.bboxes[j].cpu().tolist()
                    # xyxy -> xywh
                    bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                    predictions.append({
                        'image_id': img_id,
                        'category_id': int(pi.labels[j].cpu().item()) + 1,  # COCO category id
                        'bbox': bbox_xywh,
                        'score': float(pi.scores[j].cpu().item()),
                    })

            if (i + 1) % 50 == 0:
                print(f'  推理进度: {i+1}/{len(dataset)}')

    return predictions


def main():
    parser = argparse.ArgumentParser(description='LDMDet 瓶颈诊断')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='检查点路径')
    parser.add_argument('--ann', required=True, help='COCO 标注文件路径')
    parser.add_argument('--output', default='bottleneck_report.json', help='输出报告路径')
    parser.add_argument('--device', default='cuda:0', help='设备')
    parser.add_argument('--skip-inference', action='store_true', help='跳过推理, 使用缓存结果')
    parser.add_argument('--cache', default='bottleneck_preds.json', help='推理结果缓存路径')
    args = parser.parse_args()

    print('=' * 80)
    print('LDMDet 检测能力瓶颈诊断')
    print('=' * 80)
    print(f'Config: {args.config}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Annotation: {args.ann}')

    # 加载 GT
    coco_gt = COCO(args.ann)

    # 运行推理或加载缓存
    if args.skip_inference and os.path.exists(args.cache):
        print(f'\n加载缓存预测结果: {args.cache}')
        with open(args.cache, 'r') as f:
            predictions = json.load(f)
    else:
        print('\n运行推理...')
        predictions = run_inference(args.config, args.checkpoint, args.ann, args.device)
        with open(args.cache, 'w') as f:
            json.dump(predictions, f)
        print(f'预测结果已缓存到: {args.cache}')

    print(f'\n总预测数: {len(predictions)}')

    # 转换为 COCO DT 格式
    coco_dt = coco_gt.loadRes(predictions)

    # 运行所有诊断
    report = {}
    report['ap_iou_curve'] = analyze_ap_iou_curve(coco_gt, coco_dt)
    report['confusion_matrix'] = analyze_confusion_matrix(coco_gt, predictions)
    report['scale_analysis'] = analyze_scale_distribution(coco_gt, predictions)
    report['density_analysis'] = analyze_density(coco_gt, predictions)
    report['error_decomposition'] = analyze_error_decomposition(coco_gt, predictions)
    report['confidence_calibration'] = analyze_confidence_calibration(coco_gt, predictions)
    report['bbox_quality'] = analyze_bbox_quality(coco_gt, predictions)
    report['proposal_utilization'] = analyze_proposal_utilization(coco_gt, predictions)

    # 汇总
    print('\n' + '=' * 80)
    print('瓶颈诊断汇总')
    print('=' * 80)
    for name, result in report.items():
        print(f'\n--- {name} ---')
        if 'analysis' in result:
            print(result['analysis'])

    # 保存报告
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'\n完整报告已保存到: {args.output}')


if __name__ == '__main__':
    main()
