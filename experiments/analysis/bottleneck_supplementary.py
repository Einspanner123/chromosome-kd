#!/usr/bin/env python3
"""补充瓶颈诊断 — 解决初版分析的三个盲区

盲区 1: 误差分解仅在 IoU=0.5 下进行, 低估了定位瓶颈
        -> 在多 IoU 阈值 (0.5/0.75/0.9) 下重新分解
盲区 2: 尺度分析仅在 IoU=0.5 下计算 recall, 无法反映高 IoU 下的尺度差异
        -> 在多 IoU 阈值下计算尺度 recall
盲区 3: 密度分析无效 (染色体数据集每图固定 ~46 个目标, 无密度变化)
        -> 替换为类别不平衡分析 + 每类定位质量分析
盲区 4: 缺少每类的 IoU 分布, 无法定位哪些类的框回归最差
        -> 新增每类 IoU 分布分析

Usage:
    python experiments/analysis/bottleneck_supplementary.py \
        experiments/configs/ldmdet/rf_heun_adaln.py \
        work_dirs/multi_seed_aug/rf_heun_adaln/seed_42/best_coco_bbox_mAP_epoch_102.pth \
        --ann data/Chromosome20240904_NoAug_NoResize_coco/valid/_annotations.coco.json \
        --cache work_dirs/bottleneck/bottleneck_preds.json \
        --output work_dirs/bottleneck/supplementary_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pycocotools.coco import COCO


# ──────────────────────────────────────────────
# 工具函数 (与 bottleneck_diagnosis.py 一致)
# ──────────────────────────────────────────────

def box_iou_matrix(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-8)


def preds_to_tensors(preds: list[dict], cat_id_to_idx: dict | None = None):
    if not preds:
        return (torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long), [])
    boxes_xyxy, scores, labels, cat_ids = [], [], [], []
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


# ──────────────────────────────────────────────
# 1. 多 IoU 阈值误差分解
# ──────────────────────────────────────────────

def analyze_error_multi_iou(coco_gt, predictions, iou_thrs=(0.5, 0.75, 0.9), score_thr=0.1):
    """在多个 IoU 阈值下进行 TIDE 风格误差分解"""
    print('\n[1/4] 多 IoU 阈值误差分解...')
    results = {}

    pred_by_img = defaultdict(list)
    for p in predictions:
        pred_by_img[p['image_id']].append(p)

    for iou_thr in iou_thrs:
        print(f'  IoU={iou_thr}...')
        stats = {'TP': 0, 'Cls': 0, 'Loc': 0, 'Both': 0, 'Dupe': 0, 'Bkg': 0, 'Miss': 0}
        total_gt = 0

        for img_id in coco_gt.getImgIds():
            ann_ids = coco_gt.getAnnIds(imgIds=img_id)
            anns = coco_gt.loadAnns(ann_ids)
            gt_boxes, gt_labels = [], []
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
        results[f'iou_{iou_thr}'] = {
            'stats': stats,
            'error_distribution': error_dist,
            'total_gt': total_gt,
        }

    # 分析: Cls 占比随 IoU 变化趋势
    cls_ratios = [results[f'iou_{t}']['error_distribution']['Cls'] for t in iou_thrs]
    loc_ratios = [results[f'iou_{t}']['error_distribution']['Loc'] for t in iou_thrs]
    both_ratios = [results[f'iou_{t}']['error_distribution']['Both'] for t in iou_thrs]

    if cls_ratios[-1] < cls_ratios[0] and loc_ratios[-1] > loc_ratios[0]:
        analysis = (f'随着 IoU 阈值升高, Cls 占比从 {cls_ratios[0]:.1%} 降至 {cls_ratios[-1]:.1%}, '
                    f'Loc 占比从 {loc_ratios[0]:.1%} 升至 {loc_ratios[-1]:.1%}. '
                    f'定位瓶颈在高 IoU 下更加显著.')
    else:
        analysis = (f'Cls 占比在各 IoU 阈值下均较高 ({cls_ratios[0]:.1%} -> {cls_ratios[-1]:.1%}), '
                    f'分类是稳定的主要瓶颈.')

    return {'per_iou': results, 'cls_ratios': cls_ratios, 'loc_ratios': loc_ratios,
            'both_ratios': both_ratios, 'analysis': analysis}


# ──────────────────────────────────────────────
# 2. 多 IoU 阈值尺度分析
# ──────────────────────────────────────────────

def analyze_scale_multi_iou(coco_gt, predictions, iou_thrs=(0.5, 0.75, 0.9), score_thr=0.3):
    """在多个 IoU 阈值下分析尺度 recall"""
    print('\n[2/4] 多 IoU 阈值尺度分析...')

    # COCO 面积分界: small<32^2=1024, medium 32^2~96^2=9216, large>96^2
    area_bins = {'small': (0, 1024), 'medium': (1024, 9216), 'large': (9216, float('inf'))}

    pred_by_img = defaultdict(list)
    for p in predictions:
        pred_by_img[p['image_id']].append(p)

    results = {}
    for iou_thr in iou_thrs:
        print(f'  IoU={iou_thr}...')
        scale_stats = {s: {'total': 0, 'detected': 0} for s in area_bins}

        for img_id in coco_gt.getImgIds():
            ann_ids = coco_gt.getAnnIds(imgIds=img_id)
            anns = coco_gt.loadAnns(ann_ids)
            gt_boxes, gt_labels, gt_areas = [], [], []
            for ann in anns:
                x, y, w, h = ann['bbox']
                gt_boxes.append([x, y, x + w, y + h])
                gt_labels.append(ann['category_id'])
                gt_areas.append(ann['area'])

            if not gt_boxes:
                continue
            gt_boxes = torch.tensor(gt_boxes).float()
            gt_labels = torch.tensor(gt_labels).long()

            preds = pred_by_img.get(img_id, [])
            if preds:
                pred_boxes, pred_scores, _, pred_cat_ids = preds_to_tensors(preds)
                pred_labels = torch.tensor(pred_cat_ids).long()
                keep = pred_scores > score_thr
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
            else:
                pred_boxes = torch.zeros(0, 4)
                pred_labels = torch.zeros(0, dtype=torch.long)

            for j, (area, gt_label) in enumerate(zip(gt_areas, gt_labels)):
                scale_key = 'small' if area < 1024 else 'medium' if area < 9216 else 'large'
                scale_stats[scale_key]['total'] += 1

                if len(pred_boxes) > 0:
                    ious = box_iou_matrix(pred_boxes, gt_boxes[j:j+1])
                    for i in range(len(pred_boxes)):
                        if ious[i, 0] >= iou_thr and pred_labels[i] == gt_label:
                            scale_stats[scale_key]['detected'] += 1
                            break

        for s in scale_stats:
            total = scale_stats[s]['total']
            detected = scale_stats[s]['detected']
            scale_stats[s]['recall'] = detected / max(total, 1)

        results[f'iou_{iou_thr}'] = scale_stats

    # 分析: 小目标 recall 在高 IoU 下的下降幅度
    small_recall_05 = results['iou_0.5']['small']['recall']
    small_recall_09 = results['iou_0.9']['small']['recall']
    large_recall_05 = results['iou_0.5']['large']['recall']
    large_recall_09 = results['iou_0.9']['large']['recall']

    small_drop = small_recall_05 - small_recall_09
    large_drop = large_recall_05 - large_recall_09

    if small_drop > large_drop + 0.1:
        analysis = (f'小目标 recall 从 IoU=0.5 的 {small_recall_05:.3f} 降至 IoU=0.9 的 {small_recall_09:.3f} '
                    f'(下降 {small_drop:.3f}), 大目标下降 {large_drop:.3f}. '
                    f'小目标在高 IoU 下定位更困难, 尺度是定位瓶颈的放大因素.')
    else:
        analysis = (f'小/大目标 recall 下降幅度接近 (小:{small_drop:.3f} vs 大:{large_drop:.3f}), '
                    f'尺度不是定位瓶颈的主要因素.')

    return {'per_iou': results, 'analysis': analysis}


# ──────────────────────────────────────────────
# 3. 类别不平衡 + 每类定位质量分析
# ──────────────────────────────────────────────

def analyze_class_balance_and_localization(coco_gt, predictions, iou_thr=0.5, score_thr=0.3):
    """类别不平衡分析 + 每类 IoU 分布"""
    print('\n[3/4] 类别不平衡 + 每类定位质量...')

    # 统计每类 GT 数量
    cat_ids = sorted(coco_gt.cats.keys())
    cat_names = [coco_gt.cats[cid]['name'] for cid in cat_ids]
    cat_id_to_name = {cid: coco_gt.cats[cid]['name'] for cid in cat_ids}

    gt_counts = defaultdict(int)
    for img_id in coco_gt.getImgIds():
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        for ann in coco_gt.loadAnns(ann_ids):
            gt_counts[ann['category_id']] += 1

    # 每类 IoU 分布 (TP 框的 IoU)
    pred_by_img = defaultdict(list)
    for p in predictions:
        pred_by_img[p['image_id']].append(p)

    per_class_ious = defaultdict(list)

    for img_id in coco_gt.getImgIds():
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes, gt_labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(ann['category_id'])

        if not gt_boxes:
            continue
        gt_boxes = torch.tensor(gt_boxes).float()
        gt_labels = torch.tensor(gt_labels).long()

        preds = pred_by_img.get(img_id, [])
        if not preds:
            continue
        pred_boxes, pred_scores, _, pred_cat_ids = preds_to_tensors(preds)
        pred_labels = torch.tensor(pred_cat_ids).long()
        keep = pred_scores > score_thr
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]

        if len(pred_boxes) == 0:
            continue

        ious = box_iou_matrix(pred_boxes, gt_boxes)
        matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
        sort_inds = torch.argsort(pred_scores[keep], descending=True)

        for idx in sort_inds:
            max_iou, max_gt = ious[idx].max(0)
            if max_iou >= iou_thr and not matched_gt[max_gt]:
                if pred_labels[idx] == gt_labels[max_gt]:
                    matched_gt[max_gt] = True
                    per_class_ious[gt_labels[max_gt].item()].append(max_iou.item())

    # 汇总
    class_stats = []
    for cid, name in zip(cat_ids, cat_names):
        ious = per_class_ious.get(cid, [])
        class_stats.append({
            'class': name,
            'gt_count': gt_counts[cid],
            'tp_count': len(ious),
            'iou_mean': float(np.mean(ious)) if ious else 0,
            'iou_median': float(np.median(ious)) if ious else 0,
            'iou_p25': float(np.percentile(ious, 25)) if ious else 0,
            'recall': len(ious) / max(gt_counts[cid], 1),
        })

    # 按 GT 数量排序
    class_stats_by_count = sorted(class_stats, key=lambda x: x['gt_count'])
    # 按 IoU 均值排序 (最差的在前)
    class_stats_by_iou = sorted(class_stats, key=lambda x: x['iou_mean'])

    # 分析
    gt_counts_list = [s['gt_count'] for s in class_stats]
    gt_max = max(gt_counts_list)
    gt_min = min(gt_counts_list)
    imbalance_ratio = gt_max / max(gt_min, 1)

    worst_iou_classes = [s['class'] for s in class_stats_by_iou[:5]]
    worst_iou_values = [f'{s["class"]}({s["iou_mean"]:.3f})' for s in class_stats_by_iou[:5]]

    analysis = (
        f'类别不平衡: 最多 {gt_max} 样本, 最少 {gt_min} 样本, 不平衡比={imbalance_ratio:.1f}. '
        f'定位最差的 5 个类: {", ".join(worst_iou_values)}. '
    )
    if imbalance_ratio > 3:
        analysis += f'存在显著类别不平衡, 少样本类的检测能力受限.'
    else:
        analysis += f'类别不平衡不严重.'

    return {
        'class_stats': class_stats,
        'imbalance_ratio': imbalance_ratio,
        'worst_iou_classes': worst_iou_classes,
        'analysis': analysis,
    }


# ──────────────────────────────────────────────
# 4. 每类 AP-IoU 下降曲线 (定位 vs 分离)
# ──────────────────────────────────────────────

def analyze_per_class_iou_drop(coco_gt, predictions, iou_thrs=(0.5, 0.75, 0.9, 0.95), score_thr=0.1):
    """每类在不同 IoU 阈值下的 recall, 分析哪些类的定位最差"""
    print('\n[4/4] 每类 IoU 下降曲线...')

    cat_ids = sorted(coco_gt.cats.keys())
    cat_names = [coco_gt.cats[cid]['name'] for cid in cat_ids]

    pred_by_img = defaultdict(list)
    for p in predictions:
        pred_by_img[p['image_id']].append(p)

    # 每类每 IoU 的 recall
    results = {name: {f'iou_{t}': {'total': 0, 'detected': 0} for t in iou_thrs}
               for name in cat_names}

    for img_id in coco_gt.getImgIds():
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes, gt_labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(ann['category_id'])

        if not gt_boxes:
            continue
        gt_boxes = torch.tensor(gt_boxes).float()
        gt_labels = torch.tensor(gt_labels).long()

        preds = pred_by_img.get(img_id, [])
        if preds:
            pred_boxes, pred_scores, _, pred_cat_ids = preds_to_tensors(preds)
            pred_labels = torch.tensor(pred_cat_ids).long()
            keep = pred_scores > score_thr
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
        else:
            pred_boxes = torch.zeros(0, 4)
            pred_labels = torch.zeros(0, dtype=torch.long)

        for j in range(len(gt_boxes)):
            gt_label = gt_labels[j].item()
            gt_name = cat_id_to_name = coco_gt.cats[gt_label]['name']

            for iou_thr in iou_thrs:
                results[gt_name][f'iou_{iou_thr}']['total'] += 1

            if len(pred_boxes) > 0:
                ious = box_iou_matrix(pred_boxes, gt_boxes[j:j+1])
                sort_inds = torch.argsort(ious[:, 0], descending=True)

                for iou_thr in iou_thrs:
                    # 找到最高 IoU 且类别匹配的预测
                    for idx in sort_inds:
                        if ious[idx, 0] >= iou_thr and pred_labels[idx].item() == gt_label:
                            results[gt_name][f'iou_{iou_thr}']['detected'] += 1
                            break

    # 计算 recall 和下降幅度
    per_class = []
    for name in cat_names:
        recalls = {}
        for t in iou_thrs:
            total = results[name][f'iou_{t}']['total']
            detected = results[name][f'iou_{t}']['detected']
            recalls[f'iou_{t}'] = detected / max(total, 1)

        drop_50_75 = recalls['iou_0.5'] - recalls['iou_0.75']
        drop_50_90 = recalls['iou_0.5'] - recalls['iou_0.9']

        per_class.append({
            'class': name,
            'recalls': recalls,
            'drop_50_75': drop_50_75,
            'drop_50_90': drop_50_90,
        })

    # 按下降幅度排序
    per_class_by_drop = sorted(per_class, key=lambda x: x['drop_50_90'], reverse=True)

    worst_drop_classes = [f'{c["class"]}(drop={c["drop_50_90"]:.3f})' for c in per_class_by_drop[:5]]
    analysis = (f'IoU 0.5->0.9 recall 下降最大的 5 个类: {", ".join(worst_drop_classes)}. '
                f'这些类的定位能力最差, 是定位瓶颈的主要来源.')

    return {'per_class': per_class, 'worst_drop_classes': worst_drop_classes, 'analysis': analysis}


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='LDMDet 瓶颈补充诊断')
    parser.add_argument('config', help='配置文件')
    parser.add_argument('checkpoint', help='检查点')
    parser.add_argument('--ann', required=True, help='COCO 标注文件')
    parser.add_argument('--cache', default='bottleneck_preds.json', help='推理结果缓存')
    parser.add_argument('--output', default='supplementary_report.json', help='输出报告')
    parser.add_argument('--device', default='cuda:0', help='设备')
    args = parser.parse_args()

    print('=' * 80)
    print('LDMDet 瓶颈补充诊断')
    print('=' * 80)

    coco_gt = COCO(args.ann)

    # 加载缓存预测
    if os.path.exists(args.cache):
        print(f'加载缓存预测: {args.cache}')
        with open(args.cache) as f:
            predictions = json.load(f)
    else:
        # 运行推理
        print('缓存不存在, 运行推理...')
        from experiments.analysis.bottleneck_diagnosis import run_inference
        predictions = run_inference(args.config, args.checkpoint, args.ann, args.device)
        with open(args.cache, 'w') as f:
            json.dump(predictions, f)

    print(f'总预测数: {len(predictions)}')

    report = {}
    report['error_multi_iou'] = analyze_error_multi_iou(coco_gt, predictions)
    report['scale_multi_iou'] = analyze_scale_multi_iou(coco_gt, predictions)
    report['class_balance'] = analyze_class_balance_and_localization(coco_gt, predictions)
    report['per_class_iou_drop'] = analyze_per_class_iou_drop(coco_gt, predictions)

    print('\n' + '=' * 80)
    print('补充诊断关键发现')
    print('=' * 80)
    for key, val in report.items():
        if 'analysis' in val:
            print(f'  [{key}] {val["analysis"]}')

    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'\n报告已保存到: {args.output}')


if __name__ == '__main__':
    main()
