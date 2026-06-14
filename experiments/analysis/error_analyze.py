import argparse
import os

import torch

from mmdet.evaluation.functional import bbox_overlaps
from mmdet.utils import register_all_modules


def analyze_errors(results_path, ann_file, iou_thr=0.5, score_thr=0.1):
    """
    基于简化版 TIDE 逻辑进行错误分析
    分类：
    1. Loc: 类别正确，IoU 在 [0.1, iou_thr) 之间
    2. Cls: IoU >= iou_thr，但类别错误
    3. Dupe: 类别正确，IoU >= iou_thr，但该 GT 已被匹配
    4. Bkg: 类别错误/正确，但与任何 GT 的 IoU < 0.1
    5. Miss: 未被检测到的 GT
    """
    register_all_modules()

    # 加载结果 (假设是标准的 mmdet pkl 结果)
    if not os.path.exists(results_path):
        print(f'Error: {results_path} not found.')
        return

    results = torch.load(results_path)

    # 加载数据集以获取 GT
    # 注意：这里需要根据实际情况加载，这里简化为从结果中提取（如果结果包含 GT）
    # 或者通过 COCO API 加载
    from pycocotools.coco import COCO

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    stats = {
        'Cls': 0,
        'Loc': 0,
        'Both': 0,
        'Dupe': 0,
        'Bkg': 0,
        'Miss': 0,
        'TP': 0,
    }

    total_gt = 0

    for i, img_id in enumerate(img_ids):
        # 获取 GT
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        gt_bboxes = []
        gt_labels = []
        for ann in anns:
            if ann.get('ignore', False):
                continue
            x, y, w, h = ann['bbox']
            gt_bboxes.append([x, y, x + w, y + h])
            gt_labels.append(ann['category_id'])

        gt_bboxes = (
            torch.tensor(gt_bboxes).float()
            if gt_bboxes
            else torch.zeros((0, 4))
        )
        gt_labels = (
            torch.tensor(gt_labels).long()
            if gt_labels
            else torch.zeros((0,), dtype=torch.long)
        )
        total_gt += len(gt_bboxes)

        # 获取预测结果
        # 假设 results[i] 是一个 DetDataSample
        pred_instances = results[i].pred_instances
        pred_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        pred_labels = pred_instances.labels

        # 过滤低分框
        keep = pred_scores > score_thr
        pred_bboxes = pred_bboxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        if len(pred_bboxes) == 0:
            stats['Miss'] += len(gt_bboxes)
            continue

        if len(gt_bboxes) == 0:
            stats['Bkg'] += len(pred_bboxes)
            continue

        # 计算 IoU 矩阵 [num_pred, num_gt]
        ious = bbox_overlaps(pred_bboxes, gt_bboxes)

        # 匹配逻辑
        matched_gt = torch.zeros(len(gt_bboxes), dtype=torch.bool)

        # 按得分从高到低处理预测
        sort_inds = torch.argsort(pred_scores, descending=True)
        for idx in sort_inds:
            p_box = pred_bboxes[idx]
            p_label = pred_labels[idx]
            p_ious = ious[idx]

            max_iou, max_idx = p_ious.max(0)

            if max_iou >= iou_thr:
                if p_label == gt_labels[max_idx]:
                    if not matched_gt[max_idx]:
                        stats['TP'] += 1
                        matched_gt[max_idx] = True
                    else:
                        stats['Dupe'] += 1
                else:
                    stats['Cls'] += 1
            elif max_iou >= 0.1:
                if p_label == gt_labels[max_idx]:
                    stats['Loc'] += 1
                else:
                    stats['Both'] += 1
            else:
                stats['Bkg'] += 1

        stats['Miss'] += (~matched_gt).sum().item()

    # 打印分析结果
    print('\n' + '=' * 30)
    print('      LDMDet Error Analysis')
    print('=' * 30)
    print(f'Total GT: {total_gt}')
    print(f'True Positives: {stats["TP"]}')
    print('-' * 30)
    print(f'Classification Errors: {stats["Cls"]}')
    print(f'Localization Errors:   {stats["Loc"]}')
    print(f'Cls + Loc Errors:      {stats["Both"]}')
    print(f'Duplicate Errors:      {stats["Dupe"]}')
    print(f'Background Errors:     {stats["Bkg"]}')
    print(f'Missed GTs:            {stats["Miss"]}')
    print('=' * 30)

    # 计算影响比例 (简单示意)
    total_errors = sum(
        [v for k, v in stats.items() if k not in ['TP', 'Miss']]
    )
    if total_errors > 0:
        print('\nError Distribution:')
        for k in ['Cls', 'Loc', 'Both', 'Dupe', 'Bkg']:
            print(f'{k:4s}: {stats[k] / total_errors * 100:5.1f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', help='Path to pkl results file')
    parser.add_argument('ann', help='Path to COCO annotation file')
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--score-thr', type=float, default=0.1)
    args = parser.parse_args()

    analyze_errors(args.results, args.ann, args.iou_thr, args.score_thr)
