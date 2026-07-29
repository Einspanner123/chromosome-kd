#!/usr/bin/env python3
"""诊断 box_renewal ON 时 mAP 崩溃 (0.04) 的根因.

假设: uncommitted 的 renewal_mask 改动 (head.py 中向 dpm_solver.step() 传递
      _renewal_mask) 可能破坏了 box_renewal 路径.

验证: 对比三种配置在 baseline [1,1,1,1] + box_renewal ON 下的 mAP:
  A) 原始 (不传 renewal_mask) — 模拟 committed HEAD 行为
  B) 当前 (传 renewal_mask) — uncommitted 行为
  C) box_renewal OFF — 对照 (已知 mAP=0.862)

同时打印首图预测统计 (框数/score 分布/IoU 分布) 辅助定位.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _to_plain_tensor(boxes) -> torch.Tensor:
    if hasattr(boxes, 'tensor'):
        boxes = boxes.tensor
    if hasattr(boxes, 'cpu'):
        boxes = boxes.cpu()
    return boxes


def _iou_matrix(b1, b2):
    N, M = b1.shape[0], b2.shape[0]
    if N == 0 or M == 0:
        return torch.zeros(N, M)
    a1 = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    a2 = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
    lt = torch.max(b1[:, None, :2], b2[None, :, :2])
    rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])
    wh = (rb - lt).clamp(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (a1[:, None] + a2[None] - inter).clamp(1e-6)


def run(config_path, checkpoint, ann_file, device, max_imgs,
        box_renewal, disable_renewal_mask):
    """评估单配置, 返回 mAP + 首图诊断."""
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS, METRICS

    cfg = Config.fromfile(config_path)
    cfg.model.bbox_head.solver_type = 'dpm_solver_pp'
    cfg.model.bbox_head.box_renewal = box_renewal
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='DetLocalVisualizer', vis_backends=cfg.vis_backends, name='vis')

    model = init_detector(cfg, checkpoint, device=device)
    model.eval()
    model.bbox_head._sampler.dim_d1_mask = None

    # === 关键: 控制 renewal_mask 传递 ===
    if disable_renewal_mask:
        # 猴子补丁: 让 _renewal_mask 始终为 None (模拟 committed HEAD 行为)
        head = model.bbox_head
        _orig_predict = head.predict

        # 通过包装 step 调用屏蔽 renewal_mask
        _orig_create = head._sampler.create_dpm_solver
        _solver_holder = [None]

        def _capturing_create():
            s = _orig_create()
            if s is not None:
                _solver_holder[0] = s
                # 猴子补丁 step: 忽略 renewal_mask 参数
                _orig_step = s.step

                def _step_no_renewal(x, x0_pred, t_n, step_idx,
                                     renewal_mask=None):
                    return _orig_step(x, x0_pred, t_n, step_idx,
                                      renewal_mask=None)

                s.step = _step_no_renewal
            return s

        head._sampler.create_dpm_solver = _capturing_create

    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    evaluator = METRICS.build(dict(
        type='CocoMetric', ann_file=ann_file, metric='bbox',
        classwise=False, format_only=False))
    evaluator.dataset_meta = dataset.metainfo

    n_imgs = min(max_imgs, len(dataset)) if max_imgs else len(dataset)
    first_img_diag = None

    with torch.no_grad():
        for i in range(n_imgs):
            data = dataset[i]
            data['inputs'] = data['inputs'].unsqueeze(0).to(device)
            if not isinstance(data['data_samples'], list):
                data['data_samples'] = [data['data_samples']]

            out = model.test_step(data)
            out_list = out if isinstance(out, list) else [out]

            eval_samples = []
            for r in out_list:
                d = {}
                if hasattr(r, 'pred_instances') and r.pred_instances is not None:
                    pi = r.pred_instances
                    d['pred_instances'] = {
                        'bboxes': pi.bboxes.cpu(),
                        'scores': pi.scores.cpu(),
                        'labels': pi.labels.cpu(),
                    }
                d['img_id'] = getattr(r, 'img_id', i)
                d['ori_shape'] = getattr(r, 'ori_shape', (1, 1))
                eval_samples.append(d)

                # 首图诊断
                if first_img_diag is None and hasattr(r, 'pred_instances'):
                    pi = r.pred_instances
                    pred = _to_plain_tensor(pi.bboxes)
                    scores = pi.scores.cpu()
                    ds_in = data['data_samples'][0]
                    gt = torch.zeros(0, 4)
                    if hasattr(ds_in, 'gt_instances') and ds_in.gt_instances is not None:
                        gt = _to_plain_tensor(ds_in.gt_instances.bboxes)
                    iou = _iou_matrix(pred, gt) if pred.shape[0] > 0 and gt.shape[0] > 0 else torch.zeros(0)
                    best_iou = iou.max(dim=1)[0] if iou.numel() > 0 else torch.zeros(0)
                    first_img_diag = {
                        'n_pred': int(pred.shape[0]),
                        'n_gt': int(gt.shape[0]),
                        'score_mean': float(scores.mean()) if scores.numel() else 0,
                        'score_max': float(scores.max()) if scores.numel() else 0,
                        'best_iou_mean': float(best_iou.mean()) if best_iou.numel() else 0,
                        'best_iou_gt_0.5_ratio': float((best_iou > 0.5).float().mean()) if best_iou.numel() else 0,
                    }

            evaluator.process({}, eval_samples)

    metrics = evaluator.evaluate(n_imgs)
    return {
        'mAP': float(metrics.get('coco/bbox_mAP', 0)),
        'AP50': float(metrics.get('coco/bbox_mAP_50', 0)),
        'first_img': first_img_diag,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--ann', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-imgs', type=int, default=50)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'

    print('=' * 70)
    print('box_renewal 崩溃根因诊断')
    print('=' * 70)

    configs = [
        ('A: box_renewal ON + 无 renewal_mask (模拟 HEAD)', True, True),
        ('B: box_renewal ON + renewal_mask (当前 uncommitted)', True, False),
    ]

    for label, br, disable_rm in configs:
        print(f'\n--- {label} ---')
        r = run(args.config, args.checkpoint, args.ann, device,
                args.max_imgs, box_renewal=br, disable_renewal_mask=disable_rm)
        print(f'  mAP={r["mAP"]:.4f}  AP50={r["AP50"]:.4f}')
        fi = r['first_img']
        if fi:
            print(f'  首图: n_pred={fi["n_pred"]} n_gt={fi["n_gt"]} '
                  f'score_mean={fi["score_mean"]:.3f} score_max={fi["score_max"]:.3f}')
            print(f'        best_iou_mean={fi["best_iou_mean"]:.3f} '
                  f'iou>0.5_ratio={fi["best_iou_gt_0.5_ratio"]:.3f}')

    print('\n' + '=' * 70)
    print('判定:')
    print('  若 A≈0.86 且 B≈0.04 → renewal_mask 改动是根因')
    print('  若 A≈0.04 且 B≈0.04 → box_renewal ON 本身与 DPM++ 不兼容 (脚本无关)')
    print('  若 A≈0.86 且 B≈0.86 → 之前 0.04 是其他原因 (如 max_imgs 太小)')


if __name__ == '__main__':
    main()
