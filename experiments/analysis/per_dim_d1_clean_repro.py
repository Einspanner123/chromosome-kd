#!/usr/bin/env python3
"""per-dim D1 掩码干净重评 (2026-07-29)

在 +DPM-Solver++ checkpoint 上, 通过 dim_d1_mask 机制 (基类 RFDPMSolverMultistep,
兼容 renewal_mask) 测试 4 种配置, 干净回答两个问题:

  Q1: DPM-Solver++ 是否更适合中心 (cx/cy) 预测?
  Q2: cx/cy 与 w/h 分开用各自优势求解器是否有 mAP 收益?

配置 (dim_d1_mask: 1=保留 D1 二阶校正, 0=置零 D1 退化为一阶 Euler):
  1. [1,1,1,1] baseline: 全维度 2阶 DPM++
  2. [1,1,0,0] center-DPM++: cx/cy 2阶, w/h 1阶
  3. [0,0,1,1] size-DPM++:  cx/cy 1阶, w/h 2阶
  4. [0,0,0,0] all-Euler:   全 1阶 (对照)

测量:
  - mAP, AP50, AP75 (COCO, 500 张验证图)
  - per-dim L1 误差 (cx/cy/w/h, IoU≥0.5 匈牙利匹配 pred→GT, normalized cxcywh [0,1])

判定:
  Q1: 比较 [1,1,1,1]→[0,0,0,0] 各维 L1 误差增幅. 若 cx/cy 增幅 > w/h 增幅,
      则 DPM++ 二阶校正对中心维度更有用 (更适合中心).
  Q2: [1,1,0,0] 或 [0,0,1,1] 的 mAP 是否 > baseline [1,1,1,1].

与旧实验 (§九 RFDPMSolverPerDim / §六 缺失脚本) 的区别:
  - 使用基类 dim_d1_mask 路径, 与 renewal_mask (Path A) 兼容, 不崩溃
  - box_renewal 保持 ON (与 +DPM-Solver++ baseline 一致, 无混淆)
  - 新增 per-dim L1 误差 (旧实验仅 mAP, 无法回答 Q1)

Usage:
    python experiments/analysis/per_dim_d1_clean_repro.py \
        --config experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \
        --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \
        --ann data/24_chromosomes_object/coco/valid/_annotations.coco.json \
        --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ============================================================
# per-dim 误差计算
# ============================================================

def _to_plain_tensor(boxes) -> torch.Tensor:
    """mmdet HorizontalBoxes / 普通 tensor → 普通 [N,4] tensor (xyxy)."""
    if hasattr(boxes, 'tensor'):  # mmdet BaseBoxes 子类
        boxes = boxes.tensor
    if hasattr(boxes, 'cpu'):
        boxes = boxes.cpu()
    return boxes


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """[N,4] xyxy → [N,4] cxcywh"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1).clamp(min=0.0)
    h = (y2 - y1).clamp(min=0.0)
    return torch.stack([cx, cy, w, h], dim=-1)


def _iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """[N,4] xyxy, [M,4] xyxy → [N,M] IoU"""
    N, M = boxes1.shape[0], boxes2.shape[0]
    if N == 0 or M == 0:
        return torch.zeros(N, M)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * \
            (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * \
            (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[..., 0] * wh[..., 1]  # [N,M]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _hungarian_match(iou: torch.Tensor):
    """IoU [N,M] → 匹配 (pred_idx, gt_idx). 用 scipy 若可用, 否则贪心."""
    try:
        from scipy.optimize import linear_sum_assignment
        # 最大化 IoU = 最小化 -IoU
        row, col = linear_sum_assignment(-iou.numpy())
        return list(zip(row.tolist(), col.tolist()))
    except Exception:
        # 贪心: 按 IoU 降序逐一匹配
        N, M = iou.shape
        flat = iou.flatten()
        order = flat.argsort(descending=True).tolist()
        used_p, used_g = set(), set()
        pairs = []
        for idx in order:
            p, g = idx // M, idx % M
            if p in used_p or g in used_g:
                continue
            used_p.add(p)
            used_g.add(g)
            pairs.append((p, g))
        return pairs


def per_dim_error_for_image(
    pred_xyxy: torch.Tensor,
    gt_xyxy: torch.Tensor,
    img_shape,  # (H, W) or (H, W, 3)
    iou_thr: float = 0.5,
):
    """单图 per-dim L1 误差 (normalized cxcywh [0,1]).

    Returns: dict {cx,cy,w,h: list[float]} 每个 matched pair 的 L1, 或 None 若无匹配.
    """
    if pred_xyxy.shape[0] == 0 or gt_xyxy.shape[0] == 0:
        return None
    H = float(img_shape[0])
    W = float(img_shape[1])
    scale = torch.tensor([W, H, W, H], dtype=pred_xyxy.dtype)

    iou = _iou_matrix(pred_xyxy, gt_xyxy)  # [Np, Ng]
    pairs = _hungarian_match(iou)

    pred_cxcywh = _xyxy_to_cxcywh(pred_xyxy) / scale  # normalized [0,1]
    gt_cxcywh = _xyxy_to_cxcywh(gt_xyxy) / scale

    errs = {'cx': [], 'cy': [], 'w': [], 'h': []}
    for p, g in pairs:
        if iou[p, g].item() < iou_thr:
            continue
        diff = (pred_cxcywh[p] - gt_cxcywh[g]).abs()
        errs['cx'].append(diff[0].item())
        errs['cy'].append(diff[1].item())
        errs['w'].append(diff[2].item())
        errs['h'].append(diff[3].item())
    if not errs['cx']:
        return None
    return errs


# ============================================================
# 单配置评估
# ============================================================

def run_config(
    config_path, checkpoint, ann_file, dim_mask, device='cuda:0',
    max_imgs=None, box_renewal=True,
):
    """在给定 dim_d1_mask 下评估 mAP + per-dim L1 误差 + 延迟."""
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS, METRICS

    cfg = Config.fromfile(config_path)
    # hybrid 配置用 dpm_pp_heun_hybrid solver_type; 其余用 dpm_solver_pp + dim_d1_mask
    is_hybrid = (dim_mask == 'hybrid')
    cfg.model.bbox_head.solver_type = (
        'dpm_pp_heun_hybrid' if is_hybrid else 'dpm_solver_pp'
    )
    # box_renewal 开关: 关闭时 renewal_mask≡None, D1 作用于全体 proposal,
    # 干净隔离 dim_d1_mask 的分维度效果 (开启时 renewal_mask 掩盖 dim_d1_mask)
    cfg.model.bbox_head.box_renewal = box_renewal
    # 禁用 SwanLab
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='DetLocalVisualizer', vis_backends=cfg.vis_backends, name='visualizer',
    )

    model = init_detector(cfg, checkpoint, device=device)
    model.eval()

    # 注入 dim_d1_mask (hybrid 跳过, 用自身 dpm_dims/heun_dims 分维度)
    if is_hybrid:
        model.bbox_head._sampler.dim_d1_mask = None
        label = 'hybrid (cx/cy DPM++, w/h Heun)'
    elif dim_mask is None:
        model.bbox_head._sampler.dim_d1_mask = None
        label = '[1,1,1,1] baseline (全 DPM++)'
    else:
        t = torch.tensor(dim_mask, dtype=torch.float32)
        model.bbox_head._sampler.dim_d1_mask = t
        label = f'{dim_mask}'

    # ===== 验证 solver 配置真正传递到 solver 实例 =====
    _verify_solver = model.bbox_head._sampler.create_dpm_solver()
    if _verify_solver is not None:
        if is_hybrid:
            print(f'    [验证] RFDPMSolverHybrid: dpm_dims={_verify_solver.dpm_dims}, '
                  f'heun_dims={_verify_solver.heun_dims}, '
                  f'model_fn={"set" if _verify_solver.model_fn is not None else "None"}')
        else:
            print(f'    [验证] create_dpm_solver → solver.dim_d1_mask = '
                  f'{_verify_solver.dim_d1_mask}')

    # ===== 捕获 eta_str_per_dim (验证 D1 掩码真正应用到对应维度) =====
    # step() 中 eta_str_per_dim 在 dim_d1_mask 应用之后计算, 故 [1,1,0,0] 应使
    # w/h 维度 (index 2,3) 的 eta_str 降为 ~0, cx/cy (index 0,1) 保持高值.
    _last_solver_holder = [None]
    _orig_create = model.bbox_head._sampler.create_dpm_solver

    def _capturing_create():
        s = _orig_create()
        if s is not None:
            _last_solver_holder[0] = s
        return s

    model.bbox_head._sampler.create_dpm_solver = _capturing_create

    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    evaluator = METRICS.build(dict(
        type='CocoMetric', ann_file=ann_file, metric='bbox',
        classwise=False, format_only=False,
    ))
    evaluator.dataset_meta = dataset.metainfo

    per_dim_accum = {'cx': [], 'cy': [], 'w': [], 'h': []}
    latencies = []
    num_samples = 0
    n_imgs = len(dataset) if max_imgs is None else min(max_imgs, len(dataset))

    with torch.no_grad():
        for i in range(n_imgs):
            data = dataset[i]
            data['inputs'] = data['inputs'].unsqueeze(0).to(device)
            if not isinstance(data['data_samples'], list):
                data['data_samples'] = [data['data_samples']]
            # data_samples 留 CPU (含 GT); model.test_step 内部处理 device

            if 'cuda' in device:
                torch.cuda.synchronize()
            t0 = time.time()
            out = model.test_step(data)
            if 'cuda' in device:
                torch.cuda.synchronize()
            latencies.append(time.time() - t0)

            out_list = out if isinstance(out, list) else [out]
            eval_samples = []
            for r in out_list:
                # ---- mAP evaluator ----
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

                # ---- per-dim 误差 ----
                pred_xyxy = _to_plain_tensor(pi.bboxes) if hasattr(r, 'pred_instances') and r.pred_instances is not None else torch.zeros(0, 4)
                # GT: 从 input data_sample 取 (val 集含 gt_instances)
                gt_xyxy = torch.zeros(0, 4)
                img_shape = getattr(r, 'img_shape', None) or getattr(r, 'ori_shape', (1, 1))
                ds_in = data['data_samples'][0] if len(data['data_samples']) > 0 else None
                if ds_in is not None and hasattr(ds_in, 'gt_instances') and ds_in.gt_instances is not None:
                    gt_xyxy = _to_plain_tensor(ds_in.gt_instances.bboxes)
                    img_shape = getattr(ds_in, 'img_shape', img_shape)
                if pred_xyxy.shape[0] > 0 and gt_xyxy.shape[0] > 0:
                    errs = per_dim_error_for_image(pred_xyxy, gt_xyxy, img_shape)
                    if errs is not None:
                        for k in per_dim_accum:
                            per_dim_accum[k].extend(errs[k])

            evaluator.process({}, eval_samples)
            num_samples += len(eval_samples)
            if (i + 1) % 100 == 0:
                print(f'    进度 {i+1}/{n_imgs}')

    metrics = evaluator.evaluate(num_samples)
    avg_lat = float(np.mean(latencies) * 1000)

    per_dim_mean = {}
    for k in ('cx', 'cy', 'w', 'h'):
        vals = per_dim_accum[k]
        per_dim_mean[k] = float(np.mean(vals)) if vals else float('nan')
    per_dim_mean['n_matched'] = sum(len(v) for v in per_dim_accum.values()) // 4

    # 读取最后一图 solver 的 eta_str_per_dim (验证 D1 掩码生效)
    # eta_str_per_dim 顺序 = (cx, cy, w, h), 在 dim_d1_mask 应用后计算
    eta_str_per_dim = []
    if _last_solver_holder[0] is not None:
        eta_str_per_dim = [
            [round(float(x), 4) for x in row]
            for row in _last_solver_holder[0].eta_str_per_dim_history
        ]

    return {
        'dim_mask': dim_mask if dim_mask is not None else [1, 1, 1, 1],
        'label': label,
        'mAP': float(metrics.get('coco/bbox_mAP', 0)),
        'AP50': float(metrics.get('coco/bbox_mAP_50', 0)),
        'AP75': float(metrics.get('coco/bbox_mAP_75', 0)),
        'APs': float(metrics.get('coco/bbox_mAP_s', 0)),
        'avg_latency_ms': avg_lat,
        'fps': float(1000.0 / avg_lat) if avg_lat > 0 else 0,
        'per_dim_l1': per_dim_mean,
        'eta_str_per_dim': eta_str_per_dim,
    }


# ============================================================
# 主流程
# ============================================================

CONFIGS = [
    None,              # [1,1,1,1] baseline
    [1.0, 1.0, 0.0, 0.0],  # center-DPM++: cx/cy 2阶, w/h 1阶(Euler)
    [0.0, 0.0, 1.0, 1.0],  # size-DPM++: cx/cy 1阶, w/h 2阶
    [0.0, 0.0, 0.0, 0.0],  # all-Euler: 全 1阶
    'hybrid',          # cx/cy DPM++ 2阶, w/h Heun 2阶 (额外 NFE)
]


def main():
    parser = argparse.ArgumentParser(description='per-dim D1 干净重评')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--ann', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-imgs', type=int, default=None, help='限制评估图数(调试)')
    parser.add_argument('--output', default=None)
    parser.add_argument(
        '--no-box-renewal', action='store_true',
        help='关闭 box_renewal, 使 D1 作用于全体 proposal (干净隔离 dim_d1_mask 效果). '
             '开启 box_renewal 时 renewal_mask 会置零大部分 proposal 的 D1, 掩盖 dim_d1_mask.',
    )
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    box_renewal = not args.no_box_renewal
    suffix = '_norenewal' if not box_renewal else ''
    output_path = args.output or os.path.join(
        _PROJECT_ROOT, 'work_dirs', 'diagnosis',
        f'per_dim_d1_clean_repro{suffix}.json',
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print('=' * 80)
    print('per-dim D1 干净重评 (2026-07-29)')
    print('=' * 80)
    print(f'Config: {args.config}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Ann: {args.ann}')
    print(f'Device: {device}')
    print(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'box_renewal: {"ON (+DPM-Solver++ default)" if box_renewal else "OFF (干净隔离 D1 效果)"}')
    print()

    all_results = []
    for dim_mask in CONFIGS:
        if dim_mask is None:
            lbl = '[1,1,1,1] baseline'
        elif dim_mask == 'hybrid':
            lbl = 'hybrid (cx/cy DPM++, w/h Heun)'
        else:
            lbl = str(dim_mask)
        print(f'\n--- 配置: {lbl} ---')
        r = run_config(
            args.config, args.checkpoint, args.ann, dim_mask,
            device=device, max_imgs=args.max_imgs, box_renewal=box_renewal,
        )
        all_results.append(r)
        print(f'  mAP={r["mAP"]:.4f}  AP50={r["AP50"]:.4f}  AP75={r["AP75"]:.4f}  '
              f'lat={r["avg_latency_ms"]:.1f}ms')
        pd = r['per_dim_l1']
        print(f'  per-dim L1 (norm cxcywh): cx={pd["cx"]:.4f} cy={pd["cy"]:.4f} '
              f'w={pd["w"]:.4f} h={pd["h"]:.4f}  (n_matched={pd["n_matched"]})')
        # eta_str_per_dim 验证: 确认 D1 掩码生效 (被掩码维度应降为 ~0)
        if r.get('eta_str_per_dim'):
            print(f'  eta_str_per_dim (cx,cy,w,h) 各步:')
            for si, row in enumerate(r['eta_str_per_dim']):
                print(f'    step{si}: [{row[0]:.2f}, {row[1]:.2f}, '
                      f'{row[2]:.2f}, {row[3]:.2f}]')


    # ===================== 分析 =====================
    print('\n' + '=' * 80)
    print('汇总')
    print('=' * 80)
    print(f'{"配置":<28} {"mAP":>8} {"AP75":>8} {"cx_L1":>9} {"cy_L1":>9} '
          f'{"w_L1":>9} {"h_L1":>9} {"lat(ms)":>8}')
    print('-' * 95)
    for r in all_results:
        pd = r['per_dim_l1']
        print(f'{r["label"]:<28} {r["mAP"]:>8.4f} {r["AP75"]:>8.4f} '
              f'{pd["cx"]:>9.4f} {pd["cy"]:>9.4f} {pd["w"]:>9.4f} {pd["h"]:>9.4f} '
              f'{r["avg_latency_ms"]:>8.1f}')

    base = all_results[0]      # [1,1,1,1]
    euler = all_results[3]     # [0,0,0,0]
    center = all_results[1]    # [1,1,0,0]
    size = all_results[2]      # [0,0,1,1]

    print('\n--- Q1: DPM++ 是否更适合中心 (cx/cy)? ---')
    print('比较 [1,1,1,1]→[0,0,0,0] (全 DPM++ → 全 Euler) 各维 L1 误差增幅:')
    for k in ('cx', 'cy', 'w', 'h'):
        db = base['per_dim_l1'][k]
        de = euler['per_dim_l1'][k]
        inc = (de - db) / db if db > 0 else float('nan')
        print(f'  {k}: {db:.4f} → {de:.4f}  (增幅 {inc*100:+.1f}%)')
    cx_inc = (euler['per_dim_l1']['cx'] - base['per_dim_l1']['cx']) / base['per_dim_l1']['cx']
    wh_inc_w = (euler['per_dim_l1']['w'] - base['per_dim_l1']['w']) / base['per_dim_l1']['w']
    wh_inc_h = (euler['per_dim_l1']['h'] - base['per_dim_l1']['h']) / base['per_dim_l1']['h']
    cy_inc = (euler['per_dim_l1']['cy'] - base['per_dim_l1']['cy']) / base['per_dim_l1']['cy']
    center_avg = (cx_inc + cy_inc) / 2
    size_avg = (wh_inc_w + wh_inc_h) / 2
    print(f'  中心 (cx,cy) 平均增幅: {center_avg*100:+.1f}%')
    print(f'  尺度 (w,h)  平均增幅: {size_avg*100:+.1f}%')
    if center_avg > size_avg:
        print(f'  → Q1 结论: 中心维度增幅 > 尺度维度, DPM++ 二阶校正对中心更有用 (更适合中心)')
    else:
        print(f'  → Q1 结论: 中心维度增幅 ≤ 尺度维度, DPM++ 对尺度更有用或无差别')

    print('\n--- Q2: 分维度优势求解器 mAP 收益? ---')
    print(f'  baseline [1,1,1,1]: mAP={base["mAP"]:.4f}')
    print(f'  center   [1,1,0,0]: mAP={center["mAP"]:.4f}  Δ={center["mAP"]-base["mAP"]:+.4f}')
    print(f'  size     [0,0,1,1]: mAP={size["mAP"]:.4f}  Δ={size["mAP"]-base["mAP"]:+.4f}')
    print(f'  all-Euler[0,0,0,0]: mAP={euler["mAP"]:.4f}  Δ={euler["mAP"]-base["mAP"]:+.4f}')
    hybrid = all_results[4] if len(all_results) > 4 else None
    if hybrid is not None:
        print(f'  hybrid   cxcyDPM++/whHeun: mAP={hybrid["mAP"]:.4f}  '
              f'Δ={hybrid["mAP"]-base["mAP"]:+.4f}  lat={hybrid["avg_latency_ms"]:.1f}ms')
    best_split = max([center, size] + ([hybrid] if hybrid else []), key=lambda r: r['mAP'])
    if best_split['mAP'] > base['mAP'] + 0.001:
        print(f'  → Q2 结论: 最优配置 ({best_split["label"]}) mAP +{best_split["mAP"]-base["mAP"]:.4f}, '
              f'超 noise 阈值 0.001, 有收益')
    else:
        print(f'  → Q2 结论: 全部分维度配置 mAP Δ≤0.001 (noise 范围内), 无显著收益')

    if hybrid is not None:
        print('\n--- Q3: w/h 用 Heun 2阶 vs Euler 1阶 (cx/cy 均用 DPM++)? ---')
        print(f'  [1,1,0,0] w/h Euler 1阶: mAP={center["mAP"]:.4f}  '
              f'w_L1={center["per_dim_l1"]["w"]:.5f} h_L1={center["per_dim_l1"]["h"]:.5f}  '
              f'lat={center["avg_latency_ms"]:.1f}ms')
        print(f'  hybrid    w/h Heun  2阶: mAP={hybrid["mAP"]:.4f}  '
              f'w_L1={hybrid["per_dim_l1"]["w"]:.5f} h_L1={hybrid["per_dim_l1"]["h"]:.5f}  '
              f'lat={hybrid["avg_latency_ms"]:.1f}ms')
        d_map = hybrid["mAP"] - center["mAP"]
        d_w = hybrid["per_dim_l1"]["w"] - center["per_dim_l1"]["w"]
        d_h = hybrid["per_dim_l1"]["h"] - center["per_dim_l1"]["h"]
        print(f'  ΔmAP={d_map:+.4f}  Δw_L1={d_w:+.6f}  Δh_L1={d_h:+.6f}  '
              f'Δlat={hybrid["avg_latency_ms"]-center["avg_latency_ms"]:+.1f}ms')
        if abs(d_map) <= 0.001 and abs(d_w) < 1e-4 and abs(d_h) < 1e-4:
            print(f'  → Q3 结论: Heun 2阶 vs Euler 1阶 对 w/h 无显著差异 (低曲率维度, '
                  f'梯形≈前向; 网络重预测 x0 补偿求解器差异). 额外 NFE 无收益.')
        else:
            print(f'  → Q3 结论: Heun 对 w/h 有可测量差异, 见上方数据.')

    # 保存
    output = {
        'config': args.config, 'checkpoint': args.checkpoint, 'ann_file': args.ann,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'box_renewal': box_renewal,
        'note': (
            f'dim_d1_mask 基类路径, box_renewal {"ON" if box_renewal else "OFF"}. '
            + ('OFF 时 renewal_mask≡None, D1 作用于全体 proposal, 干净隔离分维度效果.'
               if not box_renewal else
               'ON 时 renewal_mask 置零被 renewal 的 proposal D1, 可能掩盖 dim_d1_mask 效果.')
        ),
        'results': all_results,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {output_path}')


if __name__ == '__main__':
    main()
