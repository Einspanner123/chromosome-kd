#!/usr/bin/env python3
"""v_next 范数诊断: 实证 Heun 数值不稳定根因

在 hybrid 求解器 (cx/cy DPM++, w/h Heun) 上运行少量图像,
捕获 RFDPMSolverHybrid.v_next_diag_history, 验证:

  假设: Heun 的 v_next = (x_euler - x0_next) / t_next 在 t_next 小时爆炸,
        导致 w/h 维度预测退化 (hybrid 3-seed mAP 0.858 vs baseline 0.863).

预期: ratio_v_next_over_v_t 在后期 step (t_next 小) 显著增大 (>10x),
      证明 v_next 数值不稳定是 hybrid 掉点的根因.

输出: work_dirs/diagnosis/vnext_instability.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(
        _PROJECT_ROOT,
        'experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py'))
    parser.add_argument('--checkpoint', default=os.path.join(
        _PROJECT_ROOT, 'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth'))
    parser.add_argument('--ann', default=os.path.join(
        _PROJECT_ROOT, 'data/24_chromosomes_object/coco/valid/_annotations.coco.json'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n-imgs', type=int, default=20)
    args = parser.parse_args()
    device = f'cuda:{args.gpu}'

    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS

    cfg = Config.fromfile(args.config)
    cfg.model.bbox_head.solver_type = 'dpm_pp_heun_hybrid'
    cfg.model.bbox_head.box_renewal = False
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='DetLocalVisualizer', vis_backends=cfg.vis_backends, name='vis')

    model = init_detector(cfg, args.checkpoint, device=device)
    model.eval()

    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    n_imgs = min(args.n_imgs, len(dataset))

    # 捕获 solver 实例 (通过猴子补丁 create_dpm_solver)
    _solver_holder = [None]
    _orig_create = model.bbox_head._sampler.create_dpm_solver

    def _capturing_create():
        s = _orig_create()
        if s is not None:
            _solver_holder[0] = s
        return s

    model.bbox_head._sampler.create_dpm_solver = _capturing_create

    print('=' * 70)
    print('v_next 范数诊断 (Heun 数值不稳定根因实证)')
    print('=' * 70)
    print(f'求解器: hybrid (cx/cy DPM++, w/h Heun)')
    print(f'图像数: {n_imgs}')
    print()

    all_diag = []
    with torch.no_grad():
        for i in range(n_imgs):
            data = dataset[i]
            data['inputs'] = data['inputs'].unsqueeze(0).to(device)
            if not isinstance(data['data_samples'], list):
                data['data_samples'] = [data['data_samples']]
            _ = model.test_step(data)

            # 提取本图的 v_next 诊断
            if _solver_holder[0] is not None and hasattr(_solver_holder[0], 'v_next_diag_history'):
                diag = _solver_holder[0].v_next_diag_history
                all_diag.append(list(diag))  # copy
                if i == 0:
                    # 首图详细打印
                    print(f'首图 (img {i}) v_next 诊断:')
                    print(f'{"step":>5} {"t_n":>10} {"t_next":>10} {"||v_t||":>12} {"||v_next||":>12} {"ratio":>10}')
                    print('-' * 65)
                    for d in diag:
                        print(f'{d["step_idx"]:>5} {d["t_n"]:>10.6f} {d["t_next"]:>10.6f} '
                              f'{d["v_t_norm"]:>12.6f} {d["v_next_norm"]:>12.6f} '
                              f'{d["ratio_v_next_over_v_t"]:>10.2f}')

    # 跨图统计: 每个 step 的 v_next/v_t ratio 均值
    print(f'\n跨 {n_imgs} 图统计 (每 step 的 ratio_v_next_over_v_t):')
    print(f'{"step":>5} {"t_n":>10} {"t_next":>10} {"ratio_mean":>12} {"ratio_max":>12} {"ratio_std":>12}')
    print('-' * 75)
    n_steps = len(all_diag[0]) if all_diag else 0
    step_stats = []
    for s in range(n_steps):
        ratios = [all_diag[img][s]['ratio_v_next_over_v_t'] for img in range(len(all_diag))
                  if s < len(all_diag[img])]
        t_n = all_diag[0][s]['t_n']
        t_next = all_diag[0][s]['t_next']
        r_mean = float(np.mean(ratios))
        r_max = float(np.max(ratios))
        r_std = float(np.std(ratios))
        step_stats.append({
            'step': s, 't_n': t_n, 't_next': t_next,
            'ratio_mean': round(r_mean, 4), 'ratio_max': round(r_max, 4),
            'ratio_std': round(r_std, 4),
        })
        print(f'{s:>5} {t_n:>10.6f} {t_next:>10.6f} '
              f'{r_mean:>12.4f} {r_max:>12.4f} {r_std:>12.4f}')

    # 判定
    print('\n判定:')
    max_ratio = max(s['ratio_mean'] for s in step_stats) if step_stats else 0
    if max_ratio > 10:
        print(f'  ratio 最大均值 = {max_ratio:.2f} (>10), v_next 在小 t_next step 爆炸,')
        print(f'  → 实证 Heun 数值不稳定是 hybrid w/h 退化根因')
    elif max_ratio > 3:
        print(f'  ratio 最大均值 = {max_ratio:.2f} (>3), v_next 有放大但不极端,')
        print(f'  → Heun 不稳定是部分原因, 可能还有其他因素')
    else:
        print(f'  ratio 最大均值 = {max_ratio:.2f} (≤3), v_next 未爆炸,')
        print(f'  → Heun 不稳定假设不成立, hybrid 掉点另有根因')

    # 保存
    output_path = os.path.join(
        _PROJECT_ROOT, 'work_dirs', 'diagnosis', 'vnext_instability.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {
        'solver': 'hybrid (cx/cy DPM++, w/h Heun)',
        'n_imgs': n_imgs,
        'n_steps': n_steps,
        'first_img_diag': all_diag[0] if all_diag else [],
        'step_stats': step_stats,
        'max_ratio_mean': max_ratio,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {output_path}')


if __name__ == '__main__':
    main()
