"""E1.5 Blocking 诊断: 测量 baseline 在大 t 段是否已 MMSE 最优

FEASIBLE_TRIP.md §10 Phase 1 要求 E1.5 Blocking 判据:
  ratio(t) = ||f_θ(x_t, t) - μ_p^c|| / ||x_0 - μ_p^c||   (cxcywh 归一化空间)
  - ratio → 0: 网络输出 ≈ μ_p (已 MMSE 最优, TRIP 收益有限)
  - ratio 显著 > 0: 网络远离 μ_p (非 MMSE, TRIP 有正则化收益空间)

机制:
  复用 head.loss() 前向路径 (head.py:538-575), 通过 monkey-patch:
    - _sample_t → 固定 t (而非随机采样)
    - _build_training_targets → 捕获 x_starts (x_0=GT raw cxcywh) + matched_gt_indices
  + forward hook 捕获 all_pred_bboxes = f̂_θ(x_t, t)
  不重新实现前向逻辑, 保证与训练一致.

用法:
  python experiments/runners/diag_trip_e15.py \
      --config experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \
      --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \
      --priors data/class_priors_24obj.pkl \
      --gpu-id 0 --num-samples 50 \
      --output work_dirs/trip_24obj/e15_diag.json
"""

import argparse
import json
import os
import sys

import torch

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mmengine.config import Config
from mmengine.runner import Runner

from ldmdet.data.structures import ImageMeta


def parse_args():
    p = argparse.ArgumentParser(description='E1.5 Blocking 诊断')
    p.add_argument('config', help='Baseline 配置 (a4_dpm_pp_24obj.py)')
    p.add_argument('--checkpoint', required=True, help='Baseline checkpoint')
    p.add_argument('--priors', required=True, help='class_priors pickle 路径')
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--num-samples', type=int, default=50, help='评估样本数')
    p.add_argument('--output', required=True, help='输出 JSON 路径')
    p.add_argument(
        '--t-values', type=float, nargs='+',
        default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
        help='评估的 t 网格 (RF 路径 [0,1])',
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    cfg = Config.fromfile(args.config)
    cfg.work_dir = os.path.dirname(args.checkpoint) or 'work_dirs/diag'

    # 抑制 SwanLab (诊断无需可视化)
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='DetLocalVisualizer', vis_backends=cfg.vis_backends, name='visualizer'
    )

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)

    model = runner.model
    model.eval()
    head = model.bbox_head
    device = next(model.parameters()).device

    # 加载类先验
    priors = torch.load(args.priors, map_location='cpu')
    mu_p = priors['mu'].to(device)  # [num_classes, 4] cxcywh [0,1]
    sigma_bar_sq = priors['sigma_bar_sq'].to(device)
    num_classes = head.num_classes
    assert mu_p.shape[0] == num_classes, (
        f"priors num_classes={mu_p.shape[0]} != head {num_classes}"
    )
    snr_scale = head.snr_scale

    print(f"[E1.5] model loaded from {args.checkpoint}")
    print(f"[E1.5] priors: mu={mu_p.shape}, sigma_bar_sq={sigma_bar_sq.shape}")
    print(f"[E1.5] snr_scale={snr_scale}, diffusion_type={head.diffusion_type}")
    print(f"[E1.5] t_values={args.t_values}, num_samples={args.num_samples}")

    # 保存原始方法
    orig_sample_t = head._sample_t
    orig_build_targets = head._build_training_targets

    # 捕获容器
    captured = {}

    def make_fixed_sample_t(t_value):
        def _fixed(bs, device):
            return torch.full((bs,), t_value, device=device, dtype=torch.float32)
        return _fixed

    def _capturing_build_targets(bs, device, t, targets, gt_bboxes,
                                  external_noise=None, img_ids=None):
        out = orig_build_targets(
            bs, device, t, targets, gt_bboxes,
            external_noise=external_noise, img_ids=img_ids,
        )
        captured['x_starts'] = out[1]  # list[bs] of [N, 4] raw cxcywh
        captured['matched_gt_indices'] = out[3]  # list[bs] of [N]
        return out

    # forward hook 捕获 all_pred_bboxes
    def _forward_hook(module, inputs, output):
        # output = (all_cls_logits, all_pred_bboxes, all_curr_proposals)
        captured['all_pred_bboxes'] = output[1]

    val_loader = runner.val_dataloader

    # results[t_value] = list of per-proposal ratios
    results = {tv: [] for tv in args.t_values}
    n_processed = 0

    with torch.no_grad():
        for data in val_loader:
            if n_processed >= args.num_samples:
                break

            # data_preprocessor: 接收 dataloader batch, 返回 dict (mmengine 0.10.5)
            pp_out = model.data_preprocessor(data)
            if isinstance(pp_out, dict):
                batch_inputs = pp_out['inputs']
                batch_data_samples = pp_out['data_samples']
            else:
                batch_inputs, batch_data_samples = pp_out
            features = model.extract_feat(batch_inputs)

            # 提取 img_metas / gt_bboxes / gt_labels (与 detector.loss 一致)
            img_metas, gt_bboxes, gt_labels = [], [], []
            for ds in batch_data_samples:
                img_metas.append(ImageMeta(
                    img_shape=ds.metainfo['img_shape'],
                    pad_shape=ds.metainfo.get('pad_shape'),
                    ori_shape=ds.metainfo.get('ori_shape'),
                    scale_factor=ds.metainfo.get('scale_factor'),
                    img_id=ds.metainfo.get('img_id'),
                ))
                gt_bboxes.append(ds.gt_instances.bboxes)
                gt_labels.append(ds.gt_instances.labels)

            for t_value in args.t_values:
                # 注入 monkey-patch
                head._sample_t = make_fixed_sample_t(t_value)
                head._build_training_targets = _capturing_build_targets
                handle = head.register_forward_hook(_forward_hook)

                try:
                    head.loss(features, img_metas, gt_bboxes, gt_labels)
                finally:
                    head._sample_t = orig_sample_t
                    head._build_training_targets = orig_build_targets
                    handle.remove()

                # 提取预测: all_pred_bboxes [num_heads, bs, N, 4] pixel xyxy
                all_pred = captured.get('all_pred_bboxes')
                x_starts = captured.get('x_starts')
                matched_idx = captured.get('matched_gt_indices')
                if all_pred is None or x_starts is None:
                    continue

                # 归一化预测 → main head [-1] → normalized xyxy [0,1] → cxcywh [0,1]
                norm_pred_xyxy = head._normalize_pred_bboxes(
                    all_pred.float(), img_metas
                )  # [num_heads, bs, N, 4]
                main_pred_xyxy = norm_pred_xyxy[-1]  # [bs, N, 4]

                from ldmdet.utils.box_ops import bbox_xyxy_to_cxcywh
                pred_cxcywh = bbox_xyxy_to_cxcywh(main_pred_xyxy)  # [bs, N, 4] [0,1]

                bs = len(gt_bboxes)
                for i in range(bs):
                    if gt_bboxes[i].shape[0] == 0:
                        continue
                    m_idx = matched_idx[i]  # [N] proposal→GT index
                    x_start_raw = x_starts[i]  # [N, 4] raw cxcywh [-snr, snr]
                    # x_0 归一化 cxcywh [0,1]
                    x0_norm = (x_start_raw / snr_scale + 1) / 2
                    # 每 proposal 的 GT 类
                    gt_label_per_prop = gt_labels[i][m_idx.clamp(max=gt_labels[i].shape[0]-1)]
                    mu_p_per_prop = mu_p[gt_label_per_prop]  # [N, 4]
                    # ratio = ||pred - mu_p|| / ||x0 - mu_p||  (per proposal, L2 over 4 dims)
                    num = (pred_cxcywh[i] - mu_p_per_prop).norm(dim=-1)  # [N]
                    den = (x0_norm - mu_p_per_prop).norm(dim=-1).clamp(min=1e-6)  # [N]
                    ratio = (num / den).cpu().tolist()
                    results[t_value].extend(ratio)

            n_processed += len(batch_data_samples)
            if n_processed % 10 == 0 or n_processed >= args.num_samples:
                print(f"[E1.5] processed {n_processed}/{args.num_samples} images")

    # 汇总
    summary = {}
    print(f"\n{'='*60}")
    print(f"E1.5 Blocking 诊断结果 (baseline = {os.path.basename(args.checkpoint)})")
    print(f"{'='*60}")
    print(f"{'t':>6} {'n_prop':>8} {'mean':>8} {'median':>8} {'std':>8} {'p10':>8} {'p90':>8}")
    for tv in args.t_values:
        ratios = results[tv]
        if len(ratios) == 0:
            summary[str(tv)] = None
            print(f"{tv:>6.2f} {'N/A':>8}")
            continue
        t_tensor = torch.tensor(ratios)
        s = {
            'n_proposals': len(ratios),
            'mean': t_tensor.mean().item(),
            'median': t_tensor.median().item(),
            'std': t_tensor.std().item(),
            'p10': t_tensor.quantile(0.1).item(),
            'p90': t_tensor.quantile(0.9).item(),
        }
        summary[str(tv)] = s
        print(f"{tv:>6.2f} {s['n_proposals']:>8} {s['mean']:>8.4f} "
              f"{s['median']:>8.4f} {s['std']:>8.4f} {s['p10']:>8.4f} {s['p90']:>8.4f}")

    print(f"\n判据:")
    print(f"  ratio → 0 (大 t 段): baseline 已 MMSE 最优 → TRIP 收益有限")
    print(f"  ratio 显著 > 0 (大 t 段): baseline 非 MMSE → TRIP 有正则化收益空间")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'priors': args.priors,
            'num_samples': n_processed,
            't_values': args.t_values,
            'summary': summary,
        }, f, indent=2)
    print(f"\n[E1.5] 结果已保存到 {args.output}")


if __name__ == '__main__':
    main()
