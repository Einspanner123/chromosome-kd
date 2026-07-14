#!/usr/bin/env python3
"""推理优化阈值扫描 — 快速确定可行配置

在 SOTA checkpoint 上测试不同阈值配置的:
1. 推理时间 (per-sample, CUDA synchronized)
2. 退出触发率 (IO1: 步级退出率, IO4: 头级退出率)
3. mAP (在全 val 集上用 COCO evaluator)

Usage:
    conda run -n chromo-new python experiments/analysis/threshold_sweep.py \
        --config experiments/configs/ldmdet/ldmdet_rf_heun_adaln_stochot_eps5.py \
        --checkpoint work_dirs/a3_full_sota_24obj/best_coco_bbox_mAP_epoch_114.pth \
        --gpu-id 0 --num-timing-images 50

输出:
    work_dirs/inference_opt/threshold_sweep_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

# Path setup
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import mmdet.models  # noqa: F401
import mmdet.datasets  # noqa: F401
from mmengine.config import Config
from mmengine.registry import init_default_scope

init_default_scope("mmdet")
from experiments.mmdet_bridge.registry import register_all

register_all()

from ldmdet.data.structures import ImageMeta


def load_model(config_path, checkpoint_path, device="cuda"):
    from mmdet.registry import MODELS

    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model, cfg


def load_val_samples(cfg, num_samples):
    from pycocotools.coco import COCO
    from mmcv import imread, imresize

    val_cfg = cfg.val_dataloader.dataset
    ann_file = osp.join(val_cfg.data_root, val_cfg.ann_file)
    img_prefix = osp.join(val_cfg.data_root, val_cfg.data_prefix["img"])

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    if num_samples < len(img_ids):
        indices = np.linspace(0, len(img_ids) - 1, num_samples, dtype=int)
        img_ids = [img_ids[i] for i in indices]

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    samples = []
    cat_ids = coco.getCatIds()
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = osp.join(img_prefix, img_info["file_name"])
        img = imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = min(1333 / w, 800 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = imresize(img, (new_w, new_h))

        pad_h = (32 - new_h % 32) % 32
        pad_w = (32 - new_w % 32) % 32
        img_padded = np.zeros((new_h + pad_h, new_w + pad_w, 3), dtype=np.uint8)
        img_padded[:new_h, :new_w] = img

        img_rgb = img_padded[:, :, ::-1].astype(np.float32)
        img_rgb = (img_rgb - mean) / std
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            gt_boxes.append([x * scale, y * scale, (x + bw) * scale, (y + bh) * scale])
            gt_labels.append(cat_id_to_idx[ann["category_id"]])

        samples.append({
            "img_id": img_id,
            "img": img_tensor,
            "gt_boxes": torch.tensor(gt_boxes).float() if gt_boxes else torch.zeros(0, 4),
            "gt_labels": torch.tensor(gt_labels).long() if gt_labels else torch.zeros(0, dtype=torch.long),
            "img_meta": ImageMeta(
                img_shape=(new_h + pad_h, new_w + pad_w),
                ori_shape=(h, w),
                scale_factor=scale,
            ),
        })

    return samples


def measure_timing(model, samples, device, n_images=50, warmup=5):
    """测量平均推理时间 (ms), 返回 per-sample 时间列表."""
    head = model.bbox_head
    n_eval = min(n_images, len(samples))

    # Warmup
    for i in range(min(warmup, n_eval)):
        img = samples[i]["img"].to(device)
        img_metas = [samples[i]["img_meta"]]
        with torch.no_grad():
            feats = model.backbone(img)
            feats = model.neck(feats)
            _ = head.predict(feats, img_metas, rescale=False)
    torch.cuda.synchronize()

    # Measure
    times = []
    for i in range(n_eval):
        img = samples[i]["img"].to(device)
        img_metas = [samples[i]["img_meta"]]

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            feats = model.backbone(img)
            feats = model.neck(feats)
            _ = head.predict(feats, img_metas, rescale=False)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)

    return times


def count_exits_io1(model, samples, device, n_images=50):
    """统计 IO1 步级退出率."""
    head = model.bbox_head
    n_eval = min(n_images, len(samples))
    exit_counts = 0
    total_counts = 0

    for i in range(n_eval):
        img = samples[i]["img"].to(device)
        img_metas = [samples[i]["img_meta"]]
        with torch.no_grad():
            feats = model.backbone(img)
            feats = model.neck(feats)
            _ = head.predict(feats, img_metas, rescale=False)

        total_counts += 1
        if head._step_exit_stats and 'exit_step_idx' in head._step_exit_stats:
            exit_counts += 1

    return exit_counts, total_counts


def count_exits_io4(model, samples, device, n_images=50):
    """统计 IO4 头级退出率."""
    head = model.bbox_head
    n_eval = min(n_images, len(samples))
    exit_counts = 0
    total_counts = 0
    active_heads_list = []

    for i in range(n_eval):
        img = samples[i]["img"].to(device)
        img_metas = [samples[i]["img_meta"]]
        with torch.no_grad():
            feats = model.backbone(img)
            feats = model.neck(feats)
            _ = head.predict(feats, img_metas, rescale=False)

        total_counts += 1
        if head._exit_stats and 'exit_head_idx' in head._exit_stats:
            exit_counts += 1
            active_heads_list.append(head._exit_stats['exit_head_idx'] + 1)
        else:
            active_heads_list.append(head.num_heads)

    return exit_counts, total_counts, active_heads_list


def run_io1_sweep(model, samples, device, n_images=50):
    """IO1 阈值扫描."""
    print("\n" + "=" * 70)
    print("  IO1 步级提前终止 — 阈值扫描")
    print("=" * 70)

    head = model.bbox_head
    # 保存原始状态
    orig_enabled = head.step_early_exit_enabled
    orig_thr = head.step_exit_threshold
    orig_cls_thr = head.step_exit_cls_threshold

    configs = [
        ("baseline (disabled)", None, None, None),
        ("thr=0.01, cls=0.95", 0.01, 0.95, True),
        ("thr=0.02, cls=0.92", 0.02, 0.92, True),
        ("thr=0.03, cls=0.90", 0.03, 0.90, True),
        ("thr=0.05, cls=0.85", 0.05, 0.85, True),
        ("thr=0.10, cls=0.80", 0.10, 0.80, True),
    ]

    results = []
    for name, thr, cls_thr, enabled in configs:
        if enabled is None:
            head.step_early_exit_enabled = False
        else:
            head.step_early_exit_enabled = True
            head.step_exit_threshold = thr
            head.step_exit_cls_threshold = cls_thr

        times = measure_timing(model, samples, device, n_images)
        exits, total = count_exits_io1(model, samples, device, n_images)

        result = {
            "config": name,
            "threshold": thr,
            "cls_threshold": cls_thr,
            "mean_ms": round(float(np.mean(times)), 2),
            "p50_ms": round(float(np.percentile(times, 50)), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "exit_rate": round(exits / total, 4) if total > 0 else 0,
            "exit_count": exits,
            "total_count": total,
        }
        results.append(result)
        print(f"\n  {name}")
        print(f"    Mean: {result['mean_ms']:.1f}ms | P50: {result['p50_ms']:.1f}ms | P95: {result['p95_ms']:.1f}ms")
        print(f"    Exit rate: {result['exit_rate']*100:.1f}% ({exits}/{total})")

    # 恢复
    head.step_early_exit_enabled = orig_enabled
    head.step_exit_threshold = orig_thr
    head.step_exit_cls_threshold = orig_cls_thr

    return results


def run_io4_sweep(model, samples, device, n_images=50):
    """IO4 阈值扫描."""
    print("\n" + "=" * 70)
    print("  IO4 级联头提前退出 — 阈值扫描")
    print("=" * 70)

    head = model.bbox_head
    orig_enabled = head.head_early_exit_enabled
    orig_thr = head.head_exit_box_threshold
    orig_cls_thr = head.head_exit_cls_threshold
    orig_time_aware = head.head_exit_time_aware
    orig_min_heads = head.head_exit_min_heads

    configs = [
        ("baseline (disabled)", None, None, None, None),
        ("thr=0.005, cls=0.98, time-aware", 0.005, 0.98, True, 3),
        ("thr=0.01, cls=0.95, fixed", 0.01, 0.95, False, 3),
        ("thr=0.02, cls=0.92, fixed", 0.02, 0.92, False, 3),
        ("thr=0.05, cls=0.90, fixed", 0.05, 0.90, False, 2),
        ("thr=0.10, cls=0.85, fixed", 0.10, 0.85, False, 2),
    ]

    results = []
    for name, thr, cls_thr, time_aware, min_heads in configs:
        if thr is None:
            head.head_early_exit_enabled = False
        else:
            head.head_early_exit_enabled = True
            head.head_exit_box_threshold = thr
            head.head_exit_cls_threshold = cls_thr
            head.head_exit_time_aware = time_aware if time_aware is not None else False
            head.head_exit_min_heads = min_heads if min_heads is not None else 3

        times = measure_timing(model, samples, device, n_images)
        exits, total, active_list = count_exits_io4(model, samples, device, n_images)

        avg_active = float(np.mean(active_list)) if active_list else head.num_heads
        result = {
            "config": name,
            "threshold": thr,
            "cls_threshold": cls_thr,
            "time_aware": time_aware,
            "min_heads": min_heads,
            "mean_ms": round(float(np.mean(times)), 2),
            "p50_ms": round(float(np.percentile(times, 50)), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "exit_rate": round(exits / total, 4) if total > 0 else 0,
            "exit_count": exits,
            "total_count": total,
            "avg_active_heads": round(avg_active, 2),
            "saving_ratio": round(1.0 - avg_active / head.num_heads, 4),
        }
        results.append(result)
        print(f"\n  {name}")
        print(f"    Mean: {result['mean_ms']:.1f}ms | P50: {result['p50_ms']:.1f}ms | P95: {result['p95_ms']:.1f}ms")
        print(f"    Exit rate: {result['exit_rate']*100:.1f}% ({exits}/{total})")
        print(f"    Avg active heads: {result['avg_active_heads']:.1f}/{head.num_heads} (saving {result['saving_ratio']*100:.1f}%)")

    # 恢复
    head.head_early_exit_enabled = orig_enabled
    head.head_exit_box_threshold = orig_thr
    head.head_exit_cls_threshold = orig_cls_thr
    head.head_exit_time_aware = orig_time_aware
    head.head_exit_min_heads = orig_min_heads

    return results


def run_io3_sweep(model, samples, device, n_images=50):
    """IO3 K 值扫描."""
    print("\n" + "=" * 70)
    print("  IO3 Top-K 框剪枝 — K 值扫描")
    print("=" * 70)

    head = model.bbox_head
    orig_enabled = head.topk_pruning_enabled
    orig_k = head.topk_k
    orig_step = head.topk_pruning_step

    configs = [
        ("baseline (N=500, no pruning)", None),
        ("K=300", 300),
        ("K=200", 200),
        ("K=150", 150),
        ("K=100", 100),
    ]

    results = []
    for name, k in configs:
        if k is None:
            head.topk_pruning_enabled = False
        else:
            head.topk_pruning_enabled = True
            head.topk_k = k
            head.topk_pruning_step = 0

        times = measure_timing(model, samples, device, n_images)

        result = {
            "config": name,
            "k": k,
            "mean_ms": round(float(np.mean(times)), 2),
            "p50_ms": round(float(np.percentile(times, 50)), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
        }
        results.append(result)
        print(f"\n  {name}")
        print(f"    Mean: {result['mean_ms']:.1f}ms | P50: {result['p50_ms']:.1f}ms | P95: {result['p95_ms']:.1f}ms")

    # 恢复
    head.topk_pruning_enabled = orig_enabled
    head.topk_k = orig_k
    head.topk_pruning_step = orig_step

    return results


def run_convergence_analysis(model, samples, device, n_images=50):
    """分析实际收敛分布, 为阈值选择提供依据."""
    print("\n" + "=" * 70)
    print("  收敛分布分析 (为阈值选择提供依据)")
    print("=" * 70)

    head = model.bbox_head
    device = next(head.parameters()).device
    time_pairs = head._sampler.build_time_pairs(device)
    n_steps = len(time_pairs)
    n_eval = min(n_images, len(samples))

    # IO1: x0 步间相对变化
    io1_stats = [[] for _ in range(n_steps)]  # step_idx -> list of rel_change
    io1_cls_stats = [[] for _ in range(n_steps)]

    # IO4: head 间 box 相对变化
    n_heads = head.num_heads
    io4_stats = [[] for _ in range(n_heads - 1)]  # head_pair -> list of rel_change
    io4_cls_stats = [[] for _ in range(n_heads - 1)]

    for i in range(n_eval):
        img = samples[i]["img"].to(device)
        img_metas = [samples[i]["img_meta"]]

        with torch.no_grad():
            feats = model.backbone(img)
            feats = model.neck(feats)
            results, trajectory = head.predict(
                feats, img_metas, rescale=False, return_trajectory=True
            )

        if len(trajectory) < 2:
            continue

        # IO1: x0 步间变化
        x0_list = []
        for cls_logits, pred_bboxes in trajectory:
            x0 = head._sampler.xyxy_to_raw(pred_bboxes, img_metas)
            x0_list.append(x0)

        for step_idx in range(1, n_steps):
            x0_curr = x0_list[step_idx].squeeze(0)
            x0_prev = x0_list[step_idx - 1].squeeze(0)
            delta = (x0_curr - x0_prev).norm(dim=-1)
            magnitude = x0_curr.norm(dim=-1).clamp(min=1e-6)
            rel_delta = (delta / magnitude).mean().item()
            io1_stats[step_idx].append(rel_delta)

            cls_curr = trajectory[step_idx][0].squeeze(0)
            cls_prev = trajectory[step_idx - 1][0].squeeze(0)
            consistency = (cls_curr.argmax(-1) == cls_prev.argmax(-1)).float().mean().item()
            io1_cls_stats[step_idx].append(consistency)

    # 打印 IO1 分布
    print(f"\n  IO1: x0 步间相对变化分布 ({n_eval} images)")
    print(f"  {'Step':>4} {'t_curr':>7} {'mean':>10} {'p50':>10} {'p25':>10} {'<0.01%':>8} {'<0.05%':>8} {'cls_cons':>10}")
    print(f"  {'-'*4} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")

    io1_result = []
    for step_idx in range(1, n_steps):
        vals = io1_stats[step_idx]
        cls_vals = io1_cls_stats[step_idx]
        if not vals:
            continue
        mean_v = float(np.mean(vals))
        p50 = float(np.percentile(vals, 50))
        p25 = float(np.percentile(vals, 25))
        lt_001 = sum(1 for v in vals if v < 0.01) / len(vals) * 100
        lt_005 = sum(1 for v in vals if v < 0.05) / len(vals) * 100
        cls_mean = float(np.mean(cls_vals))
        t_curr = time_pairs[step_idx][0]
        print(f"  {step_idx:>4} {t_curr:>7.3f} {mean_v:>10.5f} {p50:>10.5f} {p25:>10.5f} {lt_001:>7.1f}% {lt_005:>7.1f}% {cls_mean:>10.4f}")
        io1_result.append({
            "step": step_idx,
            "t_curr": float(t_curr),
            "mean": round(mean_v, 5),
            "p50": round(p50, 5),
            "p25": round(p25, 5),
            "lt_001_pct": round(lt_001, 1),
            "lt_005_pct": round(lt_005, 1),
            "cls_consistency": round(cls_mean, 4),
        })

    return {"io1_distribution": io1_result}


def main():
    parser = argparse.ArgumentParser(description="推理优化阈值扫描")
    parser.add_argument(
        "--config",
        default="experiments/configs/ldmdet/ldmdet_rf_heun_adaln_stochot_eps5.py",
    )
    parser.add_argument(
        "--checkpoint",
        default="work_dirs/a3_full_sota_24obj/best_coco_bbox_mAP_epoch_114.pth",
    )
    parser.add_argument("--num-timing-images", type=int, default=50)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output-dir", default="work_dirs/inference_opt")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Loading model...")
    model, cfg = load_model(args.config, args.checkpoint, device)

    print(f"Loading {args.num_timing_images} val samples...")
    samples = load_val_samples(cfg, args.num_timing_images)
    print(f"  Loaded {len(samples)} samples")

    report = {"config": args.config, "checkpoint": args.checkpoint}

    # 1. 收敛分布分析
    report["convergence"] = run_convergence_analysis(
        model, samples, device, n_images=args.num_timing_images
    )

    # 2. IO1 阈值扫描
    report["io1_sweep"] = run_io1_sweep(
        model, samples, device, n_images=args.num_timing_images
    )

    # 3. IO4 阈值扫描
    report["io4_sweep"] = run_io4_sweep(
        model, samples, device, n_images=args.num_timing_images
    )

    # 4. IO3 K 值扫描
    report["io3_sweep"] = run_io3_sweep(
        model, samples, device, n_images=args.num_timing_images
    )

    # 保存
    output_path = output_dir / "threshold_sweep_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n{'='*70}")
    print(f"报告已保存: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
