#!/usr/bin/env python3
"""LDMDet 推理优化方向验证 (IO1-IO5)

在 SOTA checkpoint 上实测验证 5 个推理优化方向的核心假设:
- IO1: 采样步数收敛性 (x0_pred 步间相对变化)
- IO3: proposal 冗余度 (score 分布 + N-scaling 延迟)
- IO4: 级联头收敛性 (head 间预测差异)
- IO5: 框位移分布 (跨步 box 位移直方图)

同时测量组件级延迟 (RoIAlign / Self-Attn / DynamicConv / FFN / cls+reg),
为实施优先级提供数据支撑。

Usage:
    python experiments/analysis/benchmark_inference.py \
        --config experiments/configs/ldmdet/ldmdet_rf_heun_adaln_stochot_eps5.py \
        --checkpoint work_dirs/reproduce_0751_stochot_eps5_v2/best_coco_bbox_mAP_epoch_59.pth \
        --num-images 50 --gpu-id 0

输出:
    work_dirs/inference_opt/benchmark_report.json
    控制台打印汇总表
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Module registration ───────────────────────────────────────────────────────
import mmdet.models  # noqa: F401
import mmdet.datasets  # noqa: F401
from mmengine.config import Config
from mmengine.registry import init_default_scope

init_default_scope("mmdet")
from experiments.mmdet_bridge.registry import register_all

register_all()

from ldmdet.core.single_head import SingleDiffusionDetHead
from ldmdet.data.structures import ImageMeta


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers: model / data loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """加载模型和 checkpoint."""
    from mmdet.registry import MODELS

    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model, cfg


def load_val_samples(cfg, num_samples: int):
    """加载验证集样本 (含 resize + ImageNet 归一化)."""
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

    # ImageNet 归一化 (与 DetDataPreprocessor 一致)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # RGB
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    samples = []
    cat_ids = coco.getCatIds()
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = osp.join(img_prefix, img_info["file_name"])
        img = imread(img_path)  # BGR (OpenCV)
        if img is None:
            continue

        # Resize: (1333, 800) keep_ratio (与 test_pipeline 一致)
        h, w = img.shape[:2]
        scale = min(1333 / w, 800 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = imresize(img, (new_w, new_h))

        # Pad to divisible by 32
        pad_h = (32 - new_h % 32) % 32
        pad_w = (32 - new_w % 32) % 32
        img_padded = np.zeros((new_h + pad_h, new_w + pad_w, 3), dtype=np.uint8)
        img_padded[:new_h, :new_w] = img

        # BGR → RGB, normalize
        img_rgb = img_padded[:, :, ::-1].astype(np.float32)
        img_rgb = (img_rgb - mean) / std

        # to tensor [1, C, H, W]
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

        # GT
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            # Scale GT boxes to resized image
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


# ═══════════════════════════════════════════════════════════════════════════════
# Latency Profiling
# ═══════════════════════════════════════════════════════════════════════════════


class ComponentProfiler:
    """组件级延迟 profiler, 使用 CUDA events.

    对 SingleDiffusionDetHead 的各子模块注册 forward hooks,
    在 hooks 内 record CUDA events, 最后统一 synchronize 并计算耗时。
    """

    def __init__(self):
        self.records: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
            defaultdict(list)
        )
        self._handles: list = []
        self._pending: dict[int, torch.cuda.Event] = {}

    def _make_hooks(self, name: str, module: torch.nn.Module):
        def pre(_module, _input):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._pending[id(_module)] = start

        def post(_module, _input, _output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start = self._pending.pop(id(_module), None)
            if start is not None:
                self.records[name].append((start, end))

        self._handles.append(module.register_forward_pre_hook(pre))
        self._handles.append(module.register_forward_hook(post))

    def attach(self, head):
        """挂载到 DiffusionDetHead 及其子模块."""
        # RoIAlign (共享提取器, 每个 single_head 调用一次)
        self._make_hooks("roi_align", head.roi_extractor)

        # 每个 SingleDiffusionDetHead 的子组件
        for i, sh in enumerate(head.head_series):
            # 整个 single_head 总耗时
            self._make_hooks(f"single_head_total", sh)
            # 子模块
            self._make_hooks("dynamic_conv", sh.inst_interact)
            self._make_hooks("ffn_linear1", sh.linear1)
            self._make_hooks("ffn_linear2", sh.linear2)
            self._make_hooks("cls_head", sh.cls_head)
            self._make_hooks("reg_head", sh.reg_head)

        # Self-Attn: SDPA 路径不经过 nn.MultiheadAttention,
        # 需要 monkey-patch _self_attn 方法
        self._patch_self_attn()

    def _patch_self_attn(self):
        """Monkey-patch _self_attn 以捕获 SDPA 路径的 attention 耗时."""
        original = SingleDiffusionDetHead._self_attn

        def timed_self_attn(self_sh, q, k=None, v=None):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(self_sh, q, k, v)
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.records["self_attn"].append((start, end))
            return result

        SingleDiffusionDetHead._self_attn = timed_self_attn
        self._original_self_attn = original

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        # Restore original _self_attn
        if hasattr(self, "_original_self_attn"):
            SingleDiffusionDetHead._self_attn = self._original_self_attn

    def summarize(self) -> dict:
        torch.cuda.synchronize()
        result = {}
        for name, pairs in self.records.items():
            times = [s.elapsed_time(e) for s, e in pairs]
            if not times:
                continue
            result[name] = {
                "count": len(times),
                "total_ms": round(sum(times), 2),
                "mean_ms": round(sum(times) / len(times), 4),
                "p50_ms": round(float(np.percentile(times, 50)), 4),
                "p95_ms": round(float(np.percentile(times, 95)), 4),
            }
        return result


def run_latency_profiling(model, samples, device, n_images=50, warmup=5):
    """组件级延迟 profiling.

    测量:
    - backbone + FPN 延迟
    - 采样循环总延迟 (含所有 NFE × heads)
    - 每个组件的延迟: RoIAlign, Self-Attn, DynamicConv, FFN, cls+reg
    - 总推理延迟
    """
    print("\n" + "=" * 70)
    print("  [1/4] 延迟 Profiling")
    print("=" * 70)

    head = model.bbox_head
    profiler = ComponentProfiler()
    profiler.attach(head)

    n_eval = min(n_images, len(samples))

    # ── Warmup ──
    print(f"  Warmup ({warmup} images)...")
    for i in range(min(warmup, n_eval)):
        img = samples[i]["img"].to(device)
        img_metas = [samples[i]["img_meta"]]
        with torch.no_grad():
            feats = model.backbone(img)
            feats = model.neck(feats)
            _ = model.bbox_head.predict(feats, img_metas, rescale=False)
    torch.cuda.synchronize()

    # ── Profiled run ──
    print(f"  Profiling ({n_eval} images)...")

    backbone_times = []
    total_times = []

    for i in range(n_eval):
        img = samples[i]["img"].to(device)
        img_metas = [samples[i]["img_meta"]]

        torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            # Backbone + FPN
            torch.cuda.synchronize()
            t_bb = time.time()
            feats = model.backbone(img)
            feats = model.neck(feats)
            torch.cuda.synchronize()
            backbone_times.append((time.time() - t_bb) * 1000)

            # Sampling loop (predict)
            t_sample = time.time()
            _ = head.predict(feats, img_metas, rescale=False)
            torch.cuda.synchronize()
            total_times.append((time.time() - t0) * 1000)

    profiler_data = profiler.summarize()
    profiler.detach()

    # ── 汇总 ──
    n_nfe = _count_nfe(head)
    n_heads = head.num_heads
    n_single_forward = n_nfe * n_heads

    # 每组件的单次调用平均耗时 (ms)
    comp_summary = {}
    for name in ["roi_align", "self_attn", "dynamic_conv", "ffn_linear1", "ffn_linear2", "cls_head", "reg_head"]:
        if name in profiler_data:
            comp_summary[name] = profiler_data[name]

    # single_head_total 的平均 × n_single_forward = 采样循环总耗时
    sh_total = profiler_data.get("single_head_total", {})
    sh_total_ms = sh_total.get("total_ms", 0) if sh_total else 0

    # FFN = linear1 + linear2 近似
    ffn_ms = 0
    for n in ["ffn_linear1", "ffn_linear2"]:
        if n in comp_summary:
            ffn_ms += comp_summary[n]["mean_ms"]

    # 组件占比 (基于 single_head_total 的 mean)
    sh_mean = sh_total.get("mean_ms", 0) if sh_total else 0
    components_pct = {}
    if sh_mean > 0:
        for name, data in comp_summary.items():
            label = {
                "roi_align": "RoIAlign",
                "self_attn": "Self-Attn",
                "dynamic_conv": "DynamicConv",
                "ffn_linear1": "FFN(linear1)",
                "ffn_linear2": "FFN(linear2)",
                "cls_head": "cls_head",
                "reg_head": "reg_head",
            }.get(name, name)
            components_pct[label] = round(data["mean_ms"] / sh_mean * 100, 1)

    result = {
        "n_images": n_eval,
        "n_nfe_per_inference": n_nfe,
        "n_heads": n_heads,
        "n_single_head_forward": n_single_forward,
        "backbone_fpn_mean_ms": round(float(np.mean(backbone_times)), 2),
        "sampling_loop_mean_ms": round(float(np.mean(total_times)) - float(np.mean(backbone_times)), 2),
        "total_inference_mean_ms": round(float(np.mean(total_times)), 2),
        "total_inference_p50_ms": round(float(np.percentile(total_times, 50)), 2),
        "total_inference_p95_ms": round(float(np.percentile(total_times, 95)), 2),
        "single_head_mean_ms": round(sh_mean, 4),
        "component_breakdown_ms": {
            "RoIAlign": comp_summary.get("roi_align", {}).get("mean_ms", 0),
            "Self-Attn": comp_summary.get("self_attn", {}).get("mean_ms", 0),
            "DynamicConv": comp_summary.get("dynamic_conv", {}).get("mean_ms", 0),
            "FFN(linear1+linear2)": round(ffn_ms, 4),
            "cls_head": comp_summary.get("cls_head", {}).get("mean_ms", 0),
            "reg_head": comp_summary.get("reg_head", {}).get("mean_ms", 0),
        },
        "component_pct_of_single_head": components_pct,
        "raw_profiler_data": profiler_data,
    }

    # ── 打印 ──
    print(f"\n  NFE per inference: {n_nfe}  (Heun {head._sampler.sampling_timesteps}步)")
    print(f"  Single head forwards per inference: {n_single_forward}")
    print(f"\n  {'Component':<25} {'Mean (ms)':>10} {'% of head':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    for label, pct in sorted(components_pct.items(), key=lambda x: -x[1]):
        ms = result["component_breakdown_ms"].get(label, 0)
        print(f"  {label:<25} {ms:>10.4f} {pct:>9.1f}%")
    print(f"\n  Backbone+FPN:    {result['backbone_fpn_mean_ms']:.1f} ms")
    print(f"  Sampling loop:   {result['sampling_loop_mean_ms']:.1f} ms")
    print(f"  Total inference: {result['total_inference_mean_ms']:.1f} ms (p50={result['total_inference_p50_ms']:.1f}, p95={result['total_inference_p95_ms']:.1f})")

    return result


def _count_nfe(head) -> int:
    """计算实际 NFE (考虑最后一步 t_next=0 时 Heun 退化为 Euler)."""
    device = next(head.parameters()).device
    time_pairs = head._sampler.build_time_pairs(device)
    n_nfe = 0
    for t_curr, t_next in time_pairs:
        n_nfe += 1  # 主 NFE
        if head.solver_type == "heun" and t_next > 0:
            n_nfe += 1  # model_fn NFE
    return n_nfe


# ═══════════════════════════════════════════════════════════════════════════════
# N-Scaling Test (IO3 validation)
# ═══════════════════════════════════════════════════════════════════════════════


def run_n_scaling_test(model, device, n_values=(500, 300, 200, 100, 50), warmup=3, repeat=30):
    """N-scaling 延迟测试: 验证 IO3 的 N² → N 加速预期.

    使用 dummy 数据 (固定输入), 仅测量计算延迟, 不测精度。
    """
    print("\n" + "=" * 70)
    print("  [2/4] N-Scaling 延迟测试 (IO3 验证)")
    print("=" * 70)

    head = model.bbox_head
    original_n = head.num_proposals

    # Dummy 输入
    dummy_img = torch.randn(1, 3, 800, 1216, device=device)
    img_meta = ImageMeta(img_shape=(800, 1216), ori_shape=(800, 1216), scale_factor=1.0)

    with torch.no_grad():
        feats = model.backbone(dummy_img)
        feats = model.neck(feats)

    results = {}
    print(f"\n  {'N':>6} {'Latency (ms)':>14} {'vs N=500':>10} {'Self-Attn ∝':>12}")
    print(f"  {'-'*6} {'-'*14} {'-'*10} {'-'*12}")

    for n in n_values:
        head.num_proposals = n

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = head.predict(feats, [img_meta], rescale=False)
        torch.cuda.synchronize()

        # Measure
        t0 = time.time()
        for _ in range(repeat):
            with torch.no_grad():
                _ = head.predict(feats, [img_meta], rescale=False)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / repeat * 1000

        results[n] = round(elapsed, 2)

    # Restore
    head.num_proposals = original_n

    baseline = results.get(n_values[0], 1)
    for n in n_values:
        ms = results[n]
        ratio = ms / baseline if baseline else 0
        # Self-Attn 理论加速比 = (N_base/N)² (仅 attention 部分)
        attn_ratio = (n_values[0] / n) ** 2
        print(f"  {n:>6} {ms:>14.1f} {ratio:>9.2f}x {attn_ratio:>11.1f}x")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Step Convergence (IO1, IO3, IO5)
# ═══════════════════════════════════════════════════════════════════════════════


class HeadOutputCapture:
    """Context manager: 捕获所有级联头的 (cls_logits, pred_bboxes) 输出."""

    def __init__(self, head):
        self.head = head
        self.captured: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        self._handles = []

    def __enter__(self):
        for i, sh in enumerate(self.head.head_series):
            def make_hook(idx):
                def hook(_module, _input, output):
                    if isinstance(output, tuple) and len(output) >= 2:
                        cls = output[0].detach().cpu()
                        bboxes = output[1].detach().cpu()
                        self.captured.append((idx, cls, bboxes))
                return hook
            self._handles.append(sh.register_forward_hook(make_hook(i)))
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def run_step_convergence(model, samples, device, n_images=50):
    """每步收敛性分析 (IO1, IO3, IO5).

    测量:
    - IO1: x0_pred 步间相对变化 + cls 一致率
    - IO3: 每步高置信度 proposal 占比 (score 分布)
    - IO5: 跨步框位移分布
    """
    print("\n" + "=" * 70)
    print("  [3/4] 每步收敛性分析 (IO1 / IO3 / IO5)")
    print("=" * 70)

    head = model.bbox_head
    device = next(head.parameters()).device
    time_pairs = head._sampler.build_time_pairs(device)
    n_steps = len(time_pairs)

    n_eval = min(n_images, len(samples))

    # 累积器: 每步的统计量列表 (跨图像)
    per_step_stats = [[] for _ in range(n_steps)]

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

        # 重建 x0_raw (扩散空间) 从 pred_bboxes (图像空间)
        x0_list = []
        for cls_logits, pred_bboxes in trajectory:
            x0 = head._sampler.xyxy_to_raw(pred_bboxes, img_metas)
            x0_list.append(x0)

        # 逐步统计
        for step_idx in range(n_steps):
            cls_curr = trajectory[step_idx][0].squeeze(0)  # [N, C]
            bboxes_curr = trajectory[step_idx][1].squeeze(0)  # [N, 4] xyxy
            x0_curr = x0_list[step_idx].squeeze(0)  # [N, 4] raw

            # Score 分布 (IO3)
            scores = torch.sigmoid(cls_curr).max(dim=-1)[0]  # [N]
            high_conf_ratio = (scores > 0.5).float().mean().item()
            med_conf_ratio = (scores > 0.2).float().mean().item()
            top100_mean_score = scores.topk(min(100, scores.shape[0]))[0].mean().item()

            stats = {
                "high_conf_ratio": high_conf_ratio,
                "med_conf_ratio": med_conf_ratio,
                "top100_mean_score": top100_mean_score,
            }

            # x0 相对变化 (IO1)
            if step_idx > 0:
                x0_prev = x0_list[step_idx - 1].squeeze(0)
                delta = (x0_curr - x0_prev).norm(dim=-1)  # [N]
                magnitude = x0_curr.norm(dim=-1).clamp(min=1e-6)
                relative_delta = (delta / magnitude).mean().item()
                stats["x0_rel_change"] = relative_delta

                # cls 一致率 (IO1)
                cls_prev = trajectory[step_idx - 1][0].squeeze(0)
                curr_label = cls_curr.argmax(dim=-1)
                prev_label = cls_prev.argmax(dim=-1)
                consistency = (curr_label == prev_label).float().mean().item()
                stats["cls_consistency"] = consistency

                # 框位移 (IO5) — 图像空间中心点距离
                bboxes_prev = trajectory[step_idx - 1][1].squeeze(0)
                center_curr = (bboxes_curr[:, :2] + bboxes_curr[:, 2:]) / 2
                center_prev = (bboxes_prev[:, :2] + bboxes_prev[:, 2:]) / 2
                displacement = (center_curr - center_prev).norm(dim=-1)  # [N] px
                stats["box_displacement_mean"] = displacement.mean().item()
                stats["box_displacement_median"] = displacement.median().item()
                stats["box_disp_lt_1px_ratio"] = (displacement < 1.0).float().mean().item()
                stats["box_disp_lt_2px_ratio"] = (displacement < 2.0).float().mean().item()
                stats["box_disp_lt_5px_ratio"] = (displacement < 5.0).float().mean().item()

            per_step_stats[step_idx].append(stats)

    # 聚合
    result = {"n_steps": n_steps, "n_images": n_eval, "per_step": []}
    for step_idx in range(n_steps):
        stats_list = per_step_stats[step_idx]
        if not stats_list:
            continue
        keys = stats_list[0].keys()
        agg = {"step_idx": step_idx, "t_curr": time_pairs[step_idx][0], "t_next": time_pairs[step_idx][1]}
        for k in keys:
            vals = [s[k] for s in stats_list]
            agg[k] = round(float(np.mean(vals)), 4)
        result["per_step"].append(agg)

    # 收敛率统计 (IO1): x0_rel_change < 0.01 的图像比例
    if n_steps >= 2:
        conv_ratios = []
        for step_idx in range(1, n_steps):
            vals = [s.get("x0_rel_change", 0) for s in per_step_stats[step_idx]]
            conv_ratio = sum(1 for v in vals if v < 0.01) / len(vals) if vals else 0
            conv_ratios.append(conv_ratio)
        result["convergence_ratio_per_step"] = conv_ratios
        result["avg_convergence_ratio"] = round(float(np.mean(conv_ratios)), 4)

    # ── 打印 ──
    print(f"\n  图像数: {n_eval}, 采样步数: {n_steps}")
    print(f"\n  {'Step':>4} {'t_curr':>7} {'x0_Δrel':>10} {'cls_cons':>10} "
          f"{'hi_conf%':>9} {'disp<1px%':>10} {'disp_mean':>10}")
    print(f"  {'-'*4} {'-'*7} {'-'*10} {'-'*10} {'-'*9} {'-'*10} {'-'*10}")
    for step in result["per_step"]:
        x0ch = step.get("x0_rel_change", "-")
        clsc = step.get("cls_consistency", "-")
        hi = step.get("high_conf_ratio", 0) * 100
        dlt1 = step.get("box_disp_lt_1px_ratio", 0) * 100
        dm = step.get("box_displacement_mean", 0)
        x0ch_s = f"{x0ch:.4f}" if isinstance(x0ch, float) else "   -"
        clsc_s = f"{clsc:.4f}" if isinstance(clsc, float) else "   -"
        dlt1_s = f"{dlt1:.1f}%" if isinstance(dlt1, float) else "   -"
        dm_s = f"{dm:.2f}" if isinstance(dm, float) else "   -"
        print(f"  {step['step_idx']:>4} {step['t_curr']:>7.3f} {x0ch_s:>10} {clsc_s:>10} "
              f"{hi:>8.1f}% {dlt1_s:>10} {dm_s:>10}")

    if "avg_convergence_ratio" in result:
        print(f"\n  IO1 收敛率 (x0_Δrel < 0.01): {result['avg_convergence_ratio']*100:.1f}%")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Head Convergence (IO4)
# ═══════════════════════════════════════════════════════════════════════════════


def run_head_convergence(model, samples, device, n_images=30):
    """级联头间收敛性分析 (IO4).

    在每个采样步, 捕获 6 级级联头的逐头输出,
    测量相邻头间的预测差异 (box 相对变化 + cls 一致率)。
    """
    print("\n" + "=" * 70)
    print("  [4/4] 级联头收敛性分析 (IO4)")
    print("=" * 70)

    head = model.bbox_head
    device = next(head.parameters()).device
    time_pairs = head._sampler.build_time_pairs(device)
    n_steps = len(time_pairs)
    n_heads = head.num_heads

    n_eval = min(n_images, len(samples))

    # 累积器: [step_idx][head_pair_idx] -> list of stats
    per_step_head_stats = [
        [[] for _ in range(n_heads - 1)] for _ in range(n_steps)
    ]

    for i in range(n_eval):
        img = samples[i]["img"].to(device)
        img_metas = [samples[i]["img_meta"]]
        bs = 1

        with torch.no_grad():
            feats = model.backbone(img)
            feats = model.neck(feats)

            x_raw = torch.randn(bs, head.num_proposals, 4, device=device)
            dpm_solver = head._sampler.create_dpm_solver()
            if dpm_solver is not None:
                dpm_solver.reset()

            for step_idx, (t_curr, t_next) in enumerate(time_pairs):
                # 捕获每头输出
                with HeadOutputCapture(head) as cap:
                    cls_logits, pred_bboxes, x0_raw = head._forward_at_t(
                        feats, x_raw, t_curr, img_metas
                    )

                # 分析头间差异
                if len(cap.captured) >= 2:
                    for h in range(len(cap.captured) - 1):
                        _, cls_h, bboxes_h = cap.captured[h]
                        _, cls_h1, bboxes_h1 = cap.captured[h + 1]

                        # box 相对变化
                        delta = (bboxes_h1 - bboxes_h).norm(dim=-1)
                        magnitude = bboxes_h1.norm(dim=-1).clamp(min=1e-6)
                        rel_delta = (delta / magnitude).mean().item()

                        # cls 一致率
                        label_h = cls_h.argmax(dim=-1)
                        label_h1 = cls_h1.argmax(dim=-1)
                        consistency = (label_h == label_h1).float().mean().item()

                        per_step_head_stats[step_idx][h].append({
                            "box_rel_change": rel_delta,
                            "cls_consistency": consistency,
                        })

                # 演化 x_raw (Heun / Euler / DPM)
                if dpm_solver is not None:
                    x_raw = dpm_solver.step(x_raw, x0_raw, t_curr, step_idx)
                elif head.solver_type == "heun" and t_next > 0:
                    def model_fn(x_tmp, t_tmp):
                        _, _, x0_tmp = head._forward_at_t(feats, x_tmp, t_tmp, img_metas)
                        return x0_tmp, None
                    x_raw = head.rf.heun_step(x_raw, x0_raw, t_curr, t_next, model_fn)
                else:
                    x_raw = head.rf.step(x_raw, x0_raw, t_curr, t_next)

                if head.box_renewal:
                    x_raw = head._sampler.apply_box_renewal(x_raw, cls_logits)

    # 聚合
    result = {"n_steps": n_steps, "n_heads": n_heads, "n_images": n_eval, "per_step": []}

    for step_idx in range(n_steps):
        step_data = {"step_idx": step_idx, "t_curr": time_pairs[step_idx][0], "head_pairs": []}
        for h in range(n_heads - 1):
            stats_list = per_step_head_stats[step_idx][h]
            if not stats_list:
                continue
            box_changes = [s["box_rel_change"] for s in stats_list]
            cls_cons = [s["cls_consistency"] for s in stats_list]
            step_data["head_pairs"].append({
                "head": f"{h+1}→{h+2}",
                "box_rel_change_mean": round(float(np.mean(box_changes)), 5),
                "box_rel_change_p50": round(float(np.median(box_changes)), 5),
                "cls_consistency_mean": round(float(np.mean(cls_cons)), 4),
            })
        result["per_step"].append(step_data)

    # ── 打印 ──
    print(f"\n  图像数: {n_eval}, 级联头数: {n_heads}")
    print(f"\n  每步的相邻头间预测差异:")
    print(f"  {'Step':>4} {'t_curr':>7} {'Head':>8} {'box_Δrel':>10} {'cls_cons':>10}")
    print(f"  {'-'*4} {'-'*7} {'-'*8} {'-'*10} {'-'*10}")
    for step in result["per_step"]:
        for hp in step["head_pairs"]:
            print(f"  {step['step_idx']:>4} {step['t_curr']:>7.3f} {hp['head']:>8} "
                  f"{hp['box_rel_change_mean']:>10.5f} {hp['cls_consistency_mean']:>10.4f}")

    # IO4 可行性: 后期头 (head 3→4, 4→5, 5→6) 的 box_rel_change < 0.005 的比例
    late_heads_converged = []
    for step in result["per_step"]:
        for hp in step["head_pairs"]:
            head_num = int(hp["head"].split("→")[0])
            if head_num >= 3 and hp["box_rel_change_mean"] < 0.005:
                late_heads_converged.append((step["step_idx"], hp["head"]))
    result["late_heads_converged"] = late_heads_converged
    if late_heads_converged:
        print(f"\n  IO4 后期头收敛 (head≥3→4, box_Δrel<0.005): {len(late_heads_converged)} 处")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="LDMDet 推理优化方向验证 (IO1-IO5)")
    parser.add_argument(
        "--config",
        default="experiments/configs/ldmdet/ldmdet_rf_heun_adaln_stochot_eps5.py",
    )
    parser.add_argument(
        "--checkpoint",
        default="work_dirs/reproduce_0751_stochot_eps5_v2/best_coco_bbox_mAP_epoch_59.pth",
    )
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="work_dirs/inference_opt",
    )
    parser.add_argument(
        "--skip-latency", action="store_true", help="跳过延迟 profiling"
    )
    parser.add_argument(
        "--skip-n-scaling", action="store_true", help="跳过 N-scaling 测试"
    )
    parser.add_argument(
        "--skip-convergence", action="store_true", help="跳过收敛性分析"
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载模型 ──
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Loading model...")
    model, cfg = load_model(args.config, args.checkpoint, device)

    # ── 加载数据 ──
    print(f"Loading {args.num_images} val samples...")
    samples = load_val_samples(cfg, args.num_images)
    print(f"  Loaded {len(samples)} samples")

    report = {"config": args.config, "checkpoint": args.checkpoint}

    # ── 1. 延迟 Profiling ──
    if not args.skip_latency:
        report["latency"] = run_latency_profiling(
            model, samples, device, n_images=args.num_images
        )

    # ── 2. N-Scaling ──
    if not args.skip_n_scaling:
        report["n_scaling"] = run_n_scaling_test(model, device)

    # ── 3. 每步收敛性 ──
    if not args.skip_convergence:
        report["step_convergence"] = run_step_convergence(
            model, samples, device, n_images=args.num_images
        )

    # ── 4. 级联头收敛性 ──
    if not args.skip_convergence:
        report["head_convergence"] = run_head_convergence(
            model, samples, device, n_images=min(30, args.num_images)
        )

    # ── 保存 ──
    output_path = output_dir / "benchmark_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n{'='*70}")
    print(f"报告已保存到: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
