"""LDMDet 训练性能 Profile 脚本 (纯 ldmdet 库)

使用 PyTorch Profiler 对训练流程进行细粒度性能分析，
定位训练速度瓶颈。基于 ldmdet 纯 PyTorch 库，零 mmdet 依赖。

用法:
    # 默认参数 (模拟染色体检测场景)
    python ldmdet/tools/profile_training.py

    # 自定义参数
    python ldmdet/tools/profile_training.py \
        --num-proposals 500 --num-heads 6 --bs 8 \
        --num-warmup 3 --num-profile 5

    # 启用 TensorBoard Profiler (详细 kernel trace)
    python ldmdet/tools/profile_training.py --use-tb-profiler

输出:
    - 终端打印: 各阶段耗时统计
    - TensorBoard 日志: ./profiler_logs/ (可选)
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ldmdet.core import DiffusionDetHead, SingleDiffusionDetHead, SingleRoIExtractor
from ldmdet.coupling import build_coupling
from ldmdet.criterion import DiffusionDetCriterion, DiffusionDetMatcher, FocalLoss, GIoULoss, L1Loss
from ldmdet.data.structures import ImageMeta, InstanceData


# ============================================================
# 计时工具
# ============================================================

class TimingStats:
    """累积计时统计"""

    def __init__(self):
        self._timings = defaultdict(list)

    def record(self, name: str, elapsed: float):
        self._timings[name].append(elapsed)

    def summary(self) -> str:
        lines = []
        lines.append(f"{'Stage':<45} {'Mean(ms)':>10} {'Std(ms)':>10} {'Total(ms)':>12} {'Count':>6} {'Pct':>6}")
        lines.append("-" * 95)

        total_time = sum(sum(v) for v in self._timings.values())

        sorted_items = sorted(
            self._timings.items(), key=lambda x: sum(x[1]), reverse=True
        )

        for name, times in sorted_items:
            arr = np.array(times) * 1000  # to ms
            pct = sum(times) / max(total_time, 1e-10) * 100
            lines.append(
                f"{name:<45} {arr.mean():>10.2f} {arr.std():>10.2f} "
                f"{arr.sum():>12.2f} {len(arr):>6d} {pct:>5.1f}%"
            )

        lines.append("-" * 95)
        lines.append(f"{'TOTAL':<45} {'':>10} {'':>10} {total_time*1000:>12.2f}")
        return "\n".join(lines)


@contextmanager
def timer(stats: TimingStats, name: str, cuda: bool = True):
    """计时上下文管理器"""
    if cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    stats.record(name, elapsed)


# ============================================================
# 构建模型
# ============================================================

def build_model(args):
    """构建完整的 DiffusionDetHead 模型"""
    feat_channels = args.feat_channels
    num_classes = args.num_classes
    num_proposals = args.num_proposals
    num_heads = args.num_heads
    pooler_resolution = args.pooler_resolution

    single_head = SingleDiffusionDetHead(
        num_classes=num_classes,
        feat_channels=feat_channels,
        dim_feedforward=feat_channels * 8,
        num_cls_convs=1,
        num_reg_convs=3,
        num_heads=8,
        dropout=0.0,
        pooler_resolution=pooler_resolution,
        time_conditioning=args.time_conditioning,
        use_flash_attn=args.use_flash_attn,
    )

    roi_extractor = SingleRoIExtractor(
        roi_layer=dict(output_size=pooler_resolution, sampling_ratio=2, aligned=True),
        out_channels=feat_channels,
        featmap_strides=[4, 8, 16, 32],
    )

    matcher = DiffusionDetMatcher(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        candidate_topk=5,
    )
    criterion = DiffusionDetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        loss_cls=FocalLoss(loss_weight=2.0),
        loss_bbox=L1Loss(loss_weight=5.0),
        loss_giou=GIoULoss(loss_weight=2.0),
        deep_supervision=True,
        scale_aware=args.scale_aware,
        scale_aware_mode='inverse',
    )

    coupling = build_coupling(args.coupling, epsilon=args.ot_epsilon, num_iters=args.ot_num_iters)

    head = DiffusionDetHead(
        num_classes=num_classes,
        feat_channels=feat_channels,
        num_proposals=num_proposals,
        num_heads=num_heads,
        snr_scale=args.snr_scale,
        timesteps=1000,
        sampling_timesteps=1,
        solver_type='euler',
        diffusion_type=args.diffusion_type,
        rf_schedule=args.rf_schedule,
        rf_shift=args.rf_shift,
        single_head=single_head,
        roi_extractor=roi_extractor,
        criterion=criterion,
        coupling=coupling,
        deep_supervision=True,
    )

    return head


# ============================================================
# 生成模拟数据
# ============================================================

def generate_synthetic_batch(args, device):
    """生成模拟训练 batch (无需真实数据集)"""
    bs = args.bs
    img_h, img_w = args.img_size, args.img_size

    features = [
        torch.randn(bs, args.feat_channels, img_h // s, img_w // s, device=device)
        for s in [4, 8, 16, 32]
    ]

    img_metas = [
        ImageMeta(img_shape=(img_h, img_w))
        for _ in range(bs)
    ]

    gt_bboxes = []
    gt_labels = []
    for _ in range(bs):
        num_gt = torch.randint(20, 50, (1,)).item()
        cx = torch.rand(num_gt)
        cy = torch.rand(num_gt)
        w = torch.rand(num_gt) * 0.1 + 0.02
        h = torch.rand(num_gt) * 0.1 + 0.02
        x1 = (cx - w / 2).clamp(0, 1)
        y1 = (cy - h / 2).clamp(0, 1)
        x2 = (cx + w / 2).clamp(0, 1)
        y2 = (cy + h / 2).clamp(0, 1)
        gt_bboxes.append(torch.stack([x1, y1, x2, y2], dim=-1).to(device))
        gt_labels.append(torch.randint(0, args.num_classes, (num_gt,)).to(device))

    return features, img_metas, gt_bboxes, gt_labels


# ============================================================
# 细粒度 Profile
# ============================================================

def profile_training_step(head, features, img_metas, gt_bboxes, gt_labels, stats, device):
    """对单个训练步骤进行细粒度计时"""
    bs = len(img_metas)

    with timer(stats, "1. _normalize_targets"):
        targets = head._normalize_targets(gt_bboxes, gt_labels, img_metas, bs)

    with timer(stats, "2. _sample_t"):
        t = head._sample_t(bs, device)

    with timer(stats, "3. _build_training_targets (coupling+diffusion)"):
        x_boxes, x_starts, x_noises, matched_gt_indices = head._build_training_targets(
            bs, device, t, targets, gt_bboxes, img_metas
        )

    # 细分 coupling
    with timer(stats, "3a. coupling (per-image loop)"):
        for i in range(bs):
            num_gt = gt_bboxes[i].shape[0]
            if num_gt == 0:
                continue
            from ldmdet.utils.box_ops import bbox_xyxy_to_cxcywh
            norm_gt_cxcywh = bbox_xyxy_to_cxcywh(targets[i].bboxes)
            gt_diffusion = (norm_gt_cxcywh * 2 - 1) * head.snr_scale
            noise = torch.randn(head.num_proposals, 4, device=device)
            with timer(stats, f"3a1. couple[{i}] ({type(head.ot_module).__name__})"):
                head.ot_module.couple(noise, gt_diffusion, targets[i].labels, device)

    x_noisy_batch = torch.stack(x_boxes)
    with timer(stats, "4. raw_to_xyxy"):
        curr_bboxes = head._sampler.raw_to_xyxy(x_noisy_batch, img_metas)

    t_input = t if head.diffusion_type == 'ddpm' else t * head.timesteps
    with timer(stats, "5. time_mlp"):
        time_emb = head.time_mlp(t_input)

    with timer(stats, "6. head_series forward (all heads)"):
        inter_cls_logits = []
        inter_pred_bboxes = []
        curr_proposals = None

        for head_idx, sh in enumerate(head.head_series):
            with timer(stats, f"6a. head[{head_idx}] total"):
                with timer(stats, f"6a1. head[{head_idx}] forward (roi+attn+predict)"):
                    cls_logits, pred_bboxes, curr_proposals = sh(
                        features, curr_bboxes, curr_proposals, head.roi_extractor, time_emb
                    )

                inter_cls_logits.append(cls_logits)
                inter_pred_bboxes.append(pred_bboxes)
                curr_bboxes = pred_bboxes.detach()

    all_cls_logits = torch.stack(inter_cls_logits)
    all_pred_bboxes = torch.stack(inter_pred_bboxes)

    with timer(stats, "7. _normalize_pred_bboxes"):
        norm_pred_bboxes = head._normalize_pred_bboxes(all_pred_bboxes, img_metas)

    with timer(stats, "8. _build_outputs"):
        outputs = head._build_outputs(all_cls_logits, norm_pred_bboxes)

    with timer(stats, "9. criterion (matcher+loss)"):
        losses = head.criterion(outputs, targets)

    with timer(stats, "9a. matcher"):
        _ = head.criterion.matcher(outputs, targets)

    return losses


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Profile LDMDet training (pure ldmdet lib)")
    parser.add_argument("--num-proposals", type=int, default=500)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=24)
    parser.add_argument("--feat-channels", type=int, default=256)
    parser.add_argument("--pooler-resolution", type=int, default=7)
    parser.add_argument("--bs", type=int, default=8, help="batch size")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--snr-scale", type=float, default=2.0)
    parser.add_argument("--diffusion-type", type=str, default='rectified_flow')
    parser.add_argument("--rf-schedule", type=str, default='shifted')
    parser.add_argument("--rf-shift", type=float, default=2.0)
    parser.add_argument("--coupling", type=str, default='ghss',
                        choices=['random', 'hard_ot', 'sinkhorn_argmax', 'sinkhorn_stochastic', 'ghss'])
    parser.add_argument("--ot-epsilon", type=float, default=5.0)
    parser.add_argument("--ot-num-iters", type=int, default=20)
    parser.add_argument("--time-conditioning", type=str, default='scale_shift',
                        choices=['scale_shift', 'adaln_zero'])
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--scale-aware", action="store_true")
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-profile", type=int, default=5)
    parser.add_argument("--use-tb-profiler", action="store_true",
                        help="enable PyTorch TensorBoard Profiler")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("LDMDet Training Profiler (pure ldmdet lib)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {args.bs}")
    print(f"Num proposals: {args.num_proposals}")
    print(f"Num heads: {args.num_heads}")
    print(f"Num classes: {args.num_classes}")
    print(f"Coupling: {args.coupling} (eps={args.ot_epsilon}, iters={args.ot_num_iters})")
    print(f"Diffusion: {args.diffusion_type}, schedule={args.rf_schedule}, shift={args.rf_shift}")
    print(f"Time conditioning: {args.time_conditioning}")
    print(f"Flash attention: {args.use_flash_attn}")
    print(f"Warmup: {args.num_warmup}, Profile: {args.num_profile}")

    # ---- 构建模型 ----
    head = build_model(args).to(device)
    head.train()

    # ---- 参数统计 ----
    total_params = sum(p.numel() for p in head.parameters())
    print(f"\nModel Parameters: {total_params:,}")
    for name, module in head.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        if n_params > 0:
            print(f"  {name:<25} {n_params:>12,}")

    # ---- Warmup ----
    print(f"\n--- Warmup ({args.num_warmup} iters) ---")
    for i in range(args.num_warmup):
        features, img_metas, gt_bboxes, gt_labels = generate_synthetic_batch(args, device)
        losses = head.loss(features, img_metas, gt_bboxes, gt_labels)
        total_loss = sum(losses.values())
        total_loss.backward()
        for p in head.parameters():
            if p.grad is not None:
                p.grad = None
        print(f"  Warmup iter {i+1}/{args.num_warmup} done, loss={total_loss.item():.4f}")

    # ---- Profile ----
    print(f"\n--- Profiling ({args.num_profile} iters) ---")
    stats = TimingStats()

    tb_profiler = None
    if args.use_tb_profiler:
        from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule as prof_schedule
        tb_profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=tensorboard_trace_handler('./profiler_logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        tb_profiler.start()

    for i in range(args.num_profile):
        features, img_metas, gt_bboxes, gt_labels = generate_synthetic_batch(args, device)

        with timer(stats, "0. FULL ITERATION (forward+backward)"):
            with timer(stats, "0a. forward (loss computation)"):
                losses = profile_training_step(
                    head, features, img_metas, gt_bboxes, gt_labels, stats, device
                )

            total_loss = sum(losses.values())
            with timer(stats, "0b. backward"):
                total_loss.backward()

            with timer(stats, "0c. zero_grad"):
                for p in head.parameters():
                    if p.grad is not None:
                        p.grad = None

        if tb_profiler is not None:
            tb_profiler.step()

        print(f"  Profile iter {i+1}/{args.num_profile} done, loss={total_loss.item():.4f}")

    if tb_profiler is not None:
        tb_profiler.stop()

    # ---- 输出结果 ----
    print("\n" + "=" * 60)
    print("PROFILE RESULTS")
    print("=" * 60)
    print(stats.summary())

    # ---- GPU 内存统计 ----
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
        print(f"  Reserved:  {torch.cuda.max_memory_reserved()/1024**2:.1f} MB")

    # ---- PyTorch Profiler 内置分析 ----
    print("\n--- Quick PyTorch Profiler (1 iter, kernel-level) ---")
    from torch.profiler import profile, ProfilerActivity

    features, img_metas, gt_bboxes, gt_labels = generate_synthetic_batch(args, device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=True) as prof:
        losses = head.loss(features, img_metas, gt_bboxes, gt_labels)
        total_loss = sum(losses.values())
        total_loss.backward()

    print("\nTop 15 GPU kernels by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print("\nTop 15 CPU ops by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    if args.use_tb_profiler:
        print(f"\nTensorBoard Profiler logs saved to: ./profiler_logs/")
        print(f"View with: tensorboard --logdir ./profiler_logs")


if __name__ == '__main__':
    main()
