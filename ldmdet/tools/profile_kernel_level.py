"""CUDA Kernel 级别细粒度 Profile 脚本

使用 PyTorch Profiler 精确测量每个 CUDA kernel 的耗时，
识别底层优化空间并评估最大可提速比。

用法:
    python ldmdet/tools/profile_kernel_level.py --gpu 0
"""

import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ldmdet.tools.profile_training import build_model, generate_synthetic_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--num-proposals", type=int, default=500)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=24)
    parser.add_argument("--feat-channels", type=int, default=256)
    parser.add_argument("--pooler-resolution", type=int, default=7)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--snr-scale", type=float, default=2.0)
    parser.add_argument("--diffusion-type", type=str, default='rectified_flow')
    parser.add_argument("--rf-schedule", type=str, default='shifted')
    parser.add_argument("--rf-shift", type=float, default=2.0)
    parser.add_argument("--coupling", type=str, default='ghss')
    parser.add_argument("--ot-epsilon", type=float, default=5.0)
    parser.add_argument("--ot-num-iters", type=int, default=20)
    parser.add_argument("--time-conditioning", type=str, default='scale_shift')
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--scale-aware", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile-iters", type=int, default=3)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda")

    # 构建模型
    head = build_model(args).to(device)
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)

    # Warmup
    print("Warmup...")
    for _ in range(args.warmup):
        features, img_metas, gt_bboxes, gt_labels = generate_synthetic_batch(args, device)
        losses = head.loss(features, img_metas, gt_bboxes, gt_labels)
        sum(losses.values()).backward()
        optimizer.step()
        optimizer.zero_grad()

    # 使用 PyTorch Profiler 进行 kernel 级别分析
    from torch.profiler import profile, ProfilerActivity, record_function

    print(f"\nProfiling {args.profile_iters} iterations with CUDA kernel trace...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.profile_iters):
            features, img_metas, gt_bboxes, gt_labels = generate_synthetic_batch(args, device)
            losses = head.loss(features, img_metas, gt_bboxes, gt_labels)
            total_loss = sum(losses.values())
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # ============================================================
    # 分析结果
    # ============================================================

    print("\n" + "=" * 80)
    print("TOP-50 CUDA KERNELS BY TOTAL CUDA TIME")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))

    print("\n" + "=" * 80)
    print("TOP-30 CPU OPERATIONS BY TOTAL TIME")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    # ============================================================
    # 分类统计: 按功能模块聚合 kernel 耗时
    # ============================================================

    print("\n" + "=" * 80)
    print("KERNEL CATEGORY ANALYSIS")
    print("=" * 80)

    categories = {
        'roi_align': ['roi_align', 'RoIAlign', 'roi_pool'],
        'attention': ['multihead', 'attention', 'softmax', 'bmm', 'matmul'],
        'box_iou': ['box_iou', 'nms', 'giou', 'iou'],
        'focal_loss': ['sigmoid', 'log_sigmoid', 'binary_cross_entropy', 'focal'],
        'sinkhorn': ['logsumexp', 'cdist', 'scatter'],
        'linear': ['addmm', 'linear', 'matmul'],
        'conv': ['conv', 'cudnn'],
        'norm': ['layer_norm', 'batch_norm', 'LayerNorm'],
        'activation': ['relu', 'silu', 'gelu'],
        'optimizer': ['adam', 'adamw'],
        'other': [],
    }

    key_avgs = prof.key_averages()
    cat_times = {}
    uncategorized = []

    for item in key_avgs:
        name = item.key
        cuda_time = item.cuda_time_total
        if cuda_time == 0:
            continue

        categorized = False
        for cat_name, keywords in categories.items():
            if cat_name == 'other':
                continue
            if any(kw.lower() in name.lower() for kw in keywords):
                cat_times[cat_name] = cat_times.get(cat_name, 0) + cuda_time
                categorized = True
                break
        if not categorized:
            cat_times['other'] = cat_times.get('other', 0) + cuda_time
            uncategorized.append((name, cuda_time))

    total_cuda_time = sum(cat_times.values())

    print(f"\n{'Category':<25} {'Time (ms)':>12} {'Percentage':>12}")
    print("-" * 52)
    for cat, t in sorted(cat_times.items(), key=lambda x: -x[1]):
        pct = t / max(total_cuda_time, 1) * 100
        print(f"{cat:<25} {t/1000:>12.2f} {pct:>11.1f}%")
    print("-" * 52)
    print(f"{'TOTAL':<25} {total_cuda_time/1000:>12.2f}")

    if uncategorized:
        print(f"\nTop uncategorized kernels:")
        for name, t in sorted(uncategorized, key=lambda x: -x[1])[:15]:
            print(f"  {name:<60} {t/1000:.2f} ms")

    # ============================================================
    # 优化空间评估
    # ============================================================

    print("\n" + "=" * 80)
    print("OPTIMIZATION POTENTIAL ANALYSIS")
    print("=" * 80)

    # 计算各模块的理论下限和优化空间
    analysis = {
        'roi_align': {
            'desc': 'ROI Align 操作 (固定 CUDA kernel，难以优化)',
            'theoretical_min_pct': 90,  # 已接近最优
        },
        'attention': {
            'desc': 'Self-Attention (可用 Flash Attention 优化)',
            'theoretical_min_pct': 40,  # Flash Attention 可减少 ~60%
        },
        'box_iou': {
            'desc': 'Box IoU / GIoU 计算 (可合并调用减少次数)',
            'theoretical_min_pct': 50,  # 合并 box_iou + giou 可减少 ~50%
        },
        'focal_loss': {
            'desc': 'Focal Loss (sigmoid/log 计算)',
            'theoretical_min_pct': 60,  # 自定义 fused kernel 可减少 ~40%
        },
        'sinkhorn': {
            'desc': 'Sinkhorn 迭代 (logsumexp + cdist)',
            'theoretical_min_pct': 70,  # 自定义 kernel 可减少 ~30%
        },
        'linear': {
            'desc': 'Linear 层 (addmm)',
            'theoretical_min_pct': 85,  # 已接近最优
        },
    }

    for cat, info in analysis.items():
        current_time = cat_times.get(cat, 0) / 1000
        if current_time > 0:
            min_pct = info['theoretical_min_pct']
            potential_saving = current_time * (1 - min_pct / 100)
            print(f"\n  {cat}:")
            print(f"    当前耗时: {current_time:.2f} ms")
            print(f"    {info['desc']}")
            print(f"    理论最低: {current_time * min_pct / 100:.2f} ms ({min_pct}% of current)")
            print(f"    可节省: {potential_saving:.2f} ms")

    # 总体评估
    total_trainable = sum(cat_times.get(c, 0) for c in ['attention', 'box_iou', 'focal_loss', 'sinkhorn'])
    total_fixed = sum(cat_times.get(c, 0) for c in ['roi_align', 'linear', 'conv', 'norm', 'activation', 'optimizer'])
    total_all = total_trainable + total_fixed + cat_times.get('other', 0)

    max_saving = 0
    for cat, info in analysis.items():
        current = cat_times.get(cat, 0)
        if current > 0:
            max_saving += current * (1 - info['theoretical_min_pct'] / 100)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total CUDA time: {total_all/1000:.2f} ms")
    print(f"  Optimizable: {total_trainable/1000:.2f} ms ({total_trainable/max(total_all,1)*100:.1f}%)")
    print(f"  Fixed (hard to optimize): {total_fixed/1000:.2f} ms ({total_fixed/max(total_all,1)*100:.1f}%)")
    print(f"  Max theoretical saving: {max_saving/1000:.2f} ms")
    print(f"  Max theoretical speedup: {total_all / max(total_all - max_saving, 1):.2f}x")
    print(f"")
    print(f"  Key optimization strategies:")
    print(f"    1. Flash Attention: ~{cat_times.get('attention', 0)/1000 * 0.6:.1f} ms saving")
    print(f"    2. Fused Focal Loss kernel: ~{cat_times.get('focal_loss', 0)/1000 * 0.4:.1f} ms saving")
    print(f"    3. Merge box_iou+giou calls: ~{cat_times.get('box_iou', 0)/1000 * 0.5:.1f} ms saving")
    print(f"    4. Custom Sinkhorn kernel: ~{cat_times.get('sinkhorn', 0)/1000 * 0.3:.1f} ms saving")
    print(f"    5. AMP (FP16): ~{total_all/1000 * 0.3:.1f} ms saving (global ~30%)")
    print(f"    6. torch.compile: ~{total_all/1000 * 0.15:.1f} ms saving (global ~15%)")


if __name__ == '__main__':
    main()
