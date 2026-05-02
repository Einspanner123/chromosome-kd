"""
离线多样性测量工具（无需模型加载）

直接从 COCO 格式标注中读取 GT 框，模拟训练时的耦合过程，
计算所有 ε 下的 H_assign、C_transport、CAM 度等指标。

用法:
    python tools/measure_diversity_offline.py \
        --ann data/selfmake_chromosome202250604_NoResizeNoAug/valid/annotations.json \
        --eps-list 0.01 0.1 0.5 1.0 5.0 10.0 50.0 100.0 \
        --num-proposals 500 --snr-scale 2.0 \
        --num-samples 500 --out diversity_results.json
"""

import argparse
import json
import os

import numpy as np
import torch


def compute_sinkhorn_transport(noise, gt_diffusion, epsilon=1.0, num_iters=20):
    cost = torch.cdist(noise, gt_diffusion, p=2)
    N, K = cost.shape
    a = torch.ones(N, device=noise.device) / N
    proposals_per_gt = max(N // max(K, 1), 1)
    gt_mass = torch.full((K,), proposals_per_gt / N, device=noise.device)
    b = gt_mass / gt_mass.sum()
    log_K_mat = -cost / epsilon
    log_u = torch.zeros(N, device=noise.device)
    log_v = torch.zeros(K, device=noise.device)
    for _ in range(num_iters):
        log_u = torch.log(a + 1e-10) - torch.logsumexp(
            log_K_mat + log_v.unsqueeze(0), dim=1
        )
        log_v = torch.log(b + 1e-10) - torch.logsumexp(
            log_K_mat + log_u.unsqueeze(1), dim=0
        )
    transport = torch.exp(log_u.unsqueeze(1) + log_K_mat + log_v.unsqueeze(0))
    return transport, cost


def compute_all_metrics(transport, cost, noise, gt_diffusion):
    N, K = transport.shape
    metrics = {}

    row_probs = transport / transport.sum(dim=1, keepdim=True)

    argmax_idx = transport.argmax(dim=1)
    stochastic_idx = torch.multinomial(row_probs, 1).squeeze(-1)

    assign_counts_argmax = torch.bincount(argmax_idx, minlength=K).float()
    assign_counts_argmax = assign_counts_argmax / assign_counts_argmax.sum()
    mask_argmax = assign_counts_argmax > 0
    metrics["H_assign_argmax"] = -(
        assign_counts_argmax[mask_argmax] * assign_counts_argmax[mask_argmax].log()
    ).sum().item()

    assign_counts_stoch = torch.bincount(stochastic_idx, minlength=K).float()
    assign_counts_stoch = assign_counts_stoch / assign_counts_stoch.sum()
    mask_stoch = assign_counts_stoch > 0
    metrics["H_assign_stochastic"] = -(
        assign_counts_stoch[mask_stoch] * assign_counts_stoch[mask_stoch].log()
    ).sum().item()

    metrics["H_assign_random"] = float(np.log(K))

    row_entropies = -(
        row_probs * torch.clamp(row_probs, min=1e-10).log()
    ).sum(dim=1)
    metrics["H_row_mean"] = row_entropies.mean().item()
    metrics["H_row_max"] = float(np.log(K))

    metrics["C_transport_argmax"] = cost[torch.arange(N), argmax_idx].mean().item()
    metrics["C_transport_stochastic"] = cost[torch.arange(N), stochastic_idx].mean().item()
    metrics["C_transport_random"] = cost.mean(dim=1).mean().item()

    denom = max(metrics["H_assign_random"] - metrics["H_assign_argmax"], 1e-10)
    metrics["CAM_degree"] = (
        metrics["H_assign_stochastic"] - metrics["H_assign_argmax"]
    ) / denom

    gt_per_noise_argmax = torch.zeros(K, dtype=torch.long)
    for k in range(K):
        gt_per_noise_argmax[k] = (argmax_idx == k).sum()
    metrics["assign_distribution_argmax"] = gt_per_noise_argmax.tolist()

    gt_per_noise_stoch = torch.zeros(K, dtype=torch.long)
    for k in range(K):
        gt_per_noise_stoch[k] = (stochastic_idx == k).sum()
    metrics["assign_distribution_stochastic"] = gt_per_noise_stoch.tolist()

    return metrics


def load_coco_annotations(ann_path):
    with open(ann_path) as f:
        coco = json.load(f)

    img_id_to_size = {}
    for img in coco["images"]:
        img_id_to_size[img["id"]] = (img["width"], img["height"])

    img_id_to_bboxes = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_id_to_bboxes:
            img_id_to_bboxes[img_id] = []
        img_id_to_bboxes[img_id].append(ann["bbox"])

    samples = []
    for img_id, bboxes in img_id_to_bboxes.items():
        if img_id not in img_id_to_size:
            continue
        w, h = img_id_to_size[img_id]
        norm_bboxes = []
        for bbox in bboxes:
            x, y, bw, bh = bbox
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            norm_bboxes.append([cx, cy, nw, nh])
        if len(norm_bboxes) >= 2:
            samples.append(torch.tensor(norm_bboxes, dtype=torch.float32))

    return samples


def main():
    parser = argparse.ArgumentParser(description="Offline diversity measurement")
    parser.add_argument("--ann", required=True, help="COCO annotation JSON path")
    parser.add_argument(
        "--eps-list",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        help="List of epsilon values",
    )
    parser.add_argument("--num-proposals", type=int, default=500)
    parser.add_argument("--snr-scale", type=float, default=2.0)
    parser.add_argument("--num-samples", type=int, default=500, help="Number of images to sample")
    parser.add_argument("--num-repeats", type=int, default=5, help="Repeats per image per eps")
    parser.add_argument("--out", default="diversity_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading annotations from {args.ann}...")
    samples = load_coco_annotations(args.ann)
    print(f"Found {len(samples)} images with >= 2 GT boxes")

    if args.num_samples < len(samples):
        indices = np.random.choice(len(samples), args.num_samples, replace=False)
        samples = [samples[i] for i in indices]

    K_values = [s.shape[0] for s in samples]
    print(f"K (GT per image): mean={np.mean(K_values):.1f}, "
          f"min={np.min(K_values)}, max={np.max(K_values)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_results = {}
    eps_list = args.eps_list

    for eps in eps_list:
        print(f"\n{'='*60}")
        print(f"Computing metrics for ε = {eps}")
        print(f"{'='*60}")

        agg_metrics = {}
        num_valid = 0

        for sample_idx, gt_cxcywh in enumerate(samples):
            gt_diffusion = (gt_cxcywh.to(device) * 2 - 1) * args.snr_scale
            K = gt_diffusion.shape[0]

            for repeat in range(args.num_repeats):
                noise = torch.randn(args.num_proposals, 4, device=device)

                transport, cost = compute_sinkhorn_transport(
                    noise, gt_diffusion, epsilon=eps, num_iters=20
                )
                metrics = compute_all_metrics(transport, cost, noise, gt_diffusion)

                for k, v in metrics.items():
                    if k.startswith("assign_distribution"):
                        continue
                    if k not in agg_metrics:
                        agg_metrics[k] = []
                    agg_metrics[k].append(v)

                num_valid += 1

        result = {"epsilon": eps, "num_measurements": num_valid}
        for k, v_list in agg_metrics.items():
            result[f"{k}_mean"] = float(np.mean(v_list))
            result[f"{k}_std"] = float(np.std(v_list))

        all_results[f"eps_{eps}"] = result

        print(f"  H_assign_argmax:     {result['H_assign_argmax_mean']:.4f} ± {result['H_assign_argmax_std']:.4f}")
        print(f"  H_assign_stochastic: {result['H_assign_stochastic_mean']:.4f} ± {result['H_assign_stochastic_std']:.4f}")
        print(f"  H_assign_random:     {result['H_assign_random_mean']:.4f}")
        print(f"  H_row_mean:          {result['H_row_mean_mean']:.4f} ± {result['H_row_mean_std']:.4f}")
        print(f"  CAM_degree:          {result['CAM_degree_mean']:.4f} ± {result['CAM_degree_std']:.4f}")
        print(f"  C_transport_argmax:     {result['C_transport_argmax_mean']:.4f} ± {result['C_transport_argmax_std']:.4f}")
        print(f"  C_transport_stochastic: {result['C_transport_stochastic_mean']:.4f} ± {result['C_transport_stochastic_std']:.4f}")
        print(f"  C_transport_random:     {result['C_transport_random_mean']:.4f}")

    print(f"\n{'='*60}")
    print("Summary Table")
    print(f"{'='*60}")
    print(f"{'ε':>8} | {'H_argmax':>10} | {'H_stoch':>10} | {'H_random':>10} | {'CAM':>8} | {'C_argmax':>10} | {'C_stoch':>10} | {'C_random':>10}")
    print("-" * 90)
    for eps in eps_list:
        r = all_results[f"eps_{eps}"]
        print(f"{eps:>8.2f} | {r['H_assign_argmax_mean']:>10.4f} | {r['H_assign_stochastic_mean']:>10.4f} | {r['H_assign_random_mean']:>10.4f} | {r['CAM_degree_mean']:>8.4f} | {r['C_transport_argmax_mean']:>10.4f} | {r['C_transport_stochastic_mean']:>10.4f} | {r['C_transport_random_mean']:>10.4f}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
