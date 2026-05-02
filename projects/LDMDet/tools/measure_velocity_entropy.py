"""
Velocity Distribution Entropy Measurement (Offline)

Measures velocity distribution entropy under different coupling strategies
to validate Theorem 1 (OT Diversity Gap) and Theorem 4.3 (Stochastic Coupling).

Key metrics:
- H(V|Z): conditional velocity entropy given noise
  - Random: log K
  - OT: 0
  - Stochastic: H_row_mean (from Sinkhorn transport matrix)
- H(V): unconditional velocity entropy (Gaussian approximation)
- H(V|X_t): conditional velocity entropy given intermediate state x_t
  - Computed via analytical posterior p(j|x_t) at multiple time steps
- Velocity variance decomposition (between-GT vs within-GT)
- Delta_H: H_rand(V|Z) - H_OT(V|Z) = log K (Theorem 1 validation)

Usage:
    python tools/measure_velocity_entropy.py \
        --ann data/selfmake_chromosome202250604_NoResizeNoAug/valid/_annotations.coco.json \
        --eps-list 0.01 0.1 0.5 1.0 5.0 10.0 50.0 100.0 \
        --num-proposals 500 --snr-scale 2.0 \
        --num-samples 200 --out velocity_entropy_results.json
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


def compute_H_V_Z(transport, K):
    N = transport.shape[0]
    row_probs = transport / transport.sum(dim=1, keepdim=True)
    row_entropies = -(row_probs * torch.clamp(row_probs, min=1e-10).log()).sum(dim=1)
    return row_entropies.mean().item()


def compute_H_V_gaussian(velocities):
    d = velocities.shape[1]
    mean = velocities.mean(dim=0)
    centered = velocities - mean
    cov = (centered.T @ centered) / (centered.shape[0] - 1)
    cov += 1e-6 * torch.eye(d, device=cov.device)
    sign, logdet = torch.linalg.slogdet(cov)
    if sign <= 0:
        eigvals = torch.linalg.eigvalsh(cov)
        logdet = torch.log(torch.clamp(eigvals, min=1e-10)).sum()
    H = 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * logdet.item()
    return H


def compute_H_V_X_t_analytical(noise, gt_diffusion, t, coupling_type="random",
                                transport=None, epsilon=None):
    N = noise.shape[0]
    K = gt_diffusion.shape[0]
    device = noise.device
    d = noise.shape[1]

    if t <= 1e-8:
        if coupling_type == "stochastic":
            return compute_H_V_Z(transport, K)
        elif coupling_type == "random":
            return float(np.log(K))
        else:
            return 0.0
    if t >= 1.0 - 1e-8:
        return 0.0

    if coupling_type == "random":
        idx = torch.randint(0, K, (N,), device=device)
    elif coupling_type == "ot":
        cost = torch.cdist(noise, gt_diffusion, p=2)
        idx = cost.argmin(dim=1)
    elif coupling_type == "stochastic":
        row_probs = transport / transport.sum(dim=1, keepdim=True)
        idx = torch.multinomial(row_probs, 1).squeeze(-1)
    else:
        raise ValueError(f"Unknown coupling type: {coupling_type}")

    x_t = (1 - t) * noise + t * gt_diffusion[idx]

    sigma_t_sq = (1 - t) ** 2
    log_p_x_given_j = torch.zeros(N, K, device=device)
    for j in range(K):
        diff = x_t - t * gt_diffusion[j].unsqueeze(0)
        mahal = (diff ** 2).sum(dim=-1) / sigma_t_sq
        log_p_x_given_j[:, j] = -0.5 * mahal - 0.5 * d * np.log(2 * np.pi * sigma_t_sq)

    if coupling_type == "random":
        log_prior = -np.log(K) * torch.ones(N, K, device=device)
    elif coupling_type == "ot":
        cost = torch.cdist(noise, gt_diffusion, p=2)
        nearest = cost.argmin(dim=1)
        log_prior = torch.full((N, K), -1e10, device=device)
        log_prior[torch.arange(N), nearest] = 0.0
    elif coupling_type == "stochastic":
        row_probs = transport / transport.sum(dim=1, keepdim=True)
        log_prior = torch.clamp(row_probs, min=1e-10).log()
    else:
        raise ValueError(f"Unknown coupling type: {coupling_type}")

    log_joint = log_p_x_given_j + log_prior
    log_norm = torch.logsumexp(log_joint, dim=1, keepdim=True)
    log_posterior = log_joint - log_norm
    posterior = torch.exp(log_posterior)
    posterior = torch.clamp(posterior, min=1e-10)
    log_posterior = torch.clamp(log_posterior, min=np.log(1e-10))

    H_per_sample = -(posterior * log_posterior).sum(dim=1)
    return H_per_sample.mean().item()


def compute_velocity_variance_decomposition(noise, gt_diffusion, coupling_idx):
    N, d = noise.shape
    K = gt_diffusion.shape[0]

    velocities = gt_diffusion[coupling_idx] - noise
    total_mean = velocities.mean(dim=0)
    total_var = ((velocities - total_mean) ** 2).sum(dim=1).mean().item()

    gt_means = []
    gt_counts = []
    for j in range(K):
        mask = coupling_idx == j
        if mask.sum() > 0:
            gt_means.append(velocities[mask].mean(dim=0))
            gt_counts.append(mask.sum().item())
        else:
            gt_means.append(total_mean)
            gt_counts.append(0)

    gt_means = torch.stack(gt_means)
    gt_counts = torch.tensor(gt_counts, dtype=torch.float, device=noise.device)
    gt_probs = gt_counts / gt_counts.sum()

    between_var = (gt_probs.unsqueeze(1) * (gt_means - total_mean.unsqueeze(0)) ** 2).sum().item()

    within_var = 0.0
    for j in range(K):
        mask = coupling_idx == j
        if mask.sum() > 1:
            v_j = velocities[mask]
            within_var += gt_probs[j].item() * ((v_j - v_j.mean(dim=0)) ** 2).sum(dim=1).mean().item()

    return total_var, between_var, within_var


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
    parser = argparse.ArgumentParser(description="Velocity entropy measurement")
    parser.add_argument("--ann", required=True, help="COCO annotation JSON path")
    parser.add_argument(
        "--eps-list",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    )
    parser.add_argument("--num-proposals", type=int, default=500)
    parser.add_argument("--snr-scale", type=float, default=2.0)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--num-repeats", type=int, default=3)
    parser.add_argument("--time-steps", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--out", default="velocity_entropy_results.json")
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

    # === Random & OT baselines (no epsilon) ===
    print(f"\n{'='*80}")
    print("Computing baselines: Random & OT coupling")
    print(f"{'='*80}")

    baseline_agg = {"random": {}, "ot": {}}
    for coupling in ["random", "ot"]:
        for metric_name in ["H_V_Z", "H_V", "total_var", "between_var", "within_var"]:
            baseline_agg[coupling][metric_name] = []
        for t in args.time_steps:
            baseline_agg[coupling][f"H_V_X_t_{t:.1f}"] = []

    for sample_idx, gt_cxcywh in enumerate(samples):
        gt_diffusion = (gt_cxcywh.to(device) * 2 - 1) * args.snr_scale
        K = gt_diffusion.shape[0]

        for repeat in range(args.num_repeats):
            noise = torch.randn(args.num_proposals, 4, device=device)
            cost = torch.cdist(noise, gt_diffusion, p=2)

            # Random coupling
            random_idx = torch.randint(0, K, (args.num_proposals,), device=device)
            random_vel = gt_diffusion[random_idx] - noise
            baseline_agg["random"]["H_V_Z"].append(float(np.log(K)))
            baseline_agg["random"]["H_V"].append(compute_H_V_gaussian(random_vel))
            tv, bv, wv = compute_velocity_variance_decomposition(
                noise, gt_diffusion, random_idx)
            baseline_agg["random"]["total_var"].append(tv)
            baseline_agg["random"]["between_var"].append(bv)
            baseline_agg["random"]["within_var"].append(wv)

            # OT coupling
            ot_idx = cost.argmin(dim=1)
            ot_vel = gt_diffusion[ot_idx] - noise
            baseline_agg["ot"]["H_V_Z"].append(0.0)
            baseline_agg["ot"]["H_V"].append(compute_H_V_gaussian(ot_vel))
            tv, bv, wv = compute_velocity_variance_decomposition(
                noise, gt_diffusion, ot_idx)
            baseline_agg["ot"]["total_var"].append(tv)
            baseline_agg["ot"]["between_var"].append(bv)
            baseline_agg["ot"]["within_var"].append(wv)

            # H(V|X_t) at multiple time steps
            for t in args.time_steps:
                H_rand_xt = compute_H_V_X_t_analytical(
                    noise, gt_diffusion, t, coupling_type="random")
                H_ot_xt = compute_H_V_X_t_analytical(
                    noise, gt_diffusion, t, coupling_type="ot")
                baseline_agg["random"][f"H_V_X_t_{t:.1f}"].append(H_rand_xt)
                baseline_agg["ot"][f"H_V_X_t_{t:.1f}"].append(H_ot_xt)

    # Aggregate baselines
    for coupling in ["random", "ot"]:
        result = {"coupling": coupling}
        for metric_name, values in baseline_agg[coupling].items():
            result[f"{metric_name}_mean"] = float(np.mean(values))
            result[f"{metric_name}_std"] = float(np.std(values))
        all_results[coupling] = result
        print(f"\n{coupling.upper()} baseline:")
        print(f"  H(V|Z) = {result['H_V_Z_mean']:.4f}")
        print(f"  H(V)   = {result['H_V_mean']:.4f}")
        print(f"  Total var = {result['total_var_mean']:.4f}, "
              f"Between = {result['between_var_mean']:.4f}, "
              f"Within = {result['within_var_mean']:.4f}")
        for t in args.time_steps:
            key = f"H_V_X_t_{t:.1f}_mean"
            print(f"  H(V|X_t={t:.1f}) = {result[key]:.4f}")

    # === Stochastic coupling at each epsilon ===
    for eps in args.eps_list:
        print(f"\n{'='*80}")
        print(f"Computing Stochastic coupling at ε = {eps}")
        print(f"{'='*80}")

        agg = {}
        for metric_name in ["H_V_Z", "H_V", "total_var", "between_var", "within_var"]:
            agg[metric_name] = []
        for t in args.time_steps:
            agg[f"H_V_X_t_{t:.1f}"] = []

        for sample_idx, gt_cxcywh in enumerate(samples):
            gt_diffusion = (gt_cxcywh.to(device) * 2 - 1) * args.snr_scale
            K = gt_diffusion.shape[0]

            for repeat in range(args.num_repeats):
                noise = torch.randn(args.num_proposals, 4, device=device)
                transport, cost = compute_sinkhorn_transport(
                    noise, gt_diffusion, epsilon=eps, num_iters=20)

                # Stochastic coupling
                row_probs = transport / transport.sum(dim=1, keepdim=True)
                stoch_idx = torch.multinomial(row_probs, 1).squeeze(-1)
                stoch_vel = gt_diffusion[stoch_idx] - noise

                # H(V|Z) = mean row entropy
                H_V_Z = compute_H_V_Z(transport, K)
                agg["H_V_Z"].append(H_V_Z)

                # H(V) = Gaussian approximation
                agg["H_V"].append(compute_H_V_gaussian(stoch_vel))

                # Velocity variance decomposition
                tv, bv, wv = compute_velocity_variance_decomposition(
                    noise, gt_diffusion, stoch_idx)
                agg["total_var"].append(tv)
                agg["between_var"].append(bv)
                agg["within_var"].append(wv)

                # H(V|X_t) at multiple time steps
                for t in args.time_steps:
                    H_xt = compute_H_V_X_t_analytical(
                        noise, gt_diffusion, t, coupling_type="stochastic",
                        transport=transport, epsilon=eps)
                    agg[f"H_V_X_t_{t:.1f}"].append(H_xt)

        result = {"coupling": "stochastic", "epsilon": eps}
        for metric_name, values in agg.items():
            result[f"{metric_name}_mean"] = float(np.mean(values))
            result[f"{metric_name}_std"] = float(np.std(values))
        all_results[f"stochastic_eps_{eps}"] = result

        print(f"\n  Stochastic ε={eps}:")
        print(f"  H(V|Z) = {result['H_V_Z_mean']:.4f}")
        print(f"  H(V)   = {result['H_V_mean']:.4f}")
        print(f"  Total var = {result['total_var_mean']:.4f}, "
              f"Between = {result['between_var_mean']:.4f}, "
              f"Within = {result['within_var_mean']:.4f}")
        for t in args.time_steps:
            key = f"H_V_X_t_{t:.1f}_mean"
            print(f"  H(V|X_t={t:.1f}) = {result[key]:.4f}")

    # === Summary ===
    print(f"\n{'='*80}")
    print("SUMMARY: Velocity Entropy by Coupling Strategy")
    print(f"{'='*80}")

    print(f"\n{'Coupling':>25} | {'H(V|Z)':>8} | {'H(V)':>8} | {'ΔH/H_rand':>10} | "
          f"{'Between%':>9} | {'Within%':>8}")
    print("-" * 85)

    H_rand_V_Z = all_results["random"]["H_V_Z_mean"]
    for key in ["random", "ot"] + [f"stochastic_eps_{e}" for e in args.eps_list]:
        r = all_results[key]
        coupling_name = key.replace("stochastic_eps_", "Stoch ε=")
        if key == "random":
            coupling_name = "Random"
        elif key == "ot":
            coupling_name = "Hard OT"

        delta_H_ratio = r["H_V_Z_mean"] / H_rand_V_Z if H_rand_V_Z > 0 else 0
        total_v = r["total_var_mean"]
        between_pct = r["between_var_mean"] / total_v * 100 if total_v > 0 else 0
        within_pct = r["within_var_mean"] / total_v * 100 if total_v > 0 else 0

        print(f"{coupling_name:>25} | {r['H_V_Z_mean']:>8.4f} | {r['H_V_mean']:>8.4f} | "
              f"{delta_H_ratio:>10.4f} | {between_pct:>8.1f}% | {within_pct:>7.1f}%")

    # Time-dependent H(V|X_t) summary
    print(f"\n{'='*80}")
    print("SUMMARY: H(V|X_t) at Different Time Steps")
    print(f"{'='*80}")

    header = f"{'Coupling':>25}"
    for t in args.time_steps:
        header += f" | t={t:.1f}"
    print(header)
    print("-" * (25 + 12 * len(args.time_steps)))

    for key in ["random", "ot"] + [f"stochastic_eps_{e}" for e in args.eps_list]:
        r = all_results[key]
        coupling_name = key.replace("stochastic_eps_", "Stoch ε=")
        if key == "random":
            coupling_name = "Random"
        elif key == "ot":
            coupling_name = "Hard OT"

        row = f"{coupling_name:>25}"
        for t in args.time_steps:
            val = r[f"H_V_X_t_{t:.1f}_mean"]
            row += f" | {val:>7.4f}"
        print(row)

    # Theorem 1 validation
    print(f"\n{'='*80}")
    print("THEOREM 1 VALIDATION: ΔH = H_rand(V|Z) - H_OT(V|Z)")
    print(f"{'='*80}")
    H_rand = all_results["random"]["H_V_Z_mean"]
    H_ot = all_results["ot"]["H_V_Z_mean"]
    delta_H = H_rand - H_ot
    mean_K = np.mean(K_values)
    log_K = np.log(mean_K)
    print(f"  H_rand(V|Z) = {H_rand:.4f}")
    print(f"  H_OT(V|Z)   = {H_ot:.4f}")
    print(f"  ΔH           = {delta_H:.4f}")
    print(f"  log(K)       = {log_K:.4f} (K_mean={mean_K:.1f})")
    print(f"  ΔH/log(K)    = {delta_H/log_K:.4f}")
    print(f"  Theorem 1 predicts ΔH = log(K) = {log_K:.4f}")
    print(f"  Validation: {'PASS' if abs(delta_H - log_K) / log_K < 0.1 else 'DEVIATION'} "
          f"(relative error: {abs(delta_H - log_K) / log_K * 100:.1f}%)")

    # Dimension-dependent severity
    H_rand_V = all_results["random"]["H_V_mean"]
    delta_H_uncond = H_rand_V - all_results["ot"]["H_V_mean"]
    severity = delta_H_uncond / H_rand_V
    print(f"\n  H_rand(V)    = {H_rand_V:.4f}")
    print(f"  H_OT(V)      = {all_results['ot']['H_V_mean']:.4f}")
    print(f"  ΔH_uncond    = {delta_H_uncond:.4f}")
    print(f"  ΔH_uncond/H_rand(V) = {severity:.4f}")
    print(f"  Theory predicts ≈ 2log(K)/(d/2*log(2πeσ²)+log(K)) ≈ "
          f"{2*log_K/(2*np.log(2*np.pi*np.e)+log_K):.4f} (d=4, σ²=1)")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
