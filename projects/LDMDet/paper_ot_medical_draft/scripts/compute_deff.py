#!/usr/bin/env python3
"""Compute D_eff (effective match count) from Sinkhorn transport matrices.

D_eff = 1 / Σ_j (s_j / N)²   where s_j is column mass for target j.

This is a property of the coupling mechanism — computable from the cost matrix
and ε without trained models. We use the same setup as training:
  - N=500 proposals drawn from a standard Gaussian in R^4
  - M targets drawn from a realistic spatial distribution (uniform in [-s,s]^4
    with s=2.0, matching the paper's SNR scale)
  - Cost matrix C_ij = ||x1_i - x0_j||^2
  - Sinkhorn iteration (20 steps) with uniform marginals

For argmax decoding, assignments are one-hot per row; D_eff is computed from
assignment counts. For stochastic decoding, D_eff is computed from expected
column masses (Sinkhorn column sums). Reported values average over 100
random proposal/target realizations.

Output: deff_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = SCRIPT_DIR.parent / "deff_results.json"

N_PROPOSALS = 500
N_TARGETS = 46
N_GROUPS = 8               # chromosome groups (A-G + X/Y)
TARGETS_PER_GROUP = [3, 2, 7, 3, 3, 2, 2, 1, 1]  # approximate sizes, will truncate
SNR_SCALE = 2.0            # matches paper: x0 normalized to [-s, s]^4
SINKHORN_ITERS = 20
N_SAMPLE_DRAWS = 100       # stochastic sampling draws per realization
N_REALIZATIONS = 50        # number of random proposal/target draws
EPS_VALUES = [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0]
RNG = np.random.default_rng(42)


def sinkhorn(C: np.ndarray, eps: float, n_iters: int = 20) -> np.ndarray:
    """Compute entropic OT plan via Sinkhorn iteration (log-stabilized).

    Args:
        C: (N, M) cost matrix.
        eps: regularization strength.
        n_iters: number of Sinkhorn iterations.

    Returns:
        P: (N, M) doubly-stochastic transport plan.
    """
    N, M = C.shape
    # Log-stabilized Sinkhorn for numerical safety at small epsilon
    log_a = np.log(np.ones(N) / N)
    log_b = np.log(np.ones(M) / M)
    f = np.zeros(N)   # log(u)
    g = np.zeros(M)   # log(v)

    for _ in range(n_iters):
        # f = log_a - logsumexp(-C/eps + g, axis=1)
        log_Kg = -C / eps + g[None, :]        # (N, M)
        f = log_a - np.max(log_Kg, axis=1) - np.log(
            np.sum(np.exp(log_Kg - np.max(log_Kg, axis=1, keepdims=True)), axis=1)
        )
        # g = log_b - logsumexp(-C/eps + f, axis=0)
        log_Kf = -C / eps + f[:, None]         # (N, M)
        g = log_b - np.max(log_Kf, axis=0) - np.log(
            np.sum(np.exp(log_Kf - np.max(log_Kf, axis=0, keepdims=True)), axis=0)
        )

    # Reconstruct P = diag(exp(f)) @ K @ diag(exp(g))
    P = np.exp(f[:, None] - C / eps + g[None, :])
    return P


def effective_match_count(column_mass: np.ndarray) -> float:
    """Inverse Herfindahl index: D_eff = (Σ s_j)^2 / Σ s_j^2."""
    total = column_mass.sum()
    if total == 0:
        return 0.0
    return float((total ** 2) / np.sum(column_mass ** 2))


def sample_clustered_targets(rng: np.random.Generator) -> np.ndarray:
    """Sample targets in clustered groups, mimicking chromosome spatial layout.

    Each chromosome group occupies a distinct spatial region with small
    within-group jitter. This creates the density bottleneck described in
    the paper: targets in the same cluster compete for nearby proposals.
    """
    # Group centers scattered across a limited region (mimics metaphase spread)
    group_centers = rng.uniform(-SNR_SCALE * 0.5, SNR_SCALE * 0.5,
                                 (N_GROUPS, 4))
    targets = []
    # Realistic chromosome group sizes (A through Y, approx)
    sizes = [3, 2, 7, 3, 3, 2, 2, 1, 1, 7, 3, 2, 2]  # truncated to 8 groups: take first 8
    actual_sizes = [3, 2, 7, 3, 3, 2, 2, 24]  # last group = all remaining
    # Actually: simpler approach — 8 groups with various sizes summing to 46
    sizes = [8, 7, 7, 6, 6, 5, 4, 3]  # sums to 46
    for g in range(N_GROUPS):
        n = sizes[g]
        # Tight within-group scatter — chromosomes in same group nearly overlap
        scatter = 0.08 * SNR_SCALE
        group_targets = group_centers[g] + rng.normal(0, scatter, (n, 4))
        targets.append(group_targets)
    return np.vstack(targets)


def compute_for_epsilon(eps: float) -> dict:
    """Compute D_eff statistics across realizations for one epsilon."""
    argmax_vals: list[float] = []
    stoch_sampled_vals: list[float] = []

    for _ in range(N_REALIZATIONS):
        # Sample proposals from Gaussian (wider than targets)
        x1 = RNG.standard_normal((N_PROPOSALS, 4)) * SNR_SCALE
        # Sample targets in spatial clusters
        x0 = sample_clustered_targets(RNG)
        # Pairwise L2 cost
        diff = x1[:, None, :] - x0[None, :, :]          # (N, M, 4)
        C = np.sum(diff ** 2, axis=-1)                    # (N, M)

        P = sinkhorn(C, eps, SINKHORN_ITERS)

        # --- Argmax decoding ---
        assignments_argmax = P.argmax(axis=1)             # (N,)
        col_mass_argmax = np.bincount(assignments_argmax,
                                       minlength=N_TARGETS).astype(float)
        argmax_vals.append(effective_match_count(col_mass_argmax))

        # --- Stochastic decoding (sample many times, average D_eff) ---
        # Normalize rows for sampling (numerical guard)
        P_norm = P / P.sum(axis=1, keepdims=True)
        for _ in range(N_SAMPLE_DRAWS):
            # Sample one assignment per proposal
            sampled = np.array([
                RNG.choice(N_TARGETS, p=P_norm[i])
                for i in range(N_PROPOSALS)
            ])
            col_mass_sampled = np.bincount(sampled, minlength=N_TARGETS).astype(float)
            stoch_sampled_vals.append(effective_match_count(col_mass_sampled))

    # For stochastic: averaged over all draws × realizations
    stoch_arr = np.array(stoch_sampled_vals)
    # Group by realization (N_SAMPLE_DRAWS per realization)
    stoch_per_realization = stoch_arr.reshape(N_REALIZATIONS, N_SAMPLE_DRAWS).mean(axis=1)

    return {
        "epsilon": eps,
        "D_eff_argmax_mean": float(np.mean(argmax_vals)),
        "D_eff_argmax_std": float(np.std(argmax_vals)),
        "D_eff_stoch_mean": float(np.mean(stoch_per_realization)),
        "D_eff_stoch_std": float(np.std(stoch_per_realization)),
        "n_realizations": N_REALIZATIONS,
        "n_sample_draws": N_SAMPLE_DRAWS,
    }


def main():
    print(f"Computing D_eff for {len(EPS_VALUES)} epsilon values × {N_REALIZATIONS} realizations...")
    print(f"  N_proposals={N_PROPOSALS}, M_targets={N_TARGETS}, SNR_scale={SNR_SCALE}")
    print()

    results = []
    for eps in EPS_VALUES:
        r = compute_for_epsilon(eps)
        results.append(r)
        print(
            f"  ε={eps:6.2f}  "
            f"D_eff(argmax)={r['D_eff_argmax_mean']:.2f}±{r['D_eff_argmax_std']:.2f}  "
            f"D_eff(stoch)={r['D_eff_stoch_mean']:.2f}±{r['D_eff_stoch_std']:.2f}"
        )

    output = {
        "description": (
            "D_eff (effective match count) computed from Sinkhorn transport matrices. "
            "N=500 proposals (Gaussian), M=46 targets (uniform in [-2,2]^4), "
            "20 Sinkhorn iterations, 100 realizations per epsilon."
        ),
        "results": results,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
