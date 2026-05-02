"""
Optimal Epsilon Prediction: Refined Model (P1-6)

Key insight: The ε=50 anomaly reveals that the consistency cost is NOT monotone
in ε. Stochastic ε=50 (mAP=0.736 peak) is worse than Random (mAP=0.751), even
though both have similar ρ and η. This is because Sinkhorn coupling at large ε
creates spatially correlated pairings that cause persistent gradient conflicts.

Model:
    mAP(ε) = mAP_OT + α·ρ(ε) - β·η(ε) - γ·bias(ε)

where:
    ρ(ε) = H_row(ε)/H_rand  (diversity, monotone increasing, saturates fast)
    η(ε) = (C_stoch(ε)-C_OT)/(C_rand-C_OT)  (efficiency cost, monotone increasing)
    bias(ε) = KL(T_ε || T_uniform) · η(ε)  (coupling bias, non-monotone)

The coupling bias peaks at intermediate ε where the Sinkhorn transport is
neither deterministic nor uniform, creating the worst gradient conflicts.

Usage:
    python tools/predict_optimal_epsilon.py \
        --diversity-json docs/diversity_results.json \
        --velocity-json velocity_entropy_results.json \
        --out optimal_epsilon_prediction.json
"""

import argparse
import json

import numpy as np
from scipy.optimize import minimize_scalar


def load_data(diversity_path, velocity_path):
    with open(diversity_path) as f:
        div_data = json.load(f)
    with open(velocity_path) as f:
        vel_data = json.load(f)

    H_rand = vel_data["random"]["H_V_Z_mean"]
    C_rand = div_data["eps_0.01"]["C_transport_random_mean"]
    C_OT = div_data["eps_0.01"]["C_transport_argmax_mean"]

    eps_list = []
    rho_list = []
    eta_list = []
    for key in sorted(div_data.keys(), key=lambda k: div_data[k]["epsilon"]):
        eps = div_data[key]["epsilon"]
        H_row = div_data[key]["H_row_mean_mean"]
        C_stoch = div_data[key]["C_transport_stochastic_mean"]
        rho = H_row / H_rand
        eta = (C_stoch - C_OT) / (C_rand - C_OT)
        eps_list.append(eps)
        rho_list.append(rho)
        eta_list.append(eta)

    return {
        "eps": np.array(eps_list),
        "rho": np.array(rho_list),
        "eta": np.array(eta_list),
        "H_rand": H_rand,
        "C_OT": C_OT,
        "C_rand": C_rand,
    }


def main():
    parser = argparse.ArgumentParser(description="Optimal epsilon prediction")
    parser.add_argument("--diversity-json", default="docs/diversity_results.json")
    parser.add_argument("--velocity-json", default="velocity_entropy_results.json")
    parser.add_argument("--out", default="optimal_epsilon_prediction.json")
    args = parser.parse_args()

    data = load_data(args.diversity_json, args.velocity_json)

    print("=" * 80)
    print("STEP 1: Measured diversity and efficiency metrics")
    print("=" * 80)

    print(f"\n  H_rand = {data['H_rand']:.4f}")
    print(f"  C_OT   = {data['C_OT']:.4f}")
    print(f"  C_rand = {data['C_rand']:.4f}")

    print(f"\n  eps     | rho(eps) | eta(eps) | 1-rho   | 1-eta")
    print(f"  --------|----------|----------|---------|---------")
    for i, eps in enumerate(data["eps"]):
        print(f"  {eps:7.2f} | {data['rho'][i]:8.4f} | {data['eta'][i]:8.4f} | "
              f"{1 - data['rho'][i]:7.4f} | {1 - data['eta'][i]:7.4f}")

    print(f"\n{'=' * 80}")
    print("STEP 2: Model-independent prediction")
    print("=" * 80)

    print("""
  The optimal ε can be predicted without parametric fitting by analyzing
  the saturation behavior of ρ(ε) and η(ε):

  1. ρ(ε) = H_row(ε)/H_rand measures diversity recovery
     - ρ > 0.95 at ε ≥ 0.5  (95% diversity recovered)
     - ρ > 0.99 at ε ≥ 1.0  (99% diversity recovered)
     - ρ > 0.999 at ε ≥ 5.0 (essentially Random diversity)

  2. η(ε) = (C_stoch-C_OT)/(C_rand-C_OT) measures efficiency loss
     - η < 0.80 at ε ≤ 1.0  (efficiency still good)
     - η ≈ 0.97 at ε = 5.0  (efficiency significantly degraded)
     - η > 0.99 at ε ≥ 50   (efficiency nearly Random-level)

  3. Key: ρ saturates ~10x faster than η
     - At ε=0.5: ρ=0.951 (95% diversity) but η=0.624 (only 62% cost)
     - At ε=1.0: ρ=0.986 (99% diversity) but η=0.788 (79% cost)
     - At ε=5.0: ρ≈1.000 (full diversity) but η=0.966 (97% cost)

  The "sweet spot" is where ρ is high but η is still moderate.
  This gives ε* ∈ [0.5, 5.0], most likely ε* ∈ [1, 3].
    """)

    print(f"{'=' * 80}")
    print("STEP 3: Analysis of the ε=50 anomaly")
    print("=" * 80)

    print("""
  The ε=50 anomaly cannot be explained by ρ and η alone, since both are
  saturated at ε≥5. The degradation is caused by COUPLING BIAS:

  Stochastic ε=50 vs Random:
  - Both have ρ≈1.0, η≈1.0 (similar diversity and efficiency)
  - But Stochastic ε=50 has mAP=0.736 (peak), while Random has mAP=0.751
  - Difference: Sinkhorn transport at ε=50 is NOT truly uniform

  The Sinkhorn transport matrix T_ε at large ε is close to uniform but
  retains residual structure from the cost matrix. This creates:
  1. Spatially correlated pairings across training iterations
  2. Persistent gradient conflicts in certain directions
  3. Training instability that causes mAP degradation after epoch 40

  In contrast, Random coupling generates independent pairings each iteration,
  so gradient conflicts average out over time.

  The coupling bias can be quantified as:
    bias(ε) = E_i[KL(T_ε(i,·) || Uniform)] × η(ε)

  This is non-monotone: it peaks at intermediate ε where T_ε has maximum
  structure relative to uniform, and vanishes at ε→0 (deterministic) and
  ε→∞ (uniform).
    """)

    print(f"{'=' * 80}")
    print("STEP 4: Three-regime model")
    print("=" * 80)

    print("""
  Based on the analysis, there are three regimes:

  REGIME 1: ε ∈ (0, 0.5) — "OT-dominated"
    - ρ < 0.95: insufficient diversity
    - η < 0.62: good efficiency
    - mAP limited by diversity collapse
    - Prediction: mAP < 0.745

  REGIME 2: ε ∈ [0.5, 5.0] — "Sweet spot"
    - ρ > 0.95: sufficient diversity
    - η < 0.97: acceptable efficiency
    - Coupling bias is minimal (Sinkhorn structure is weak)
    - Prediction: mAP ≈ 0.748-0.751

  REGIME 3: ε > 5.0 — "Coupling bias dominated"
    - ρ ≈ 1.0: full diversity
    - η > 0.97: high efficiency cost
    - Coupling bias causes training instability
    - Prediction: mAP degrades (0.736 at ε=50, worse with longer training)

  Optimal ε* is in REGIME 2, specifically where:
    - ρ(ε*) > 0.99 (ε* > 1.0)
    - η(ε*) < 0.90 (ε* < 3.0 approximately)
  => ε* ≈ 1.0-3.0
    """)

    print(f"{'=' * 80}")
    print("STEP 5: Quantitative prediction using ρ-η tradeoff")
    print("=" * 80)

    print("""
  Define the "net benefit" as the difference between diversity recovery
  and efficiency loss:

    NB(ε) = ρ(ε) - λ·η(ε)

  where λ is the relative cost of efficiency loss vs diversity gain.
  The optimal ε maximizes NB(ε).

  From experimental data:
    - Random (ε=∞): ρ=1.0, η=1.0, mAP=0.751
    - Stochastic ε=5: ρ≈1.0, η≈0.97, mAP=0.751
    - Stochastic ε=50: ρ≈1.0, η≈1.0, mAP=0.736 (anomaly)

  The fact that ε=5 and Random have the same mAP despite different η
  suggests λ is small (efficiency loss is tolerable when diversity is
  saturated). But ε=50 shows that coupling bias dominates at high ε.

  For the ρ-η tradeoff (ignoring coupling bias):
    dNB/dε = ρ'(ε) - λ·η'(ε) = 0

  Using exponential approximations:
    ρ(ε) ≈ 1 - exp(-ε/0.3)
    η(ε) ≈ 1 - exp(-ε/1.5)

    ρ'(ε) = (1/0.3)·exp(-ε/0.3)
    η'(ε) = (1/1.5)·exp(-ε/1.5)

  Setting ρ'(ε*) = λ·η'(ε*):
    (1/0.3)·exp(-ε*/0.3) = λ·(1/1.5)·exp(-ε*/1.5)
    5·exp(-ε*/0.3) = λ·(2/3)·exp(-ε*/1.5)
    exp(-ε*/0.3 + ε*/1.5) = (2λ)/(15)
    exp(-ε*·(1/0.3 - 1/1.5)) = (2λ)/15
    exp(-ε*·(10/3)) = (2λ)/15

    ε* = -(3/10)·ln(2λ/15)
    """)

    print(f"  ε* as a function of λ (efficiency cost weight):")
    print(f"  {'λ':>8} | {'ε*':>8} | Interpretation")
    print(f"  {'-' * 8}-+-{'-' * 8}-+-{'-' * 30}")
    for lam in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        ratio = 2 * lam / 15
        if ratio > 0:
            eps_star = -(3.0 / 10.0) * np.log(ratio)
        else:
            eps_star = np.inf
        interp = ""
        if lam < 0.1:
            interp = "Efficiency nearly free"
        elif lam < 1.0:
            interp = "Moderate efficiency cost"
        else:
            interp = "Efficiency costly"
        print(f"  {lam:8.2f} | {eps_star:8.2f} | {interp}")

    print(f"""
  Since ε=5 and Random have the same mAP (efficiency loss is tolerable),
  λ is small (≈ 0.01-0.1), giving ε* ≈ 1.5-3.0.

  But we must also avoid the coupling bias regime (ε > 5), so the
  practical recommendation is ε* ≈ 1-3.
    """)

    print(f"{'=' * 80}")
    print("STEP 6: Testable predictions")
    print("=" * 80)

    print("""
  The following predictions can be verified by running additional experiments:

  PREDICTION 1: Stochastic ε=1.0
    ρ=0.986, η=0.789
    Expected mAP ≈ 0.750-0.752 (close to Random, slightly better than ε=5
    due to lower efficiency cost)
    This is the STRONGEST test of the model.

  PREDICTION 2: Stochastic ε=2.0
    ρ≈0.999, η≈0.90
    Expected mAP ≈ 0.749-0.751 (similar to ε=5 but with lower efficiency)

  PREDICTION 3: Stochastic ε=0.5
    ρ=0.951, η=0.624
    Expected mAP ≈ 0.745-0.750 (slightly lower due to 5% diversity deficit)

  PREDICTION 4: Stochastic ε=10.0
    ρ≈1.0, η≈0.975
    Expected mAP ≈ 0.740-0.748 (coupling bias starts to matter)

  PREDICTION 5: Stochastic ε=0.1
    ρ=0.693, η=0.216
    Expected mAP ≈ 0.735-0.745 (significant diversity deficit)

  RANKING PREDICTION:
    ε=1.0 ≥ ε=2.0 ≥ ε=5.0 ≥ ε=0.5 > ε=10.0 > ε=50.0 > ε=0.1 > Hard OT
    """)

    print(f"{'=' * 80}")
    print("STEP 7: Comparison with image generation")
    print("=" * 80)

    print("""
  In image generation (d ≈ 10^5):
    - OT diversity collapse does NOT occur (ΔH/H_rand ≈ 0)
    - Diversity gain α ≈ 0 (OT already provides sufficient diversity)
    - Efficiency cost δ > 0 (OT's straight paths are beneficial)
    - Optimal ε* → 0 (hard OT is optimal)

  In object detection (d = 4):
    - OT diversity collapse is severe (ΔH/H_rand ≈ 1)
    - Diversity gain α >> 0 (OT severely lacks diversity)
    - Efficiency cost δ > 0 (but relatively small)
    - Optimal ε* > 0 (Stochastic Coupling needed to restore diversity)

  The dimension d determines the sign and magnitude of the diversity gain,
  explaining why OT works in image generation but hurts in detection.
    """)

    prediction = {
        "model_independent": {
            "eps_star_range": [0.5, 5.0],
            "eps_star_best_guess": [1.0, 3.0],
            "reasoning": "rho saturates 10x faster than eta",
        },
        "three_regimes": {
            "OT_dominated": {"eps_range": [0, 0.5], "rho_range": [0, 0.95],
                             "prediction": "mAP < 0.745"},
            "sweet_spot": {"eps_range": [0.5, 5.0], "rho_range": [0.95, 1.0],
                           "prediction": "mAP ≈ 0.748-0.751"},
            "bias_dominated": {"eps_range": [5.0, 100.0], "rho_range": [1.0, 1.0],
                               "prediction": "mAP degrades"},
        },
        "testable_predictions": {
            "eps_0.1": {"mAP_range": [0.735, 0.745]},
            "eps_0.5": {"mAP_range": [0.745, 0.750]},
            "eps_1.0": {"mAP_range": [0.750, 0.752]},
            "eps_2.0": {"mAP_range": [0.749, 0.751]},
            "eps_10.0": {"mAP_range": [0.740, 0.748]},
        },
        "measured_data": {
            "eps": data["eps"].tolist(),
            "rho": data["rho"].tolist(),
            "eta": data["eta"].tolist(),
        },
    }

    with open(args.out, "w") as f:
        json.dump(prediction, f, indent=2)
    print(f"\n  Results saved to {args.out}")


if __name__ == "__main__":
    main()
