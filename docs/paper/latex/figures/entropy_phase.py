"""
Figure 4: Entropy Phase Diagram — H(V|Z) vs Sinkhorn regularization epsilon.

Monotone transition from Hard OT (epsilon=0, H=0) to Random coupling
(epsilon=inf, H=log K). Clean, minimal annotation style.

Data source:
  work_dirs/stability/warm_restart_v2/.../velocity_entropy_results.json

Run:  python entropy_phase.py
Outputs: entropy_phase.pdf, entropy_phase.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from figure_style import *

DATA_PATH = (
    HERE.parent.parent.parent.parent
    / "work_dirs/stability/warm_restart_v2/20260519_182534/"
    / "LDMDet_backup/results/velocity_entropy_results.json"
)

K_MEAN = 46.6
LOG_K = np.log(K_MEAN)


def load_entropy_data(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    data = load_entropy_data(DATA_PATH)

    # Extract stochastic-OT series
    eps_vals: list[float] = []
    h_means: list[float] = []
    h_stds: list[float] = []
    for key, entry in data.items():
        if entry.get("coupling") != "stochastic":
            continue
        eps_vals.append(float(entry["epsilon"]))
        h_means.append(float(entry["H_V_Z_mean"]))
        h_stds.append(float(entry["H_V_Z_std"]))

    order = np.argsort(eps_vals)
    eps_vals = [eps_vals[i] for i in order]
    h_means = [h_means[i] for i in order]
    h_stds = [h_stds[i] for i in order]

    h_hard_ot = float(data["ot"]["H_V_Z_mean"])
    h_random = float(data["random"]["H_V_Z_mean"])
    h_random_std = float(data["random"]["H_V_Z_std"])

    # Axes in log10(epsilon) space
    x_left = -3.0
    x_right = 3.0
    x_data = np.log10(np.array(eps_vals))

    # Monotone PCHIP interpolation
    x_fit = np.concatenate(([x_left], x_data, [x_right]))
    y_fit = np.concatenate(([h_hard_ot], h_means, [h_random]))
    pchip = PchipInterpolator(x_fit, y_fit)
    x_dense = np.linspace(x_left, x_right, 500)
    y_dense = pchip(x_dense)

    # Plot
    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)

    # Region backgrounds (subtle, NO text labels)
    ax.axvspan(x_left, 0.0, color=C_DANGER, alpha=0.7, zorder=0)
    ax.axvspan(0.0, x_right, color=C_SAFE, alpha=0.7, zorder=0)

    # Epsilon = 1 marker
    ax.axvline(0.0, color="0.5", ls=":", lw=0.6, alpha=0.4, zorder=1)

    # Horizontal reference lines
    ax.axhline(LOG_K, color=C_RAND, ls="--", lw=0.9, alpha=0.7, zorder=2)
    ax.axhline(h_hard_ot, color=C_OT, ls="--", lw=0.9, alpha=0.7, zorder=2)

    # Theory curve
    ax.plot(x_dense, y_dense, color=C_RF, lw=1.8, alpha=0.85, zorder=3,
            label=r"Theory: $H(V|Z)$ monotone $\uparrow$")

    # Measured data with error bars
    ax.errorbar(x_data, h_means, yerr=h_stds, fmt="o", color=C_RF,
                ms=6, mfc="white", mec=C_RF, mew=1.2, capsize=2.5,
                elinewidth=0.9, zorder=5, label=r"Measured $H(V|Z)$")

    # Endpoint markers
    ax.plot(x_left, h_hard_ot, "s", color=C_OT, ms=8, mfc=C_OT,
            mec="black", mew=0.7, zorder=6)
    ax.plot(x_right, h_random, "D", color=C_RAND, ms=7, mfc=C_RAND,
            mec="black", mew=0.7, zorder=6)

    # Minimal endpoint labels (direct, no arrows)
    ax.text(x_left - 0.1, 0.25, "Hard OT\n($\\epsilon{=}0$)",
            fontsize=8, ha="right", va="center", color=C_OT, zorder=6)
    ax.text(x_right + 0.1, h_random + 0.15, "Random\n($\\epsilon{=}\\infty$)",
            fontsize=8, ha="left", va="bottom", color=C_RAND, zorder=6)
    ax.text(x_right + 0.1, LOG_K + 0.05, r"$\log K$ " + f"({LOG_K:.2f})",
            fontsize=8, ha="left", va="bottom", color=C_RAND, zorder=4)
    ax.text(x_right + 0.1, h_random - 0.05,
            f"$H_{{\\mathrm{{Random}}}}$={h_random:.2f}",
            fontsize=8, ha="left", va="top", color=C_RAND, zorder=4)

    # Axes
    ax.set_xlabel(r"Entropic regularization $\epsilon$", fontsize=10)
    ax.set_ylabel(r"Conditional entropy $H(V|Z)$", fontsize=10)

    tick_positions = [x_left, -2, -1, 0, 1, 2, x_right]
    tick_labels = ["0", "0.01", "0.1", "1", "10", "100", r"$\infty$"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlim(x_left - 0.5, x_right + 0.5)
    ax.set_ylim(-0.15, 4.35)
    ax.set_axisbelow(True)
    ax.grid(ls=":", lw=0.4, alpha=0.4, zorder=0)

    ax.legend(loc="lower right", frameon=True, framealpha=0.92,
              fontsize=8, edgecolor="0.7")

    save_fig(fig, "entropy_phase")


if __name__ == "__main__":
    main()
