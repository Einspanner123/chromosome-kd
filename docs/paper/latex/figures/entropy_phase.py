"""
Figure 4: Entropy Phase Diagram — H(V|Z) vs Sinkhorn regularization epsilon.

Optimized: manual label placement with strategic positioning to avoid overlap.
Data: work_dirs/stability/warm_restart_v2/.../velocity_entropy_results.json

Run:  python entropy_phase.py
Outputs: entropy_phase.pdf, entropy_phase.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
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

    eps_vals: list[float] = []
    h_means: list[float] = []
    for entry in data.values():
        if entry.get("coupling") != "stochastic":
            continue
        eps_vals.append(float(entry["epsilon"]))
        h_means.append(float(entry["H_V_Z_mean"]))

    order = np.argsort(eps_vals)
    eps_vals = np.array(eps_vals)[order]
    h_means = np.array(h_means)[order]

    h_hard_ot = float(data["ot"]["H_V_Z_mean"])
    h_random = float(data["random"]["H_V_Z_mean"])

    x_left = -3.0
    x_right = 3.0
    x_data = np.log10(eps_vals)

    x_fit = np.concatenate(([x_left], x_data, [x_right]))
    y_fit = np.concatenate(([h_hard_ot], h_means, [h_random]))
    pchip = PchipInterpolator(x_fit, y_fit)
    x_dense = np.linspace(x_left, x_right, 500)
    y_dense = pchip(x_dense)

    df_meas = pd.DataFrame({
        "log_eps": x_data,
        "H": h_means,
    })

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)

    ax.axvspan(x_left, 0.0, color=C_DANGER, alpha=0.4, zorder=0)
    ax.axvspan(0.0, x_right, color=C_SAFE, alpha=0.4, zorder=0)
    ax.axvline(0.0, color="0.5", ls=":", lw=0.6, alpha=0.4, zorder=1)

    ax.axhline(LOG_K, color=C_RAND, ls="--", lw=0.9, alpha=0.5, zorder=2)
    ax.axhline(h_hard_ot, color=C_OT, ls="--", lw=0.9, alpha=0.5, zorder=2)

    df_theory = pd.DataFrame({"x": x_dense, "y": y_dense})
    sns.lineplot(data=df_theory, x="x", y="y", color=C_RF,
                 linewidth=2.0, alpha=0.9, zorder=3, ax=ax,
                 label=r"Empirical monotone interpolation")

    sns.scatterplot(data=df_meas, x="log_eps", y="H", color=C_RF,
                    s=65, edgecolor="white", linewidth=1.5,
                    zorder=5, ax=ax)

    ax.plot(x_left, h_hard_ot, "s", color=C_OT, ms=10, mfc=C_OT,
            mec="black", mew=0.8, zorder=6)
    ax.plot(x_right, h_random, "D", color=C_RAND, ms=9, mfc=C_RAND,
            mec="black", mew=0.8, zorder=6)

    fs = 9.5

    ax.annotate(
        f"Hard OT (ε=0)\nH={h_hard_ot:.2f}",
        xy=(x_left, h_hard_ot),
        xytext=(-2.2, 0.35),
        fontsize=fs,
        color=C_OT,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_OT, lw=0.8, alpha=0.9),
        arrowprops=dict(arrowstyle="-", color=C_OT, lw=0.8),
        zorder=7,
    )

    ax.annotate(
        r"$\log K$" + f"\n={LOG_K:.2f}",
        xy=(2.8, LOG_K),
        xytext=(2.2, 4.15),
        fontsize=fs,
        color=C_RAND,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_RAND, lw=0.8, alpha=0.9),
        arrowprops=dict(arrowstyle="-", color=C_RAND, lw=0.8),
        zorder=7,
    )

    ax.annotate(
        f"Random (ε=∞)\nH={h_random:.2f}",
        xy=(x_right, h_random),
        xytext=(2.2, 3.55),
        fontsize=fs,
        color=C_RAND,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_RAND, lw=0.8, alpha=0.9),
        arrowprops=dict(arrowstyle="-", color=C_RAND, lw=0.8),
        zorder=7,
    )

    ax.set_xlabel(r"Entropic regularization $\epsilon$", fontsize=11)
    ax.set_ylabel(r"Conditional entropy $H(V|Z)$", fontsize=11)

    tick_positions = [x_left, -2, -1, 0, 1, 2, x_right]
    tick_labels = ["0", "0.01", "0.1", "1", "10", "100", r"$\infty$"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.set_xlim(-3.8, 3.8)
    ax.set_ylim(-0.2, 4.4)
    ax.set_axisbelow(True)
    ax.grid(ls=":", lw=0.4, alpha=0.4, zorder=0)

    legend = ax.legend(loc="upper left", fontsize=9, frameon=True,
                       framealpha=0.9, edgecolor="0.7")

    save_fig(fig, "entropy_phase")


if __name__ == "__main__":
    main()
