"""
Figure 4: Entropy Phase Diagram — H(V|Z) vs Sinkhorn regularization epsilon.

Rewritten with adjustText for automatic label placement.
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
from adjustText import adjust_text

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

    # Axes in log10(epsilon) space
    x_left = -3.0
    x_right = 3.0
    x_data = np.log10(eps_vals)

    # Monotone PCHIP interpolation for theory curve
    x_fit = np.concatenate(([x_left], x_data, [x_right]))
    y_fit = np.concatenate(([h_hard_ot], h_means, [h_random]))
    pchip = PchipInterpolator(x_fit, y_fit)
    x_dense = np.linspace(x_left, x_right, 500)
    y_dense = pchip(x_dense)

    # Tidy DataFrame
    df_meas = pd.DataFrame({
        "log_eps": x_data,
        "H": h_means,
    })

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)

    # Region backgrounds
    ax.axvspan(x_left, 0.0, color=C_DANGER, alpha=0.7, zorder=0)
    ax.axvspan(0.0, x_right, color=C_SAFE, alpha=0.7, zorder=0)
    ax.axvline(0.0, color="0.5", ls=":", lw=0.6, alpha=0.4, zorder=1)

    # Horizontal reference lines
    ax.axhline(LOG_K, color=C_RAND, ls="--", lw=0.9, alpha=0.7, zorder=2)
    ax.axhline(h_hard_ot, color=C_OT, ls="--", lw=0.9, alpha=0.7, zorder=2)

    # Theory curve
    df_theory = pd.DataFrame({"x": x_dense, "y": y_dense})
    sns.lineplot(data=df_theory, x="x", y="y", color=C_RF,
                 linewidth=1.8, alpha=0.85, zorder=3, ax=ax,
                 label=r"Theory: $H(V|Z)$ monotone $\uparrow$")

    # Measured data: scatter only (no error bars)
    sns.scatterplot(data=df_meas, x="log_eps", y="H", color=C_RF,
                    s=55, edgecolor="white", linewidth=1.2,
                    zorder=5, ax=ax)

    # Endpoint markers
    ax.plot(x_left, h_hard_ot, "s", color=C_OT, ms=8, mfc=C_OT,
            mec="black", mew=0.7, zorder=6)
    ax.plot(x_right, h_random, "D", color=C_RAND, ms=7, mfc=C_RAND,
            mec="black", mew=0.7, zorder=6)

    # ── Labels with adjustText ──
    # Increase fontsize and add bbox so labels are readable
    fs = 9
    texts_to_adjust = []

    t1 = ax.text(x_left, h_hard_ot, "Hard OT ($\\epsilon{=}0$)",
                 fontsize=fs, ha="left", va="bottom", color=C_OT,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white",
                           ec="none", alpha=0.85))
    texts_to_adjust.append(t1)

    t2 = ax.text(x_right, h_random, "Random ($\\epsilon{=}\\infty$)",
                 fontsize=fs, ha="right", va="bottom", color=C_RAND,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white",
                           ec="none", alpha=0.85))
    texts_to_adjust.append(t2)

    # Reference annotations on the right
    t3 = ax.text(x_right + 0.5, LOG_K, rf"$\log K$={LOG_K:.2f}",
                 fontsize=fs, ha="left", va="center", color=C_RAND,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white",
                           ec="none", alpha=0.85))
    texts_to_adjust.append(t3)

    t4 = ax.text(x_right + 0.5, h_random,
                 f"$H_{{\\mathrm{{Random}}}}$={h_random:.2f}",
                 fontsize=fs, ha="left", va="center", color=C_RAND,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white",
                           ec="none", alpha=0.85))
    texts_to_adjust.append(t4)

    # adjustText — push labels away from data AND keep them off the curve
    adjust_text(
        texts_to_adjust,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.4, shrinkA=5),
        force_text=(1.5, 1.5),     # strong text-text repulsion
        force_points=(2.0, 2.0),   # strong text↔datapoint repulsion (curve)
        expand=(1.5, 1.5),         # generous bounding box margin
        lim=400,
        precision=0.005,
        va="center",
    )

    # ── Axes ──
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

    save_fig(fig, "entropy_phase")


if __name__ == "__main__":
    main()
