"""
Figure 4: Entropy Phase Diagram — H(V|Z) vs Sinkhorn regularization epsilon. (seaborn)

Monotone transition from Hard OT (epsilon=0, H=0) to Random coupling
(epsilon=inf, H=log K). Uses sns.lineplot() for theory curve and
sns.scatterplot() for measured data points with error bars.

Data: work_dirs/stability/warm_restart_v2/.../velocity_entropy_results.json

Run:  python entropy_phase.py
Outputs: entropy_phase.pdf, entropy_phase.png
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
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

    # Monotone PCHIP interpolation for theory curve
    x_fit = np.concatenate(([x_left], x_data, [x_right]))
    y_fit = np.concatenate(([h_hard_ot], h_means, [h_random]))
    pchip = PchipInterpolator(x_fit, y_fit)
    x_dense = np.linspace(x_left, x_right, 500)
    y_dense = pchip(x_dense)

    # Tidy DataFrame for measured data
    df_meas = pd.DataFrame({
        "log_eps": x_data,
        "H": h_means,
        "H_std": h_stds,
    })

    # Plot
    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)

    # Region backgrounds
    ax.axvspan(x_left, 0.0, color=C_DANGER, alpha=0.7, zorder=0)
    ax.axvspan(0.0, x_right, color=C_SAFE, alpha=0.7, zorder=0)
    ax.axvline(0.0, color="0.5", ls=":", lw=0.6, alpha=0.4, zorder=1)

    # Horizontal reference lines
    ax.axhline(LOG_K, color=C_RAND, ls="--", lw=0.9, alpha=0.7, zorder=2)
    ax.axhline(h_hard_ot, color=C_OT, ls="--", lw=0.9, alpha=0.7, zorder=2)

    # Theory curve via seaborn
    df_theory = pd.DataFrame({"x": x_dense, "y": y_dense})
    sns.lineplot(data=df_theory, x="x", y="y", color=C_RF,
                 linewidth=1.8, alpha=0.85, zorder=3, ax=ax,
                 label=r"Theory: $H(V|Z)$ monotone $\uparrow$")

    # Measured data via seaborn scatter + manual error bars
    sns.scatterplot(data=df_meas, x="log_eps", y="H", color=C_RF,
                    s=50, edgecolor="white", linewidth=1.2,
                    zorder=5, ax=ax, label=r"Measured $H(V|Z)$")
    # Error bars (seaborn doesn't support yerr natively)
    for _, row in df_meas.iterrows():
        ax.plot([row["log_eps"], row["log_eps"]],
                [row["H"] - row["H_std"], row["H"] + row["H_std"]],
                color=C_RF, lw=0.9, zorder=4)
        ax.plot([row["log_eps"] - 0.04, row["log_eps"] + 0.04],
                [row["H"] - row["H_std"], row["H"] - row["H_std"]],
                color=C_RF, lw=0.9, zorder=4)
        ax.plot([row["log_eps"] - 0.04, row["log_eps"] + 0.04],
                [row["H"] + row["H_std"], row["H"] + row["H_std"]],
                color=C_RF, lw=0.9, zorder=4)

    # Endpoint markers
    ax.plot(x_left, h_hard_ot, "s", color=C_OT, ms=8, mfc=C_OT,
            mec="black", mew=0.7, zorder=6)
    ax.plot(x_right, h_random, "D", color=C_RAND, ms=7, mfc=C_RAND,
            mec="black", mew=0.7, zorder=6)

    # Endpoint labels with lead lines
    lead_label(ax, "Hard OT ($\\epsilon{=}0$)",
               xy=(x_left, h_hard_ot), xytext=(x_left + 0.5, 0.5),
               fontsize=8, ha="left", va="center", color=C_OT)
    lead_label(ax, "Random ($\\epsilon{=}\\infty$)",
               xy=(x_right, h_random), xytext=(x_right - 0.5, 3.5),
               fontsize=8, ha="right", va="center", color=C_RAND)

    # Reference line annotations
    ax.text(x_right + 0.15, LOG_K, r"$\log K$" + f"={LOG_K:.2f}",
            fontsize=7.5, ha="left", va="center", color=C_RAND, zorder=4)
    ax.text(x_right + 0.15, h_random,
            f"$H_{{\\mathrm{{Random}}}}$={h_random:.2f}",
            fontsize=7.5, ha="left", va="center", color=C_RAND, zorder=4)

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

    sns.move_legend(ax, "lower right", frameon=True, framealpha=0.92,
                    fontsize=8, edgecolor="0.7")

    save_fig(fig, "entropy_phase")


if __name__ == "__main__":
    main()
