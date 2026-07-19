"""
Figure: RF Training Pipeline + Shifted Schedule + DPM-Solver++ Acceleration.

Three-panel technical illustration with cleaner layout:
  (a) RF training pipeline: simplified flowchart
  (b) Shifted noise schedule: sampling density comparison
  (c) DPM-Solver++ vs Heun: comparison table style

Run:  python tech_pipeline.py
Outputs: tech_pipeline.pdf, tech_pipeline.png
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from figure_style import *


def panel_pipeline(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a) RF Training Pipeline", **PANEL_LABEL_KW)
    hide_spines(ax)

    def box(x, y, w, h, text, fc="white", ec=C_DARKGRAY, tc="black", fs=8, wt="normal"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                              fc=fc, ec=ec, lw=0.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight=wt)

    def arrow(x1, y1, x2, y2, color=C_DARKGRAY, lw=0.9):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", mutation_scale=12,
                                    color=color, lw=lw), zorder=3)

    box(0.3, 3.2, 2.2, 0.9, "Input\nImage", fc=C_LIGHTGRAY, fs=9)
    arrow(2.5, 3.65, 3.8, 3.65)
    box(3.8, 3.2, 2.5, 0.9, "Backbone\nResNet-50+FPN", fc=C_RF, tc="white", fs=9, wt="bold")
    arrow(6.3, 3.65, 7.8, 3.65)
    box(7.8, 3.2, 2.0, 0.9, "Random t\n(shifted)", fc=C_ADALN, tc="white", fs=8, wt="bold")

    box(0.3, 1.8, 2.2, 0.8, "GT Boxes\n(xyxy->cxcywh)", fc=C_LIGHTGRAY, fs=8)
    arrow(2.5, 2.2, 3.8, 2.2)
    box(3.8, 1.7, 2.5, 1.0, "RF Interp.\n" + r"$x_t=(1-t)x_0+t\epsilon$", fc=C_RF, tc="white", fs=8.5, wt="bold")
    arrow(6.3, 2.2, 7.8, 2.2)
    box(7.8, 1.8, 2.0, 0.8, "Cascade Heads\n(6 layers)", fc=C_DARKGRAY, tc="white", fs=8, wt="bold")

    box(0.3, 0.4, 2.2, 0.8, r"Noise $\epsilon$" + "\n" + r"$\sim\mathcal{N}(0,\sigma^2)$",
        fc=C_SOURCE, tc="white", fs=8, wt="bold")
    arrow(2.5, 0.8, 3.8, 1.7)

    box(7.8, 0.4, 2.0, 0.8, r"$v_\theta$", fc=C_RF, tc="white", fs=9, wt="bold")

    box(3.0, 0.1, 5.0, 0.5,
        r"$\mathcal{L}_{FM} = \mathbb{E}\|v_\theta - (\epsilon - x_0)\|^2$",
        fc="#f8f8f8", ec=C_RF, fs=8.5)

    arrow(5.0, 2.7, 5.0, 1.7, color=C_ADALN, lw=0.7)
    arrow(5.0, 1.2, 5.0, 0.6, color=C_ADALN, lw=0.7)


def panel_schedule(ax: plt.Axes) -> None:
    t = np.linspace(0, 1, 1000)
    linear = np.ones_like(t)
    shift = 3.0
    shifted = shift / (1 + (shift - 1) * t) ** 2

    df_sched = pd.DataFrame({
        "t": np.concatenate([t, t]),
        "density": np.concatenate([linear, shifted]),
        "schedule": ["Linear (uniform)"] * len(t) + [f"Shifted ($s$={shift})"] * len(t),
    })

    sns.lineplot(data=df_sched, x="t", y="density", hue="schedule",
                 palette={"Linear (uniform)": C_RAND, f"Shifted ($s$={shift})": C_RF},
                 linewidth=1.8, ax=ax)
    ax.fill_between(t, 0, shifted, color=C_RF, alpha=0.15)
    ax.fill_between(t, 0, linear, color=C_RAND, alpha=0.1)

    ax.annotate("More samples\nnear $t=0$\n(fine details)",
                xy=(0.1, 2.0), xytext=(0.25, 2.8),
                fontsize=8.5, ha="left", color=C_RF,
                arrowprops=dict(arrowstyle="->", color=C_RF, lw=0.8), zorder=5)
    ax.annotate("Fewer samples\nnear $t=1$",
                xy=(0.85, 0.5), xytext=(0.8, 0.8),
                fontsize=8.5, ha="center", color=C_RF,
                arrowprops=dict(arrowstyle="->", color=C_RF, lw=0.8), zorder=5)

    ax.set_xlabel("Time step $t$", fontsize=10)
    ax.set_ylabel("Sampling Density $p(t)$", fontsize=10)
    ax.set_title("(b) Shifted Timestep Sampling", **PANEL_LABEL_KW)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.5)
    ax.set_axisbelow(True)
    ax.grid(ls=":", lw=0.5, alpha=0.4)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.92)
    ax.tick_params(labelsize=9)


def panel_dpm_solver(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(c) DPM-Solver++ vs Heun: NFE Comparison", **PANEL_LABEL_KW)
    hide_spines(ax)

    def box(x, y, w, h, text, fc="white", ec=C_DARKGRAY, tc="black", fs=8, wt="normal"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              fc=fc, ec=ec, lw=0.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight=wt)

    steps = [1.0, 0.75, 0.5, 0.25, 0.0]
    step_y = 3.2
    step_x = [1.0, 2.8, 4.6, 6.4, 8.2]

    for i, (t_val, xp) in enumerate(zip(steps, step_x)):
        label = f"$t_{i}$={t_val:.2f}" if i < len(steps) - 1 else "$t_4$=0"
        fc = C_LIGHTGRAY if t_val > 0 else C_RF
        tc = C_DARKGRAY if t_val > 0 else "white"
        box(xp - 0.45, step_y - 0.35, 0.9, 0.7, label, fc=fc, tc=tc, fs=9,
            wt="bold" if t_val == 0 else "normal")

        if i < len(steps) - 1:
            ax.annotate("", xy=(step_x[i + 1] - 0.45, step_y),
                        xytext=(xp + 0.45, step_y),
                        arrowprops=dict(arrowstyle="->", mutation_scale=10,
                                        color=C_DARKGRAY, lw=0.8))

    ax.text(4.5, 2.55, "1 NFE / step (no corrector)", fontsize=8.5,
            ha="center", va="center", color=C_RF, weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#eef6ff", ec=C_RF,
                      lw=0.7, alpha=0.9))

    heun_y = 1.2
    ax.text(0.2, heun_y + 0.4, "Heun (2 NFE/step):", fontsize=9,
            ha="left", va="center", weight="bold", color=C_DARKGRAY)

    for i, (t_val, xp) in enumerate(zip(steps, step_x)):
        if t_val == 0:
            box(xp - 0.45, heun_y - 0.35, 0.9, 0.7, "$x_0$", fc=C_RF, tc="white", fs=9, wt="bold")
        else:
            box(xp - 0.45, heun_y - 0.35, 0.9, 0.7, "P+C\n(2 NFE)",
                fc=C_DDPM, tc="white", fs=8, wt="bold")

    ax.annotate("", xy=(8.2, 3.1), xytext=(8.2, 1.5),
                arrowprops=dict(arrowstyle="<->", color="#E63946", lw=1.2),
                zorder=5)

    ax.text(9.0, 2.3, r"\textbf{1.71$\times$ NFE Saving}", fontsize=9,
            ha="left", va="center", color="#E63946", weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E63946", lw=1.0))

    formula = (r"$\mathbf{DPM^{++}\ Step:}\ x_{t_{n+1}} = \frac{t_{n+1}}{t_n} x_{t_n}$"
               r"$\ +\ (1-\frac{t_{n+1}}{t_n})x_0^{(n)}\ +\ \varphi_1 D_1$")
    ax.text(5.0, 0.35, formula, fontsize=9,
            ha="center", va="center", color=C_DARKGRAY,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec=C_RF, lw=0.8))


def main() -> None:
    fig = plt.figure(figsize=(8.5, 9.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0],
                          hspace=0.35, left=0.08, right=0.96,
                          bottom=0.04, top=0.95)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])

    panel_pipeline(ax_a)
    panel_schedule(ax_b)
    panel_dpm_solver(ax_c)

    save_fig(fig, "tech_pipeline")


if __name__ == "__main__":
    main()
