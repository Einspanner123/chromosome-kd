"""
Figure: RF Training Pipeline + Shifted Schedule + DPM-Solver++ Acceleration.

Three-panel technical illustration:
  (a) RF training pipeline: image → features → RF interpolation → velocity prediction
  (b) Shifted noise schedule: sampling density with shift=3.0 vs linear
  (c) DPM-Solver++ 4-step mechanism: 1 NFE/step vs Heun 2 NFE/step

Run:  python tech_pipeline.py
Outputs: tech_pipeline.pdf, tech_pipeline.png
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

from figure_style import *


# =====================================================================
# Panel (a): RF Training Pipeline
# =====================================================================
def panel_pipeline(ax: plt.Axes) -> None:
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(a) RF Training Pipeline", **PANEL_LABEL_KW)
    hide_spines(ax)

    def bx(x, y, w, h, text, fc="#ffffff", ec=C_DARKGRAY,
           tc="black", fs=7.5, wt="normal"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              fc=fc, ec=ec, lw=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight=wt)

    def ar(x1, y1, x2, y2, color=C_DARKGRAY, lw=0.9, ms=10):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", mutation_scale=ms,
                                    color=color, lw=lw), zorder=3)

    # Top row: Image → Backbone+FPN
    bx(0.3, 4.2, 2.0, 1.0, "Input\nimage", fc=C_LIGHTGRAY, fs=8)
    ar(2.3, 4.7, 3.5, 4.7)
    bx(3.5, 4.2, 2.2, 1.0, "Backbone\nResNet-50+FPN", fc=C_RF,
       tc="white", fs=8, wt="bold")

    # Bottom row: GT boxes → diffusion space
    bx(0.3, 2.5, 2.0, 0.8, "GT boxes\n(xyxy→cxcywh,\n[-s,+s])", fc=C_LIGHTGRAY, fs=7)
    ar(2.3, 2.9, 3.5, 2.9)

    # Noise
    bx(0.3, 0.5, 2.0, 0.8, r"Noise $\epsilon$" + "\n" + r"$\sim\mathcal{N}(0,\sigma^2)$",
       fc=C_SOURCE, tc="white", fs=7.5, wt="bold")
    ar(2.3, 0.9, 5.5, 0.9)

    # Feature → concat flow → Transformer heads
    bx(3.5, 2.5, 2.2, 0.8, "RF interp.\n" + r"$x_t=(1-t)x_0+t\epsilon$",
       fc=C_RF, tc="white", fs=7.5, wt="bold")
    ar(5.7, 2.5, 7.0, 2.5)

    # Time t with shifted schedule
    bx(5.5, 4.2, 2.5, 0.8, "Random $t$" + "\n(shifted schedule)", fc=C_ADALN,
       tc="white", fs=7.5, wt="bold")
    ar(5.5, 4.2, 4.5, 3.2, lw=0.7, ms=8)  # t down to concat
    ar(5.5, 4.2, 5.5, 3.3, lw=0.7, ms=8)  # t down to head

    bx(7.0, 2.0, 2.5, 1.0, "Cascade\nTransformer Heads", fc=C_DARKGRAY,
       tc="white", fs=8, wt="bold")
    ar(9.5, 2.5, 10.8, 2.5)

    # Velocity prediction → loss
    bx(10.8, 2.0, 1.0, 1.0, r"$v_\theta$", fc=C_RF, tc="white", fs=8, wt="bold")

    # Loss
    bx(7.5, 0.3, 4.0, 0.7,
       r"$\mathcal{L}_{FM} = \mathbb{E}\|v_\theta - (\epsilon - x_0)\|^2$",
       fc="#f5f5f5", ec=C_RF, fs=7.5)
    ar(11.3, 2.0, 9.5, 1.0, lw=0.7, ms=8)  # v_theta down to loss

    # Arrow from bottom noise → RF interp
    ar(4.5, 1.3, 4.5, 2.5, lw=0.7, ms=8)

    # Annotation
    ax.text(6.0, 5.0, "feature concat", fontsize=7, ha="center",
            color=C_MIDGRAY, style="italic")


# =====================================================================
# Panel (b): Shifted Noise Schedule — sampling density
# =====================================================================
def panel_schedule(ax: plt.Axes) -> None:
    t = np.linspace(0, 1, 1000)

    # Linear (uniform) schedule: t stays uniform
    linear = np.ones_like(t)

    # Shifted schedule transformation: t_shifted = shift * t / (1 + (shift-1) * t)
    # The sampling density is p(t) = dt_shifted/dt = shift / (1 + (shift-1)*t)^2
    shift = 3.0
    shifted = shift / (1 + (shift - 1) * t) ** 2

    # Build tidy DataFrame
    df_sched = pd.DataFrame({
        "t": np.concatenate([t, t]),
        "density": np.concatenate([linear, shifted]),
        "schedule": (["Linear (uniform)"] * len(t)
                     + [f"Shifted ($s$={shift})"] * len(t)),
    })

    sns.lineplot(data=df_sched, x="t", y="density", hue="schedule",
                 palette={"Linear (uniform)": C_RAND,
                          f"Shifted ($s$={shift})": C_RF},
                 linewidth=1.5, ax=ax)
    # Fill between for shifted (seaborn has no fill_between equivalent)
    ax.fill_between(t, 0, shifted, color=C_RF, alpha=0.12)
    ax.fill_between(t, 0, linear, color=C_RAND, alpha=0.08)

    # Annotate the shift effect
    ax.annotate("More samples\nnear $t{=}0$\n(fine details)",
                xy=(0.1, shifted.max() * 0.8), xytext=(0.3, 2.5),
                fontsize=7, ha="center", color=C_RF,
                arrowprops=dict(arrowstyle="->", color=C_RF, lw=0.7), zorder=5)
    ax.annotate("Fewer samples\nnear $t{=}1$",
                xy=(0.8, shifted[t > 0.7][-1]),
                xytext=(0.7, 0.5), fontsize=7, ha="center", color=C_RF,
                arrowprops=dict(arrowstyle="->", color=C_RF, lw=0.7), zorder=5)

    ax.set_xlabel("Time step $t$", fontsize=9)
    ax.set_ylabel("Sampling density $p(t)$", fontsize=9)
    ax.set_title("(b) Timestep Sampling: Shifted Schedule",
                 **PANEL_LABEL_KW)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.5)
    ax.set_axisbelow(True)
    ax.grid(ls=":", lw=0.4, alpha=0.4)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92)
    ax.tick_params(labelsize=8)


# =====================================================================
# Panel (c): DPM-Solver++ 4-Step Mechanism
# =====================================================================
def panel_dpm_solver(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(c) DPM-Solver++: 4-Step Inference",
                 **PANEL_LABEL_KW)
    hide_spines(ax)

    def bx(x, y, w, h, text, fc="#ffffff", ec=C_DARKGRAY,
           tc="black", fs=7.5, wt="normal"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              fc=fc, ec=ec, lw=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight=wt)

    def ar(x1, y1, x2, y2, color=C_DARKGRAY, lw=0.9, ms=10):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", mutation_scale=ms,
                                    color=color, lw=lw), zorder=3)

    # Top: 4 step nodes (t=1 → t=0.75 → t=0.5 → t=0.25 → t=0)
    steps = [1.0, 0.75, 0.5, 0.25, 0.0]
    step_y = 4.0
    step_x_positions = [1.0, 3.0, 5.0, 7.0, 9.0]

    for i, (t_val, xp) in enumerate(zip(steps, step_x_positions)):
        label = f"$t_{{{i}}}$={t_val:.2f}" if i < len(steps) - 1 else f"$t_{{{i}}}$=0"
        fc = C_LIGHTGRAY if t_val > 0 else C_RF
        tc = C_DARKGRAY if t_val > 0 else "white"
        bx(xp - 0.5, step_y - 0.4, 1.0, 0.8, label, fc=fc, tc=tc, fs=8,
           wt="bold" if t_val == 0 else "normal")

        if i < len(steps) - 1:
            ar(xp + 0.5, step_y, step_x_positions[i+1] - 0.5, step_y)

    # Arrow label
    ax.text(5.0, 3.3, "1 NFE per step (no corrector)", fontsize=7,
            ha="center", va="center", color=C_RF, weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#f0f8ff", ec=C_RF,
                      lw=0.5, alpha=0.85))

    # Bottom: Heun comparison
    heun_y = 1.8
    ax.text(0.5, heun_y + 0.3, "Heun (2 NFE/step):", fontsize=7.5,
            ha="left", va="center", weight="bold", color=C_DARKGRAY)

    for i, (t_val, xp) in enumerate(zip(steps, step_x_positions)):
        if t_val == 0:
            bx(xp - 0.5, heun_y - 0.3, 1.0, 0.6, "$x_0$", fc=C_RF,
               tc="white", fs=7, wt="bold")
        else:
            # Heun: predictor-corrector = 2 NFE
            bx(xp - 0.5, heun_y - 0.3, 1.0, 0.6,
               f"pred+corr\n(2 NFE)", fc=C_DDPM, tc="white", fs=6.5, wt="bold")

    # Comparison annotation
    ax.annotate("", xy=(8.5, 3.9), xytext=(8.5, 2.5),
                arrowprops=dict(arrowstyle="<->", color="#E63946", lw=1.0),
                zorder=5)
    ax.text(9.2, 3.2, "1.71$\ imes$\nNFE\nsaving", fontsize=7,
            ha="left", va="center", color="#E63946", weight="bold")

    # x0 prediction formula
    bx(0.5, 0.1, 9.0, 0.8,
       r"$x_{t_{n+1}} = \frac{t_{n+1}}{t_n} x_{t_n} + (1-\frac{t_{n+1}}{t_n})x_0^{(n)} + \varphi_1 D_1$" +
       "\n" + r"$\varphi_1 = t_{n+1}\log\frac{t_n}{t_{n+1}} - t_n + t_{n+1}, \quad D_1 = \frac{x_0^{(n)}-x_0^{(n-1)}}{t_n-t_{n-1}}$",
       fc="#f5f5ff", ec=C_RF, fs=7)

    ax.text(9.5, 4.2, "NFE saving vs Heun", fontsize=8,
            ha="right", va="bottom", color="#E63946", weight="bold",
            style="italic")


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    fig = plt.figure(figsize=(7.5, 7.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0],
                           hspace=0.28, left=0.06, right=0.97,
                           bottom=0.04, top=0.97)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])

    panel_pipeline(ax_a)
    panel_schedule(ax_b)
    panel_dpm_solver(ax_c)

    save_fig(fig, "tech_pipeline")


if __name__ == "__main__":
    main()
