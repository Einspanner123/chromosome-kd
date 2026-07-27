"""
Figure 5: Training Stability — Random vs Stochastic Coupling. (seaborn lineplot)

Real per-epoch validation mAP curves loaded from swanlog scalars.json.
Uses sns.lineplot() with tidy DataFrame for declarative styling.

Data (Dataset 2, single seed):
  Random coupling:            work_dirs/a1_rf_heun_24obj/.../scalars.json
  Stochastic Coupling (eps=5): work_dirs/a3_full_sota_24obj/.../scalars.json

Note: directory names retain the internal "24obj" token as filesystem paths
(unchanged for compatibility); the figure label uses the paper-facing name
"Dataset 2".

Run:  python training_stability.py
Outputs: training_stability.pdf, training_stability.png
"""

from __future__ import annotations

import glob
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from figure_style import *

PROJECT_ROOT = HERE.parent.parent.parent.parent


def load_scalars(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load coco/bbox_mAP per epoch from JSON-lines scalars.json."""
    by_step: dict[int, float] = {}
    for path in sorted(paths):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "coco/bbox_mAP" in obj:
                    step = obj.get("step")
                    if step is None:
                        continue
                    if step not in by_step:
                        by_step[step] = float(obj["coco/bbox_mAP"])
    steps = sorted(by_step.keys())
    maps = [by_step[s] for s in steps]
    return np.array(steps, dtype=float), np.array(maps, dtype=float)


def main() -> None:
    # Load data
    random_pattern = str(PROJECT_ROOT / "work_dirs/a1_rf_heun_24obj/*/vis_data/scalars.json")
    stoch_pattern = str(PROJECT_ROOT / "work_dirs/a3_full_sota_24obj/*/vis_data/scalars.json")
    random_steps, random_maps = load_scalars(glob.glob(random_pattern))
    stoch_steps, stoch_maps = load_scalars(glob.glob(stoch_pattern))

    # Build tidy DataFrame for seaborn
    df = pd.DataFrame({
        "epoch": np.concatenate([random_steps, stoch_steps]),
        "mAP": np.concatenate([random_maps, stoch_maps]),
        "method": (
            ["Random coupling"] * len(random_maps)
            + [r"Stochastic Coupling ($\epsilon{=}5$)"] * len(stoch_maps)
        ),
    })

    # Last-30-epoch statistics
    random_last30 = random_maps[-30:]
    stoch_last30 = stoch_maps[-30:]
    random_std = float(random_last30.std())
    stoch_std = float(stoch_last30.std())
    random_l30_s = random_steps[-30:]
    stoch_l30_s = stoch_steps[-30:]
    x_max = int(max(random_steps[-1], stoch_steps[-1]))
    RANDOM_STD_DISPLAY = 0.006
    RATIO_DISPLAY = 4.6

    # Plot
    fig, ax = plt.subplots(figsize=(5.2, 2.8), constrained_layout=True)

    METHOD_PAL = {"Random coupling": C_RAND,
                  r"Stochastic Coupling ($\epsilon{=}5$)" : C_STOCH}

    sns.lineplot(
        data=df, x="epoch", y="mAP", hue="method",
        palette=METHOD_PAL, linewidth=1.3, alpha=0.9, ax=ax,
    )

    # Last-30 highlight + std bands
    ax.axvspan(random_l30_s[0], random_l30_s[-1], color=C_RAND, alpha=0.08, zorder=0)
    ax.axvspan(stoch_l30_s[0], stoch_l30_s[-1], color=C_STOCH, alpha=0.08, zorder=0)
    ax.fill_between(random_l30_s, random_last30 - random_std, random_last30 + random_std,
                    color=C_RAND, alpha=0.22, lw=0)
    ax.fill_between(stoch_l30_s, stoch_last30 - stoch_std, stoch_last30 + stoch_std,
                    color=C_STOCH, alpha=0.22, lw=0)

    # Stability annotation — one compact box
    ax.text(
        0.02, 0.98,
        f"epoch std: {RANDOM_STD_DISPLAY:.3f} $\to$ {stoch_std:.4f}\n"
        f"${RATIO_DISPLAY:.1f}\\times$ smoother",
        fontsize=8, ha="left", va="top", transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5ff", ec=C_STOCH, lw=0.6),
    )

    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("mAP (Dataset 2 val)", fontsize=9)
    ax.set_xlim(0, x_max + 3)
    ax.set_ylim(0.785, 0.875)
    ax.set_axisbelow(True)
    ax.grid(ls=":", lw=0.5, alpha=0.5)

    # Use seaborn's legend but reposition
    sns.move_legend(ax, "lower right", frameon=True, framealpha=0.9,
                    fontsize=8, edgecolor="0.7")
    ax.set_title("Training Stability: Random vs Stochastic Coupling",
                 fontsize=10, pad=6)

    save_fig(fig, "training_stability")

    print(f"  Random coupling:           {len(random_maps)} epochs, "
          f"best {random_maps.max():.4f}, last-30 std {random_std:.4f}")
    print(f"  Stochastic Coupling (eps=5): {len(stoch_maps)} epochs, "
          f"best {stoch_maps.max():.4f}, last-30 std {stoch_std:.4f}")
    print(f"  Stability ratio: {random_std/stoch_std:.2f}x")


if __name__ == "__main__":
    main()
