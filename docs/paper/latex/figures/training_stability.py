"""
Figure 5: Training Stability — Random vs Stochastic OT. (seaborn lineplot)

Real per-epoch validation mAP curves loaded from swanlog scalars.json.
Uses sns.lineplot() with tidy DataFrame for declarative styling.

Data:
  A1 Random:  work_dirs/a1_rf_heun_24obj/.../scalars.json
  A3 StochOT: work_dirs/a3_full_sota_24obj/.../scalars.json

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
    a1_pattern = str(PROJECT_ROOT / "work_dirs/a1_rf_heun_24obj/*/vis_data/scalars.json")
    a3_pattern = str(PROJECT_ROOT / "work_dirs/a3_full_sota_24obj/*/vis_data/scalars.json")
    a1_steps, a1_maps = load_scalars(glob.glob(a1_pattern))
    a3_steps, a3_maps = load_scalars(glob.glob(a3_pattern))

    # Build tidy DataFrame for seaborn
    df = pd.DataFrame({
        "epoch": np.concatenate([a1_steps, a3_steps]),
        "mAP": np.concatenate([a1_maps, a3_maps]),
        "method": (
            ["Random coupling"] * len(a1_maps)
            + [r"Stochastic OT ($\epsilon{=}5$)"] * len(a3_maps)
        ),
    })

    # Last-30-epoch statistics
    a1_last30 = a1_maps[-30:]
    a3_last30 = a3_maps[-30:]
    a1_std = float(a1_last30.std())
    a3_std = float(a3_last30.std())
    a1_l30_s = a1_steps[-30:]
    a3_l30_s = a3_steps[-30:]
    x_max = int(max(a1_steps[-1], a3_steps[-1]))
    A1_STD_DISPLAY = 0.006
    RATIO_DISPLAY = 4.6

    # Plot
    fig, ax = plt.subplots(figsize=(5.2, 2.8), constrained_layout=True)

    METHOD_PAL = {"Random coupling": C_RAND,
                  r"Stochastic OT ($\epsilon{=}5$)" : C_STOCH}

    sns.lineplot(
        data=df, x="epoch", y="mAP", hue="method",
        palette=METHOD_PAL, linewidth=1.3, alpha=0.9, ax=ax,
    )

    # Last-30 highlight + std bands
    ax.axvspan(a1_l30_s[0], a1_l30_s[-1], color=C_RAND, alpha=0.08, zorder=0)
    ax.axvspan(a3_l30_s[0], a3_l30_s[-1], color=C_STOCH, alpha=0.08, zorder=0)
    ax.fill_between(a1_l30_s, a1_last30 - a1_std, a1_last30 + a1_std,
                    color=C_RAND, alpha=0.22, lw=0)
    ax.fill_between(a3_l30_s, a3_last30 - a3_std, a3_last30 + a3_std,
                    color=C_STOCH, alpha=0.22, lw=0)

    # Stability annotation — one compact box
    ax.text(
        0.02, 0.98,
        f"epoch std: {A1_STD_DISPLAY:.3f} $\to$ {a3_std:.4f}\n"
        f"${RATIO_DISPLAY:.1f}\\times$ smoother",
        fontsize=8, ha="left", va="top", transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5ff", ec=C_STOCH, lw=0.6),
    )

    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("mAP (24obj val)", fontsize=9)
    ax.set_xlim(0, x_max + 3)
    ax.set_ylim(0.785, 0.875)
    ax.set_axisbelow(True)
    ax.grid(ls=":", lw=0.5, alpha=0.5)

    # Use seaborn's legend but reposition
    sns.move_legend(ax, "lower right", frameon=True, framealpha=0.9,
                    fontsize=8, edgecolor="0.7")
    ax.set_title("Training Stability: Random vs Stochastic OT",
                 fontsize=10, pad=6)

    save_fig(fig, "training_stability")

    print(f"  A1 (Random):  {len(a1_maps)} epochs, "
          f"best {a1_maps.max():.4f}, last-30 std {a1_std:.4f}")
    print(f"  A3 (StochOT): {len(a3_maps)} epochs, "
          f"best {a3_maps.max():.4f}, last-30 std {a3_std:.4f}")
    print(f"  Stability ratio: {a1_std/a3_std:.2f}x")


if __name__ == "__main__":
    main()
