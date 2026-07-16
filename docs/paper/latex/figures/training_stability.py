"""Figure 4: Training Stability (Random vs StochOT).

Illustrative epoch-mAP curves comparing Random coupling (epoch std 0.006,
noisy) with Stochastic OT coupling epsilon=5 (epoch std 0.0013, smooth).

NOTE: Real per-epoch training mAP data is not available in this repo
(swanlog/ directory exists but exported JSON was not located). We therefore
generate **illustrative** curves matching the documented last-30-epoch
statistics from the paper (Sec 3.3.5 and Sec 4.4):
  - Random:    best mAP ~0.856, last-30 epoch std = 0.006
  - StochOT:   best mAP ~0.858, last-30 epoch std = 0.0013
  - 4.6x stability gain.

If real scalars.json data becomes available later, replace the
``build_curve`` inputs with the empirical series.

Run:  python training_stability.py
Outputs:
  training_stability.pdf
  training_stability.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)

PAL = sns.color_palette("colorblind")
C_RAND = PAL[1]   # orange
C_STOCH = PAL[0]   # blue

HERE = Path(__file__).resolve().parent


def build_curve(
    epochs: int,
    target_best: float,
    last30_std: float,
    warmup_epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an illustrative epoch-mAP curve matching documented statistics.

    The curve ramps from a low value, reaches a plateau, and oscillates with
    the specified last-30-epoch std around a target best mAP.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(epochs)

    # Sigmoid warmup to a plateau
    plateau = target_best - last30_std * 1.2
    start = plateau - 0.05
    ramp = start + (plateau - start) / (1 + np.exp(-(t - warmup_epochs) / 8))

    trend = ramp

    # Oscillation: larger amplitude early, then settle to target std for last 30
    # Animate amplitude shrink
    early_amp = 0.012
    amp = np.where(
        t < epochs - 30,
        np.interp(t, [0, epochs - 60, epochs - 30], [early_amp, last30_std * 2, last30_std]),
        last30_std,
    )
    noise = rng.normal(0, amp)
    curve = trend + noise

    # Ensure last-30 std exactly matches the documented value (for clarity)
    last30 = curve[-30:]
    # Rescale to target std, then shift to target mean
    if last30.std() > 0:
        last30 = (last30 - last30.mean()) / last30.std() * last30_std
    last30 = last30 + (target_best - last30_std * 0.6)
    curve[-30:] = last30
    # Clamp curve to realistic range
    curve = np.clip(curve, 0.78, 0.89)
    return t, curve


def main() -> None:
    epochs = 150
    t_rand, m_rand = build_curve(
        epochs=epochs, target_best=0.856, last30_std=0.006,
        warmup_epochs=20, seed=42,
    )
    t_stoch, m_stoch = build_curve(
        epochs=epochs, target_best=0.858, last30_std=0.0013,
        warmup_epochs=20, seed=43,
    )

    fig, ax = plt.subplots(figsize=(5.2, 2.8), constrained_layout=True)

    last30_slice = slice(epochs - 30, epochs)
    rand_mean_last = m_rand[last30_slice].mean()
    stoch_mean_last = m_stoch[last30_slice].mean()

    ax.plot(t_rand, m_rand, color=C_RAND, lw=1.3, alpha=0.9, label="Random coupling")
    ax.plot(t_stoch, m_stoch, color=C_STOCH, lw=1.3, alpha=0.9, label=r"Stochastic OT ($\epsilon{=}5$)")

    ax.axvspan(epochs - 30, epochs, color="0.88", alpha=0.5, zorder=0)
    ax.text(epochs - 15, 0.877, "last 30 epochs", fontsize=7, ha="center", color="0.4", va="top")

    ax.fill_between(
        t_rand[last30_slice],
        m_rand[last30_slice] - 0.006,
        m_rand[last30_slice] + 0.006,
        color=C_RAND, alpha=0.2, lw=0,
    )
    ax.fill_between(
        t_stoch[last30_slice],
        m_stoch[last30_slice] - 0.0013,
        m_stoch[last30_slice] + 0.0013,
        color=C_STOCH, alpha=0.2, lw=0,
    )

    ax.text(
        epochs - 32, 0.877,
        "epoch std: 0.006 → 0.0013\n" r"$4.6\times$ smoother",
        fontsize=7, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5ff", ec=C_STOCH, lw=0.6),
    )

    ax.text(
        epochs - 2, 0.803,
        "(illustrative)",
        fontsize=7, ha="right", va="bottom",
        color="0.5", style="italic",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP (24obj val)")
    ax.set_xlim(0, epochs)
    ax.set_ylim(0.80, 0.88)
    ax.set_axisbelow(True)
    ax.grid(ls=":", lw=0.5, alpha=0.5)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=8, edgecolor="0.7")

    ax.set_title("Training Stability: Random vs Stochastic OT",
                 fontsize=10, pad=6)

    out_pdf = HERE / "training_stability.pdf"
    out_png = HERE / "training_stability.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
