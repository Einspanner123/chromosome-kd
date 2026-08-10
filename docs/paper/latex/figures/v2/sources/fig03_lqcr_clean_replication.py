"""Figure 3: clean fixed-checkpoint LQCR replication across training seeds.

The builder is intentionally evidence-gated. It must not emit a placeholder
figure when the clean paired export is absent or violates causal provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style_v2 import (
    C_FOUNDATION,
    C_LIGHT,
    C_LQCR,
    C_LINE,
    C_MUTED,
    configure_style,
    panel_title,
    save_vector_figure,
)


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "source_lqcr_clean_multiseed.json"
REQUIRED_SEEDS = {42, 123, 789}


def load_evidence() -> list[dict[str, float]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Evidence gate: missing {DATA_PATH}. Export clean paired final-only results first."
        )
    payload = json.loads(DATA_PATH.read_text())
    provenance = payload.get("provenance", {})
    required_flags = ("same_checkpoint", "same_boxes", "same_classes", "final_only")
    if not all(provenance.get(flag) is True for flag in required_flags):
        raise ValueError(f"Invalid LQCR causal provenance; all flags required: {required_flags}")
    rows = payload.get("results", [])
    seeds = {int(row["seed"]) for row in rows}
    if seeds != REQUIRED_SEEDS:
        raise ValueError(f"Expected exactly seeds {sorted(REQUIRED_SEEDS)}, received {sorted(seeds)}")
    for row in rows:
        for key in ("baseline_map", "lqcr_map", "baseline_ap90", "lqcr_ap90", "baseline_ap95", "lqcr_ap95"):
            if key not in row:
                raise ValueError(f"Missing required field {key!r} for seed {row.get('seed')}")
    return sorted(rows, key=lambda row: int(row["seed"]))


def paired_panel(ax: plt.Axes, rows: list[dict[str, float]], base_key: str,
                 lqcr_key: str, title: str, label: str) -> None:
    panel_title(ax, label, title)
    y = np.arange(len(rows))
    baseline = np.array([row[base_key] for row in rows], dtype=float)
    lqcr = np.array([row[lqcr_key] for row in rows], dtype=float)
    for idx, (left, right) in enumerate(zip(baseline, lqcr)):
        ax.plot([left, right], [idx, idx], color=C_LINE, linewidth=1.2, zorder=1)
    ax.scatter(baseline, y, color=C_FOUNDATION, s=25, label="baseline", zorder=3)
    ax.scatter(lqcr, y, color=C_LQCR, s=28, label="+ LQCR", zorder=3)
    ax.set_yticks(y, [f"seed {int(row['seed'])}" for row in rows])
    ax.invert_yaxis()
    ax.grid(axis="x", color=C_LIGHT, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(title)


def main() -> None:
    rows = load_evidence()
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    paired_panel(axes[0], rows, "baseline_map", "lqcr_map", "mAP", "a")
    paired_panel(axes[1], rows, "baseline_ap90", "lqcr_ap90", "AP90", "b")

    panel_title(axes[2], "c", "Paired effect by seed")
    deltas = np.array([row["lqcr_map"] - row["baseline_map"] for row in rows]) * 100
    x = np.arange(len(rows))
    axes[2].axhline(0, color=C_LINE, linewidth=0.8)
    axes[2].scatter(x, deltas, color=C_LQCR, s=30, zorder=3)
    axes[2].plot(x, deltas, color=C_LQCR, linewidth=1.0, alpha=0.7)
    axes[2].axhline(deltas.mean(), color=C_FOUNDATION, linestyle="--", linewidth=1.0,
                    label=f"mean {deltas.mean():+.2f} pt")
    axes[2].set_xticks(x, [str(int(row["seed"])) for row in rows])
    axes[2].set_xlabel("training seed")
    axes[2].set_ylabel(r"$\Delta$mAP (points)")
    axes[2].grid(axis="y", color=C_LIGHT, linewidth=0.8)
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].legend(frameon=False, fontsize=6.8, loc="best")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.32, -0.01), fontsize=7.0)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.25, top=0.86, wspace=0.42)
    save_vector_figure(fig, "fig03_lqcr_clean_replication")


if __name__ == "__main__":
    main()
