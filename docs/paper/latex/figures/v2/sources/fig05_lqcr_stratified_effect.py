"""Figure 5: clean paired LQCR effects by scale and local overlap stratum."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style_v2 import C_LIGHT, C_LQCR, C_LINE, configure_style, panel_title, save_vector_figure


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "source_lqcr_stratified_clean.json"
SCALE = ("small", "medium", "large")
OVERLAP = ("isolated", "near", "overlap")
REQUIRED_SEEDS = {42, 123, 789}


def load_evidence() -> list[dict[str, object]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Evidence gate: missing {DATA_PATH}. Export paired stratum metrics first."
        )
    payload = json.loads(DATA_PATH.read_text())
    provenance = payload.get("provenance", {})
    required_flags = ("same_checkpoint", "same_boxes", "same_classes", "same_coordinates", "final_only")
    if not all(provenance.get(flag) is True for flag in required_flags):
        raise ValueError(f"Invalid stratified provenance; all flags required: {required_flags}")
    rows = payload.get("results", [])
    seeds = {int(row["seed"]) for row in rows}
    if seeds != REQUIRED_SEEDS:
        raise ValueError(f"Expected exactly seeds {sorted(REQUIRED_SEEDS)}, received {sorted(seeds)}")
    for row in rows:
        metrics = row.get("strata", {})
        if set(metrics) != set(SCALE + OVERLAP):
            raise ValueError(f"Seed {row.get('seed')} must contain exactly {SCALE + OVERLAP}")
        for name, values in metrics.items():
            if not {"baseline_ap", "lqcr_ap"}.issubset(values):
                raise ValueError(f"Stratum {name!r} lacks baseline_ap/lqcr_ap")
    return sorted(rows, key=lambda row: int(row["seed"]))


def effect_panel(ax: plt.Axes, rows: list[dict[str, object]], names: tuple[str, ...],
                 title: str, label: str) -> None:
    panel_title(ax, label, title)
    x = np.arange(len(names))
    for row in rows:
        deltas = [
            100 * (row["strata"][name]["lqcr_ap"] - row["strata"][name]["baseline_ap"])
            for name in names
        ]
        ax.plot(x, deltas, color=C_LQCR, alpha=0.35, linewidth=0.9)
        ax.scatter(x, deltas, color=C_LQCR, alpha=0.65, s=18)
    means = np.array([
        np.mean([
            100 * (row["strata"][name]["lqcr_ap"] - row["strata"][name]["baseline_ap"])
            for row in rows
        ])
        for name in names
    ])
    ax.scatter(x, means, color=C_LQCR, edgecolor="white", linewidth=0.7,
               s=58, zorder=4, label="seed mean")
    ax.axhline(0, color=C_LINE, linewidth=0.8)
    ax.set_xticks(x, names)
    ax.set_ylabel(r"paired $\Delta$AP (points)")
    ax.grid(axis="y", color=C_LIGHT, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    rows = load_evidence()
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    effect_panel(axes[0], rows, SCALE, "Effect by object scale", "a")
    effect_panel(axes[1], rows, OVERLAP, "Effect by local geometry", "b")
    axes[1].legend(frameon=False, fontsize=7.0, loc="best")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.84, wspace=0.18)
    save_vector_figure(fig, "fig05_lqcr_stratified_effect")


if __name__ == "__main__":
    main()
