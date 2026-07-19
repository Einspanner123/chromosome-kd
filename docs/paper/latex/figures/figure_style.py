"""
Unified figure style for KaryoFlow / LDMDet paper figures.

All figure scripts should import from this module instead of duplicating
rcParams, color definitions, and helpers.

Usage:
    from figure_style import *
    import matplotlib.pyplot as plt
    # ... plotting code ...
    save_fig(fig, "my_figure")

Design principles (from craft.md):
  - seaborn base: set_theme + set_context for consistent defaults
  - Layout contract: FIG_CONFIG standardizes multi-panel dimensions
  - Lead lines: lead_label() for annotations at risk of overlap
  - Restrained palette: colorblind-safe, one color per semantic concept
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
from matplotlib.lines import Line2D

# =====================================================================
# Seaborn base style + paper context
# =====================================================================
sns.set_theme(
    style="white",              # white background, no top/right spines
    font="serif",
    font_scale=1.0,             # we control sizes via rcParams below
    palette="colorblind",
    rc={
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,     # editable text in vector PDF
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.facecolor": "white",
        "axes.edgecolor": "0.2",
        "grid.color": "0.85",
        "grid.alpha": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.7",
        "legend.fancybox": False,
    },
)

# Apply paper context with our exact font sizes
# (seaborn's "paper" context is a starting point; we override for IEEEtran)
sns.set_context("paper", rc={
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
})

# =====================================================================
# Color palette — colorblind-safe, consistent across all figures
# Each semantic concept maps to ONE color across the entire paper.
# =====================================================================
PAL = sns.color_palette("colorblind")

# --- Semantic colors ---
C_RF       = PAL[0]   # blue         — Rectified Flow / KaryoFlow
C_DDPM     = PAL[3]   # reddish      — DDPM baseline / Euler
C_ADALN    = PAL[2]   # green        — AdaLN-Zero time conditioning
C_OT       = PAL[4]   # purple       — OT coupling / Sinkhorn
C_RAND     = PAL[1]   # orange       — Random coupling
C_SOURCE   = PAL[7]   # sky blue     — noise / source distribution
C_GT       = "black"  # black        — ground truth / target
C_STOCH    = PAL[0]   # blue (same as RF) — Stochastic Coupling

# --- Method-level colors ---
C_HEUN     = PAL[0]   # blue         — Heun solver
C_EULER    = PAL[1]   # orange       — Euler solver
C_DPMPP    = PAL[2]   # green        — DPM-Solver++

# --- Per-class AP ---
C_LARGE    = PAL[0]   # blue         — large chromosomes (A-C)
C_MED      = PAL[1]   # orange       — medium (D-E)
C_SMALL    = PAL[2]   # green        — small (F-G)
C_SEX      = PAL[4]   # purple       — sex (X, Y)
C_OVERALL  = "0.3"    # dark gray    — overall mean line

# --- Gray scale ---
C_DARKGRAY  = "#333333"
C_MIDGRAY   = "#666666"
C_LIGHTGRAY = "#E8E8E8"
C_GRID      = "#CCCCCC"

# --- Region fills for entropy phase diagram ---
C_DANGER  = "#fdecea"  # light red
C_SAFE    = "#eafaf1"  # light green

HERE = Path(__file__).resolve().parent


# =====================================================================
# Helper functions
# =====================================================================

def save_fig(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    """Save figure as PDF (vector) and PNG (raster preview)."""
    out_pdf = HERE / f"{stem}.pdf"
    out_png = HERE / f"{stem}.png"
    fig.savefig(out_pdf, dpi=dpi)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  \u2713 {out_pdf.name}  ({out_pdf.stat().st_size/1024:.0f} KB)")
    print(f"  \u2713 {out_png.name}  ({out_png.stat().st_size/1024:.0f} KB)")


def box(ax: plt.Axes, x: float, y: float, w: float, h: float,
        text: str, fc: str = "#ffffff", ec: str = C_DARKGRAY,
        text_color: str = "black", fontsize: float = 8.5,
        weight: str = "normal") -> None:
    """Draw a rounded-rect box with centered text."""
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.05",
        fc=fc, ec=ec, lw=0.9,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight=weight)


def arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float,
          color: str = C_DARKGRAY, lw: float = 1.0,
          mutation_scale: float = 12) -> None:
    """Draw an arrow annotation."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->",
                                mutation_scale=mutation_scale,
                                color=color, lw=lw),
                zorder=3)


def hide_spines(ax: plt.Axes) -> None:
    """Hide all spines (axis borders). Uses seaborn's despine."""
    sns.despine(ax=ax, left=True, bottom=True)


def annotation_box(ax: plt.Axes, x: float, y: float, text: str,
                   fontsize: float = 8, color: str = C_DARKGRAY,
                   ha: str = "center", va: str = "bottom",
                   ec: str = "0.7", fc: str = "white",
                   alpha: float = 0.92) -> None:
    """Add a text annotation with a rounded box background."""
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
            transform=ax.transAxes if isinstance(x, float) and x <= 1 else None,
            bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=0.5,
                      alpha=alpha), zorder=7)


def lead_label(ax: plt.Axes, text: str, xy: tuple,
               xytext: tuple, fontsize: float = 8,
               color: str = C_DARKGRAY, ha: str = "left",
               va: str = "center", lw: float = 0.5,
               arrow_color: str = "0.6",
               bbox_fc: str = "white", bbox_ec: str = "0.7",
               bbox_alpha: float = 0.85) -> None:
    """Place a text label with a short lead line connecting it to a data point.

    Parameters
    ----------
    xy : (x, y) data coordinates of the target point
    xytext : (x, y) data coordinates of the label
    """
    # Lead line (short connector) — 15% of the way from data to label
    ax.annotate("", xy=(xy[0] + (xytext[0] - xy[0]) * 0.15,
                        xy[1] + (xytext[1] - xy[1]) * 0.15),
                xytext=xy,
                arrowprops=dict(arrowstyle="-", color=arrow_color, lw=lw),
                zorder=2)
    # Label with subtle background
    ax.text(xytext[0], xytext[1], text, fontsize=fontsize, color=color,
            ha=ha, va=va, zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", fc=bbox_fc, ec=bbox_ec,
                      lw=0.4, alpha=bbox_alpha))


# =====================================================================
# Layout contract: standard multi-panel figure dimensions
# Used by method_overview (1x3), ot_theory (1x2),
# qual_mosaic (3x2), tech_pipeline (3x1)
# =====================================================================
FIG_CONFIG = {
    "1x2": {"figsize": (7.2, 3.0), "gridspec": {"wspace": 0.15}},
    "1x3": {"figsize": (7.2, 3.2), "gridspec": {"wspace": 0.12}},
    "3x2": {"figsize": (5.5, 5.0), "gridspec": {"wspace": 0.08, "hspace": 0.18}},
    "3x1": {"figsize": (7.5, 7.5), "gridspec": {"hspace": 0.28}},
}
PANEL_LABEL_KW = dict(loc="left", fontsize=10, weight="bold", pad=3)
