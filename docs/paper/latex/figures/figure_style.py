"""
Unified figure style for KaryoFlow / LDMDet paper figures.

All figure scripts should import from this module instead of duplicating
rcParams, color definitions, and helpers.

Usage:
    from figure_style import *
    import matplotlib.pyplot as plt
    # ... plotting code ...
    save_fig(fig, "my_figure")
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
from matplotlib.lines import Line2D

# =====================================================================
# rcParams — single source of truth
# Font sizes optimized for IEEEtran column width (~89 mm / 3.5 in).
# Minimum readable annotation: 8pt in figure; 7pt only for very minor labels.
# =====================================================================
RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,  # editable text in vector PDF
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
}

plt.rcParams.update(RC_PARAMS)

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
    """Hide all spines (axis borders)."""
    for spine in ax.spines.values():
        spine.set_visible(False)


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
