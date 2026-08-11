"""Shared vector-first style for the rebuilt paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE.parent / "generated"

# Okabe--Ito inspired, colorblind-safe semantic palette.
C_TEXT = "#202124"
C_MUTED = "#667085"
C_LINE = "#98A2B3"
C_LIGHT = "#F2F4F7"
C_FOUNDATION = "#667085"
C_RF = "#0072B2"
C_RF_LIGHT = "#E5F3FA"
C_OCGR = "#009E73"
C_OCGR_LIGHT = "#E4F6F0"
C_LQCR = "#D55E00"
C_LQCR_LIGHT = "#FCEEE6"
C_OUTPUT = "#7A5195"


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.color": C_TEXT,
        "axes.edgecolor": C_TEXT,
        "lines.solid_capstyle": "round",
    })


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str = "",
    *,
    facecolor: str = "white",
    edgecolor: str = C_LINE,
    textcolor: str = C_TEXT,
    linewidth: float = 0.9,
    linestyle: str = "-",
    fontsize: float = 7.2,
    weight: str = "normal",
    radius: float = 0.025,
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            xy[0] + width / 2,
            xy[1] + height / 2,
            text,
            ha="center",
            va="center",
            color=textcolor,
            fontsize=fontsize,
            weight=weight,
            zorder=zorder + 1,
        )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = C_LINE,
    linewidth: float = 0.9,
    style: str = "-|>",
    zorder: int = 2,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": style,
            "color": color,
            "linewidth": linewidth,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 8,
        },
        zorder=zorder,
    )


def panel_title(ax: plt.Axes, label: str, title: str) -> None:
    """Draw only a subfigure identifier; explain panel content in the caption."""
    ax.text(
        0.0,
        1.025,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        weight="bold",
        color=C_TEXT,
    )


def save_vector_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.025,
            transparent=False,
        )
    plt.close(fig)
