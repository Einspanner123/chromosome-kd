"""
Figure 3: OT Diversity Collapse Theory.

Two panels:
  (a) OT coupling on real chromosome detection image.
      GT boxes, noise samples, OT (nearest-neighbor) and random assignments.
  (b) Empirical validation: theoretical log K vs empirical ΔH.

Run:  python ot_theory.py
Outputs: ot_theory.pdf, ot_theory.png
"""

from __future__ import annotations

import json as json_mod

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from figure_style import *

# Real chromosome image crop region (contains 8 GT boxes)
CROP = (0, 150, 300, 450)  # x1, y1, x2, y2 in original image coords

CAT_SHORT = {
    1: "A1", 2: "A2", 3: "A3", 4: "B4", 5: "B5", 6: "C6", 7: "C7", 8: "C8",
    9: "C9", 10: "C10", 11: "C11", 12: "C12", 13: "D13", 14: "D14", 15: "D15",
    16: "E16", 17: "E17", 18: "E18", 19: "F19", 20: "F20", 21: "G21", 22: "G22",
    23: "X", 24: "Y",
}

NOISE_AREA = 90  # vertical space above image for noise samples


def panel_voronoi(ax: plt.Axes) -> None:
    """OT coupling visualization on real chromosome detection image."""
    DATA_ROOT = HERE.parent.parent.parent.parent / "data"
    JEPG_DIR = DATA_ROOT / "24_chromosomes_object" / "JEPG"
    ANN_FILE = DATA_ROOT / "24_chromosomes_object" / "coco" / "valid" / "_annotations.coco.json"

    # Load image and annotations
    gt_data = json_mod.load(open(ANN_FILE))
    img_info = gt_data["images"][0]  # id=1, file=1060521.jpg
    img_path = JEPG_DIR / img_info["file_name"]

    # Get GT boxes within crop region
    img_anns = [a for a in gt_data["annotations"] if a["image_id"] == 1]
    crop_anns = []
    for a in img_anns:
        x, y, w, h = a["bbox"]
        if x >= CROP[0] and y >= CROP[1] and x + w <= CROP[2] and y + h <= CROP[3]:
            crop_anns.append(a)

    # Load and crop image
    image = Image.open(img_path).convert("RGB")
    image = image.crop(CROP)
    img_arr = np.array(image)
    img_h, img_w = img_arr.shape[:2]

    # GT box centers in crop coords
    gt_centers = []
    gt_labels = []
    for a in crop_anns:
        x, y, w, h = a["bbox"]
        cx = x - CROP[0] + w / 2
        cy = y - CROP[1] + h / 2
        gt_centers.append((cx, cy))
        gt_labels.append(CAT_SHORT.get(a["category_id"], "?"))
    gt_centers = np.array(gt_centers)

    # Noise samples (in space above image)
    rng = np.random.default_rng(42)
    n_noise = 10
    noise = np.column_stack([
        rng.uniform(20, img_w - 20, size=n_noise),
        -rng.uniform(15, NOISE_AREA - 10, size=n_noise),
    ])

    # OT assignment (nearest-neighbor in 2D center space)
    nn_idx = np.argmin(
        np.linalg.norm(noise[:, None, :] - gt_centers[None, :, :], axis=2),
        axis=1,
    )

    # Random assignment
    rand_perm = rng.permutation(len(gt_centers))
    rand_perm = np.resize(rand_perm, len(noise))

    # --- Plot ---
    # Show image
    ax.imshow(img_arr, extent=[0, img_w, 0, img_h], zorder=0)

    # Draw GT boxes
    for a in crop_anns:
        x, y, w, h = a["bbox"]
        x_c = x - CROP[0]
        y_c = y - CROP[1]
        cat = CAT_SHORT.get(a["category_id"], "?")
        rect = plt.Rectangle((x_c, y_c), w, h, linewidth=1.5,
                             edgecolor=C_GT, facecolor="none", zorder=3)
        ax.add_patch(rect)
        ax.text(x_c + 1, y_c - 1, cat, fontsize=6.5, color=C_GT,
                fontweight="bold", ha="left", va="bottom", zorder=4,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))

    # Noise samples
    ax.scatter(noise[:, 0], noise[:, 1], s=35, color=C_SOURCE,
              edgecolor="k", lw=0.5, zorder=5, label="noise $z_i$")

    # OT assignments (solid lines)
    for z, k in zip(noise, nn_idx):
        ax.plot([z[0], gt_centers[k, 0]], [z[1], gt_centers[k, 1]],
               color=C_OT, lw=1.0, alpha=0.85, zorder=2)

    # Random assignments (dashed, subset for clarity)
    n_rand_show = 4
    rand_indices = rng.choice(len(noise), size=n_rand_show, replace=False)
    for idx in rand_indices:
        z = noise[idx]
        k = rand_perm[idx]
        ax.plot([z[0], gt_centers[k, 0]], [z[1], gt_centers[k, 1]],
               color=C_RAND, lw=0.9, alpha=0.75, ls="--", zorder=1)

    # Separator between noise area and image
    ax.axhline(y=0, color="#cccccc", lw=0.5, ls=":", zorder=0)
    ax.text(img_w / 2, -NOISE_AREA + 5, "noise space", ha="center", va="top",
           fontsize=7, color="#888888", style="italic")

    ax.set_xlim(-5, img_w + 5)
    ax.set_ylim(-NOISE_AREA - 5, img_h + 5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a) OT coupling on detection image", **PANEL_LABEL_KW)
    hide_spines(ax)

    handles = [
        Line2D([0], [0], color=C_OT, lw=1.4, label="OT assignment"),
        Line2D([0], [0], color=C_RAND, lw=1.4, ls="--",
               label="Random assignment"),
        plt.Rectangle((0, 0), 1, 1, fc="none", ec=C_GT, lw=1.5,
                      label="GT boxes"),
    ]
    legend = ax.legend(handles=handles, loc="lower right", frameon=True,
                       fontsize=7, framealpha=0.9, edgecolor="#cccccc",
                       borderpad=0.5, handletextpad=0.5)
    legend.get_frame().set_facecolor("white")



def panel_dh(ax: plt.Axes) -> None:
    K_mean = 46.6
    theory = np.log(K_mean)       # 3.8427
    empirical = 3.8415
    rel_err = abs(theory - empirical) / theory * 100

    # Tidy DataFrame
    df_bar = pd.DataFrame({
        "label": [r"$\log K$ (theory)", r"$\Delta H$ (empirical)"],
        "value": [theory, empirical],
    })
    PAL_BAR = [C_RF, C_OT]

    sns.barplot(data=df_bar, x="label", y="value",
                hue="label", palette=PAL_BAR, edgecolor="black",
                linewidth=0.7, saturation=1, width=0.45,
                ax=ax, legend=False)

    # Value labels inside bars — adaptive text color
    for bar, val, col in zip(ax.patches, [theory, empirical], PAL_BAR):
        brightness = 0.299*col[0] + 0.587*col[1] + 0.114*col[2]
        txt_color = "white" if brightness < 0.5 else "black"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", ha="center", va="center",
                fontsize=9, color=txt_color, fontweight="bold")

    ax.set_ylabel(r"Conditional entropy reduction $\Delta H$", fontsize=9)
    ax.set_ylim(0.0, 4.7)
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.set_title("(b) Theory vs. empirical", **PANEL_LABEL_KW)
    ax.tick_params(axis="y", labelsize=8)

    # Arrow connecting the two bars
    y_top = max(theory, empirical) + 0.35
    ax.annotate("", xy=(0, theory + 0.1), xytext=(1, empirical + 0.1),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.0,
                                connectionstyle="arc3,rad=0"))
    ax.text(0.5, y_top,
            f"rel. err. {rel_err:.2f}%",
            ha="center", va="bottom", fontsize=8, color=C_DARKGRAY,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.95))


def main() -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=FIG_CONFIG["1x2"]["figsize"],
        gridspec_kw={"width_ratios": [1.2, 1]},
        constrained_layout=True,
    )
    panel_voronoi(axes[0])
    panel_dh(axes[1])

    save_fig(fig, "ot_theory")


if __name__ == "__main__":
    main()
