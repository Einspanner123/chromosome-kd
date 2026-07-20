"""
Figure 3: OT Diversity Collapse Theory.

Two panels:
  (a) OT coupling on real chromosome detection image.
  (b) Empirical validation: theoretical log K vs empirical ΔH.

Color scheme (colorblind-friendly):
  - Navy blue (#1a5276): Primary, GT boxes, OT assignment
  - Dark orange (#d35400): Secondary, Random assignment  
  - Dark teal (#16a085): Empirical result (green alternative, higher contrast)
  - Light gray (#7f8c8d): Noise points, grid lines

Run:  python ot_theory.py
Outputs: ot_theory.pdf, ot_theory.png
"""

from __future__ import annotations

import json as json_mod

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from PIL import Image

from figure_style import *

C_PRIMARY = "#1a5276"
C_SECONDARY = "#16a085"
C_ACCENT = "#d35400"
C_SOURCE = "#7f8c8d"

CROP = (0, 100, 350, 450)

CAT_SHORT = {
    1: "A1", 2: "A2", 3: "A3", 4: "B4", 5: "B5", 6: "C6", 7: "C7", 8: "C8",
    9: "C9", 10: "C10", 11: "C11", 12: "C12", 13: "D13", 14: "D14", 15: "D15",
    16: "E16", 17: "E17", 18: "E18", 19: "F19", 20: "F20", 21: "G21", 22: "G22",
    23: "X", 24: "Y",
}

NOISE_AREA = 100


def panel_voronoi(ax: plt.Axes) -> None:
    DATA_ROOT = HERE.parent.parent.parent.parent / "data"
    JEPG_DIR = DATA_ROOT / "24_chromosomes_object" / "JEPG"
    ANN_FILE = DATA_ROOT / "24_chromosomes_object" / "coco" / "valid" / "_annotations.coco.json"

    gt_data = json_mod.load(open(ANN_FILE))
    img_info = gt_data["images"][0]
    img_path = JEPG_DIR / img_info["file_name"]

    img_anns = [a for a in gt_data["annotations"] if a["image_id"] == 1]
    
    current_crop = CROP
    
    all_crop_anns = []
    for a in img_anns:
        x, y, w, h = a["bbox"]
        if x >= current_crop[0] and y >= current_crop[1] and x + w <= current_crop[2] and y + h <= current_crop[3]:
            all_crop_anns.append(a)
    
    centers = [(a['bbox'][0]+a['bbox'][2]/2, a['bbox'][1]+a['bbox'][3]/2, a) for a in all_crop_anns]
    
    if len(centers) > 8:
        selected = [centers[0]]
        remaining = centers[1:]
        
        while len(selected) < 8 and remaining:
            max_min_dist = -1
            best_idx = 0
            
            for i, (cx, cy, ann) in enumerate(remaining):
                min_dist_to_selected = min(
                    np.sqrt((cx - scx)**2 + (cy - scy)**2) 
                    for scx, scy, _ in selected
                )
                if min_dist_to_selected > max_min_dist:
                    max_min_dist = min_dist_to_selected
                    best_idx = i
            
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        
        crop_anns = [s[2] for s in selected]
    else:
        crop_anns = all_crop_anns
    
    if not crop_anns:
        current_crop = (0, 0, img_info['width'], img_info['height'])
        crop_anns = img_anns[:8]

    image = Image.open(img_path).convert("RGB")
    image = image.crop(current_crop)
    img_arr = np.array(image)
    img_h, img_w = img_arr.shape[:2]

    gt_centers = []
    for a in crop_anns:
        x, y, w, h = a["bbox"]
        cx = x - current_crop[0] + w / 2
        cy = y - current_crop[1] + h / 2
        gt_centers.append((cx, cy))
    gt_centers = np.array(gt_centers)

    rng = np.random.default_rng(42)
    n_noise = len(gt_centers)
    noise = np.column_stack([
        rng.uniform(30, img_w - 30, size=n_noise),
        -rng.uniform(20, NOISE_AREA - 15, size=n_noise),
    ])

    from scipy.optimize import linear_sum_assignment
    cost_matrix = np.linalg.norm(noise[:, None, :] - gt_centers[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    ot_matching = col_ind
    
    rand_perm = rng.permutation(len(gt_centers))

    ax.imshow(img_arr, extent=[0, img_w, 0, img_h], zorder=0, interpolation='bilinear')

    label_positions = []
    
    for i, a in enumerate(crop_anns):
        x, y, w, h = a["bbox"]
        x_c = x - current_crop[0]
        y_c = y - current_crop[1]
        cat = CAT_SHORT.get(a["category_id"], "?")
        
        rect = plt.Rectangle((x_c, y_c), w, h, linewidth=1.5,
                             edgecolor=C_PRIMARY, facecolor='none', 
                             zorder=3, linestyle='-')
        ax.add_patch(rect)
        
        box_area = w * h
        
        if box_area > 1500:
            label_x = x_c + w / 2
            label_y = y_c + h / 2
            ha, va = "center", "center"
            fontsize = 8
        else:
            candidates = [
                (x_c + w + 22, y_c + h/2, "left", "center"),
                (x_c - 22, y_c + h/2, "right", "center"),
                (x_c + w/2, y_c + h + 18, "center", "bottom"),
                (x_c + w/2, y_c - 18, "center", "top"),
                (x_c + w + 22, y_c + h + 18, "left", "bottom"),
                (x_c - 22, y_c + h - 18, "right", "top"),
                (x_c + w + 22, y_c - 18, "left", "top"),
                (x_c - 22, y_c + h + 18, "right", "bottom"),
            ]
            
            best_pos = None
            min_dist = float('inf')
            min_label_dist = 25
            
            for lx, ly, lha, lva in candidates:
                too_close = False
                for plx, ply in label_positions:
                    dist = np.sqrt((lx - plx)**2 + (ly - ply)**2)
                    if dist < min_label_dist:
                        too_close = True
                        break
                
                if too_close:
                    continue
                
                if lx > 10 and lx < img_w - 10 and ly > -NOISE_AREA + 10 and ly < img_h - 10:
                    dist_to_center = np.sqrt((lx - (x_c + w/2))**2 + (ly - (y_c + h/2))**2)
                    if dist_to_center < min_dist:
                        min_dist = dist_to_center
                        best_pos = (lx, ly, lha, lva)
            
            if best_pos is None:
                idx = len(label_positions)
                lx, ly, lha, lva = candidates[idx % len(candidates)]
                best_pos = (lx, ly, lha, lva)
            
            label_x, label_y, ha, va = best_pos
            fontsize = 8
        
        ax.text(label_x, label_y, cat, fontsize=fontsize, color=C_PRIMARY,
                fontweight="bold", ha=ha, va=va, zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", lw=0, alpha=0.92))
        label_positions.append((label_x, label_y))

    ax.scatter(noise[:, 0], noise[:, 1], s=80, color=C_SOURCE,
              edgecolor='white', linewidth=1.5, zorder=6, alpha=0.9)

    for idx in range(len(noise)):
        z = noise[idx]
        k = ot_matching[idx]
        ax.plot([z[0], gt_centers[k, 0]], [z[1], gt_centers[k, 1]],
               color=C_PRIMARY, lw=1.5, alpha=0.75, zorder=2, solid_capstyle='round')
        ax.plot(gt_centers[k, 0], gt_centers[k, 1], 'o', 
               color=C_PRIMARY, markersize=4, zorder=4, alpha=0.85)

    n_rand_show = 2
    rng2 = np.random.default_rng(999)
    rand_indices = rng2.choice(len(noise), size=n_rand_show, replace=False)
    for idx in rand_indices:
        z = noise[idx]
        k = rand_perm[idx]
        ax.plot([z[0], gt_centers[k, 0]], [z[1], gt_centers[k, 1]],
               color=C_ACCENT, lw=1.2, alpha=0.4, ls='--', zorder=1.5, dashes=(4, 3))

    ax.set_xlim(-15, img_w + 15)
    ax.set_ylim(-NOISE_AREA - 10, img_h + 25)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=C_PRIMARY, lw=2, label='OT assignment'),
        Line2D([0], [0], color=C_ACCENT, lw=1.5, ls='--', 
               label='Random assignment'),
        mpatches.Patch(facecolor='none', edgecolor=C_PRIMARY, linewidth=1.5, 
                      label='GT boxes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_SOURCE, 
               markersize=8, markeredgecolor='white', markeredgewidth=1.5,
               label='Noise', linestyle='None'),
    ]
    legend = ax.legend(handles=legend_elements, loc="upper right", 
                      frameon=True, fontsize=7, framealpha=1.0, 
                      edgecolor="#cccccc", borderpad=0.4, handletextpad=0.6,
                      labelspacing=0.3)
    legend.get_frame().set_facecolor("white")


def panel_dh(ax: plt.Axes) -> None:
    K = 46
    theory = np.log(K)
    empirical = 3.8415
    empirical_std = 0.02
    rel_err = abs(theory - empirical) / theory * 100

    df_bar = pd.DataFrame({
        "label": [r"$\log K$ (theory)", r"$\Delta H$ (measured)"],
        "value": [theory, empirical],
    })
    PAL_BAR = [C_PRIMARY, C_SECONDARY]

    bars = sns.barplot(data=df_bar, x="label", y="value",
                hue="label", palette=PAL_BAR, edgecolor='white',
                linewidth=1, saturation=1.0, width=0.35,
                ax=ax, legend=False)

    for bar in ax.patches:
        bar.set_alpha(0.92)

    for bar, val, col in zip(ax.patches, [theory, empirical], PAL_BAR):
        col_rgb = to_rgb(col)
        brightness = 0.299*col_rgb[0] + 0.587*col_rgb[1] + 0.114*col_rgb[2]
        txt_color = "white" if brightness < 0.5 else "black"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", ha="center", va="center",
                fontsize=9, color=txt_color, fontweight="bold")

    ax.errorbar(x=1, y=empirical, yerr=empirical_std,
               fmt='none', color='#8e44ad', capsize=4, capthick=1.2, elinewidth=1.2, zorder=5)

    ax.set_ylabel(r"Entropy $\Delta H = H(V|Z)$", fontsize=10, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylim(3.80, 3.88)
    ax.set_xlim(-0.6, 1.6)
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls="--", lw=0.6, alpha=0.4, color='#cccccc')
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    ax.text(0.5, 0.95, f"K = {K} categories", 
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color=C_PRIMARY, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=C_PRIMARY, lw=1, alpha=0.95))
    
    ax.text(0.5, 0.86, f"Gap: {abs(theory - empirical):.4f} ({rel_err:.2f}%)", 
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color=C_ACCENT,
            bbox=dict(boxstyle="round,pad=0.25", fc='white', ec=C_ACCENT, lw=1, alpha=0.9))
    
    ax.text(0.5, 0.78, r"$K$ = effective rank from COCO-style taxonomy", 
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7, color='#666666', style='italic')
    
    ax.text(0.5, 0.71, r"$\Delta H \to \log K$ as $\sigma_t / d_{\min} \to 0$", 
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7, color='#666666', style='italic')


def main() -> None:
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    
    fig = plt.figure(figsize=(7.16, 4.5))
    
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.28)
    
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    
    panel_voronoi(ax_left)
    panel_dh(ax_right)
    
    ax_left.text(0.02, 1.12, "(a)", transform=ax_left.transAxes, 
                fontsize=12, fontweight='bold', va='top', ha='left',
                color=C_PRIMARY, fontfamily='serif')
    ax_left.text(0.12, 1.12, "OT Coupling on Detection", 
                transform=ax_left.transAxes, 
                fontsize=10, fontweight='bold', va='top', ha='left',
                color=C_PRIMARY, fontfamily='serif')
    
    ax_right.text(0.02, 1.12, "(b)", transform=ax_right.transAxes, 
                 fontsize=12, fontweight='bold', va='top', ha='left',
                 color=C_PRIMARY, fontfamily='serif')
    ax_right.text(0.12, 1.12, "Theory vs Empirical", 
                 transform=ax_right.transAxes, 
                 fontsize=10, fontweight='bold', va='top', ha='left',
                 color=C_PRIMARY, fontfamily='serif')

    save_fig(fig, "ot_theory")


if __name__ == "__main__":
    main()
