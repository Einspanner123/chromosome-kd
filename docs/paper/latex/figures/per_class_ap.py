"""
Figure 6: Per-Class AP Analysis — horizontal bar chart.

24 classes grouped by chromosome size (Large A-C, Medium D-E, Small F-G, Sex X/Y).
Clean layout with proper spacing.

Run:  python per_class_ap.py
Outputs: per_class_ap.pdf, per_class_ap.png
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from figure_style import *

DATA = [
    ("Large", "A1", 0.913), ("Large", "A2", 0.907), ("Large", "A3", 0.905),
    ("Large", "B4", 0.905), ("Large", "B5", 0.908),
    ("Large", "C6", 0.900), ("Large", "C7", 0.896), ("Large", "C8", 0.883),
    ("Large", "C9", 0.880), ("Large", "C10", 0.877), ("Large", "C11", 0.871),
    ("Large", "C12", 0.890),
    ("Medium", "D13", 0.857), ("Medium", "D14", 0.856),
    ("Medium", "D15", 0.845),
    ("Medium", "E16", 0.854), ("Medium", "E17", 0.842),
    ("Medium", "E18", 0.834),
    ("Small", "F19", 0.821), ("Small", "F20", 0.818),
    ("Small", "G21", 0.789), ("Small", "G22", 0.790),
    ("Sex", "X", 0.885), ("Sex", "Y", 0.776),
]

GROUP_COLOR = {"Large": C_LARGE, "Medium": C_MED, "Small": C_SMALL, "Sex": C_SEX}
GROUP_ORDER = ["Large", "Medium", "Small", "Sex"]

GROUP_FILL = {
    "Large": C_LIGHTGRAY,
    "Medium": "#f8f8f8",
    "Small": C_LIGHTGRAY,
    "Sex": "#f0f0f0",
}


def main() -> None:
    df = pd.DataFrame(DATA, columns=["group", "class", "ap"])
    
    classes = df["class"].tolist()
    groups = df["group"].tolist()
    aps = df["ap"].tolist()
    
    n = len(classes)
    y_pos = np.arange(n)
    
    fig, ax = plt.subplots(figsize=(7.0, 7.5), constrained_layout=True)
    
    bar_height = 0.7
    bars = ax.barh(y_pos, aps, height=bar_height, 
                  color=[GROUP_COLOR[g] for g in groups],
                  edgecolor='black', linewidth=0.5)
    
    prev_group = None
    group_start = 0
    for i, g in enumerate(groups):
        if g != prev_group:
            if prev_group is not None:
                group_end = i - 1
                ax.axhspan(group_start - 0.5, group_end + 0.5, 
                          color=GROUP_FILL[prev_group], alpha=0.4, zorder=0)
            group_start = i
            prev_group = g
    ax.axhspan(group_start - 0.5, n - 0.5, 
              color=GROUP_FILL[prev_group], alpha=0.4, zorder=0)
    
    prev_group = None
    for i, g in enumerate(groups):
        if g != prev_group and prev_group is not None:
            ax.axhline(i - 0.5, color='#cccccc', lw=1.0, ls='-', alpha=0.8, zorder=1)
        prev_group = g
    
    for i, (ap, cls) in enumerate(zip(aps, classes)):
        ax.text(ap - 0.005, y_pos[i], f'{ap:.3f}', 
                va='center', ha='right', fontsize=9, 
                color='white', fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel('Average Precision (AP)', fontsize=11)
    ax.set_xlim(0.65, 1.00)
    ax.set_xticks(np.arange(0.65, 1.01, 0.10))
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis='x', ls=':', lw=0.4, alpha=0.5)
    
    ax.tick_params(axis='y', labelsize=10, pad=6)
    ax.tick_params(axis='x', labelsize=9)
    
    for label, g in zip(ax.get_yticklabels(), groups):
        label.set_color(GROUP_COLOR[g])
        label.set_fontweight('bold')
    
    group_labels = [
        ('Large (A–C)', n - 0.5 - 12/2, C_LARGE),
        ('Medium (D–E)', n - 0.5 - (12 + 6/2), C_MED),
        ('Small (F–G)', n - 0.5 - (12 + 6 + 4/2), C_SMALL),
        ('Sex (X, Y)', n - 0.5 - (12 + 6 + 4 + 2/2), C_SEX),
    ]
    
    for text, y, color in group_labels:
        ax.text(-0.08, y, text, fontsize=9, ha='right', va='center',
                color=color, fontweight='bold',
                transform=ax.get_yaxis_transform())
    
    overall_mean = df["ap"].mean()
    ax.axvline(overall_mean, color=C_OVERALL, lw=1.2, ls='--', alpha=0.7, zorder=2)
    mean_label_x = min(overall_mean + 0.01, 0.98)
    ax.text(mean_label_x, n - 0.5, f'Mean={overall_mean:.3f}',
            fontsize=9, color=C_OVERALL, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_OVERALL, lw=0.8))
    
    legend_elements = [
        Patch(facecolor=C_LARGE, edgecolor='black', label='Large (A–C)'),
        Patch(facecolor=C_MED, edgecolor='black', label='Medium (D–E)'),
        Patch(facecolor=C_SMALL, edgecolor='black', label='Small (F–G)'),
        Patch(facecolor=C_SEX, edgecolor='black', label='Sex (X, Y)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', 
              fontsize=9, frameon=True, framealpha=0.95)
    
    ax.set_title('Per-Class Average Precision by Chromosome Group',
                 fontsize=12, pad=10, fontweight='bold')
    
    save_fig(fig, 'per_class_ap')


if __name__ == '__main__':
    main()
