import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(20, 11), dpi=200, facecolor='white')
fig.suptitle('Mechanism Comparison: Order-Sensitivity Gap vs Pairwise-Coupling Effect',
             fontsize=16, fontweight='bold', y=0.99, color='#1a1a2e')

gs = fig.add_gridspec(3, 2, height_ratios=[0.42, 0.42, 0.16], hspace=0.45, wspace=0.3)

ax_left_top = fig.add_subplot(gs[0, 0])
ax_left_bottom = fig.add_subplot(gs[1, 0])
ax_right = fig.add_subplot(gs[0:2, 1])
ax_table = fig.add_subplot(gs[2, :])

fig.text(0.25, 0.94, 'Order-Sensitivity Gap', fontsize=13,
         fontweight='bold', ha='center', color='#2c3e50',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#eaf2f8', edgecolor='#5d6d7e', alpha=0.8))
fig.text(0.75, 0.94, 'Pairwise-Coupling Effect', fontsize=13,
         fontweight='bold', ha='center', color='#1e8449',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', edgecolor='#1e8449', alpha=0.8))

# ============================================================
# LEFT COLUMN TOP: Independent Diffusion
# ============================================================
ax_left_top.set_xlim(0, 10)
ax_left_top.set_ylim(0, 10)
ax_left_top.set_aspect('equal')
ax_left_top.axis('off')
ax_left_top.set_title('A. Independent Diffusion', fontsize=11, fontweight='bold', pad=10, loc='left')

cool_colors = ['#a8c5e0', '#b8d4e3', '#8ab4d6', '#c5dbe8', '#9ec5d8', '#7fb3d5']
np.random.seed(42)
positions_top = [(1.8, 6.5), (4.0, 7.8), (6.0, 5.8), (2.8, 3.5), (7.0, 3.2), (8.2, 6.8)]
sizes_top = [(1.6, 1.0), (1.4, 1.2), (1.8, 0.9), (1.5, 1.1), (1.3, 1.3), (1.6, 1.0)]

for i, ((x, y), (w, h), c) in enumerate(zip(positions_top, sizes_top, cool_colors)):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.08",
                           facecolor=c, edgecolor='#5d6d7e', linewidth=1.5, alpha=0.9)
    ax_left_top.add_patch(rect)
    ax_left_top.text(x, y, f'bbox{i+1}', ha='center', va='center', fontsize=7.5, fontweight='bold')

overlap_circle = plt.Circle((4.0, 7.8), 1.1, color='#e74c3c', fill=False, linestyle='--', linewidth=2, alpha=0.7)
ax_left_top.add_patch(overlap_circle)
ax_left_top.annotate('Overlap', xy=(4.0, 9.2), fontsize=9, color='#e74c3c', fontweight='bold',
                     ha='center', va='center',
                     arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

missing_circle = plt.Circle((8.5, 4.0), 0.7, color='#e74c3c', fill=False, linestyle='--', linewidth=2, alpha=0.7)
ax_left_top.add_patch(missing_circle)
ax_left_top.annotate('Missing', xy=(8.5, 5.2), fontsize=9, color='#e74c3c', fontweight='bold',
                     ha='center', va='center',
                     arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

ax_left_top.text(5.0, 0.5, r'Independent Diffusion $\rightarrow$ No Bbox Correlation',
                ha='center', va='center', fontsize=9, style='italic', color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#eaf2f8', edgecolor='#5d6d7e', alpha=0.8))

# ============================================================
# LEFT COLUMN BOTTOM: Ground Truth Pairing
# ============================================================
ax_left_bottom.set_xlim(0, 10)
ax_left_bottom.set_ylim(0, 10)
ax_left_bottom.set_aspect('equal')
ax_left_bottom.axis('off')
ax_left_bottom.set_title('B. Ground Truth Pairing', fontsize=11, fontweight='bold', pad=10, loc='left')

positions_bottom = [(2.0, 7.0), (2.0, 3.0), (5.0, 7.0), (5.0, 3.0), (8.0, 7.0), (8.0, 3.0)]
sizes_bottom = [(1.5, 1.0)] * 6

for i, ((x, y), (w, h)) in enumerate(zip(positions_bottom, sizes_bottom)):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.08",
                           facecolor=cool_colors[i % len(cool_colors)], edgecolor='#5d6d7e', linewidth=1.5, alpha=0.9)
    ax_left_bottom.add_patch(rect)
    ax_left_bottom.text(x, y, f'bbox{i+1}', ha='center', va='center', fontsize=7.5, fontweight='bold')

pair_labels = [r'$P(X_2|X_1)$', r'$P(X_4|X_3)$', r'$P(X_6|X_5)$']
for idx, (i, j) in enumerate([(0, 1), (2, 3), (4, 5)]):
    x1, y1 = positions_bottom[i]
    x2, y2 = positions_bottom[j]
    ax_left_bottom.annotate('', xy=(x2, y2 + 0.55), xytext=(x1, y1 - 0.55),
                           arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.0))
    mid_x = (x1 + x2) / 2 + 0.7
    mid_y = (y1 + y2) / 2
    ax_left_bottom.text(mid_x, mid_y, pair_labels[idx], fontsize=8, color='#2c3e50', fontweight='bold')

ax_left_bottom.text(5.0, 0.5, r'Ground Truth Pairing: $P(X_i \mid X_j)$',
                   ha='center', va='center', fontsize=9, style='italic', color='#2c3e50',
                   bbox=dict(boxstyle='round,pad=0.25', facecolor='#eaf2f8', edgecolor='#5d6d7e', alpha=0.8))

# ============================================================
# Gap annotation
# ============================================================
fig.text(0.25, 0.505, r'Gap = $P(X_i) \cdot P(X_j)$ vs $P(X_i, X_j)$', fontsize=10.5,
         ha='center', va='center', fontweight='bold', color='#c0392b',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#fadbd8', edgecolor='#c0392b', alpha=0.9))

# ============================================================
# RIGHT COLUMN: Pairwise-Coupling Effect
# ============================================================
ax_right.set_xlim(0, 10)
ax_right.set_ylim(0, 10)
ax_right.set_aspect('equal')
ax_right.axis('off')
ax_right.set_title('C. Fully-Connected Coupling Graph', fontsize=11, fontweight='bold', pad=10, loc='left')

warm_colors = ['#82e0aa', '#f9e79f', '#f5b041', '#e74c3c']
node_positions = {
    'a': (3.0, 7.5),
    'b': (7.0, 7.5),
    'c': (3.0, 2.5),
    'd': (7.0, 2.5),
}

for idx, (key, (x, y)) in enumerate(node_positions.items()):
    rect = FancyBboxPatch((x - 1.0, y - 0.6), 2.0, 1.2, boxstyle="round,pad=0.12",
                           facecolor=warm_colors[idx], edgecolor='#1e8449', linewidth=2.0, alpha=0.92)
    ax_right.add_patch(rect)
    ax_right.text(x, y, f'bbox_{key}', ha='center', va='center', fontsize=9, fontweight='bold')

edge_configs = [
    ('a', 'b', '#27ae60', 0.0, 0.45),
    ('a', 'c', '#e67e22', 0.0, -0.45),
    ('a', 'd', '#c0392b', 0.2, 0.3),
    ('b', 'c', '#8e44ad', -0.2, -0.3),
    ('b', 'd', '#2980b9', 0.0, -0.45),
    ('c', 'd', '#d35400', 0.0, 0.45),
]

for n1, n2, color, rad, label_offset in edge_configs:
    x1, y1 = node_positions[n1]
    x2, y2 = node_positions[n2]

    ax_right.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2.2,
                                    connectionstyle=f'arc3,rad={rad}', alpha=0.75))

    mid_x = (x1 + x2) / 2 + rad * 2
    mid_y = (y1 + y2) / 2 + label_offset * 0.4
    ax_right.text(mid_x, mid_y, f'P({n1}|{n2})', fontsize=6.5,
                  ha='center', va='center', color=color, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.88))

zoom_x, zoom_y = 4.3, 5.0
zoom_w, zoom_h = 1.4, 1.4
zoom_box = FancyBboxPatch((zoom_x, zoom_y), zoom_w, zoom_h, boxstyle="round,pad=0.1",
                           facecolor='#fff9e6', edgecolor='#f39c12', linewidth=2.5, linestyle='--', alpha=0.6)
ax_right.add_patch(zoom_box)
ax_right.text(zoom_x + zoom_w/2, zoom_y + zoom_h + 0.25, 'Zoom-in: Feature Exchange',
              fontsize=8.5, ha='center', fontweight='bold', color='#d68910')

inner_boxes = [
    (zoom_x + 0.15, zoom_y + 0.8, '#82e0aa', r'$f_a$'),
    (zoom_x + 0.75, zoom_y + 0.35, '#e74c3c', r'$f_b$'),
]
for bx, by, bc, bl in inner_boxes:
    rect = FancyBboxPatch((bx - 0.2, by - 0.15), 0.4, 0.3, boxstyle="round,pad=0.05",
                           facecolor=bc, edgecolor='#1e8449', linewidth=1, alpha=0.9)
    ax_right.add_patch(rect)
    ax_right.text(bx, by, bl, ha='center', va='center', fontsize=7, fontweight='bold')

ax_right.annotate('', xy=(inner_boxes[1][0], inner_boxes[1][1]),
                  xytext=(inner_boxes[0][0], inner_boxes[0][1]),
                  arrowprops=dict(arrowstyle='<->', color='#1e8449', lw=1.5))

ax_right.text(5.0, 0.4, 'Pairwise-Coupling: Each bbox can perceive all others via coupling matrix',
              ha='center', va='center', fontsize=9, style='italic', color='#1e8449',
              bbox=dict(boxstyle='round,pad=0.25', facecolor='#d5f5e3', edgecolor='#1e8449', alpha=0.8))

# ============================================================
# BOTTOM: Comparison Table
# ============================================================
ax_table.axis('off')

table_data = [
    ['Metric', 'Independent Diffusion', 'Pairwise-Coupling'],
    ['Order-Sensitivity', 'High', 'Low'],
    ['Performance Gain', '+0.0%', '+2.1% (24obj mAP)'],
    ['Chromosome Suitability', 'Not Suitable', 'Suitable'],
]

table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_facecolor('#1a1a2e')
        cell.set_text_props(color='white', fontweight='bold')
    elif j == 0:
        cell.set_facecolor('#f8f9fa')
        cell.set_text_props(fontweight='bold')
    elif j == 1:
        cell.set_facecolor('#eaf2f8')
    elif j == 2:
        cell.set_facecolor('#d5f5e3')

    cell.set_edgecolor('#bdc3c7')
    cell.set_linewidth(0.8)
    cell.set_height(0.18)

# ============================================================
# Formulas
# ============================================================
fig.text(0.25, 0.015,
         r'$\mathbf{Order\text{-}Sensitivity\ Gap} = H_p(X,Y) - H_q(X,Y) > 0$',
         fontsize=11, ha='center', va='center', fontweight='bold', color='#c0392b',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', edgecolor='#c0392b', alpha=0.9))

fig.text(0.75, 0.015,
         r'$\mathbf{Pairwise\text{-}Coupling}: \hat{P}(X_i | X_{j \neq i}) = \mathrm{softmax}(W_c \cdot f(X_i, X_j))$',
         fontsize=11, ha='center', va='center', fontweight='bold', color='#1e8449',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', edgecolor='#1e8449', alpha=0.9))

# ============================================================
# Legend (compact, inside right margin)
# ============================================================
legend_colors = ['#27ae60', '#e67e22', '#c0392b', '#8e44ad', '#2980b9', '#d35400']
legend_labels = [
    r'$P(a|b)$: bbox$_a$ $\rightarrow$ bbox$_b$',
    r'$P(a|c)$: bbox$_a$ $\rightarrow$ bbox$_c$',
    r'$P(a|d)$: bbox$_a$ $\rightarrow$ bbox$_d$',
    r'$P(b|c)$: bbox$_b$ $\rightarrow$ bbox$_c$',
    r'$P(b|d)$: bbox$_b$ $\rightarrow$ bbox$_d$',
    r'$P(c|d)$: bbox$_c$ $\rightarrow$ bbox$_d$',
]

legend_x = 0.985
legend_start_y = 0.93
fig.text(legend_x, legend_start_y, 'Coupling Edges:', fontsize=8, fontweight='bold',
         ha='right', color='#2c3e50')
for idx, (color, label) in enumerate(zip(legend_colors, legend_labels)):
    fig.text(legend_x, legend_start_y - 0.022 * (idx + 1), label,
             fontsize=7, ha='right', color=color, fontweight='bold')

plt.savefig('/home/linkst/workspace/chromosome-kd/docs/paper_figures/scene2_mechanism_comparison/mechanism_comparison.png',
            dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.3)

import os
file_path = '/home/linkst/workspace/chromosome-kd/docs/paper_figures/scene2_mechanism_comparison/mechanism_comparison.png'
file_size = os.path.getsize(file_path)
from PIL import Image
img = Image.open(file_path)
print(f"Image saved successfully!")
print(f"Resolution: {img.size[0]}x{img.size[1]} pixels")
print(f"File size: {file_size / 1024:.1f} KB")
