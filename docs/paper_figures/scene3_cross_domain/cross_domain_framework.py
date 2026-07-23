"""
Cross-Domain Application Framework for KaryoFlow.

A four-zone horizontal layout:
  Data Zone  ->  AI Model Zone  ->  Clinical Tasks Zone  ->  Validation Zone

Style: Nature Medicine / IEEE TMI quality, flat-design icons,
       subtle color blocks, thin lines, no 3D or glow effects.

Run: python cross_domain_framework.py
Output: cross_domain_framework.png, cross_domain_framework.pdf
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
from matplotlib.path import Path
import matplotlib.patheffects as pe
from matplotlib.font_manager import FontProperties
from pathlib import Path

HERE = Path(__file__).resolve().parent

C_DATA_BG = '#EBF5FB'
C_DATA_BORDER = '#5DADE2'
C_DATA_ICON = '#1E88E5'

C_MODEL_BG = '#E8F8F0'
C_MODEL_BORDER = '#2ECC71'
C_MODEL_TITLE = '#1ABC9C'

C_TASKS_BG = '#F4F6F7'
C_TASKS_BORDER = '#BDC3C7'

C_VALID_BG = '#FEF9E7'
C_VALID_BORDER = '#F4D03F'

C_TEXT_PRIMARY = '#1C2833'
C_TEXT_SECONDARY = '#566573'
C_TEXT_TERTIARY = '#95A5A6'
C_ACCENT = '#1ABC9C'
C_HIGHLIGHT = '#2980B9'
C_WARN = '#E74C3C'
C_SUCCESS = '#27AE60'

ARROW_COLOR = '#7F8C8D'
CONNECTOR_COLOR = '#34495E'

FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_PATH_BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
FONT_PATH_EN = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_PATH_EN_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

CN_REG = FontProperties(fname=FONT_PATH)
CN_BOLD = FontProperties(fname=FONT_PATH_BOLD, weight='bold')
EN_REG = FontProperties(fname=FONT_PATH_EN)
EN_BOLD = FontProperties(fname=FONT_PATH_EN_BOLD, weight='bold')

plt.rcParams.update({
    'mathtext.fontset': 'cm',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.unicode_minus': False,
})


def draw_rounded_rect(ax, x, y, w, h, fc, ec, lw=1.2, alpha=1.0, round_size=0.15):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0.03,rounding_size={round_size}',
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha,
        zorder=2
    )
    ax.add_patch(rect)
    return rect


def draw_arrow(ax, x1, y1, x2, y2, color=ARROW_COLOR, lw=1.2):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->',
            mutation_scale=14,
            color=color,
            lw=lw,
            connectionstyle='arc3,rad=0'
        ),
        zorder=5
    )


def text_cn_center(ax, x, y, text, fontsize=9, color=C_TEXT_PRIMARY, bold=False):
    fp = CN_BOLD if bold else CN_REG
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, color=color, fontproperties=fp, zorder=6)


def text_en_center(ax, x, y, text, fontsize=9, color=C_TEXT_PRIMARY, bold=False):
    fp = EN_BOLD if bold else EN_REG
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, color=color, fontproperties=fp, zorder=6)


def text_cn_left(ax, x, y, text, fontsize=9, color=C_TEXT_PRIMARY, bold=False):
    fp = CN_BOLD if bold else CN_REG
    ax.text(x, y, text, ha='left', va='center',
            fontsize=fontsize, color=color, fontproperties=fp, zorder=6)


def text_en_left(ax, x, y, text, fontsize=9, color=C_TEXT_PRIMARY, bold=False):
    fp = EN_BOLD if bold else EN_REG
    ax.text(x, y, text, ha='left', va='center',
            fontsize=fontsize, color=color, fontproperties=fp, zorder=6)


def draw_badge(ax, x, y, w, h, text, fc='#FFFFFF', ec='#CCCCCC',
               fontsize=7, color=C_TEXT_SECONDARY, round_size=0.15):
    draw_rounded_rect(ax, x, y, w, h, fc, ec, lw=0.8, round_size=round_size)
    text_en_center(ax, x + w/2, y + h/2, text, fontsize=fontsize, color=color)


def draw_microscope_icon(ax, cx, cy, scale=1.0):
    s = scale
    lw = 1.8
    ax.plot([cx - 0.35*s, cx + 0.35*s], [cy - 0.55*s, cy - 0.55*s],
            color=C_DATA_ICON, lw=lw, zorder=3)
    ax.plot([cx, cx], [cy - 0.55*s, cy + 0.35*s],
            color=C_DATA_ICON, lw=lw, zorder=3)
    ax.plot([cx - 0.55*s, cx + 0.55*s], [cy, cy],
            color=C_DATA_ICON, lw=lw, zorder=3)
    ax.plot([cx + 0.3*s, cx + 0.3*s], [cy, cy + 0.5*s],
            color=C_DATA_ICON, lw=lw, zorder=3)
    ax.plot([cx + 0.1*s, cx + 0.5*s], [cy + 0.5*s, cy + 0.5*s],
            color=C_DATA_ICON, lw=lw, zorder=3)
    ax.plot([cx - 0.5*s, cx - 0.25*s], [cy + 0.65*s, cy + 0.35*s],
            color=C_DATA_ICON, lw=lw, zorder=3)
    ax.plot([cx - 0.6*s, cx - 0.1*s], [cy + 0.75*s, cy + 0.75*s],
            color=C_DATA_ICON, lw=lw, zorder=3)


def draw_chromosome_icon(ax, cx, cy, scale=1.0):
    s = scale
    lw = 1.5
    color = '#3498DB'
    x = cx
    ax.plot([x, x], [cy + 0.45*s, cy - 0.45*s], color=color, lw=lw + 1.5, zorder=3)
    ax.plot([x, x - 0.2*s], [cy + 0.45*s, cy + 0.3*s], color=color, lw=lw, zorder=3)
    ax.plot([x, x + 0.2*s], [cy + 0.45*s, cy + 0.3*s], color=color, lw=lw, zorder=3)
    ax.plot([x, x - 0.2*s], [cy - 0.45*s, cy - 0.3*s], color=color, lw=lw, zorder=3)
    ax.plot([x, x + 0.2*s], [cy - 0.45*s, cy - 0.3*s], color=color, lw=lw, zorder=3)
    ax.plot([x - 0.15*s, x + 0.15*s], [cy, cy], color=color, lw=lw + 2, zorder=4)


def draw_neuron_icon(ax, cx, cy, scale=1.0):
    s = scale
    r = 0.28 * s
    circle = Circle((cx, cy), r, facecolor='#FF8A65', edgecolor='#E64A19',
                    lw=1.2, zorder=3)
    ax.add_patch(circle)
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        ex = cx + 0.48*s * np.cos(angle)
        ey = cy + 0.48*s * np.sin(angle)
        ax.plot([cx + r*np.cos(angle), ex],
                [cy + r*np.sin(angle), ey],
                color='#E64A19', lw=1.0, zorder=3)
        ax.plot([ex - 0.03*s, ex + 0.03*s], [ey - 0.03*s, ey + 0.03*s],
                color='#E64A19', lw=1.5, zorder=4)


def draw_wave_icon(ax, cx, cy, scale=1.0):
    s = scale
    x = np.linspace(cx - 0.45*s, cx + 0.45*s, 60)
    y = cy + 0.25*s * np.sin((x - cx) / 0.15*s * np.pi)
    ax.plot(x, y, color='#43A047', lw=2.0, zorder=3)
    x2 = np.linspace(cx - 0.35*s, cx + 0.35*s, 40)
    y2 = cy + 0.35*s * np.sin((x2 - cx) / 0.12*s * np.pi + 0.5)
    ax.plot(x2, y2, color='#81C784', lw=1.3, ls='--', zorder=3)


def draw_detection_icon(ax, cx, cy, scale=1.0):
    s = scale
    ax.plot([cx - 0.4*s, cx - 0.4*s, cx + 0.4*s, cx + 0.4*s, cx - 0.4*s],
            [cy - 0.3*s, cy + 0.3*s, cy + 0.3*s, cy - 0.3*s, cy - 0.3*s],
            color='#1565C0', lw=1.8, zorder=3)
    ax.plot([cx - 0.5*s, cx - 0.4*s], [cy + 0.4*s, cy + 0.3*s],
            color='#1565C0', lw=1.5, zorder=3)
    ax.plot([cx - 0.5*s, cx - 0.4*s], [cy + 0.4*s, cy - 0.3*s],
            color='#1565C0', lw=1.5, zorder=3)
    ax.plot([cx - 0.6*s, cx - 0.5*s], [cy + 0.4*s, cy + 0.4*s],
            color='#1565C0', lw=1.5, zorder=3)


def draw_counting_icon(ax, cx, cy, scale=1.0):
    s = scale
    draw_rounded_rect(ax, cx - 0.38*s, cy - 0.28*s, 0.76*s, 0.56*s,
                      '#FFFFFF', '#1976D2', lw=1.8, round_size=0.2)
    text_en_center(ax, cx, cy, '46', fontsize=15*s, color='#1976D2', bold=True)


def draw_karyotype_icon(ax, cx, cy, scale=1.0):
    s = scale
    n_pairs = 11
    spacing = 0.14 * s
    total_w = n_pairs * spacing
    start_x = cx - total_w / 2
    colors = ['#1E88E5', '#1E88E5', '#1E88E5', '#43A047', '#43A047',
              '#43A047', '#43A047', '#FF8A65', '#FF8A65', '#FF8A65', '#8E24AA']
    heights = [0.35, 0.35, 0.33, 0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.15]
    for i in range(n_pairs):
        x_base = start_x + i * spacing
        h = heights[i] * s
        color = colors[i]
        ax.plot([x_base - 0.025*s, x_base - 0.025*s],
                [cy - h/2, cy + h/2], color=color, lw=2.0*s, zorder=3)
        ax.plot([x_base + 0.025*s, x_base + 0.025*s],
                [cy - h/2, cy + h/2], color=color, lw=2.0*s, zorder=3)
    text_en_center(ax, cx, cy + 0.5*s, '1-22 + X, Y', fontsize=7*s, color=C_TEXT_TERTIARY)


def draw_abnormality_icon(ax, cx, cy, scale=1.0):
    s = scale
    circle = Circle((cx - 0.22*s, cy), 0.14*s, facecolor='#E57373', edgecolor='#C62828',
                    lw=1.2, zorder=3)
    ax.add_patch(circle)
    ax.plot([cx - 0.13*s, cx - 0.03*s], [cy - 0.08*s, cy + 0.02*s],
            color='#FFFFFF', lw=2.2, zorder=4)
    ax.plot([cx - 0.13*s, cx - 0.03*s], [cy + 0.02*s, cy - 0.08*s],
            color='#FFFFFF', lw=2.2, zorder=4)
    ax.text(cx + 0.3*s, cy, '?', fontsize=16*s, color='#C62828',
            ha='center', va='center', weight='bold', zorder=3)


def draw_segmentation_icon(ax, cx, cy, scale=1.0):
    s = scale
    draw_rounded_rect(ax, cx - 0.4*s, cy - 0.3*s, 0.8*s, 0.6*s,
                      '#FFFFFF', '#CCCCCC', lw=1.2)
    colors_fill = ['#FFCDD2', '#C8E6C9', '#BBDEFB', '#FFE0B2']
    positions = [(-0.22, -0.12), (0.02, 0.08), (-0.12, 0.12), (0.15, -0.08)]
    sizes = [(0.18*s, 0.16*s), (0.18*s, 0.14*s), (0.14*s, 0.12*s), (0.16*s, 0.14*s)]
    for color, (px, py), (w, h) in zip(colors_fill, positions, sizes):
        rect = FancyBboxPatch(
            (cx + px*s, cy + py*s), w, h,
            boxstyle='round,pad=0.02',
            facecolor=color, edgecolor='#888888', lw=0.8, alpha=0.75,
            zorder=3
        )
        ax.add_patch(rect)


def draw_doctor_icon(ax, cx, cy, scale=1.0):
    s = scale
    circle = Circle((cx, cy + 0.25*s), 0.16*s, facecolor='#FFFFFF',
                    edgecolor='#455A64', lw=1.5, zorder=3)
    ax.add_patch(circle)
    ax.plot([cx - 0.28*s, cx + 0.28*s], [cy, cy],
            color='#455A64', lw=2.2*s, zorder=3)
    ax.plot([cx - 0.2*s, cx - 0.2*s], [cy, cy - 0.22*s],
            color='#455A64', lw=1.6*s, zorder=3)
    ax.plot([cx + 0.2*s, cx + 0.2*s], [cy, cy - 0.22*s],
            color='#455A64', lw=1.6*s, zorder=3)
    ax.plot([cx - 0.16*s, cx + 0.16*s], [cy - 0.22*s, cy - 0.22*s],
            color='#455A64', lw=1.6*s, zorder=3)


def draw_report_icon(ax, cx, cy, scale=1.0):
    s = scale
    draw_rounded_rect(ax, cx - 0.22*s, cy - 0.28*s, 0.44*s, 0.56*s,
                      '#FFFFFF', '#607D8B', lw=1.5, round_size=0.1)
    for i in range(3):
        y = cy + 0.18*s - i * 0.15*s
        ax.plot([cx - 0.14*s, cx + 0.14*s], [y, y],
                color='#B0BEC5', lw=1.2*s, zorder=3)


def draw_patient_icon(ax, cx, cy, scale=1.0):
    s = scale
    circle = Circle((cx, cy + 0.25*s), 0.14*s, facecolor='#FFFFFF',
                    edgecolor='#455A64', lw=1.5, zorder=3)
    ax.add_patch(circle)
    ax.plot([cx - 0.22*s, cx + 0.22*s], [cy + 0.08*s, cy + 0.08*s],
            color='#455A64', lw=2.0*s, zorder=3)
    ax.plot([cx - 0.16*s, cx + 0.16*s], [cy - 0.2*s, cy - 0.2*s],
            color='#455A64', lw=1.8*s, zorder=3)
    ax.plot([cx - 0.16*s, cx - 0.22*s], [cy + 0.08*s, cy - 0.2*s],
            color='#455A64', lw=1.8*s, zorder=3)
    ax.plot([cx + 0.16*s, cx + 0.22*s], [cy + 0.08*s, cy - 0.2*s],
            color='#455A64', lw=1.8*s, zorder=3)


def draw_data_zone(ax, x, y, w, h):
    draw_rounded_rect(ax, x, y, w, h, C_DATA_BG, C_DATA_BORDER, lw=1.2)

    text_en_center(ax, x + w/2, y + h - 0.35, 'Domain Data', fontsize=11, color=C_HIGHLIGHT, bold=True)
    text_cn_center(ax, x + w/2, y + h - 0.12, '领域数据', fontsize=7.5, color=C_TEXT_TERTIARY)

    icon_y = y + h * 0.68
    draw_microscope_icon(ax, x + 0.65, icon_y, scale=1.1)
    draw_chromosome_icon(ax, x + 1.75, icon_y, scale=1.0)
    draw_chromosome_icon(ax, x + 2.45, icon_y, scale=1.0)
    draw_chromosome_icon(ax, x + 3.15, icon_y, scale=1.0)

    data_items = [
        ('Cell Imaging Dataset', '1,540 training images', '#3498DB'),
        ('Karyotyping Annotation', '24 chromosomes x 4 objects', '#2ECC71'),
        ('24 Chromosome Classes', 'Per-class balanced', '#9B59B6'),
    ]

    start_y = y + h * 0.40
    for i, (label, val, color) in enumerate(data_items):
        item_y = start_y - i * 0.55
        draw_rounded_rect(ax, x + 0.4, item_y - 0.32, 3.4, 0.45,
                          '#FFFFFF', color, lw=1.0, round_size=0.12)
        text_en_left(ax, x + 0.55, item_y - 0.18, label, fontsize=8.5, color=color, bold=True)
        text_en_left(ax, x + 0.55, item_y - 0.05, val, fontsize=7, color=C_TEXT_SECONDARY)


def draw_model_zone(ax, x, y, w, h):
    draw_rounded_rect(ax, x, y, w, h, C_MODEL_BG, C_MODEL_BORDER, lw=2.0)

    text_en_center(ax, x + w/2, y + h - 0.35, 'AI Model Core', fontsize=11, color=C_MODEL_TITLE, bold=True)
    text_cn_center(ax, x + w/2, y + h - 0.12, 'AI模型核心', fontsize=7.5, color=C_TEXT_TERTIARY)

    text_en_center(ax, x + w/2, y + h * 0.78, 'KaryoFlow', fontsize=18, color=C_MODEL_TITLE, bold=True)

    divider_y = y + h * 0.62
    ax.plot([x + 0.5, x + w - 0.5], [divider_y, divider_y],
            color=C_MODEL_BORDER, lw=0.8, ls='--', alpha=0.5, zorder=3)

    modules_y = y + h * 0.48
    module_xs = [x + 1.3, x + w/2, x + w - 1.3]

    draw_neuron_icon(ax, module_xs[0], modules_y + 0.2, scale=0.75)
    text_en_center(ax, module_xs[0], modules_y - 0.2, 'Feature Learning', fontsize=7.5, color='#E64A19', bold=True)

    draw_wave_icon(ax, module_xs[1], modules_y + 0.2, scale=0.85)
    text_en_center(ax, module_xs[1], modules_y - 0.2, 'Rectified Flow Denoising', fontsize=7.5, color='#2E7D32', bold=True)

    draw_detection_icon(ax, module_xs[2], modules_y + 0.2, scale=0.75)
    text_en_center(ax, module_xs[2], modules_y - 0.2, 'DPM-Solver++ Detection', fontsize=7.5, color='#1565C0', bold=True)

    specs = [
        ('DLA34-FPN Backbone', '#1ABC9C'),
        ('1-4 Steps Inference', '#27AE60'),
        ('83-92% mAP (24 obj)', '#E74C3C'),
    ]

    start_y = y + h * 0.20
    for i, (spec, color) in enumerate(specs):
        sy = start_y - i * 0.42
        draw_badge(ax, x + w*0.12, sy, w*0.76, 0.35, spec, fc='#FFFFFF', ec=color, fontsize=8, color=color)


def draw_tasks_zone(ax, x, y, w, h):
    draw_rounded_rect(ax, x, y, w, h, C_TASKS_BG, C_TASKS_BORDER, lw=1.2)

    text_en_center(ax, x + w/2, y + h - 0.35, 'Clinical Tasks', fontsize=11, color=C_TEXT_PRIMARY, bold=True)
    text_cn_center(ax, x + w/2, y + h - 0.12, '临床任务', fontsize=7.5, color=C_TEXT_TERTIARY)

    task_configs = [
        ('Chromosome Counting', '46', draw_counting_icon, '#1976D2'),
        ('Karyotype Matching', 'F1', draw_karyotype_icon, '#388E3C'),
        ('Aneuploidy Detection', 'Sens', draw_abnormality_icon, '#C62828'),
        ('Instance Segmentation', '89%', draw_segmentation_icon, '#7B1FA2'),
    ]

    n_tasks = len(task_configs)
    margin = 0.3
    task_w = (w - 2 * margin) / n_tasks
    icon_y = y + h * 0.55
    label_y = y + h * 0.28

    for i, (name, metric, icon_func, color) in enumerate(task_configs):
        tx = x + margin + i * task_w + task_w / 2

        draw_rounded_rect(ax, tx - task_w * 0.42, icon_y - 0.75,
                          task_w * 0.84, 1.45,
                          '#FFFFFF', color, lw=1.2, round_size=0.15)

        icon_func(ax, tx, icon_y, scale=0.7)

        text_en_center(ax, tx, icon_y - 0.55, name, fontsize=7.5, color=color, bold=True)

        draw_rounded_rect(ax, tx - 0.32, label_y - 0.25, 0.64, 0.42,
                          color, color, lw=1.0, round_size=0.2)
        text_en_center(ax, tx, label_y - 0.04, metric, fontsize=7.5, color='#FFFFFF', bold=True)


def draw_validation_zone(ax, x, y, w, h):
    draw_rounded_rect(ax, x, y, w, h, C_VALID_BG, C_VALID_BORDER, lw=1.5)

    text_en_center(ax, x + w/2, y + h - 0.35, 'Validation & Impact', fontsize=10, color='#E65100', bold=True)
    text_cn_center(ax, x + w/2, y + h - 0.12, '验证与影响', fontsize=7, color=C_TEXT_TERTIARY)

    cx = x + w/2

    draw_rounded_rect(ax, cx - 1.25, y + h * 0.72, 2.5, 0.5,
                      '#FFE0B2', '#FF9800', lw=1.2, round_size=0.15)
    text_en_center(ax, cx, y + h * 0.72 + 0.25, 'Clinical Validation Required', fontsize=7.5, color='#E65100', bold=True)

    impact_items = [
        ('Automation', '#1976D2'),
        ('XAI for Cytologists', '#388E3C'),
        ('Diagnostic Assistance', '#7B1FA2'),
    ]

    start_y = y + h * 0.52
    for i, (item, color) in enumerate(impact_items):
        iy = start_y - i * 0.38
        ax.plot([cx - 0.85, cx - 0.35], [iy, iy],
                color=color, lw=2.2, zorder=3)
        text_en_left(ax, cx - 0.2, iy, item, fontsize=8, color=color, bold=True)

    flow_y = y + h * 0.22
    draw_doctor_icon(ax, cx - 1.4, flow_y, scale=0.65)
    text_en_center(ax, cx - 1.4, flow_y - 0.45, 'Doctor', fontsize=6.5, color=C_TEXT_SECONDARY)

    draw_arrow(ax, cx - 1.0, flow_y, cx - 0.45, flow_y, color='#FF9800', lw=1.5)

    draw_report_icon(ax, cx, flow_y, scale=0.65)
    text_en_center(ax, cx, flow_y - 0.45, 'Report', fontsize=6.5, color=C_TEXT_SECONDARY)

    draw_arrow(ax, cx + 0.45, flow_y, cx + 1.0, flow_y, color='#FF9800', lw=1.5)

    draw_patient_icon(ax, cx + 1.4, flow_y, scale=0.65)
    text_en_center(ax, cx + 1.4, flow_y - 0.45, 'Patient', fontsize=6.5, color=C_TEXT_SECONDARY)


def draw_connections(ax, x_data, x_model, x_tasks, x_valid,
                      w_data, w_model, w_tasks, w_valid, zone_y, zone_h):
    data_right = x_data + w_data
    model_left = x_model
    model_right = x_model + w_model
    tasks_left = x_tasks
    tasks_right = x_tasks + w_tasks
    valid_left = x_valid

    conn_y = zone_y + zone_h * 0.55

    draw_arrow(ax, data_right, conn_y, model_left, conn_y, color='#5DADE2', lw=2.0)
    draw_rounded_rect(ax, (data_right + model_left) / 2 - 0.7, conn_y + 0.35,
                      1.4, 0.42, '#FFFFFF', '#5DADE2', lw=1.0, round_size=0.2)
    text_en_center(ax, (data_right + model_left) / 2, conn_y + 0.56, 'Training', fontsize=8, color='#2980B9', bold=True)

    n_tasks = 4
    for i in range(n_tasks):
        task_x = tasks_left + (i + 0.5) * (w_tasks / n_tasks)
        draw_arrow(ax, model_right, conn_y, task_x - 0.3, conn_y, color='#2ECC71', lw=1.3)

    draw_arrow(ax, tasks_right, conn_y, valid_left, conn_y, color='#F4D03F', lw=1.8)


def main():
    fig_width = 17
    fig_height = 7
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')

    zone_x_start = 0.5
    zone_y = 1.0
    zone_h = 6.5

    zone_w_data = 4.2
    zone_w_model = 4.8
    zone_w_tasks = 5.0
    zone_w_valid = 3.8

    x_data = zone_x_start
    x_model = x_data + zone_w_data + 0.2
    x_tasks = x_model + zone_w_model + 0.2
    x_valid = x_tasks + zone_w_tasks + 0.2

    draw_data_zone(ax, x_data, zone_y, zone_w_data, zone_h)
    draw_model_zone(ax, x_model, zone_y, zone_w_model, zone_h)
    draw_tasks_zone(ax, x_tasks, zone_y, zone_w_tasks, zone_h)
    draw_validation_zone(ax, x_valid, zone_y, zone_w_valid, zone_h)

    draw_connections(ax, x_data, x_model, x_tasks, x_valid,
                     zone_w_data, zone_w_model, zone_w_tasks, zone_w_valid,
                     zone_y, zone_h)

    fig.suptitle('KaryoFlow: Cross-Domain Application Framework',
                 fontsize=14, color=C_TEXT_PRIMARY, weight='bold', y=0.98)

    plt.tight_layout(pad=0.8)

    out_png = HERE / 'cross_domain_framework.png'
    out_pdf = HERE / 'cross_domain_framework.pdf'
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close(fig)

    print(f'Saved: {out_png}')
    print(f'Saved: {out_pdf}')


if __name__ == '__main__':
    main()
