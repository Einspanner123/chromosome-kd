"""Generate baseline-vs-SOTA comparison figures (EN + ZH).

Reads:  experiments/analysis/baseline_vs_sota_results.json
Outputs (docs/figures/):
  comparison_per_class_ap[_zh].png           — 24-class AP bar chart
  comparison_confusion_matrix[_zh].png       — 3-panel heatmap (DDPM | RF+Heun | Δ)
  comparison_pr_table[_zh].png               — aggregate P/R metrics table

Usage:
    python docs/figures/generate_comparison_figures.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ────────────────────────── Language helper ──────────────────────────
LANG = 'en'

def T(en: str, zh: str) -> str:
    return zh if LANG == 'zh' else en


def setup_rc(lang: str):
    global LANG
    LANG = lang
    if lang == 'zh':
        plt.rcParams.update({
            'font.family': 'SimHei',
            'font.size': 9,
            'axes.unicode_minus': False,
            'axes.titlesize': 11,
            'axes.titleweight': 'bold',
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 7.5,
            'savefig.dpi': 300,
            'figure.dpi': 110,
        })
    else:
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 9,
            'axes.titlesize': 11,
            'axes.titleweight': 'bold',
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 7.5,
            'savefig.dpi': 300,
            'figure.dpi': 110,
        })


# Academic palette
C_BASELINE = '#6B7280'   # gray — DDPM
C_SOTA = '#D97706'       # orange — RF+Heun
C_POS = '#059669'        # green — improvement
C_NEG = '#DC2626'        # red — regression
C_BG = '#F9FAFB'

# ────────────────────────── Data loading ──────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / 'experiments/analysis/baseline_vs_sota_results.json'
OUTPUT_DIR = Path(__file__).resolve().parent


def load_results() -> dict:
    if not RESULTS_PATH.exists():
        print(f'ERROR: Results file not found: {RESULTS_PATH}')
        print('Run first: python experiments/analysis/baseline_vs_sota.py')
        sys.exit(1)
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ────────────────────────── 1. Per-class AP bar chart ──────────────────────────

def plot_per_class_ap(data: dict, lang: str, output_path: Path):
    """24-class AP bar chart: DDPM vs RF+Heun (3-seed mean), with delta annotation"""
    setup_rc(lang)

    ddpm = data['models']['DDPM']['mean']
    sota = data['models']['RF+Heun']['mean']
    cat_names = ddpm.get('per_class_ap', {})
    # Use cat_names from the seeds structure (ordered)
    cat_names_list = data['models']['DDPM']['seeds'][list(data['models']['DDPM']['seeds'].keys())[0]]['cat_names']

    classes = cat_names_list
    x = np.arange(len(classes))
    width = 0.38

    ddpm_aps = [ddpm['per_class_ap'][c] for c in classes]
    sota_aps = [sota['per_class_ap'][c] for c in classes]

    # Also get per-seed values for error bars
    ddpm_seeds = data['models']['DDPM']['seeds']
    sota_seeds = data['models']['RF+Heun']['seeds']
    ddpm_std = np.array([[ddpm_seeds[s]['per_class_ap'][c] for s in ddpm_seeds] for c in classes]).std(axis=1)
    sota_std = np.array([[sota_seeds[s]['per_class_ap'][c] for s in sota_seeds] for c in classes]).std(axis=1)

    fig, ax = plt.subplots(figsize=(14, 5.5))

    bars1 = ax.bar(x - width/2, ddpm_aps, width, yerr=ddpm_std,
                   label='DDPM (baseline)', color=C_BASELINE, capsize=2,
                   edgecolor='white', linewidth=0.5, error_kw={'linewidth': 0.7})
    bars2 = ax.bar(x + width/2, sota_aps, width, yerr=sota_std,
                   label='RF+Heun (SOTA)', color=C_SOTA, capsize=2,
                   edgecolor='white', linewidth=0.5, error_kw={'linewidth': 0.7})

    # Delta annotations
    for i, (d, s) in enumerate(zip(ddpm_aps, sota_aps)):
        delta = s - d
        color = C_POS if delta >= 0 else C_NEG
        ymax = max(d + ddpm_std[i], s + sota_std[i])
        ax.annotate(f'{delta:+.3f}', xy=(i, ymax + 0.008),
                    ha='center', fontsize=6.5, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=7.5)
    ax.set_ylabel(T('AP (IoU=0.5:0.95)', 'AP (IoU=0.5:0.95)'))
    ax.set_title(T('Per-Class AP: DDPM vs RF+Heun  (3-seed mean ± std)',
                   '每类 AP 对比: DDPM vs RF+Heun  (3 seeds 均值 ± 标准差)'),
                 pad=12, loc='left', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, max(max(ddpm_aps), max(sota_aps)) * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # Mean mAP annotation
    ddpm_map = ddpm['aggregate']['mAP']
    sota_map = sota['aggregate']['mAP']
    ax.axhline(ddpm_map, color=C_BASELINE, linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(sota_map, color=C_SOTA, linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(len(classes) - 0.5, ddpm_map + 0.005, f'mAP={ddpm_map:.3f}',
            fontsize=7, color=C_BASELINE, ha='right', style='italic')
    ax.text(len(classes) - 0.5, sota_map + 0.005, f'mAP={sota_map:.3f}',
            fontsize=7, color=C_SOTA, ha='right', style='italic')

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ────────────────────────── 2. Confusion matrix heatmap ──────────────────────────

def plot_confusion_matrix(data: dict, lang: str, output_path: Path):
    """3-panel confusion matrix: DDPM (norm) | RF+Heun (norm) | Delta"""
    setup_rc(lang)

    ddpm = data['models']['DDPM']['mean']
    sota = data['models']['RF+Heun']['mean']
    cat_names = data['models']['DDPM']['seeds'][list(data['models']['DDPM']['seeds'].keys())[0]]['cat_names']
    num_classes = len(cat_names)

    # Row-normalized confusion matrices (recall per GT class)
    cm_ddpm = np.array(ddpm['confusion_matrix_norm'])[:num_classes, :num_classes]
    cm_sota = np.array(sota['confusion_matrix_norm'])[:num_classes, :num_classes]
    cm_delta = cm_sota - cm_ddpm

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5),
                             gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.35})

    labels = cat_names

    # Panel 1: DDPM
    im1 = axes[0].imshow(cm_ddpm, cmap='YlOrBr', vmin=0, vmax=1, aspect='equal')
    axes[0].set_title(T('DDPM (baseline)', 'DDPM (基线)'), pad=10, fontsize=11, fontweight='bold')
    axes[0].set_xticks(range(num_classes))
    axes[0].set_yticks(range(num_classes))
    axes[0].set_xticklabels(labels, fontsize=5.5, rotation=90)
    axes[0].set_yticklabels(labels, fontsize=5.5)
    axes[0].set_xlabel(T('Predicted', '预测类别'), fontsize=9)
    axes[0].set_ylabel(T('Ground Truth', '真实类别'), fontsize=9)

    # Annotate diagonal
    for i in range(num_classes):
        val = cm_ddpm[i, i]
        color = 'white' if val > 0.5 else 'black'
        axes[0].text(i, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=4.5, color=color, fontweight='bold')

    # Panel 2: RF+Heun
    im2 = axes[1].imshow(cm_sota, cmap='YlOrBr', vmin=0, vmax=1, aspect='equal')
    axes[1].set_title(T('RF+Heun (SOTA)', 'RF+Heun (SOTA)'), pad=10, fontsize=11, fontweight='bold')
    axes[1].set_xticks(range(num_classes))
    axes[1].set_yticks(range(num_classes))
    axes[1].set_xticklabels(labels, fontsize=5.5, rotation=90)
    axes[1].set_yticklabels(labels, fontsize=5.5)
    axes[1].set_xlabel(T('Predicted', '预测类别'), fontsize=9)

    for i in range(num_classes):
        val = cm_sota[i, i]
        color = 'white' if val > 0.5 else 'black'
        axes[1].text(i, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=4.5, color=color, fontweight='bold')

    # Panel 3: Delta (SOTA - baseline)
    max_abs = max(abs(cm_delta.min()), abs(cm_delta.max()), 0.01)
    im3 = axes[2].imshow(cm_delta, cmap='RdYlGn', vmin=-max_abs, vmax=max_abs, aspect='equal')
    axes[2].set_title(T('Δ (SOTA − baseline)', 'Δ (SOTA − 基线)'), pad=10, fontsize=11, fontweight='bold')
    axes[2].set_xticks(range(num_classes))
    axes[2].set_yticks(range(num_classes))
    axes[2].set_xticklabels(labels, fontsize=5.5, rotation=90)
    axes[2].set_yticklabels(labels, fontsize=5.5)
    axes[2].set_xlabel(T('Predicted', '预测类别'), fontsize=9)

    # Annotate significant deltas (off-diagonal changes > threshold)
    threshold = 0.01
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_delta[i, j]
            if abs(val) >= threshold:
                color = C_POS if val > 0 else C_NEG
                axes[2].text(j, i, f'{val:+.2f}', ha='center', va='center',
                             fontsize=3.8, color=color)

    # Colorbars
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label(T('Row-normalized', '行归一化'), fontsize=8)
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label(T('Row-normalized', '行归一化'), fontsize=8)
    cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar3.set_label(T('Δ recall', 'Δ recall'), fontsize=8)

    fig.suptitle(T('Confusion Matrix (IoU=0.5, score=0.3, 3-seed mean, row-normalized)',
                   '混淆矩阵 (IoU=0.5, score=0.3, 3 seeds 均值, 行归一化)'),
                 fontsize=13, fontweight='bold', y=1.01)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ────────────────────────── 3. Aggregate P/R table ──────────────────────────

def plot_pr_table(data: dict, lang: str, output_path: Path):
    """Aggregate precision/recall metrics table: COCO AP/AR + detection P/R/F1"""
    setup_rc(lang)

    ddpm = data['models']['DDPM']['mean']
    sota = data['models']['RF+Heun']['mean']

    # Build table rows: (metric, ddpm_value, sota_value, delta, higher_is_better)
    rows = []

    # COCO precision metrics
    coco_precision_metrics = [
        ('mAP (IoU=0.5:0.95)', 'mAP'),
        ('AP50', 'AP50'),
        ('AP75', 'AP75'),
        ('AP_s (small)', 'AP_s'),
        ('AP_m (medium)', 'AP_m'),
        ('AP_l (large)', 'AP_l'),
    ]
    for label, key in coco_precision_metrics:
        d_val = ddpm['aggregate'][key]
        s_val = sota['aggregate'][key]
        rows.append((T('Precision (COCO)', '准确率 (COCO)'), label, d_val, s_val, True))

    # COCO recall metrics
    coco_recall_metrics = [
        ('AR@100', 'AR@100'),
        ('AR_s (small)', 'AR_s'),
        ('AR_m (medium)', 'AR_m'),
        ('AR_l (large)', 'AR_l'),
    ]
    for label, key in coco_recall_metrics:
        d_val = ddpm['aggregate'][key]
        s_val = sota['aggregate'][key]
        rows.append((T('Recall (COCO)', '召回率 (COCO)'), label, d_val, s_val, True))

    # Detection P/R from confusion matrix
    det_metrics = [
        (T('Precision (det, IoU=0.5)', '检测准确率 (IoU=0.5)'), 'precision'),
        (T('Recall (det, IoU=0.5)', '检测召回率 (IoU=0.5)'), 'recall'),
        ('F1', 'f1'),
    ]
    for label, key in det_metrics:
        d_val = ddpm['detection_pr'][key]
        s_val = sota['detection_pr'][key]
        rows.append((T('Detection (confusion matrix)', '检测指标 (混淆矩阵)'), label, d_val, s_val, True))

    # Create figure
    n_rows = len(rows)
    # Group separators: after COCO precision (6), after COCO recall (4), det (3)
    fig, ax = plt.subplots(figsize=(12, 0.55 * n_rows + 2.5))
    ax.axis('off')

    # Title
    ax.set_title(T('Aggregate Precision / Recall: DDPM vs RF+Heun  (3-seed mean)',
                   '聚合 准确率 / 召回率: DDPM vs RF+Heun  (3 seeds 均值)'),
                 pad=16, fontsize=13, fontweight='bold', loc='left')

    # Table data
    col_labels = [
        T('Category', '类别'),
        T('Metric', '指标'),
        'DDPM\n(baseline)',
        'RF+Heun\n(SOTA)',
        T('Δ', 'Δ'),
    ]

    table_data = []
    cell_colors = []
    for cat, metric, d_val, s_val, hib in rows:
        delta = s_val - d_val
        delta_str = f'{delta:+.4f}'
        delta_color = C_POS if (delta > 0) else (C_NEG if delta < 0 else C_BASELINE)
        table_data.append([cat, metric, f'{d_val:.4f}', f'{s_val:.4f}', delta_str])
        cell_colors.append(['white', 'white', '#F3F4F6', '#FEF3C7', delta_color + '20'])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellColours=cell_colors,
                     loc='center', cellLoc='center',
                     colWidths=[0.18, 0.28, 0.16, 0.16, 0.10])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#1F2937')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9.5)
        cell.set_height(0.08)

    # Style delta cells with color text
    for i in range(1, n_rows + 1):
        d_val = rows[i-1][3]
        s_val = rows[i-1][4]
        delta = s_val - d_val
        delta_cell = table[i, 4]
        if delta > 0:
            delta_cell.set_text_props(color=C_POS, fontweight='bold')
        elif delta < 0:
            delta_cell.set_text_props(color=C_NEG, fontweight='bold')

        # Bold the SOTA value if better
        if s_val > d_val:
            table[i, 3].set_text_props(fontweight='bold', color=C_SOTA)
        elif d_val > s_val:
            table[i, 2].set_text_props(fontweight='bold', color=C_BASELINE)

    # Group separators (thicker lines)
    group_ends = [6, 10, n_rows]  # after precision, recall, det
    for ge in group_ends:
        for j in range(len(col_labels)):
            cell = table[ge, j]
            cell.visible_edges = 'BT'
            # Add bottom border
            from matplotlib.lines import Line2D
        # Draw a line
        y_pos = table[ge, 0].get_y() - table[ge, 0].get_height() / 2
        ax.axhline(y=y_pos, color='#D1D5DB', linewidth=1.5, clip_on=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ────────────────────────── 4. Per-class P/R from confusion matrix ──────────────────────────

def plot_per_class_pr(data: dict, lang: str, output_path: Path):
    """Per-class precision and recall (from confusion matrix) as grouped bar chart"""
    setup_rc(lang)

    ddpm = data['models']['DDPM']['mean']
    sota = data['models']['RF+Heun']['mean']
    cat_names = data['models']['DDPM']['seeds'][list(data['models']['DDPM']['seeds'].keys())[0]]['cat_names']

    num_classes = len(cat_names)
    x = np.arange(num_classes)
    width = 0.2

    ddpm_p = [ddpm['detection_pr_per_class'][str(i)]['precision'] for i in range(num_classes)]
    sota_p = [sota['detection_pr_per_class'][str(i)]['precision'] for i in range(num_classes)]
    ddpm_r = [ddpm['detection_pr_per_class'][str(i)]['recall'] for i in range(num_classes)]
    sota_r = [sota['detection_pr_per_class'][str(i)]['recall'] for i in range(num_classes)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Precision
    ax1.bar(x - width/2, ddpm_p, width, label='DDPM', color=C_BASELINE, edgecolor='white', linewidth=0.5)
    ax1.bar(x + width/2, sota_p, width, label='RF+Heun', color=C_SOTA, edgecolor='white', linewidth=0.5)
    ax1.set_ylabel(T('Precision', '准确率'))
    ax1.set_title(T('Per-Class Precision (IoU=0.5, score=0.3)',
                    '每类准确率 (IoU=0.5, score=0.3)'),
                  pad=8, fontsize=10, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax1.set_ylim(0, 1.05)

    # Recall
    ax2.bar(x - width/2, ddpm_r, width, label='DDPM', color=C_BASELINE, edgecolor='white', linewidth=0.5)
    ax2.bar(x + width/2, sota_r, width, label='RF+Heun', color=C_SOTA, edgecolor='white', linewidth=0.5)
    ax2.set_ylabel(T('Recall', '召回率'))
    ax2.set_title(T('Per-Class Recall (IoU=0.5, score=0.3)',
                    '每类召回率 (IoU=0.5, score=0.3)'),
                  pad=8, fontsize=10, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_names, fontsize=7.5)

    fig.suptitle(T('Per-Class Detection P/R: DDPM vs RF+Heun  (3-seed mean)',
                   '每类检测 P/R: DDPM vs RF+Heun  (3 seeds 均值)'),
                 fontsize=12, fontweight='bold', y=1.01)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ────────────────────────── Main ──────────────────────────

def main():
    print('=' * 60)
    print('Generate Baseline vs SOTA Comparison Figures')
    print('=' * 60)

    data = load_results()
    print(f'Loaded: {RESULTS_PATH}')

    for lang in ['en', 'zh']:
        suffix = '_zh' if lang == 'zh' else ''
        print(f'\n--- {lang.upper()} version ---')

        plot_per_class_ap(data, lang, OUTPUT_DIR / f'comparison_per_class_ap{suffix}.png')
        plot_confusion_matrix(data, lang, OUTPUT_DIR / f'comparison_confusion_matrix{suffix}.png')
        plot_pr_table(data, lang, OUTPUT_DIR / f'comparison_pr_table{suffix}.png')
        plot_per_class_pr(data, lang, OUTPUT_DIR / f'comparison_per_class_pr{suffix}.png')

    print(f'\nAll figures saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
