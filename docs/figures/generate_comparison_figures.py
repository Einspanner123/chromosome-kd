"""Generate baseline-vs-SOTA comparison figures (EN + ZH).

Reads:  experiments/analysis/baseline_vs_sota_results.json
Outputs (docs/figures/):
  comparison_per_class_ap[_zh].png           — 24-class AP bar chart (chromo)
  comparison_confusion_matrix[_zh].png       — 3-panel heatmap (DDPM | SOTA | Δ) on chromo
  comparison_pr_table[_zh].png               — aggregate P/R metrics table
  comparison_per_class_pr[_zh].png           — per-class Precision/Recall
  per_dataset_confusion_matrix[_zh].png      — cross-dataset confusion matrices
  per_dataset_summary_table[_zh].png         — cross-dataset aggregate metrics

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
            'font.family': 'Noto Sans CJK SC',
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
C_SOTA = '#D97706'       # orange — SOTA
C_24OBJ = '#2563EB'      # blue — 24obj dataset
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


def get_dataset_models(data: dict, ds_name: str) -> dict:
    """Extract models dict for a given dataset from new multi-dataset structure.

    Falls back to legacy single-dataset structure (data['models']) for backward compat.
    """
    if 'datasets' in data:
        if ds_name not in data['datasets']:
            return {}
        return data['datasets'][ds_name].get('models', {})
    # Legacy fallback: treat top-level as a single dataset
    if ds_name == 'chromo':
        return data.get('models', {})
    return {}


def get_model_mean(data: dict, ds_name: str, model_name: str) -> dict:
    """Get mean result for a model on a dataset."""
    models = get_dataset_models(data, ds_name)
    if model_name not in models:
        return {}
    return models[model_name].get('mean', {})


def get_model_seeds(data: dict, ds_name: str, model_name: str) -> dict:
    """Get per-seed results for a model on a dataset."""
    models = get_dataset_models(data, ds_name)
    if model_name not in models:
        return {}
    return models[model_name].get('seeds', {})


def get_cat_names(data: dict, ds_name: str, model_name: str) -> list:
    """Get ordered category names from a model's seed data."""
    seeds = get_model_seeds(data, ds_name, model_name)
    if not seeds:
        return []
    first_seed = list(seeds.keys())[0]
    return seeds[first_seed].get('cat_names', [])


# ────────────────────────── 1. Per-class AP bar chart ──────────────────────────

def plot_per_class_ap(data: dict, lang: str, output_path: Path):
    """24-class AP bar chart on chromo.

    2-bar (DDPM vs SOTA) when both available; 1-bar (SOTA) when only SOTA.
    """
    setup_rc(lang)

    # Try SOTA_0753 first, fall back to RF+Heun for legacy data
    sota_key = 'SOTA_0753'
    if sota_key not in get_dataset_models(data, 'chromo'):
        sota_key = 'RF+Heun'

    ddpm = get_model_mean(data, 'chromo', 'DDPM')
    sota = get_model_mean(data, 'chromo', sota_key)
    if not sota:
        print(f'  [SKIP] per_class_ap: missing {sota_key} on chromo')
        return

    classes = get_cat_names(data, 'chromo', 'DDPM') or get_cat_names(data, 'chromo', sota_key)
    if not classes:
        print('  [SKIP] per_class_ap: no cat_names')
        return

    x = np.arange(len(classes))
    sota_aps = [sota['per_class_ap'][c] for c in classes]
    sota_seeds = get_model_seeds(data, 'chromo', sota_key)
    sota_std = np.array([[sota_seeds[s]['per_class_ap'][c] for s in sota_seeds] for c in classes]).std(axis=1) if len(sota_seeds) > 1 else np.zeros(len(classes))
    sota_label = 'SOTA 0.753 (RF+Heun)' if sota_key == 'SOTA_0753' else 'RF+Heun (SOTA)'
    sota_map = sota['aggregate']['mAP']

    has_ddpm = bool(ddpm)

    fig, ax = plt.subplots(figsize=(14, 5.5))

    if has_ddpm:
        width = 0.38
        ddpm_aps = [ddpm['per_class_ap'][c] for c in classes]
        ddpm_seeds = get_model_seeds(data, 'chromo', 'DDPM')
        ddpm_std = np.array([[ddpm_seeds[s]['per_class_ap'][c] for s in ddpm_seeds] for c in classes]).std(axis=1) if len(ddpm_seeds) > 1 else np.zeros(len(classes))

        ax.bar(x - width/2, ddpm_aps, width, yerr=ddpm_std,
               label='DDPM (baseline)', color=C_BASELINE, capsize=2,
               edgecolor='white', linewidth=0.5, error_kw={'linewidth': 0.7})
        ax.bar(x + width/2, sota_aps, width, yerr=sota_std,
               label=sota_label, color=C_SOTA, capsize=2,
               edgecolor='white', linewidth=0.5, error_kw={'linewidth': 0.7})

        # Delta annotations
        for i, (d, s) in enumerate(zip(ddpm_aps, sota_aps)):
            delta = s - d
            color = C_POS if delta >= 0 else C_NEG
            ymax = max(d + ddpm_std[i], s + sota_std[i])
            ax.annotate(f'{delta:+.3f}', xy=(i, ymax + 0.008),
                        ha='center', fontsize=6.5, color=color, fontweight='bold')

        ddpm_map = ddpm['aggregate']['mAP']
        ax.axhline(ddpm_map, color=C_BASELINE, linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(len(classes) - 0.5, ddpm_map + 0.005, f'mAP={ddpm_map:.3f}',
                fontsize=7, color=C_BASELINE, ha='right', style='italic')
        title = T('Per-Class AP on Chromo: DDPM vs SOTA 0.753',
                  'Chromo 每类 AP 对比: DDPM vs SOTA 0.753')
    else:
        # SOTA-only: single bar
        width = 0.6
        ax.bar(x, sota_aps, width, yerr=sota_std,
               label=sota_label, color=C_SOTA, capsize=2,
               edgecolor='white', linewidth=0.5, error_kw={'linewidth': 0.7})
        title = T(f'Per-Class AP on Chromo — {sota_label}',
                  f'Chromo 每类 AP — {sota_label}')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=7.5)
    ax.set_ylabel(T('AP (IoU=0.5:0.95)', 'AP (IoU=0.5:0.95)'))
    ax.set_title(title, pad=12, loc='left', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, max(sota_aps) * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # Mean mAP annotation
    ax.axhline(sota_map, color=C_SOTA, linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(len(classes) - 0.5, sota_map + 0.005, f'mAP={sota_map:.3f}',
            fontsize=7, color=C_SOTA, ha='right', style='italic')

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ────────────────────────── 2. Confusion matrix heatmap (chromo) ──────────────────────────

def plot_confusion_matrix(data: dict, lang: str, output_path: Path):
    """Confusion matrix on chromo.

    3-panel (DDPM | SOTA | Δ) when both available; 1-panel (SOTA) when only SOTA.
    """
    setup_rc(lang)

    sota_key = 'SOTA_0753'
    if sota_key not in get_dataset_models(data, 'chromo'):
        sota_key = 'RF+Heun'

    ddpm = get_model_mean(data, 'chromo', 'DDPM')
    sota = get_model_mean(data, 'chromo', sota_key)
    cat_names = get_cat_names(data, 'chromo', 'DDPM') or get_cat_names(data, 'chromo', sota_key)

    if not sota or not cat_names:
        print(f'  [SKIP] confusion_matrix: missing SOTA data')
        return

    num_classes = len(cat_names)
    labels = cat_names
    sota_title = 'SOTA 0.753' if sota_key == 'SOTA_0753' else 'RF+Heun (SOTA)'
    cm_sota = np.array(sota['confusion_matrix_norm'])[:num_classes, :num_classes]

    has_ddpm = bool(ddpm)

    if has_ddpm:
        cm_ddpm = np.array(ddpm['confusion_matrix_norm'])[:num_classes, :num_classes]
        cm_delta = cm_sota - cm_ddpm

        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5),
                                 gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.35})

        # Panel 1: DDPM
        im1 = axes[0].imshow(cm_ddpm, cmap='YlOrBr', vmin=0, vmax=1, aspect='equal')
        axes[0].set_title(T('DDPM (baseline)', 'DDPM (基线)'), pad=10, fontsize=11, fontweight='bold')
        axes[0].set_xticks(range(num_classes))
        axes[0].set_yticks(range(num_classes))
        axes[0].set_xticklabels(labels, fontsize=5.5, rotation=90)
        axes[0].set_yticklabels(labels, fontsize=5.5)
        axes[0].set_xlabel(T('Predicted', '预测类别'), fontsize=9)
        axes[0].set_ylabel(T('Ground Truth', '真实类别'), fontsize=9)
        for i in range(num_classes):
            val = cm_ddpm[i, i]
            color = 'white' if val > 0.5 else 'black'
            axes[0].text(i, i, f'{val:.2f}', ha='center', va='center',
                         fontsize=4.5, color=color, fontweight='bold')

        # Panel 2: SOTA
        im2 = axes[1].imshow(cm_sota, cmap='YlOrBr', vmin=0, vmax=1, aspect='equal')
        axes[1].set_title(sota_title, pad=10, fontsize=11, fontweight='bold')
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

        # Panel 3: Delta
        max_abs = max(abs(cm_delta.min()), abs(cm_delta.max()), 0.01)
        im3 = axes[2].imshow(cm_delta, cmap='RdYlGn', vmin=-max_abs, vmax=max_abs, aspect='equal')
        axes[2].set_title(T(f'Δ ({sota_title} − DDPM)', f'Δ ({sota_title} − DDPM)'),
                          pad=10, fontsize=11, fontweight='bold')
        axes[2].set_xticks(range(num_classes))
        axes[2].set_yticks(range(num_classes))
        axes[2].set_xticklabels(labels, fontsize=5.5, rotation=90)
        axes[2].set_yticklabels(labels, fontsize=5.5)
        axes[2].set_xlabel(T('Predicted', '预测类别'), fontsize=9)

        threshold = 0.01
        for i in range(num_classes):
            for j in range(num_classes):
                val = cm_delta[i, j]
                if abs(val) >= threshold:
                    color = C_POS if val > 0 else C_NEG
                    axes[2].text(j, i, f'{val:+.2f}', ha='center', va='center',
                                 fontsize=3.8, color=color)

        cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.set_label(T('Row-normalized', '行归一化'), fontsize=8)
        cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.set_label(T('Row-normalized', '行归一化'), fontsize=8)
        cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        cbar3.set_label(T('Δ recall', 'Δ recall'), fontsize=8)

        fig.suptitle(T('Confusion Matrix on Chromo (IoU=0.5, score=0.3, row-normalized)',
                       'Chromo 混淆矩阵 (IoU=0.5, score=0.3, 行归一化)'),
                     fontsize=13, fontweight='bold', y=1.01)
    else:
        # SOTA-only: single panel
        fig, ax = plt.subplots(1, 1, figsize=(9, 8))
        im = ax.imshow(cm_sota, cmap='YlOrBr', vmin=0, vmax=1, aspect='equal')
        ax.set_title(sota_title, pad=10, fontsize=12, fontweight='bold')
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel(T('Predicted', '预测类别'), fontsize=10)
        ax.set_ylabel(T('Ground Truth', '真实类别'), fontsize=10)
        for i in range(num_classes):
            val = cm_sota[i, i]
            color = 'white' if val > 0.5 else 'black'
            ax.text(i, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6, color=color, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(T('Row-normalized', '行归一化'), fontsize=9)

        fig.suptitle(T(f'Confusion Matrix on Chromo — {sota_title} (IoU=0.5, score=0.3, row-normalized)',
                       f'Chromo 混淆矩阵 — {sota_title} (IoU=0.5, score=0.3, 行归一化)'),
                     fontsize=13, fontweight='bold', y=1.01)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ────────────────────────── 3. Aggregate P/R table ──────────────────────────

def plot_pr_table(data: dict, lang: str, output_path: Path):
    """Aggregate precision/recall metrics table on chromo: COCO AP/AR + detection P/R/F1"""
    setup_rc(lang)

    sota_key = 'SOTA_0753'
    if sota_key not in get_dataset_models(data, 'chromo'):
        sota_key = 'RF+Heun'

    ddpm = get_model_mean(data, 'chromo', 'DDPM')
    sota = get_model_mean(data, 'chromo', sota_key)
    if not ddpm or not sota:
        print('  [SKIP] pr_table: missing data')
        return

    rows = []
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

    det_metrics = [
        (T('Precision (det, IoU=0.5)', '检测准确率 (IoU=0.5)'), 'precision'),
        (T('Recall (det, IoU=0.5)', '检测召回率 (IoU=0.5)'), 'recall'),
        ('F1', 'f1'),
    ]
    for label, key in det_metrics:
        d_val = ddpm['detection_pr'][key]
        s_val = sota['detection_pr'][key]
        rows.append((T('Detection (confusion matrix)', '检测指标 (混淆矩阵)'), label, d_val, s_val, True))

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(12, 0.55 * n_rows + 2.5))
    ax.axis('off')

    sota_label = 'SOTA 0.753' if sota_key == 'SOTA_0753' else 'RF+Heun'
    ax.set_title(T(f'Aggregate P/R on Chromo: DDPM vs {sota_label}',
                   f'Chromo 聚合 P/R: DDPM vs {sota_label}'),
                 pad=16, fontsize=13, fontweight='bold', loc='left')

    col_labels = [
        T('Category', '类别'),
        T('Metric', '指标'),
        'DDPM\n(baseline)',
        f'{sota_label}\n(SOTA)',
        T('Δ', 'Δ'),
    ]

    table_data = []
    cell_colors = []
    for cat, metric, d_val, s_val, hib in rows:
        delta = s_val - d_val
        delta_str = f'{delta:+.4f}'
        table_data.append([cat, metric, f'{d_val:.4f}', f'{s_val:.4f}', delta_str])
        cell_colors.append(['white', 'white', '#F3F4F6', '#FEF3C7', 'white'])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellColours=cell_colors,
                     loc='center', cellLoc='center',
                     colWidths=[0.18, 0.28, 0.16, 0.16, 0.10])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#1F2937')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9.5)
        cell.set_height(0.08)

    for i in range(1, n_rows + 1):
        d_val = rows[i-1][3]
        s_val = rows[i-1][4]
        delta = s_val - d_val
        delta_cell = table[i, 4]
        if delta > 0:
            delta_cell.set_text_props(color=C_POS, fontweight='bold')
        elif delta < 0:
            delta_cell.set_text_props(color=C_NEG, fontweight='bold')

        if s_val > d_val:
            table[i, 3].set_text_props(fontweight='bold', color=C_SOTA)
        elif d_val > s_val:
            table[i, 2].set_text_props(fontweight='bold', color=C_BASELINE)

    group_ends = [6, 10, n_rows]
    for ge in group_ends:
        from matplotlib.lines import Line2D
        y_pos = table[ge, 0].get_y() - table[ge, 0].get_height() / 2
        ax.axhline(y=y_pos, color='#D1D5DB', linewidth=1.5, clip_on=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ────────────────────────── 4. Per-class P/R from confusion matrix ──────────────────────────

def plot_per_class_pr(data: dict, lang: str, output_path: Path):
    """Per-class precision and recall on chromo (from confusion matrix)"""
    setup_rc(lang)

    sota_key = 'SOTA_0753'
    if sota_key not in get_dataset_models(data, 'chromo'):
        sota_key = 'RF+Heun'

    ddpm = get_model_mean(data, 'chromo', 'DDPM')
    sota = get_model_mean(data, 'chromo', sota_key)
    cat_names = get_cat_names(data, 'chromo', 'DDPM')
    if not ddpm or not sota or not cat_names:
        print('  [SKIP] per_class_pr: missing data')
        return

    num_classes = len(cat_names)
    x = np.arange(num_classes)
    width = 0.2

    ddpm_p = [ddpm['detection_pr_per_class'][str(i)]['precision'] for i in range(num_classes)]
    sota_p = [sota['detection_pr_per_class'][str(i)]['precision'] for i in range(num_classes)]
    ddpm_r = [ddpm['detection_pr_per_class'][str(i)]['recall'] for i in range(num_classes)]
    sota_r = [sota['detection_pr_per_class'][str(i)]['recall'] for i in range(num_classes)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.bar(x - width/2, ddpm_p, width, label='DDPM', color=C_BASELINE, edgecolor='white', linewidth=0.5)
    sota_label = 'SOTA 0.753' if sota_key == 'SOTA_0753' else 'RF+Heun'
    ax1.bar(x + width/2, sota_p, width, label=sota_label, color=C_SOTA, edgecolor='white', linewidth=0.5)
    ax1.set_ylabel(T('Precision', '准确率'))
    ax1.set_title(T('Per-Class Precision on Chromo (IoU=0.5, score=0.3)',
                    'Chromo 每类准确率 (IoU=0.5, score=0.3)'),
                  pad=8, fontsize=10, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax1.set_ylim(0, 1.05)

    ax2.bar(x - width/2, ddpm_r, width, label='DDPM', color=C_BASELINE, edgecolor='white', linewidth=0.5)
    ax2.bar(x + width/2, sota_r, width, label=sota_label, color=C_SOTA, edgecolor='white', linewidth=0.5)
    ax2.set_ylabel(T('Recall', '召回率'))
    ax2.set_title(T('Per-Class Recall on Chromo (IoU=0.5, score=0.3)',
                    'Chromo 每类召回率 (IoU=0.5, score=0.3)'),
                  pad=8, fontsize=10, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_names, fontsize=7.5)

    sota_full = 'SOTA 0.753' if sota_key == 'SOTA_0753' else 'RF+Heun'
    fig.suptitle(T(f'Per-Class Detection P/R on Chromo: DDPM vs {sota_full}',
                   f'Chromo 每类检测 P/R: DDPM vs {sota_full}'),
                 fontsize=12, fontweight='bold', y=1.01)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ────────────────────────── 5. Per-dataset confusion matrix ──────────────────────────

def plot_per_dataset_confusion_matrix(data: dict, lang: str, output_path: Path):
    """Cross-dataset confusion matrix comparison.

    Panels: chromo-DDPM | chromo-SOTA | 24obj-DDPM  (row-normalized)
    Skips panels for models with no data.
    """
    setup_rc(lang)

    # Collect available (title, mean_result) pairs in order
    panels = []

    chromo_models = get_dataset_models(data, 'chromo')
    if 'DDPM' in chromo_models and chromo_models['DDPM']['mean']:
        panels.append((T('Chromo · DDPM', 'Chromo · DDPM'), chromo_models['DDPM']['mean'], C_BASELINE))
    sota_key = 'SOTA_0753' if 'SOTA_0753' in chromo_models else 'RF+Heun'
    if sota_key in chromo_models and chromo_models[sota_key]['mean']:
        sota_title = 'Chromo · SOTA 0.753' if sota_key == 'SOTA_0753' else 'Chromo · RF+Heun'
        panels.append((T(sota_title, sota_title), chromo_models[sota_key]['mean'], C_SOTA))

    obj_models = get_dataset_models(data, '24obj')
    if 'DDPM' in obj_models and obj_models['DDPM']['mean']:
        panels.append((T('24obj · DDPM', '24obj · DDPM'), obj_models['DDPM']['mean'], C_24OBJ))
    sota_24_key = 'SOTA_0753' if 'SOTA_0753' in obj_models else 'RF+Heun'
    if sota_24_key in obj_models and obj_models[sota_24_key]['mean']:
        panels.append((T('24obj · SOTA', '24obj · SOTA'), obj_models[sota_24_key]['mean'], C_SOTA))

    if not panels:
        print('  [SKIP] per_dataset_confusion_matrix: no data')
        return

    n_panels = len(panels)
    cat_names = panels[0][1].get('per_class_ap', {}).keys() if panels else []
    # Use cat_names from the first panel's confusion matrix
    # Need to fetch from seeds
    for title, mean_result, _ in panels:
        # Try to infer num_classes from confusion_matrix_norm shape
        cm = np.array(mean_result['confusion_matrix_norm'])
        num_classes = cm.shape[0] - 1  # last row/col is background
        break

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6.5),
                             gridspec_kw={'wspace': 0.35} if n_panels > 1 else {})
    if n_panels == 1:
        axes = [axes]

    for ax, (title, mean_result, accent_color) in zip(axes, panels):
        cm = np.array(mean_result['confusion_matrix_norm'])[:num_classes, :num_classes]
        im = ax.imshow(cm, cmap='YlOrBr', vmin=0, vmax=1, aspect='equal')
        ax.set_title(title, pad=10, fontsize=11, fontweight='bold')
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))

        # Use class names from the GT categories (try chromo first, then 24obj)
        ds_for_names = 'chromo' if 'Chromo' in title or 'chromo' in title.lower() else '24obj'
        model_for_names = 'DDPM' if 'DDPM' in title else sota_key
        names = get_cat_names(data, ds_for_names, model_for_names)
        if names:
            ax.set_xticklabels(names, fontsize=5.5, rotation=90)
            ax.set_yticklabels(names, fontsize=5.5)
        ax.set_xlabel(T('Predicted', '预测类别'), fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel(T('Ground Truth', '真实类别'), fontsize=9)

        # Annotate diagonal
        for i in range(num_classes):
            val = cm[i, i]
            color = 'white' if val > 0.5 else 'black'
            ax.text(i, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=4.5, color=color, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(T('Row-normalized', '行归一化'), fontsize=8)

    fig.suptitle(T('Per-Dataset Confusion Matrix (IoU=0.5, score=0.3, row-normalized)',
                   '各数据集混淆矩阵 (IoU=0.5, score=0.3, 行归一化)'),
                 fontsize=13, fontweight='bold', y=1.01)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ────────────────────────── 6. Per-dataset summary table ──────────────────────────

def plot_per_dataset_summary_table(data: dict, lang: str, output_path: Path):
    """Cross-dataset aggregate metrics summary table.

    Rows: (dataset, model) pairs for all available configurations.
    Columns: mAP, AP50, AP75, AP_s, AP_m, AP_l, AR@100, Det P, Det R, Det F1.
    """
    setup_rc(lang)

    rows = []
    for ds_name in ['chromo', '24obj']:
        models = get_dataset_models(data, ds_name)
        for model_name in ['DDPM', 'SOTA_0753', 'RF+Heun']:
            if model_name not in models:
                continue
            mean = models[model_name].get('mean', {})
            if not mean:
                continue
            agg = mean['aggregate']
            pr = mean['detection_pr']
            n_seeds = mean.get('n_seeds', 0)
            label = f'{ds_name} · {model_name}' + (f' (n={n_seeds})' if n_seeds > 1 else '')
            rows.append((
                ds_name, model_name, label,
                agg['mAP'], agg['AP50'], agg['AP75'],
                agg['AP_s'], agg['AP_m'], agg['AP_l'],
                agg['AR@100'],
                pr['precision'], pr['recall'], pr['f1'],
            ))

    if not rows:
        print('  [SKIP] per_dataset_summary_table: no data')
        return

    col_labels = [
        T('Dataset · Model', '数据集 · 模型'),
        'mAP', 'AP50', 'AP75',
        'AP_s', 'AP_m', 'AP_l',
        'AR@100',
        T('Det P', '检测 P'), T('Det R', '检测 R'), 'F1',
    ]

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(14, 0.55 * (n_rows + 2) + 2))
    ax.axis('off')

    ax.set_title(T('Per-Dataset Aggregate Metrics Summary',
                   '各数据集聚合指标汇总'),
                 pad=16, fontsize=13, fontweight='bold', loc='left')

    table_data = []
    cell_colors = []
    for r in rows:
        ds_name, model_name, label = r[0], r[1], r[2]
        vals = r[3:]
        table_data.append([label] + [f'{v:.4f}' for v in vals])
        # Color by model
        if model_name == 'DDPM':
            row_color = '#F3F4F6'
        elif model_name == 'SOTA_0753':
            row_color = '#FEF3C7'
        else:
            row_color = '#FCE7F3'
        cell_colors.append([row_color] + ['white'] * len(vals))

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellColours=cell_colors,
                     loc='center', cellLoc='center',
                     colWidths=[0.18] + [0.07] * len(col_labels[1:]))

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#1F2937')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9.5)

    # Bold best mAP per dataset
    for ds_name in ['chromo', '24obj']:
        ds_rows = [(i, r) for i, r in enumerate(rows) if r[0] == ds_name]
        if not ds_rows:
            continue
        best_idx = max(ds_rows, key=lambda x: x[1][3])  # x[1][3] = mAP
        i_table = best_idx[0] + 1  # +1 for header
        table[i_table, 1].set_text_props(fontweight='bold', color=C_POS)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ────────────────────────── Main ──────────────────────────

def main():
    print('=' * 60)
    print('Generate Baseline vs SOTA Comparison Figures (multi-dataset)')
    print('=' * 60)

    data = load_results()
    print(f'Loaded: {RESULTS_PATH}')

    # Print available datasets/models
    if 'datasets' in data:
        for ds_name, ds_data in data['datasets'].items():
            models = list(ds_data.get('models', {}).keys())
            print(f'  Dataset {ds_name}: {models}')

    for lang in ['en', 'zh']:
        suffix = '_zh' if lang == 'zh' else ''
        print(f'\n--- {lang.upper()} version ---')

        # Chromo comparison figures (DDPM vs SOTA)
        plot_per_class_ap(data, lang, OUTPUT_DIR / f'comparison_per_class_ap{suffix}.png')
        plot_confusion_matrix(data, lang, OUTPUT_DIR / f'comparison_confusion_matrix{suffix}.png')
        plot_pr_table(data, lang, OUTPUT_DIR / f'comparison_pr_table{suffix}.png')
        plot_per_class_pr(data, lang, OUTPUT_DIR / f'comparison_per_class_pr{suffix}.png')

        # Cross-dataset figures
        plot_per_dataset_confusion_matrix(data, lang,
                                          OUTPUT_DIR / f'per_dataset_confusion_matrix{suffix}.png')
        plot_per_dataset_summary_table(data, lang,
                                       OUTPUT_DIR / f'per_dataset_summary_table{suffix}.png')

    print(f'\nAll figures saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
