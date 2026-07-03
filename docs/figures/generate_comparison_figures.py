"""Generate baseline-vs-SOTA comparison figures (EN + ZH) — multi-dataset.

Reads:  experiments/analysis/baseline_vs_sota_results.json
Outputs (docs/figures/):
  comparison_per_class_ap[_zh].png           — chromo: 3-model AP bar (DDPM | RF+Heun | SOTA)
  comparison_confusion_matrix[_zh].png       — chromo: 3-panel heatmap (DDPM | SOTA | Δ)
  comparison_pr_table[_zh].png               — chromo: DDPM vs SOTA aggregate P/R
  comparison_per_class_pr[_zh].png           — chromo: DDPM vs SOTA per-class P/R
  24obj_per_class_ap[_zh].png                — 24obj: 2-model AP bar (DDPM | SOTA)
  24obj_confusion_matrix[_zh].png            — 24obj: 3-panel heatmap (DDPM | SOTA | Δ)
  24obj_pr_table[_zh].png                    — 24obj: DDPM vs SOTA aggregate P/R
  24obj_per_class_pr[_zh].png                — 24obj: DDPM vs SOTA per-class P/R
  per_dataset_summary_table[_zh].png         — cross-dataset aggregate metrics

Usage:
    python docs/figures/generate_comparison_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

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
C_BASELINE = '#6B7280'   # gray — DDPM / baseline
C_RFHEUN = '#2563EB'     # blue — RF+Heun
C_SOTA = '#D97706'       # orange — SOTA
C_POS = '#059669'        # green — improvement
C_NEG = '#DC2626'        # red — regression

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


def get_models(data: dict, ds_name: str) -> dict:
    if 'datasets' not in data or ds_name not in data['datasets']:
        return {}
    return data['datasets'][ds_name].get('models', {})


def get_mean(data: dict, ds_name: str, model_name: str) -> dict:
    models = get_models(data, ds_name)
    if model_name not in models:
        return {}
    return models[model_name].get('mean', {})


def get_seeds(data: dict, ds_name: str, model_name: str) -> dict:
    models = get_models(data, ds_name)
    if model_name not in models:
        return {}
    return models[model_name].get('seeds', {})


def get_cat_names(data: dict, ds_name: str, model_name: str) -> list:
    seeds = get_seeds(data, ds_name, model_name)
    if not seeds:
        return []
    first_seed = list(seeds.keys())[0]
    return seeds[first_seed].get('cat_names', [])


def get_std(data: dict, ds_name: str, model_name: str, metric_key: str,
            cat_names: list, sub_key: str = None) -> np.ndarray:
    """Get per-class std across seeds for a metric."""
    seeds = get_seeds(data, ds_name, model_name)
    if len(seeds) <= 1:
        return np.zeros(len(cat_names))
    vals = []
    for s in seeds:
        seed_data = seeds[s]
        if sub_key:
            row = [seed_data[metric_key][str(i)][sub_key] for i in range(len(cat_names))]
        else:
            row = [seed_data[metric_key][c] for c in cat_names]
        vals.append(row)
    # shape: (n_seeds, n_classes) -> std per class -> (n_classes,)
    return np.array(vals).std(axis=0)


# ────────────────────────── 1. Per-class AP bar chart ──────────────────────────

def plot_per_class_ap(data: dict, ds_name: str, lang: str,
                      baseline_key: str, sota_key: str,
                      output_path: Path, mid_key: str = None):
    """Per-class AP bar chart.

    3-bar (baseline | mid | sota) when mid_key provided; 2-bar otherwise.
    """
    setup_rc(lang)

    baseline = get_mean(data, ds_name, baseline_key)
    sota = get_mean(data, ds_name, sota_key)
    mid = get_mean(data, ds_name, mid_key) if mid_key else {}

    if not sota:
        print(f'  [SKIP] per_class_ap [{ds_name}]: missing {sota_key}')
        return

    classes = get_cat_names(data, ds_name, sota_key) or \
              (get_cat_names(data, ds_name, baseline_key) if baseline else []) or \
              (get_cat_names(data, ds_name, mid_key) if mid else [])
    if not classes:
        print(f'  [SKIP] per_class_ap [{ds_name}]: no cat_names')
        return

    x = np.arange(len(classes))
    sota_aps = [sota['per_class_ap'][c] for c in classes]
    sota_map = sota['aggregate']['mAP']

    has_mid = bool(mid) and mid_key
    has_baseline = bool(baseline)

    if has_mid:
        # 3-bar layout
        width = 0.25
        fig, ax = plt.subplots(figsize=(16, 5.5))

        base_aps = [baseline['per_class_ap'][c] for c in classes]
        mid_aps = [mid['per_class_ap'][c] for c in classes]
        base_std = get_std(data, ds_name, baseline_key, 'per_class_ap', classes)
        mid_std = get_std(data, ds_name, mid_key, 'per_class_ap', classes)
        sota_std = get_std(data, ds_name, sota_key, 'per_class_ap', classes)

        ax.bar(x - width, base_aps, width, yerr=base_std,
               label=f'{baseline_key} (baseline)', color=C_BASELINE, capsize=2,
               edgecolor='white', linewidth=0.5, error_kw={'linewidth': 0.7})
        ax.bar(x, mid_aps, width, yerr=mid_std,
               label=f'{mid_key}', color=C_RFHEUN, capsize=2,
               edgecolor='white', linewidth=0.5, error_kw={'linewidth': 0.7})
        ax.bar(x + width, sota_aps, width, yerr=sota_std,
               label=f'{sota_key} (SOTA)', color=C_SOTA, capsize=2,
               edgecolor='white', linewidth=0.5, error_kw={'linewidth': 0.7})

        # Delta annotations (sota - baseline, 展示完整提升)
        for i, (b, s) in enumerate(zip(base_aps, sota_aps)):
            delta = s - b
            color = C_POS if delta >= 0 else C_NEG
            ymax = max(b + base_std[i], s + sota_std[i])
            ax.annotate(f'{delta:+.3f}', xy=(i + width/2, ymax + 0.008),
                        ha='center', fontsize=5.5, color=color, fontweight='bold')

        mid_map = mid['aggregate']['mAP']
        ax.axhline(mid_map, color=C_RFHEUN, linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(len(classes) - 0.5, mid_map + 0.005, f'mAP={mid_map:.3f}',
                fontsize=7, color=C_RFHEUN, ha='right', style='italic')

        title = T(f'Per-Class AP on {ds_name}: {baseline_key} | {mid_key} | {sota_key}',
                  f'{ds_name} 每类 AP: {baseline_key} | {mid_key} | {sota_key}')

    elif has_baseline:
        # 2-bar layout
        width = 0.38
        fig, ax = plt.subplots(figsize=(14, 5.5))

        base_aps = [baseline['per_class_ap'][c] for c in classes]
        base_std = get_std(data, ds_name, baseline_key, 'per_class_ap', classes)
        sota_std = get_std(data, ds_name, sota_key, 'per_class_ap', classes)

        ax.bar(x - width/2, base_aps, width, yerr=base_std,
               label=f'{baseline_key} (baseline)', color=C_BASELINE, capsize=2,
               edgecolor='white', linewidth=0.5, error_kw={'linewidth': 0.7})
        ax.bar(x + width/2, sota_aps, width, yerr=sota_std,
               label=f'{sota_key} (SOTA)', color=C_SOTA, capsize=2,
               edgecolor='white', linewidth=0.5, error_kw={'linewidth': 0.7})

        for i, (b, s) in enumerate(zip(base_aps, sota_aps)):
            delta = s - b
            color = C_POS if delta >= 0 else C_NEG
            ymax = max(b + base_std[i], s + sota_std[i])
            ax.annotate(f'{delta:+.3f}', xy=(i, ymax + 0.008),
                        ha='center', fontsize=6.5, color=color, fontweight='bold')

        base_map = baseline['aggregate']['mAP']
        ax.axhline(base_map, color=C_BASELINE, linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(len(classes) - 0.5, base_map + 0.005, f'mAP={base_map:.3f}',
                fontsize=7, color=C_BASELINE, ha='right', style='italic')

        title = T(f'Per-Class AP on {ds_name}: {baseline_key} vs {sota_key}',
                  f'{ds_name} 每类 AP: {baseline_key} vs {sota_key}')
    else:
        # 1-bar (SOTA only)
        width = 0.6
        fig, ax = plt.subplots(figsize=(14, 5.5))
        sota_std = get_std(data, ds_name, sota_key, 'per_class_ap', classes)
        ax.bar(x, sota_aps, width, yerr=sota_std,
               label=f'{sota_key}', color=C_SOTA, capsize=2,
               edgecolor='white', linewidth=0.5, error_kw={'linewidth': 0.7})
        title = T(f'Per-Class AP on {ds_name} — {sota_key}',
                  f'{ds_name} 每类 AP — {sota_key}')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=7.5)
    ax.set_ylabel(T('AP (IoU=0.5:0.95)', 'AP (IoU=0.5:0.95)'))
    ax.set_title(title, pad=12, loc='left', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, max(sota_aps) * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    ax.axhline(sota_map, color=C_SOTA, linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(len(classes) - 0.5, sota_map + 0.005, f'mAP={sota_map:.3f}',
            fontsize=7, color=C_SOTA, ha='right', style='italic')

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ────────────────────────── 2. Confusion matrix heatmap ──────────────────────────

def plot_confusion_matrix(data: dict, ds_name: str, lang: str,
                          baseline_key: str, sota_key: str,
                          output_path: Path):
    """3-panel confusion matrix (baseline | sota | Δ)."""
    setup_rc(lang)

    baseline = get_mean(data, ds_name, baseline_key)
    sota = get_mean(data, ds_name, sota_key)
    cat_names = get_cat_names(data, ds_name, baseline_key) or \
                get_cat_names(data, ds_name, sota_key)

    if not baseline or not sota or not cat_names:
        print(f'  [SKIP] confusion_matrix [{ds_name}]: missing data')
        return

    num_classes = len(cat_names)
    labels = cat_names
    cm_base = np.array(baseline['confusion_matrix_norm'])[:num_classes, :num_classes]
    cm_sota = np.array(sota['confusion_matrix_norm'])[:num_classes, :num_classes]
    cm_delta = cm_sota - cm_base

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5),
                             gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.35})

    for ax, cm, title, cmap in [
        (axes[0], cm_base, f'{baseline_key} (baseline)', 'YlOrBr'),
        (axes[1], cm_sota, f'{sota_key} (SOTA)', 'YlOrBr'),
    ]:
        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1, aspect='equal')
        ax.set_title(title, pad=10, fontsize=11, fontweight='bold')
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(labels, fontsize=5.5, rotation=90)
        ax.set_yticklabels(labels, fontsize=5.5)
        ax.set_xlabel(T('Predicted', '预测类别'), fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel(T('Ground Truth', '真实类别'), fontsize=9)
        for i in range(num_classes):
            val = cm[i, i]
            color = 'white' if val > 0.5 else 'black'
            ax.text(i, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=4.5, color=color, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(T('Row-normalized', '行归一化'), fontsize=8)

    # Delta panel
    max_abs = max(abs(cm_delta.min()), abs(cm_delta.max()), 0.01)
    im3 = axes[2].imshow(cm_delta, cmap='RdYlGn', vmin=-max_abs, vmax=max_abs, aspect='equal')
    axes[2].set_title(T(f'Δ ({sota_key} − {baseline_key})',
                        f'Δ ({sota_key} − {baseline_key})'),
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

    cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar3.set_label(T('Δ recall', 'Δ recall'), fontsize=8)

    fig.suptitle(T(f'Confusion Matrix on {ds_name} (IoU=0.5, score=0.3, row-normalized)',
                   f'{ds_name} 混淆矩阵 (IoU=0.5, score=0.3, 行归一化)'),
                 fontsize=13, fontweight='bold', y=1.01)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ────────────────────────── 3. Aggregate P/R table ──────────────────────────

def plot_pr_table(data: dict, ds_name: str, lang: str,
                  baseline_key: str, sota_key: str,
                  output_path: Path):
    """Aggregate P/R metrics table: COCO AP/AR + detection P/R/F1."""
    setup_rc(lang)

    baseline = get_mean(data, ds_name, baseline_key)
    sota = get_mean(data, ds_name, sota_key)
    if not baseline or not sota:
        print(f'  [SKIP] pr_table [{ds_name}]: missing data')
        return

    rows = []
    coco_precision = [
        ('mAP (IoU=0.5:0.95)', 'mAP'),
        ('AP50', 'AP50'),
        ('AP75', 'AP75'),
        ('AP_s (small)', 'AP_s'),
        ('AP_m (medium)', 'AP_m'),
        ('AP_l (large)', 'AP_l'),
    ]
    for label, key in coco_precision:
        rows.append((T('Precision (COCO)', '准确率 (COCO)'), label,
                     baseline['aggregate'][key], sota['aggregate'][key]))

    coco_recall = [
        ('AR@100', 'AR@100'),
        ('AR_s (small)', 'AR_s'),
        ('AR_m (medium)', 'AR_m'),
        ('AR_l (large)', 'AR_l'),
    ]
    for label, key in coco_recall:
        rows.append((T('Recall (COCO)', '召回率 (COCO)'), label,
                     baseline['aggregate'][key], sota['aggregate'][key]))

    det_metrics = [
        (T('Precision (det, IoU=0.5)', '检测准确率 (IoU=0.5)'), 'precision'),
        (T('Recall (det, IoU=0.5)', '检测召回率 (IoU=0.5)'), 'recall'),
        ('F1', 'f1'),
    ]
    for label, key in det_metrics:
        rows.append((T('Detection (confusion matrix)', '检测指标 (混淆矩阵)'), label,
                     baseline['detection_pr'][key], sota['detection_pr'][key]))

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(12, 0.55 * n_rows + 2.5))
    ax.axis('off')

    ax.set_title(T(f'Aggregate P/R on {ds_name}: {baseline_key} vs {sota_key}',
                   f'{ds_name} 聚合 P/R: {baseline_key} vs {sota_key}'),
                 pad=16, fontsize=13, fontweight='bold', loc='left')

    col_labels = [
        T('Category', '类别'),
        T('Metric', '指标'),
        f'{baseline_key}\n(baseline)',
        f'{sota_key}\n(SOTA)',
        T('Δ', 'Δ'),
    ]

    table_data = []
    cell_colors = []
    for cat, metric, b_val, s_val in rows:
        delta = s_val - b_val
        delta_str = f'{delta:+.4f}'
        table_data.append([cat, metric, f'{b_val:.4f}', f'{s_val:.4f}', delta_str])
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
        b_val = rows[i-1][2]
        s_val = rows[i-1][3]
        delta = s_val - b_val
        delta_cell = table[i, 4]
        if delta > 0:
            delta_cell.set_text_props(color=C_POS, fontweight='bold')
        elif delta < 0:
            delta_cell.set_text_props(color=C_NEG, fontweight='bold')
        if s_val > b_val:
            table[i, 3].set_text_props(fontweight='bold', color=C_SOTA)
        elif b_val > s_val:
            table[i, 2].set_text_props(fontweight='bold', color=C_BASELINE)

    group_ends = [6, 10, n_rows]
    for ge in group_ends:
        y_pos = table[ge, 0].get_y() - table[ge, 0].get_height() / 2
        ax.axhline(y=y_pos, color='#D1D5DB', linewidth=1.5, clip_on=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ────────────────────────── 4. Per-class P/R ──────────────────────────

def plot_per_class_pr(data: dict, ds_name: str, lang: str,
                      baseline_key: str, sota_key: str,
                      output_path: Path):
    """Per-class precision and recall from confusion matrix."""
    setup_rc(lang)

    baseline = get_mean(data, ds_name, baseline_key)
    sota = get_mean(data, ds_name, sota_key)
    cat_names = get_cat_names(data, ds_name, baseline_key)
    if not baseline or not sota or not cat_names:
        print(f'  [SKIP] per_class_pr [{ds_name}]: missing data')
        return

    num_classes = len(cat_names)
    x = np.arange(num_classes)
    width = 0.2

    base_p = [baseline['detection_pr_per_class'][str(i)]['precision'] for i in range(num_classes)]
    sota_p = [sota['detection_pr_per_class'][str(i)]['precision'] for i in range(num_classes)]
    base_r = [baseline['detection_pr_per_class'][str(i)]['recall'] for i in range(num_classes)]
    sota_r = [sota['detection_pr_per_class'][str(i)]['recall'] for i in range(num_classes)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.bar(x - width/2, base_p, width, label=baseline_key, color=C_BASELINE,
            edgecolor='white', linewidth=0.5)
    ax1.bar(x + width/2, sota_p, width, label=sota_key, color=C_SOTA,
            edgecolor='white', linewidth=0.5)
    ax1.set_ylabel(T('Precision', '准确率'))
    ax1.set_title(T(f'Per-Class Precision on {ds_name} (IoU=0.5, score=0.3)',
                    f'{ds_name} 每类准确率 (IoU=0.5, score=0.3)'),
                  pad=8, fontsize=10, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax1.set_ylim(0, 1.05)

    ax2.bar(x - width/2, base_r, width, label=baseline_key, color=C_BASELINE,
            edgecolor='white', linewidth=0.5)
    ax2.bar(x + width/2, sota_r, width, label=sota_key, color=C_SOTA,
            edgecolor='white', linewidth=0.5)
    ax2.set_ylabel(T('Recall', '召回率'))
    ax2.set_title(T(f'Per-Class Recall on {ds_name} (IoU=0.5, score=0.3)',
                    f'{ds_name} 每类召回率 (IoU=0.5, score=0.3)'),
                  pad=8, fontsize=10, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_names, fontsize=7.5)

    fig.suptitle(T(f'Per-Class Detection P/R on {ds_name}: {baseline_key} vs {sota_key}',
                   f'{ds_name} 每类检测 P/R: {baseline_key} vs {sota_key}'),
                 fontsize=12, fontweight='bold', y=1.01)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ────────────────────────── 5. Cross-dataset summary table ──────────────────────────

def plot_per_dataset_summary_table(data: dict, lang: str, output_path: Path):
    """Cross-dataset aggregate metrics summary table."""
    setup_rc(lang)

    rows = []
    for ds_name in ['chromo', '24obj']:
        models = get_models(data, ds_name)
        for model_name in ['DDPM', 'RF+Heun', 'SOTA']:
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
        if model_name == 'DDPM':
            row_color = '#F3F4F6'
        elif model_name == 'RF+Heun':
            row_color = '#DBEAFE'
        else:
            row_color = '#FEF3C7'
        cell_colors.append([row_color] + ['white'] * len(vals))

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellColours=cell_colors,
                     loc='center', cellLoc='center',
                     colWidths=[0.18] + [0.07] * len(col_labels[1:]))

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#1F2937')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9.5)

    # Bold best mAP per dataset
    for ds_name in ['chromo', '24obj']:
        ds_rows = [(i, r) for i, r in enumerate(rows) if r[0] == ds_name]
        if not ds_rows:
            continue
        best_idx = max(ds_rows, key=lambda x: x[1][3])
        i_table = best_idx[0] + 1
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

    if 'datasets' in data:
        for ds_name, ds_data in data['datasets'].items():
            models = list(ds_data.get('models', {}).keys())
            print(f'  Dataset {ds_name}: {models}')

    for lang in ['en', 'zh']:
        suffix = '_zh' if lang == 'zh' else ''
        print(f'\n--- {lang.upper()} version ---')

        # chromo: 3-model AP (DDPM | RF+Heun | SOTA), rest DDPM vs SOTA (完整提升)
        plot_per_class_ap(data, 'chromo', lang, 'DDPM', 'SOTA',
                          OUTPUT_DIR / f'comparison_per_class_ap{suffix}.png',
                          mid_key='RF+Heun')
        plot_confusion_matrix(data, 'chromo', lang, 'DDPM', 'SOTA',
                              OUTPUT_DIR / f'comparison_confusion_matrix{suffix}.png')
        plot_pr_table(data, 'chromo', lang, 'DDPM', 'SOTA',
                      OUTPUT_DIR / f'comparison_pr_table{suffix}.png')
        plot_per_class_pr(data, 'chromo', lang, 'DDPM', 'SOTA',
                          OUTPUT_DIR / f'comparison_per_class_pr{suffix}.png')

        # 24obj: DDPM vs SOTA
        plot_per_class_ap(data, '24obj', lang, 'DDPM', 'SOTA',
                          OUTPUT_DIR / f'24obj_per_class_ap{suffix}.png')
        plot_confusion_matrix(data, '24obj', lang, 'DDPM', 'SOTA',
                              OUTPUT_DIR / f'24obj_confusion_matrix{suffix}.png')
        plot_pr_table(data, '24obj', lang, 'DDPM', 'SOTA',
                      OUTPUT_DIR / f'24obj_pr_table{suffix}.png')
        plot_per_class_pr(data, '24obj', lang, 'DDPM', 'SOTA',
                          OUTPUT_DIR / f'24obj_per_class_pr{suffix}.png')

        # Cross-dataset summary
        plot_per_dataset_summary_table(data, lang,
                                       OUTPUT_DIR / f'per_dataset_summary_table{suffix}.png')

    print(f'\nAll figures saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
