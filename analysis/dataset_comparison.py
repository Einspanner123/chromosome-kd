#!/usr/bin/env python3
"""
Comprehensive multi-dimensional comparison of all registered datasets.

Datasets are defined in datasets.py (DATASETS dict).

Outputs: analysis/comparison/figs/*.png
"""

import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from datasets import DATASETS, get_dataset_path, get_splits

# ── Config ──────────────────────────────────────────────────────────
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.0)

# Custom colormap — minimal, academic aesthetic
COLORS = {
    'ds1': '#4C72B0',     # muted blue
    'ds2': '#DD8452',     # muted orange
    'ds1_light': '#8CB5E0',
    'ds2_light': '#E8B87A',
    'accent': '#55A868',
    'accent2': '#C44E52',
    'neutral': '#8172B2',
    'bg': '#F5F5F5',
}
DPI = 150
FIGS_DIR = Path(__file__).parent / 'comparison' / 'figs'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# COCO size thresholds (pixels²)
S_THRESH = 32 ** 2    # 1024
L_THRESH = 96 ** 2    # 9216


# ── Data Loading ────────────────────────────────────────────────────
def load_coco_annotations(data_dir: Path, splits_list: list):
    """Load COCO-style annotations from the given splits."""
    all_images = []
    all_anns = []
    all_categories = []
    for split in splits_list:
        json_path = data_dir / split / '_annotations.coco.json'
        if not json_path.exists():
            print(f'  [WARN] {json_path} not found, skipping')
            continue
        with open(json_path) as f:
            data = json.load(f)
        # Add split info
        for img in data.get('images', []):
            img['split'] = split
        all_images.extend(data.get('images', []))
        all_anns.extend(data.get('annotations', []))
        if not all_categories and 'categories' in data:
            all_categories = data.get('categories', [])

    return all_images, all_anns, all_categories


def build_dataframes():
    """Load all datasets and return DataFrames for analysis."""
    dfs = {}
    for ds_name in DATASETS:
        ds_path = get_dataset_path(ds_name)
        splits_list = get_splits(ds_name)
        print(f'Loading {ds_name}...')
        images, annotations, categories = load_coco_annotations(ds_path, splits_list)

        cat_map = {c['id']: c['name'] for c in categories}
        img_map = {img['id']: img for img in images}

        # ── Image-level DataFrame ──
        img_rows = []
        for img in images:
            img_rows.append({
                'img_id': img['id'],
                'file_name': img['file_name'],
                'width': img['width'],
                'height': img['height'],
                'split': img['split'],
                'dataset': ds_name,
            })
        df_img = pd.DataFrame(img_rows)

        # ── Annotation-level DataFrame ──
        ann_rows = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            area = w * h
            if area < S_THRESH:
                size_class = 'S'
            elif area < L_THRESH:
                size_class = 'M'
            else:
                size_class = 'L'

            img = img_map.get(ann['image_id'], {})
            ann_rows.append({
                'img_id': ann['image_id'],
                'ann_id': ann['id'],
                'category_id': ann['category_id'],
                'category': cat_map.get(ann['category_id'], f'cat_{ann["category_id"]}'),
                'bbox_x': x,
                'bbox_y': y,
                'bbox_w': w,
                'bbox_h': h,
                'area': area,
                'cx': x + w / 2,
                'cy': y + h / 2,
                'size_class': size_class,
                'aspect_ratio': w / h if h > 0 else np.nan,
                'dataset': ds_name,
                'split': img.get('split', 'unknown'),
                'img_width': img.get('width', np.nan),
                'img_height': img.get('height', np.nan),
            })
        df_ann = pd.DataFrame(ann_rows)

        # ── Per-image summary DataFrame ──
        per_img = df_ann.groupby('img_id').agg(
            num_objects=('ann_id', 'count'),
            mean_area=('area', 'mean'),
            mean_ar=('aspect_ratio', 'mean'),
        ).reset_index()
        per_img['dataset'] = ds_name
        # Merge with image info
        per_img = per_img.merge(df_img[['img_id', 'width', 'height', 'split']], on='img_id')
        per_img['img_area'] = per_img['width'] * per_img['height']
        per_img['density'] = per_img['num_objects'] / per_img['img_area'] * 1e6  # objects per Mpixel

        dfs[ds_name] = {
            'df_img': df_img,
            'df_ann': df_ann,
            'df_per_img': per_img,
        }
        print(f'  {ds_name}: {len(df_img)} images, {len(df_ann)} annotations')
    return dfs


# ── Plotting Functions ──────────────────────────────────────────────

def plot_category_distribution(dfs):
    """Fig 1: Stacked bar of category counts for both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)

    for ax, (ds_name, d) in zip(axes, dfs.items()):
        cat_counts = d['df_ann']['category'].value_counts().sort_index()
        # Drop the redundant prefix by keeping the chromosome name only
        colors = [COLORS['ds1'] if ds_name == list(dfs.keys())[0] else COLORS['ds2']] * len(cat_counts)
        bars = ax.bar(range(len(cat_counts)), cat_counts.values, color=colors[0], alpha=0.85,
                      edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(cat_counts)))
        ax.set_xticklabels(cat_counts.index, rotation=45, ha='right', fontsize=9)
        ax.set_title(ds_name, fontsize=14, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel('Category', fontsize=12)
        # Value labels on top
        for i, (_, v) in enumerate(cat_counts.items()):
            ax.text(i, v + max(cat_counts.values) * 0.01, str(v),
                    ha='center', va='bottom', fontsize=7, rotation=0)

    # Also add a combined side-by-side bar
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ds_names = list(dfs.keys())
    cat_set = sorted(set(dfs[ds_names[0]]['df_ann']['category'].unique()) |
                     set(dfs[ds_names[1]]['df_ann']['category'].unique()))
    x = np.arange(len(cat_set))
    w = 0.35
    for i, ds_name in enumerate(ds_names):
        counts = dfs[ds_name]['df_ann']['category'].value_counts()
        vals = [counts.get(c, 0) for c in cat_set]
        ax2.bar(x + i * w, vals, w, label=ds_name,
                color=[COLORS['ds1'], COLORS['ds2']][i], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax2.set_xticks(x + w / 2)
    ax2.set_xticklabels(cat_set, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_xlabel('Category', fontsize=12)
    ax2.set_title('Category Distribution — Side-by-Side Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGS_DIR / '01_category_distribution.png', dpi=DPI, bbox_inches='tight')
    fig2.tight_layout()
    fig2.savefig(FIGS_DIR / '01b_category_distribution_side_by_side.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    plt.close(fig2)
    print('  Saved 01_category_distribution*.png')


def plot_objects_per_image(dfs):
    """Fig 2: Histogram of objects per image."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = [COLORS['ds1'], COLORS['ds2']]
    ds_names = list(dfs.keys())

    for ax, (ds_name, d), color in zip(axes, dfs.items(), colors):
        data = d['df_per_img']['num_objects']
        ax.hist(data, bins=60, color=color, alpha=0.7, edgecolor='white', linewidth=0.5, density=True)
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='#C44E52', ls='--', lw=1.5, label=f'Mean={mean_val:.1f}')
        ax.axvline(median_val, color='#55A868', ls=':', lw=1.5, label=f'Median={median_val:.0f}')
        ax.set_xlabel('Objects per Image', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{ds_name}\n(Total: {len(data)} images)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(PercentFormatter(1))

    fig.tight_layout()
    fig.savefig(FIGS_DIR / '02_objects_per_image.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('  Saved 02_objects_per_image.png')


def plot_size_class_distribution(dfs):
    """Fig 3: S/M/L size class distribution (COCO thresholds)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = [COLORS['ds1'], COLORS['ds2']]
    order = ['S', 'M', 'L']

    for ax, (ds_name, d), color in zip(axes, dfs.items(), colors):
        counts = d['df_ann']['size_class'].value_counts()
        vals = [counts.get(s, 0) for s in order]
        total = sum(vals)
        pcts = [v / total * 100 for v in vals]
        bars = ax.bar(order, vals, color=[color, color, color], alpha=0.75,
                      edgecolor='white', linewidth=0.5)
        # Gradient fade for size classes
        bars[0].set_alpha(0.9)
        bars[1].set_alpha(0.65)
        bars[2].set_alpha(0.4)

        # Labels: y-offset = 5% of max bar above the tallest bar
        y_max = max(vals)
        y_offset = y_max * 0.06
        for i, (v, p) in enumerate(zip(vals, pcts)):
            ax.text(i, v + y_offset, f'{v:,}\n({p:.1f}%)',
                    ha='center', va='bottom', fontsize=10)
        # Push y-limit further so labels don't clip
        ax.set_ylim(top=y_max + y_offset * 3.5)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel('Size Class', fontsize=12)
        ax.set_title(f'{ds_name} — S/M/L Distribution\n(S: <32², M: 32²–96², L: >96²)',
                     fontsize=12, fontweight='bold')

    # Combined pie
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (ds_name, d), color in zip(axes2, dfs.items(), colors):
        counts = d['df_ann']['size_class'].value_counts()
        vals = [counts.get(s, 0) for s in order]
        colors_pie = ['#55A868', '#4C72B0', '#C44E52']  # green, blue, red
        wedges, texts, autotexts = ax.pie(
            vals, labels=order, autopct='%1.1f%%',
            colors=colors_pie, startangle=90,
            textprops={'fontsize': 11},
            pctdistance=0.75,
        )
        for t in autotexts:
            t.set_fontweight('bold')
        ax.set_title(f'{ds_name}', fontsize=12, fontweight='bold')

    fig.tight_layout()
    fig.savefig(FIGS_DIR / '03_size_class_distribution.png', dpi=DPI, bbox_inches='tight')
    fig2.tight_layout()
    fig2.savefig(FIGS_DIR / '03b_size_class_pie.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    plt.close(fig2)
    print('  Saved 03_size_class_distribution*.png')


def plot_size_scatter(dfs):
    """Fig 3c: BBox width vs height scatter colored by S/M/L size class."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    order = ['S', 'M', 'L']
    scatter_colors = {'S': '#55A868', 'M': '#4C72B0', 'L': '#C44E52'}
    scatter_alphas = {'S': 0.3, 'M': 0.15, 'L': 0.2}

    for ax, (ds_name, d) in zip(axes, dfs.items()):
        sub = d['df_ann'][['bbox_w', 'bbox_h', 'size_class']].copy()
        # Clip extreme values for visual clarity
        w_max, h_max = sub['bbox_w'].quantile(0.99), sub['bbox_h'].quantile(0.99)
        sub = sub[(sub['bbox_w'] <= w_max) & (sub['bbox_h'] <= h_max)]

        for sc in order:
            mask = sub['size_class'] == sc
            pts = sub[mask]
            # Sample: cap large classes for performance
            n = min(15000, len(pts))
            idx = np.random.choice(len(pts), n, replace=False) if len(pts) > n else slice(None)
            pts = pts.iloc[idx] if isinstance(idx, np.ndarray) else pts
            ax.scatter(
                pts['bbox_w'], pts['bbox_h'],
                s=3, c=scatter_colors[sc], alpha=scatter_alphas[sc],
                label=f'{sc} (n={len(pts):,})', rasterized=True,
                edgecolors='none',
            )

        # Iso-area boundary curves
        w_range = np.linspace(4, max(sub['bbox_w'].max(), 300), 500)
        ax.plot(w_range, S_THRESH / w_range, 'k--', lw=0.8, alpha=0.5,
                label=f'S boundary (area={S_THRESH})')
        ax.plot(w_range, L_THRESH / w_range, 'k:', lw=0.8, alpha=0.5,
                label=f'L boundary (area={L_THRESH})')

        ax.set_xlabel('BBox Width (pixels)', fontsize=12)
        ax.set_ylabel('BBox Height (pixels)', fontsize=12)
        ax.set_title(f'{ds_name}\n99th percentile cap', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, markerscale=4, loc='upper left')
        ax.set_xlim(0, w_max * 1.05)
        ax.set_ylim(0, h_max * 1.05)
        ax.set_aspect('equal')
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIGS_DIR / '03c_size_scatter.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('  Saved 03c_size_scatter.png')


def plot_bbox_area_distribution(dfs):
    """Fig 4: BBox area histogram + violins."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = [COLORS['ds1'], COLORS['ds2']]

    for ax, (ds_name, d), color in zip(axes, dfs.items(), colors):
        areas = d['df_ann']['area'].values
        ax.hist(areas, bins=80, color=color, alpha=0.7, edgecolor='white', linewidth=0.3)
        ax.axvline(np.median(areas), color='#C44E52', ls='--', lw=1.5, label=f'Median: {np.median(areas):.0f}')
        ax.axvline(np.mean(areas), color='#55A868', ls=':', lw=1.5, label=f'Mean: {np.mean(areas):.0f}')
        ax.set_xlabel('BBox Area (pixels²)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{ds_name}\n(min={areas.min():.0f}, max={areas.max():.0f})',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        # Inset log-scale
        ins = ax.inset_axes([0.6, 0.55, 0.35, 0.35])
        ins.hist(areas, bins=60, color=color, alpha=0.7, edgecolor='white', linewidth=0.3)
        ins.set_xscale('log')
        ins.set_yscale('log')
        ins.set_title('log-log', fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGS_DIR / '04_bbox_area_distribution.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    # Combined boxen plot
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    plot_data = []
    plot_labels = []
    for ds_name, d in dfs.items():
        plot_data.append(d['df_ann']['area'].values)
        plot_labels.append(ds_name)
    bp = ax2.boxplot(plot_data, labels=plot_labels, patch_artist=True, widths=0.5,
                      showfliers=False, showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='#C44E52', markersize=6))
    for patch, color in zip(bp['boxes'], [COLORS['ds1'], COLORS['ds2']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel('BBox Area (pixels²)', fontsize=12)
    ax2.set_title('BBox Area Distribution — Box Plot (outliers hidden)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIGS_DIR / '04b_bbox_area_boxplot.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig2)
    print('  Saved 04_bbox_area_distribution*.png')


def plot_bbox_aspect_ratio(dfs):
    """Fig 5: Aspect ratio distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = [COLORS['ds1'], COLORS['ds2']]

    for ax, (ds_name, d), color in zip(axes, dfs.items(), colors):
        ar = d['df_ann']['aspect_ratio'].dropna().values
        ar = np.clip(ar, 0.1, 10)  # clip extreme values
        ax.hist(ar, bins=80, color=color, alpha=0.7, edgecolor='white', linewidth=0.3)
        ax.axvline(np.median(ar), color='#C44E52', ls='--', lw=1.5, label=f'Median: {np.median(ar):.2f}')
        ax.set_xlabel('Aspect Ratio (W/H)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{ds_name}\n(n={len(ar):,})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGS_DIR / '05_aspect_ratio_distribution.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('  Saved 05_aspect_ratio_distribution.png')


def plot_center_scatter(dfs):
    """Fig 6: BBox center scatter plot — shows spatial distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = [COLORS['ds1'], COLORS['ds2']]

    for ax, (ds_name, d), color in zip(axes, dfs.items(), colors):
        # Normalize centers to [0, 1] relative to image dimensions
        cx_norm = d['df_ann']['cx'] / d['df_ann']['img_width']
        cy_norm = d['df_ann']['cy'] / d['df_ann']['img_height']

        # Sample for performance
        n = min(15000, len(cx_norm))
        idx = np.random.choice(len(cx_norm), n, replace=False)

        ax.scatter(cx_norm.iloc[idx], cy_norm.iloc[idx], s=1.5, c=color, alpha=0.3, rasterized=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # flip Y so origin is top-left (image convention)
        ax.set_xlabel('Normalized X (cx / img_width)', fontsize=12)
        ax.set_ylabel('Normalized Y (cy / img_height)', fontsize=12)
        ax.set_title(f'{ds_name}\nCenter Distribution ({n:,} sampled points)',
                     fontsize=12, fontweight='bold')
        ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(FIGS_DIR / '06_center_scatter.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    # Kernel density version
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (ds_name, d), color in zip(axes2, dfs.items(), colors):
        cx_norm = d['df_ann']['cx'] / d['df_ann']['img_width']
        cy_norm = d['df_ann']['cy'] / d['df_ann']['img_height']
        n = min(15000, len(cx_norm))
        idx = np.random.choice(len(cx_norm), n, replace=False)
        ax.hexbin(cx_norm.iloc[idx], cy_norm.iloc[idx],
                  gridsize=60, cmap='Blues' if ds_name == list(dfs.keys())[0] else 'Oranges',
                  mincnt=1, alpha=0.8, edgecolors='none')
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_xlabel('Normalized X', fontsize=12)
        ax.set_ylabel('Normalized Y', fontsize=12)
        ax.set_title(f'{ds_name}\nCenter Heatmap (hexbin)', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')

    fig2.tight_layout()
    fig2.savefig(FIGS_DIR / '06b_center_heatmap.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig2)
    print('  Saved 06_center_scatter*.png')


def plot_image_size_distribution(dfs):
    """Fig 7: Image size scatter + histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = [COLORS['ds1'], COLORS['ds2']]

    for ax, (ds_name, d), color in zip(axes, dfs.items(), colors):
        widths = d['df_img']['width']
        heights = d['df_img']['height']
        unique_sizes = d['df_img'].groupby(['width', 'height']).size().reset_index(name='count')
        sizes = ax.scatter(unique_sizes['width'], unique_sizes['height'],
                          s=unique_sizes['count'] / unique_sizes['count'].max() * 300 + 20,
                          c=color, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=3)
        ax.set_xlabel('Width (pixels)', fontsize=12)
        ax.set_ylabel('Height (pixels)', fontsize=12)
        n_unique = len(unique_sizes)
        ax.set_title(f'{ds_name}\n{n_unique} unique sizes',
                     fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        # Add annotations for each point
        for _, row in unique_sizes.iterrows():
            ax.annotate(f'{int(row["width"])}×{int(row["height"])} (×{int(row["count"])})',
                       (row['width'], row['height']),
                       textcoords='offset points', xytext=(5, 5), fontsize=7, alpha=0.7)

    fig.tight_layout()
    fig.savefig(FIGS_DIR / '07_image_size_scatter.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('  Saved 07_image_size_scatter.png')


def plot_bbox_width_vs_height(dfs):
    """Fig 8: BBox width vs height scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = [COLORS['ds1'], COLORS['ds2']]

    for ax, (ds_name, d), color in zip(axes, dfs.items(), colors):
        n = min(20000, len(d['df_ann']))
        idx = np.random.choice(len(d['df_ann']), n, replace=False)
        w = d['df_ann']['bbox_w'].iloc[idx]
        h = d['df_ann']['bbox_h'].iloc[idx]

        ax.scatter(w, h, s=2, c=color, alpha=0.3, rasterized=True)
        # Diagonal line (square)
        lim_max = max(w.max(), h.max())
        ax.plot([0, lim_max], [0, lim_max], 'k--', lw=0.8, alpha=0.4, label='Square (w=h)')
        ax.set_xlabel('Width (pixels)', fontsize=12)
        ax.set_ylabel('Height (pixels)', fontsize=12)
        ax.set_title(f'{ds_name}\n(n={n:,} sampled)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(FIGS_DIR / '08_bbox_width_vs_height.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('  Saved 08_bbox_width_vs_height.png')


def plot_density_vs_objects(dfs):
    """Fig 9: Density (objects per Mpixel) comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    for ds_name, d in dfs.items():
        data.append(d['df_per_img']['density'].values)
        labels.append(ds_name)

    # Violin + swarm
    parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=True, widths=0.5)
    for pc, color in zip(parts['bodies'], [COLORS['ds1'], COLORS['ds2']]):
        pc.set_facecolor(color)
        pc.set_alpha(0.4)
        pc.set_edgecolor('none')
    for key in ['cmaxes', 'cmins', 'cbars', 'cmeans', 'cmedians']:
        if key in parts:
            parts[key].set_color('#333333')
            parts[key].set_linewidth(1)

    # Swarm
    for i, (ds_name, d) in enumerate(dfs.items()):
        y = d['df_per_img']['density'].values
        jitter = np.random.normal(0, 0.04, len(y))
        ax.scatter(1 + i + jitter, y, s=6, color=[COLORS['ds1'], COLORS['ds2']][i],
                   alpha=0.3, rasterized=True, zorder=3)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Object Density (objects / Mpixel)', fontsize=12)
    ax.set_title('Detection Density per Image', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / '09_density_violin.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('  Saved 09_density_violin.png')


def plot_area_by_category(dfs):
    """Fig 10: BBox area per category (boxen plot)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    colors = [COLORS['ds1'], COLORS['ds2']]

    for ax, (ds_name, d), color in zip(axes, dfs.items(), colors):
        # Top 12 categories by count
        top_cats = d['df_ann']['category'].value_counts().head(12).index
        sub = d['df_ann'][d['df_ann']['category'].isin(top_cats)]

        bp = sub.boxplot(
            column='area', by='category', ax=ax,
            patch_artist=True, showfliers=False, showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='#C44E52', markersize=4, markeredgecolor='none'),
            widths=0.6, return_type='dict',
        )
        for patch in bp['area']['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_title(f'{ds_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('BBox Area (pixels²)', fontsize=11)
        ax.tick_params(axis='x', rotation=45)

    fig.suptitle('BBox Area by Category', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / '10_area_by_category.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('  Saved 10_area_by_category.png')


def plot_summary_table(dfs):
    """Fig 11: Summary statistics table."""
    rows = []
    for ds_name, d in dfs.items():
        ann = d['df_ann']
        img = d['df_img']
        pi = d['df_per_img']
        total_ann = len(ann)
        total_img = len(img)

        # Split counts
        split_counts = img['split'].value_counts().to_dict()

        row = {
            'Dataset': ds_name,
            'Total Images': total_img,
            'Total Objects': total_ann,
            'Avg Objects/Image': f'{pi["num_objects"].mean():.1f}',
            'Avg BBox Area': f'{ann["area"].mean():.0f}',
            'Median BBox Area': f'{ann["area"].median():.0f}',
            'S (≤32²)': f'{((ann["area"] < S_THRESH).sum() / total_ann * 100):.1f}%',
            'M (32²–96²)': f'{((ann["area"] >= S_THRESH) & (ann["area"] < L_THRESH)).sum() / total_ann * 100:.1f}%',
            'L (≥96²)': f'{(ann["area"] >= L_THRESH).sum() / total_ann * 100:.1f}%',
            'Train/Test/Valid': f'{split_counts.get("train", 0)} / {split_counts.get("test", 0)} / {split_counts.get("valid", 0)}',
        }
        rows.append(row)

    df_table = pd.DataFrame(rows).T
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    table = ax.table(cellText=df_table.values,
                     rowLabels=df_table.index,
                     colLabels=df_table.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for i in range(len(df_table.index)):
        for j in range(len(df_table.columns)):
            cell = table[i + 1, j]
            cell.set_facecolor('#F0F0F0' if i % 2 == 0 else 'white')
    for j in range(len(df_table.columns)):
        table[0, j].set_facecolor('#4C72B0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(len(df_table.index)):
        table[i + 1, -1].set_facecolor('#E8E8E8')

    ax.set_title('Dataset Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / '11_summary_table.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('  Saved 11_summary_table.png')


def plot_category_per_image(dfs):
    """Fig 12: Category presence per image — binary presence heatmap (random 100 images)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, (ds_name, d) in zip(axes, dfs.items()):
        # Pivot: img_id × category, value = count → binarize
        pivot = d['df_ann'].pivot_table(
            index='img_id', columns='category', aggfunc='size', fill_value=0
        )
        # Randomly sample 100 images
        n_imgs = len(pivot)
        sample_size = min(100, n_imgs)
        sampled = pivot.sample(n=sample_size, random_state=42)
        pivot_binary = (sampled > 0).astype(int)

        sns.heatmap(pivot_binary, ax=ax, cmap='Blues' if ds_name == list(dfs.keys())[0] else 'Oranges',
                    cbar_kws={'label': 'Presence (1=yes, 0=no)'},
                    linewidths=0.3, linecolor='white',
                    vmin=0, vmax=1, annot=False,
                    xticklabels=True, yticklabels=True)
        ax.set_title(f'{ds_name}\nCategory Presence ({sample_size} random images)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Category', fontsize=11)
        ax.set_ylabel('Image ID', fontsize=11)

    fig.tight_layout()
    fig.savefig(FIGS_DIR / '12_category_per_image_heatmap.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('  Saved 12_category_per_image_heatmap.png (random 100)')


def plot_combined_center_scatter(dfs):
    """Fig 13: Combined center scatter — both datasets on one plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = [COLORS['ds1'], COLORS['ds2']]
    labels = list(dfs.keys())

    for i, (ds_name, d) in enumerate(dfs.items()):
        cx_norm = d['df_ann']['cx'] / d['df_ann']['img_width']
        cy_norm = d['df_ann']['cy'] / d['df_ann']['img_height']
        n = min(10000, len(cx_norm))
        idx = np.random.choice(len(cx_norm), n, replace=False)
        ax.scatter(cx_norm.iloc[idx], cy_norm.iloc[idx], s=1.5,
                   c=colors[i], alpha=0.25, label=ds_name, rasterized=True)

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel('Normalized X (cx / img_width)', fontsize=12)
    ax.set_ylabel('Normalized Y (cy / img_height)', fontsize=12)
    ax.set_title('BBox Center Distribution — Combined', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, markerscale=8)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(FIGS_DIR / '13_combined_center_scatter.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('  Saved 13_combined_center_scatter.png')


# ── Main ────────────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('  Dataset Comparison Visualization')
    print('=' * 60)
    print()

    # 1. Load data
    dfs = build_dataframes()
    print()

    # 2. Print numerical summary
    print('─' * 60)
    print('  Numerical Summary')
    print('─' * 60)
    for ds_name, d in dfs.items():
        ann = d['df_ann']
        img = d['df_img']
        pi = d['df_per_img']
        print(f'\n  [{ds_name}]')
        print(f'    Images: {len(img)} (train={len(img[img["split"]=="train"])}, '
              f'test={len(img[img["split"]=="test"])}, valid={len(img[img["split"]=="valid"])})')
        print(f'    Total annotations: {len(ann):,}')
        print(f'    Objects/image: mean={pi["num_objects"].mean():.1f}, '
              f'median={pi["num_objects"].median():.0f}, '
              f'min={pi["num_objects"].min()}, max={pi["num_objects"].max()}')
        print(f'    BBox area: mean={ann["area"].mean():.0f}, '
              f'median={ann["area"].median():.0f}, '
              f'min={ann["area"].min():.0f}, max={ann["area"].max():.0f}')
        print(f'    S/M/L: {(ann["area"] < S_THRESH).sum():,} / '
              f'{((ann["area"] >= S_THRESH) & (ann["area"] < L_THRESH)).sum():,} / '
              f'{(ann["area"] >= L_THRESH).sum():,} '
              f'({(ann["area"] < S_THRESH).mean()*100:.1f}% / '
              f'{((ann["area"] >= S_THRESH) & (ann["area"] < L_THRESH)).mean()*100:.1f}% / '
              f'{(ann["area"] >= L_THRESH).mean()*100:.1f}%)')
        print(f'    Image sizes: {sorted(img["width"].unique())} × {sorted(img["height"].unique())}')

        # Per-category breakdown
        print(f'    Category counts:')
        cat_counts = ann['category'].value_counts()
        for cat, count in cat_counts.items():
            cat_area = ann[ann['category'] == cat]['area']
            pct = count / len(ann) * 100
            print(f'      {cat:>4s}: {count:>7,} ({pct:5.2f}%), '
                  f'area: median={cat_area.median():.0f}, mean={cat_area.mean():.0f}')

    print()
    print('─' * 60)
    print('  Generating Figures...')
    print('─' * 60)

    plot_category_distribution(dfs)
    plot_objects_per_image(dfs)
    plot_size_class_distribution(dfs)
    plot_size_scatter(dfs)
    plot_bbox_area_distribution(dfs)
    plot_bbox_aspect_ratio(dfs)
    plot_center_scatter(dfs)
    plot_image_size_distribution(dfs)
    plot_bbox_width_vs_height(dfs)
    plot_density_vs_objects(dfs)
    plot_area_by_category(dfs)
    plot_summary_table(dfs)
    plot_category_per_image(dfs)
    plot_combined_center_scatter(dfs)

    print()
    print('─' * 60)
    print(f'  All figures saved to: {FIGS_DIR}')
    print(f'  Total files: {len(list(FIGS_DIR.glob("*.png")))}')
    print('─' * 60)


if __name__ == '__main__':
    main()
