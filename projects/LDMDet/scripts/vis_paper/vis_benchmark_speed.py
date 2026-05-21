#!/usr/bin/env python3
"""Visualization for inference speed benchmark results.

Usage:
    python projects/LDMDet/scripts/vis_paper/vis_benchmark_speed.py \
        --input projects/LDMDet/results/benchmark_speed_results.json \
        --output-dir projects/LDMDet/scripts/vis_paper/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


PALETTE = {
    'standard': '#4C72B0',
    'ldmdet': '#DD8452',
}

MODEL_COLORS = [
    '#4C72B0', '#55A868', '#C44E52', '#8172B3',
    '#CCB974', '#64B5CD', '#E5AE38', '#6D904F',
]

LDMDet_STEPS_COLORS = {
    1: '#DD8452',
    2: '#E89040',
    4: '#F09C30',
    8: '#F5A820',
}


def parse_args():
    p = argparse.ArgumentParser(description='Visualize benchmark speed results')
    p.add_argument(
        '--input',
        type=str,
        default='projects/LDMDet/results/benchmark_speed_results.json',
        help='Path to benchmark results JSON',
    )
    p.add_argument(
        '--output-dir',
        type=str,
        default='projects/LDMDet/scripts/vis_paper/',
        help='Directory to save figures',
    )
    p.add_argument(
        '--format',
        type=str,
        default='png',
        choices=['png', 'pdf', 'svg'],
        help='Output figure format',
    )
    return p.parse_args()


def load_results(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def plot_fps_comparison(results: List[Dict], output_dir: str, fmt: str):
    names = []
    e2e_fps = []
    fwd_fps = []

    for r in results:
        names.append(r['name'].replace('-', '\n'))
        e2e_fps.append(r['e2e']['fps'])
        fwd_fps.append(r['forward']['fps'])

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 6))
    bars1 = ax.bar(x - width / 2, e2e_fps, width, label='E2E FPS', color='#4C72B0', edgecolor='white')
    bars2 = ax.bar(x + width / 2, fwd_fps, width, label='Forward FPS', color='#55A868', edgecolor='white')

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('FPS (images/sec)', fontsize=12)
    ax.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'benchmark_fps_comparison.{fmt}'), dpi=200)
    plt.close(fig)
    print(f'  Saved: benchmark_fps_comparison.{fmt}')


def plot_latency_comparison(results: List[Dict], output_dir: str, fmt: str):
    names = []
    e2e_mean = []
    e2e_std = []
    fwd_mean = []
    fwd_std = []

    for r in results:
        names.append(r['name'].replace('-', '\n'))
        e2e_mean.append(r['e2e']['latency_mean_ms'])
        e2e_std.append(r['e2e']['latency_std_ms'])
        fwd_mean.append(r['forward']['latency_mean_ms'])
        fwd_std.append(r['forward']['latency_std_ms'])

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 6))
    bars1 = ax.bar(x - width / 2, e2e_mean, width, yerr=e2e_std,
                   label='E2E Latency', color='#C44E52', edgecolor='white', capsize=3)
    bars2 = ax.bar(x + width / 2, fwd_mean, width, yerr=fwd_std,
                   label='Forward Latency', color='#8172B3', edgecolor='white', capsize=3)

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'benchmark_latency_comparison.{fmt}'), dpi=200)
    plt.close(fig)
    print(f'  Saved: benchmark_latency_comparison.{fmt}')


def plot_speed_accuracy(results: List[Dict], output_dir: str, fmt: str,
                        map_data: Optional[Dict[str, float]] = None):
    if map_data is None:
        print('  Skipping speed-accuracy plot: no mAP data provided.')
        print('  Add a "map_data" field to the results JSON, e.g.:')
        print('    {"map_data": {"CascadeRCNN-R50": 0.75, "LDMDet-RF-1step": 0.751, ...}}')
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        name = r['name']
        if name not in map_data:
            continue
        latency = r['e2e']['latency_mean_ms']
        map_val = map_data[name]
        params = r['params_M']
        color = PALETTE.get(r['type'], '#999999')
        size = max(50, min(400, params * 3))
        ax.scatter(latency, map_val, s=size, c=color, alpha=0.8, edgecolors='white', linewidth=1.5, zorder=5)
        ax.annotate(name, (latency, map_val), textcoords='offset points',
                    xytext=(8, 5), fontsize=8, ha='left')

    ax.set_xlabel('E2E Latency (ms)', fontsize=12)
    ax.set_ylabel('mAP', fontsize=12)
    ax.set_title('Speed-Accuracy Trade-off', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    std_patch = mpatches.Patch(color=PALETTE['standard'], label='Standard Detector')
    ldm_patch = mpatches.Patch(color=PALETTE['ldmdet'], label='LDMDet')
    ax.legend(handles=[std_patch, ldm_patch], fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'benchmark_speed_accuracy.{fmt}'), dpi=200)
    plt.close(fig)
    print(f'  Saved: benchmark_speed_accuracy.{fmt}')


def plot_ldmdet_steps_curve(results: List[Dict], output_dir: str, fmt: str):
    ldmdet_results = [r for r in results if r['type'] == 'ldmdet' and 'sampling_steps' in r]
    if not ldmdet_results:
        print('  Skipping LDMDet steps curve: no LDMDet results with sampling_steps.')
        return

    ldmdet_by_name = {}
    for r in ldmdet_results:
        base_name = r['name'].rsplit('-', 1)[0]
        if base_name not in ldmdet_by_name:
            ldmdet_by_name[base_name] = []
        ldmdet_by_name[base_name].append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for base_name, group in ldmdet_by_name.items():
        group.sort(key=lambda x: x['sampling_steps'])
        steps = [r['sampling_steps'] for r in group]
        e2e_fps = [r['e2e']['fps'] for r in group]
        fwd_fps = [r['forward']['fps'] for r in group]
        e2e_lat = [r['e2e']['latency_mean_ms'] for r in group]
        fwd_lat = [r['forward']['latency_mean_ms'] for r in group]

        ax1.plot(steps, e2e_fps, 'o-', label=f'{base_name} (E2E)', color='#DD8452', markersize=8)
        ax1.plot(steps, fwd_fps, 's--', label=f'{base_name} (Forward)', color='#DD8452', alpha=0.6, markersize=8)

        ax2.plot(steps, e2e_lat, 'o-', label=f'{base_name} (E2E)', color='#C44E52', markersize=8)
        ax2.plot(steps, fwd_lat, 's--', label=f'{base_name} (Forward)', color='#C44E52', alpha=0.6, markersize=8)

    ax1.set_xlabel('Sampling Steps', fontsize=12)
    ax1.set_ylabel('FPS (images/sec)', fontsize=12)
    ax1.set_title('LDMDet: Steps vs FPS', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xticks([1, 2, 4, 8])

    ax2.set_xlabel('Sampling Steps', fontsize=12)
    ax2.set_ylabel('Latency (ms)', fontsize=12)
    ax2.set_title('LDMDet: Steps vs Latency', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xticks([1, 2, 4, 8])

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'benchmark_ldmdet_steps_curve.{fmt}'), dpi=200)
    plt.close(fig)
    print(f'  Saved: benchmark_ldmdet_steps_curve.{fmt}')


def plot_radar(results: List[Dict], output_dir: str, fmt: str,
               map_data: Optional[Dict[str, float]] = None):
    if map_data is None:
        print('  Skipping radar plot: no mAP data provided.')
        return

    filtered = [r for r in results if r['name'] in map_data]
    if not filtered:
        print('  Skipping radar plot: no matching mAP data.')
        return

    categories = ['FPS', 'mAP', 'Params\n(inverse)', 'FLOPs\n(inverse)', 'Memory\n(inverse)']

    max_fps = max(r['e2e']['fps'] for r in filtered)
    max_map = max(map_data[r['name']] for r in filtered)
    max_params = max(r['params_M'] for r in filtered)
    max_flops = max(r['flops_G'] for r in filtered if r['flops_G'] > 0) or 1.0
    max_mem = max(r['peak_mem_MB'] for r in filtered)

    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, r in enumerate(filtered):
        name = r['name']
        fps_norm = r['e2e']['fps'] / max_fps
        map_norm = map_data[name] / max_map
        params_inv = 1.0 - (r['params_M'] / max_params)
        flops_inv = 1.0 - (r['flops_G'] / max_flops) if r['flops_G'] > 0 else 0.5
        mem_inv = 1.0 - (r['peak_mem_MB'] / max_mem)

        values = [fps_norm, map_norm, params_inv, flops_inv, mem_inv]
        values += values[:1]

        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color, markersize=5)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title('Multi-dimensional Comparison', fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'benchmark_radar.{fmt}'), dpi=200)
    plt.close(fig)
    print(f'  Saved: benchmark_radar.{fmt}')


def plot_model_complexity(results: List[Dict], output_dir: str, fmt: str):
    names = [r['name'].replace('-', '\n') for r in results]
    params = [r['params_M'] for r in results]
    flops = [r['flops_G'] for r in results]
    mems = [r['peak_mem_MB'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(names) * 1.0), 5))

    x = np.arange(len(names))

    axes[0].barh(x, params, color='#4C72B0', edgecolor='white')
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlabel('Params (M)', fontsize=10)
    axes[0].set_title('Parameters', fontsize=12, fontweight='bold')
    for i, v in enumerate(params):
        axes[0].text(v + 0.5, i, f'{v:.1f}', va='center', fontsize=8)

    axes[1].barh(x, flops, color='#55A868', edgecolor='white')
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].set_xlabel('FLOPs (G)', fontsize=10)
    axes[1].set_title('FLOPs', fontsize=12, fontweight='bold')
    for i, v in enumerate(flops):
        if v > 0:
            axes[1].text(v + 0.5, i, f'{v:.1f}', va='center', fontsize=8)

    axes[2].barh(x, mems, color='#C44E52', edgecolor='white')
    axes[2].set_yticks(x)
    axes[2].set_yticklabels(names, fontsize=8)
    axes[2].set_xlabel('Peak Memory (MB)', fontsize=10)
    axes[2].set_title('GPU Memory', fontsize=12, fontweight='bold')
    for i, v in enumerate(mems):
        axes[2].text(v + 10, i, f'{v:.0f}', va='center', fontsize=8)

    fig.suptitle('Model Complexity Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'benchmark_model_complexity.{fmt}'), dpi=200)
    plt.close(fig)
    print(f'  Saved: benchmark_model_complexity.{fmt}')


def main():
    args = parse_args()
    data = load_results(args.input)
    results = data.get('results', [])
    map_data = data.get('map_data', None)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Generating visualizations from {len(results)} results...')

    plot_fps_comparison(results, args.output_dir, args.format)
    plot_latency_comparison(results, args.output_dir, args.format)
    plot_speed_accuracy(results, args.output_dir, args.format, map_data)
    plot_ldmdet_steps_curve(results, args.output_dir, args.format)
    plot_radar(results, args.output_dir, args.format, map_data)
    plot_model_complexity(results, args.output_dir, args.format)

    print(f'\nAll figures saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
