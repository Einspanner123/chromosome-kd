#!/usr/bin/env python3
"""Generate publication-quality figures for BMVC 2026 submission.

Reads experimental data from JSON files and produces all figures.
Output: paper_ot_medical_draft/figures/*.pdf

Usage:
    python scripts/generate_figures.py
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---- Paths ----
SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_DIR = SCRIPT_DIR.parent
FIG_DIR = DRAFT_DIR / 'figures'
PROJ_DIR = SCRIPT_DIR.parent.parent  # LDMDet root
REPO = PROJ_DIR.parent  # chromo-kd root

os.makedirs(FIG_DIR, exist_ok=True)

# ---- Style ----
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'random': '#2E86AB',  # blue
    'hard_ot': '#D8315B',  # red
    'argmax': '#F18F01',  # orange
    'stochastic': '#3A7D44',  # green
    'ot_dominated': '#F9A8A8',
    'sweet_spot': '#A8E6CF',
    'bias_dominated': '#FFD3B6',
}

CLASS_NAMES = [
    'A1',
    'A2',
    'A3',
    'B4',
    'B5',
    'C10',
    'C11',
    'C12',
    'C6',
    'C7',
    'C8',
    'C9',
    'D13',
    'D14',
    'D15',
    'E16',
    'E17',
    'E18',
    'F19',
    'F20',
    'G21',
    'G22',
    'X',
    'Y',
]
GROUPS = [
    'A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'D', 'D', 'D',
    'E', 'E', 'E', 'F', 'F', 'G', 'G', 'X', 'Y'
]


# ---- Data loading ----
def load_json(name: str) -> dict:
    """Load a JSON file from the LDMDet project directory."""
    path = PROJ_DIR / name
    if not path.exists():
        print(f'WARNING: {path} not found')
        return {}
    with open(path) as f:
        return json.load(f)


def load_per_class_ap() -> dict:
    return load_json('per_class_ap_results.json')


def load_epsilon_prediction() -> dict:
    return load_json('optimal_epsilon_prediction.json')


def load_velocity_entropy() -> dict:
    return load_json('velocity_entropy_results.json')


# ---- Figure 1: Overview ----
def fig1_overview(per_class):
    """Paper overview figure: schematic + key numbers."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')

    # Title
    ax.text(
        5,
        5.6,
        'Diversity Over Efficiency: Sinkhorn Sampling for Dense Chromosome Detection',
        ha='center',
        va='center',
        fontsize=16,
        fontweight='bold')

    # ---- Row 1: The Problem ----
    # Box 1: OT intuition
    box1 = FancyBboxPatch((0.5, 3.8),
                          3.5,
                          1.4,
                          boxstyle='round,pad=0.1',
                          facecolor='#FFF3E0',
                          edgecolor='#E65100',
                          linewidth=2)
    ax.add_patch(box1)
    ax.text(
        2.25,
        4.8,
        'Common Belief',
        ha='center',
        fontsize=12,
        fontweight='bold',
        color='#E65100')
    ax.text(
        2.25,
        4.4,
        'Optimal Transport coupling\nshortens transport paths &\nbenefits generation models',
        ha='center',
        fontsize=9.5,
        color='#333')

    # Arrow to Box 2
    ax.annotate(
        '',
        xy=(5.5, 4.5),
        xytext=(4.0, 4.5),
        arrowprops=dict(arrowstyle='->', color='#555', lw=2))

    # Box 2: The Discovery
    box2 = FancyBboxPatch((5.5, 3.8),
                          3.5,
                          1.4,
                          boxstyle='round,pad=0.1',
                          facecolor='#FFEBEE',
                          edgecolor='#B71C1C',
                          linewidth=2)
    ax.add_patch(box2)
    ax.text(
        7.25,
        4.8,
        'Our Discovery',
        ha='center',
        fontsize=12,
        fontweight='bold',
        color='#B71C1C')
    ax.text(
        7.25,
        4.35,
        f"OT harms dense detection:\nRandom mAP = {per_class.get('Random (AdaLN)', {}).get('mAP', 0.751):.3f}\nHard OT mAP = {per_class.get('Hard OT', {}).get('mAP', 0.735):.3f}",
        ha='center',
        fontsize=9.5,
        color='#333')

    # ---- Row 2: The Mechanism ----
    arrow_down = FancyArrowPatch((4.0, 3.8), (2.5, 3.0),
                                 arrowstyle='->',
                                 color='#555',
                                 lw=1.5,
                                 connectionstyle='arc3,rad=-0.2')
    ax.add_patch(arrow_down)

    box3 = FancyBboxPatch((0.5, 1.6),
                          4.0,
                          1.3,
                          boxstyle='round,pad=0.1',
                          facecolor='#E3F2FD',
                          edgecolor='#1565C0',
                          linewidth=2)
    ax.add_patch(box3)
    ax.text(
        2.5,
        2.55,
        'Mechanism 1: Diversity Collapse',
        ha='center',
        fontsize=11,
        fontweight='bold',
        color='#1565C0')
    ax.text(
        2.5,
        2.15,
        'Conditional velocity entropy\nH(V|Z): 3.841 → 0.000 nats\nTotal variance: 5.169 → 2.401',
        ha='center',
        fontsize=9,
        color='#333')

    box4 = FancyBboxPatch((5.5, 1.6),
                          4.0,
                          1.3,
                          boxstyle='round,pad=0.1',
                          facecolor='#E8F5E9',
                          edgecolor='#2E7D32',
                          linewidth=2)
    ax.add_patch(box4)
    ax.text(
        7.5,
        2.55,
        'Mechanism 2: Argmax Bottleneck',
        ha='center',
        fontsize=11,
        fontweight='bold',
        color='#2E7D32')
    ax.text(
        7.5,
        2.15,
        'Argmax destroys ε control\nD_eff frozen at 2.80 across ε\nStochastic restores D_eff to 5.31',
        ha='center',
        fontsize=9,
        color='#333')

    # ---- Row 3: The Solution ----
    ax.annotate(
        '',
        xy=(7.25, 1.6),
        xytext=(4.75, 2.9),
        arrowprops=dict(
            arrowstyle='->',
            color='#555',
            lw=1.5,
            connectionstyle='arc3,rad=0.2'))

    box5 = FancyBboxPatch((3, 0.2),
                          4.5,
                          0.9,
                          boxstyle='round,pad=0.1',
                          facecolor='#C8E6C9',
                          edgecolor='#1B5E20',
                          linewidth=2)
    ax.add_patch(box5)
    ax.text(
        5.25,
        0.9,
        'Stochastic Coupling',
        ha='center',
        fontsize=12,
        fontweight='bold',
        color='#1B5E20')
    ax.text(
        5.25,
        0.5,
        f"Sample from Sinkhorn matrix → mAP = {per_class.get('Sinkhorn sample eps=5', {}).get('mAP', 0.750):.3f}\nPreserves diversity + transport structure",
        ha='center',
        fontsize=9.5,
        color='#333')

    fig.tight_layout(pad=0.5)
    fig.savefig(FIG_DIR / 'figure1_overview.pdf')
    fig.savefig(FIG_DIR / 'figure1_overview.png')
    plt.close(fig)
    print('✓ Figure 1: Overview saved')


# ---- Figure 2: Transport Matrix Heatmaps ----
def fig2_coupling():
    """Illustrate coupling mechanisms with simulated transport matrices."""
    np.random.seed(42)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))

    titles = [
        '(a) Random Coupling', '(b) Hard OT', '(c) Sinkhorn + Argmax',
        '(d) Stochastic Coupling'
    ]
    N, M = 30, 20  # proposals × targets for visualization

    # (a) Random: uniform assignment
    P_rand = np.random.rand(N, M)
    P_rand = P_rand / P_rand.sum(axis=1, keepdims=True)

    # (b) Hard OT: one-hot per row
    P_hard = np.zeros((N, M))
    for i in range(N):
        P_hard[i, np.random.randint(0, M)] = 1.0

    # (c) Sinkhorn + argmax: concentrated but not totally
    # Simulate low-epsilon Sinkhorn (peaked but soft)
    cost = np.random.rand(N, M)
    eps = 0.1
    K = np.exp(-cost / eps)
    u = np.ones(N) / N
    v = np.ones(M) / M
    for _ in range(20):
        u = (np.ones(N) / N) / (K @ v)
        v = (np.ones(M) / M) / (K.T @ u)
    P_sink_low = np.diag(u) @ K @ np.diag(v)
    # Argmax decode
    P_argmax = np.zeros((N, M))
    P_argmax[np.arange(N), P_sink_low.argmax(axis=1)] = 1.0

    # (d) Stochastic: sample from moderate-epsilon Sinkhorn
    eps = 5.0
    K5 = np.exp(-cost / eps)
    u5 = np.ones(N) / N
    v5 = np.ones(M) / M
    for _ in range(20):
        u5 = (np.ones(N) / N) / (K5 @ v5)
        v5 = (np.ones(M) / M) / (K5.T @ u5)
    P_stoch = np.diag(u5) @ K5 @ np.diag(v5)
    # Sample (visualize as soft matrix)
    P_stoch_viz = P_stoch / P_stoch.sum(axis=1, keepdims=True)

    matrices = [P_rand, P_hard, P_argmax, P_stoch_viz]
    for ax, P, title in zip(axes, matrices, titles):
        im = ax.imshow(P, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.15)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('GT index')
        ax.set_ylabel('Proposal index')
        # Add diversity metric
        row_entropy = -np.sum(P * np.log(P + 1e-10), axis=1)
        mean_entropy = np.mean(row_entropy)
        ax.text(
            0.5,
            -0.3,
            f'Mean row entropy: {mean_entropy:.2f}',
            transform=ax.transAxes,
            ha='center',
            fontsize=8,
            color='#555')

    plt.colorbar(
        im, ax=axes, label='Assignment probability', shrink=0.6, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'figure2_coupling.pdf')
    fig.savefig(FIG_DIR / 'figure2_coupling.png')
    plt.close(fig)
    print('✓ Figure 2: Coupling mechanisms saved')


# ---- Figure 3: Epsilon Regimes ----
def fig3_epsilon():
    """Plot rho(eps), eta(eps), mAP(eps) with regime shading."""
    pred = load_epsilon_prediction()
    vel = load_velocity_entropy()

    eps_vals = np.array([0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])

    # rho from optimal_epsilon_prediction measured_data
    rho = np.array([0.184, 0.693, 0.951, 0.986, 1.000, 1.000, 1.000, 1.000])
    # eta from same
    eta = np.array([0.016, 0.216, 0.624, 0.788, 0.966, 0.973, 0.998, 1.001])

    # mAP from experiments (verified: ε=1,5 from checkpoint inference; others from training logs)
    mAP = np.array([np.nan, np.nan, 0.742, 0.749, 0.750, 0.747, 0.736, np.nan])
    mAP_eps = np.array([0.5, 1.0, 5.0, 10.0, 50.0])

    fig, ax1 = plt.subplots(1, 1, figsize=(9, 5.5))

    # Regime backgrounds
    ax1.axvspan(0.001, 0.5, alpha=0.12, color='#D8315B', label='OT-dominated')
    ax1.axvspan(0.5, 5.0, alpha=0.12, color='#3A7D44', label='Sweet spot')
    ax1.axvspan(5.0, 100, alpha=0.12, color='#F18F01', label='Bias-dominated')

    # rho and eta curves
    ax1.semilogx(
        eps_vals,
        rho,
        'o-',
        color=COLORS['random'],
        lw=2.5,
        markersize=8,
        label=r'$\rho(\varepsilon)$ Diversity Recovery')
    ax1.semilogx(
        eps_vals,
        eta,
        's--',
        color=COLORS['hard_ot'],
        lw=2.5,
        markersize=8,
        label=r'$\eta(\varepsilon)$ Transport Efficiency')
    ax1.set_xlabel(r'$\varepsilon$ (log scale)')
    ax1.set_ylabel(r'$\rho(\varepsilon)$ / $\eta(\varepsilon)$', color='#333')
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_xlim(0.008, 110)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, alpha=0.3, which='both')

    # mAP on twin axis
    ax2 = ax1.twinx()
    mAP_vals = np.array([0.742, 0.749, 0.750, 0.747, 0.736])
    ax2.semilogx(
        mAP_eps,
        mAP_vals,
        'D-',
        color='#333',
        lw=2.5,
        markersize=10,
        label='mAP')
    ax2.set_ylabel('mAP', color='#333')
    ax2.set_ylim(0.730, 0.756)
    ax2.legend(loc='upper right', frameon=True)

    # Annotations
    ax1.annotate(
        'OT-dominated\n($\\rho<0.95$)',
        xy=(0.05, 0.35),
        fontsize=10,
        color='#B71C1C',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax1.annotate(
        'Sweet Spot\n($0.5\\leq\\varepsilon\\leq 5$)',
        xy=(1.5, 0.92),
        fontsize=10,
        color='#1B5E20',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax1.annotate(
        'Bias-dominated\n($\\varepsilon>5$)',
        xy=(20, 1.05),
        fontsize=10,
        color='#E65100',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax2.annotate(
        f'mAP={0.750:.3f}',
        xy=(5, 0.750),
        xytext=(8, 0.754),
        fontsize=10,
        fontweight='bold',
        ha='center',
        arrowprops=dict(arrowstyle='->', color='#333'))

    ax1.set_title(
        'Epsilon Sweep: Three-Regime Structure', fontweight='bold', pad=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'figure3_epsilon.pdf')
    fig.savefig(FIG_DIR / 'figure3_epsilon.png')
    plt.close(fig)
    print('✓ Figure 3: Epsilon regimes saved')


# ---- Figure 4: Dataset Overview ----
def fig4_dataset():
    """Show chromosome image examples with annotations."""
    import cv2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    main_root = Path('data/Chromosome20240904_NoAug_NoResize_coco')

    img_paths = []
    if main_root.exists():
        train_dir = main_root / 'train'
        if train_dir.exists():
            imgs = sorted(train_dir.glob('*.jpg'))[:2]
            img_paths.extend(imgs)

    for i, ax in enumerate(axes.flat):
        if i < len(img_paths):
            try:
                img = cv2.imread(str(img_paths[i]))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img)
            except Exception:
                pass
        ax.set_title(
            f'Dataset A: 24-class chromosome\n(avg 46 objects/image)',
            fontsize=10,
            fontweight='bold')
        ax.axis('off')

    fig.suptitle(
        'Dataset: Dense Multi-Instance Chromosome Detection',
        fontweight='bold',
        fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'figure4_dataset.pdf')
    fig.savefig(FIG_DIR / 'figure4_dataset.png')
    plt.close(fig)
    print('✓ Figure 4: Dataset overview saved')


# ---- Figure 5: Per-Class AP Bar Chart ----
def fig5_per_class_ap(per_class):
    """Grouped bar chart: Random vs Hard OT vs Stochastic per class."""
    data = per_class.get('results', {})
    if not data:
        print('WARNING: No per-class AP data')
        return

    random_ap = [
        data['Random (AdaLN)']['per_class'].get(c, 0) for c in CLASS_NAMES
    ]
    hardot_ap = [data['Hard OT']['per_class'].get(c, 0) for c in CLASS_NAMES]
    stoch_ap = [
        data['Sinkhorn sample eps=5']['per_class'].get(c, 0)
        for c in CLASS_NAMES
    ]
    argmax_ap = [
        data.get('Sinkhorn argmax eps=5', {}).get('per_class', {}).get(c, 0)
        for c in CLASS_NAMES
    ]

    x = np.arange(len(CLASS_NAMES))
    width = 0.2

    fig, ax = plt.subplots(1, 1, figsize=(16, 5.5))

    bars1 = ax.bar(
        x - 1.5 * width,
        random_ap,
        width,
        color=COLORS['random'],
        label='Random (0.751 mAP)',
        edgecolor='white',
        linewidth=0.5)
    bars2 = ax.bar(
        x - 0.5 * width,
        hardot_ap,
        width,
        color=COLORS['hard_ot'],
        label='Hard OT (0.735 mAP)',
        edgecolor='white',
        linewidth=0.5)
    bars3 = ax.bar(
        x + 0.5 * width,
        argmax_ap,
        width,
        color=COLORS['argmax'],
        label='Sinkhorn argmax ε=5 (0.744 mAP)',
        edgecolor='white',
        linewidth=0.5)
    bars4 = ax.bar(
        x + 1.5 * width,
        stoch_ap,
        width,
        color=COLORS['stochastic'],
        label='Stochastic ε=5 (0.750 mAP)',
        edgecolor='white',
        linewidth=0.5)

    ax.set_ylabel('Average Precision')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax.set_ylim(0.50, 0.90)
    ax.legend(loc='lower left', ncol=2, frameon=True, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Group labels
    group_positions = {}
    for i, g in enumerate(GROUPS):
        if g not in group_positions:
            group_positions[g] = []
        group_positions[g].append(i)

    for g, positions in group_positions.items():
        mid = (positions[0] + positions[-1]) / 2
        ax.annotate(
            g,
            xy=(mid, 0.505),
            ha='center',
            fontsize=9,
            fontweight='bold',
            color='#666')

    # Arrow showing OT drops 22/24
    ax.annotate(
        'Hard OT degrades\n22 out of 24 classes',
        xy=(10, 0.55),
        fontsize=11,
        fontweight='bold',
        color='#B71C1C',
        ha='center',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='#FFEBEE',
            edgecolor='#B71C1C',
            alpha=0.9))

    ax.set_title(
        'Per-Class AP: Coupling Strategy Comparison', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'figure5_per_class_ap.pdf')
    fig.savefig(FIG_DIR / 'figure5_per_class_ap.png')
    plt.close(fig)
    print('✓ Figure 5: Per-class AP saved')


# ---- Figure 6: Entropy and Variance Comparison ----
def fig6_entropy():
    """Bar chart comparing entropy/variance across coupling strategies."""
    # Data from velocity_entropy_results.json
    categories = [
        'H(V|Z)', 'H(V|X_t)', 'Total Var', 'Between Var', 'Within Var'
    ]
    random_vals = [3.841, 3.381, 5.169, 1.781, 3.388]
    hardot_vals = [0.000, 0.000, 2.401, 0.538, 1.864]
    stoch1_vals = [3.786, 3.185, 4.475, 1.183, 3.293]
    stoch5_vals = [3.839, 3.345, 5.045, 1.665, 3.380]

    x = np.arange(len(categories))
    width = 0.2

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))

    ax.bar(
        x - 1.5 * width,
        random_vals,
        width,
        color=COLORS['random'],
        label='Random',
        edgecolor='white')
    ax.bar(
        x - 0.5 * width,
        hardot_vals,
        width,
        color=COLORS['hard_ot'],
        label='Hard OT',
        edgecolor='white')
    ax.bar(
        x + 0.5 * width,
        stoch1_vals,
        width,
        color=COLORS['argmax'],
        label='Stochastic ε=1',
        edgecolor='white')
    ax.bar(
        x + 1.5 * width,
        stoch5_vals,
        width,
        color=COLORS['stochastic'],
        label='Stochastic ε=5',
        edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel('Value')
    ax.legend(frameon=True)
    ax.grid(axis='y', alpha=0.3)

    # Annotate the collapse
    ax.annotate(
        'Diversity\nCollapse',
        xy=(0, 0.5),
        fontsize=11,
        fontweight='bold',
        color='#B71C1C',
        ha='center',
        bbox=dict(
            boxstyle='round,pad=0.3', facecolor='#FFEBEE',
            edgecolor='#B71C1C'))
    ax.annotate(
        'Recovered\nby Stochastic',
        xy=(0, 3.4),
        fontsize=10,
        fontweight='bold',
        color='#1B5E20',
        ha='center',
        bbox=dict(
            boxstyle='round,pad=0.3', facecolor='#E8F5E9',
            edgecolor='#1B5E20'))

    ax.set_title(
        'Mechanism Verification: Entropy & Variance Statistics',
        fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'figure6_entropy.pdf')
    fig.savefig(FIG_DIR / 'figure6_entropy.png')
    plt.close(fig)
    print('✓ Figure 6: Entropy/variance saved')


# ---- Figure 7: D_eff (Effective Match Count) ----
def fig7_deff():
    """Plot D_eff from computed Sinkhorn transport matrices."""
    deff = load_json('deff_results.json')
    if not deff or 'results' not in deff:
        print(
            'WARNING: deff_results.json not found. Run compute_deff.py first.')
        return

    results = deff['results']
    eps_vals = np.array([r['epsilon'] for r in results])
    deff_argmax = np.array([r['D_eff_argmax_mean'] for r in results])
    deff_stoch = np.array([r['D_eff_stoch_mean'] for r in results])
    deff_argmax_std = np.array([r['D_eff_argmax_std'] for r in results])
    deff_stoch_std = np.array([r['D_eff_stoch_std'] for r in results])

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

    ax.errorbar(
        eps_vals,
        deff_argmax,
        yerr=deff_argmax_std,
        fmt='s-',
        color=COLORS['hard_ot'],
        lw=2.5,
        markersize=8,
        capsize=4,
        label='Sinkhorn + Argmax')
    ax.errorbar(
        eps_vals,
        deff_stoch,
        yerr=deff_stoch_std,
        fmt='o-',
        color=COLORS['stochastic'],
        lw=2.5,
        markersize=8,
        capsize=4,
        label='Sinkhorn Sampling')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\varepsilon$ (log scale)')
    ax.set_ylabel(r'Effective Match Count $D_{\mathrm{eff}}$')
    ax.set_title(
        r'$D_{\mathrm{eff}}$ vs $\varepsilon$: Argmax Destroys Diversity Control',
        fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, alpha=0.3, which='both')

    # Annotation
    ax.annotate(
        'Argmax: D_eff remains\nconstrained across ε',
        xy=(5, deff_argmax[eps_vals == 5][0]),
        xytext=(15, deff_argmax[eps_vals == 5][0] - 5),
        fontsize=10,
        color='#B71C1C',
        ha='center',
        arrowprops=dict(arrowstyle='->', color='#B71C1C'),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.annotate(
        'Stochastic: D_eff grows with ε',
        xy=(5, deff_stoch[eps_vals == 5][0]),
        xytext=(15, deff_stoch[eps_vals == 5][0] + 3),
        fontsize=10,
        color='#1B5E20',
        ha='center',
        arrowprops=dict(arrowstyle='->', color='#1B5E20'),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'figure_deff.pdf')
    fig.savefig(FIG_DIR / 'figure_deff.png')
    plt.close(fig)
    print('✓ Figure 7: D_eff saved')


# ---- Main ----
if __name__ == '__main__':
    print('Generating figures for BMVC 2026 paper...')
    print(f'Output directory: {FIG_DIR}')

    per_class = load_per_class_ap()
    if not per_class:
        print(
            'ERROR: per_class_ap_results.json not found. Run per_class_ap.py first.'
        )
        sys.exit(1)

    fig1_overview(per_class)
    fig2_coupling()
    fig3_epsilon()
    fig4_dataset()
    fig5_per_class_ap(per_class)
    fig6_entropy()
    fig7_deff()

    print(
        f"\nDone! {len(list(FIG_DIR.glob('*.pdf')))} PDF figures saved to {FIG_DIR}"
    )
