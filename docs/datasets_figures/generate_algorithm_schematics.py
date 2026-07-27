"""Generate experiment-lineage algorithm schematics (top-conference style).

Outputs (PNG, EN + ZH versions):
  experiment_lineage_schematics[_zh].png       (combined 2x4 grid, 7 panels)
  panel_{a..g}_<name>[_zh].png                 (individual panels)

7 panels (Box Refine Net removed — ΔmAP ≈ 0, on par with baseline):
  (a) DiffusionDet DDPM (root baseline)
  (b) RF + Heun + Shifted + AdaLN-Zero
  (c) DPM-Solver++ (RF multistep)  — efficiency: iso-quality, NFE -37%
  (d) Hard OT Coupling
  (e) Sinkhorn Stochastic OT
  (f) Focal Loss gamma=3
  (g) OT Flow Coupling

ZH version: annotations in Chinese, proper nouns kept in English.
"""
from __future__ import annotations

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
import numpy as np

# ────────────────────────── Language helper ──────────────────────────
LANG = 'en'

def T(en: str, zh: str) -> str:
    """Return text in current LANG. Proper nouns stay English in both."""
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


# Academic muted palette
C_NOISE = '#6B7280'
C_DATA = '#2E5C8A'
C_ACCENT = '#D97706'
C_POS = '#059669'
C_NEG = '#DC2626'
C_PURPLE = '#7C3AED'
C_TEAL = '#0D9488'
C_LIGHT = '#E5E7EB'


# ────────────────────────── Helpers ──────────────────────────

def style_ax(ax, title, delta_str=None, delta_color=C_POS):
    ax.set_title(title, pad=10, loc='left', fontsize=11, fontweight='bold')
    if delta_str:
        ax.text(1.0, 1.02, delta_str, transform=ax.transAxes,
                fontsize=8.5, fontweight='bold', color=delta_color,
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec=delta_color, lw=0.8))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def add_arrow(ax, p0, p1, color=C_DATA, lw=1.4, style='-|>',
              mutation_scale=12, alpha=1.0, ls='-'):
    arrow = FancyArrowPatch(p0, p1, arrowstyle=style,
                            color=color, lw=lw, alpha=alpha,
                            mutation_scale=mutation_scale,
                            shrinkA=0, shrinkB=0, linestyle=ls,
                            zorder=3)
    ax.add_patch(arrow)


# ════════════════════════════════════════════════════════════
# Panel (a): DiffusionDet DDPM
# ════════════════════════════════════════════════════════════
def panel_ddpm(ax):
    style_ax(ax,
             T('(a) DiffusionDet DDPM  (root baseline)',
               '(a) DiffusionDet DDPM（根基线）'),
             '0.729 ± 0.003', C_DATA)

    n = 6
    xs = np.linspace(0.12, 0.88, n)
    y = 0.60
    for i, x in enumerate(xs):
        if i == 0:
            color, label = C_DATA, r'$x_0$'
        elif i == n - 1:
            color, label = C_NOISE, r'$x_T$'
        else:
            color, label = C_NOISE, r'$x_{{{}}}$'.format(i)
        ax.add_patch(Circle((x, y), 0.035, fc=color, ec='white', lw=1.2, zorder=4))
        ax.text(x, y, label, ha='center', va='center',
                fontsize=7, color='white', fontweight='bold', zorder=5)

    for i in range(n - 1):
        add_arrow(ax, (xs[i] + 0.04, y + 0.03), (xs[i+1] - 0.04, y + 0.03),
                  color=C_NOISE, lw=1.0, mutation_scale=10)
    ax.text(0.5, y + 0.13,
            T(r'$q(x_t \mid x_{t-1})$  forward noising',
              r'$q(x_t \mid x_{t-1})$  前向加噪'),
            ha='center', fontsize=7.5, color=C_NOISE, style='italic')

    for i in range(n - 1, 0, -1):
        add_arrow(ax, (xs[i] - 0.04, y - 0.03), (xs[i-1] + 0.04, y - .03),
                  color=C_ACCENT, lw=1.4, mutation_scale=11)
    ax.text(0.5, y - 0.13,
            T(r'$p_\theta(x_{t-1} \mid x_t)$  reverse denoising',
              r'$p_\theta(x_{t-1} \mid x_t)$  反向去噪'),
            ha='center', fontsize=7.5, color=C_ACCENT, style='italic')

    ax.text(0.5, 0.22,
            T('DDIM sampler,  1 step (sampling_timesteps=1)',
              'DDIM 采样器, 1 步 (sampling_timesteps=1)'),
            ha='center', fontsize=7.8, color='#374151',
            bbox=dict(boxstyle='round,pad=0.3', fc=C_LIGHT, ec='none'))
    ax.text(0.5, 0.12,
            T('step-aligned check:  4-step = 8-step = 0.729  (no gain from more steps)',
              '步数对齐验证: 4 步 = 8 步 = 0.729（加步数不提升）'),
            ha='center', fontsize=7, color=C_POS, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc='#ECFDF5', ec=C_POS, lw=0.7))
    ax.text(0.5, 0.04,
            T('time conditioning:  scale_shift',
              '时间条件: scale_shift'),
            ha='center', fontsize=7.5, color='#6B7280', style='italic')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


# ════════════════════════════════════════════════════════════
# Panel (b): RF + Heun + Shifted + AdaLN-Zero
# ════════════════════════════════════════════════════════════
def panel_rf_heun(ax):
    style_ax(ax, '(b) RF + Heun + Shifted + AdaLN-Zero',
             T('+0.017  (step-aligned)', '+0.017（步数对齐）'), C_POS)

    y_traj = 0.74
    t_vals = np.linspace(0.0, 1.0, 100)
    ax.plot(0.15 + 0.7 * (1 - t_vals), y_traj + 0.0 * t_vals,
            color=C_DATA, lw=2.2, zorder=3)

    ax.add_patch(Circle((0.85, y_traj), 0.038, fc=C_NOISE, ec='white', lw=1.2, zorder=4))
    ax.text(0.85, y_traj, r'$x_1$', ha='center', va='center', fontsize=7.5,
            color='white', fontweight='bold', zorder=5)
    ax.text(0.85, y_traj + 0.07,
            T(r'$t{=}1$  (noise)', r'$t{=}1$（噪声）'),
            ha='center', fontsize=7, color=C_NOISE)

    ax.add_patch(Circle((0.15, y_traj), 0.038, fc=C_DATA, ec='white', lw=1.2, zorder=4))
    ax.text(0.15, y_traj, r'$x_0$', ha='center', va='center', fontsize=7.5,
            color='white', fontweight='bold', zorder=5)
    ax.text(0.15, y_traj + 0.07,
            T(r'$t{=}0$  (data)', r'$t{=}0$（数据）'),
            ha='center', fontsize=7, color=C_DATA)

    heun_ts = np.linspace(1.0, 0.0, 5)
    heun_xs = 0.15 + 0.7 * (1 - heun_ts)
    y_mid = y_traj - 0.06
    for i in range(len(heun_xs) - 1):
        mid_x = (heun_xs[i] + heun_xs[i+1]) / 2
        ax.plot([heun_xs[i], mid_x], [y_traj, y_mid],
                color=C_ACCENT, lw=0.8, ls=':', alpha=0.7, zorder=2)
        ax.add_patch(Circle((mid_x, y_mid), 0.011,
                            fc=C_ACCENT, ec='white', lw=0.5, zorder=4))
        add_arrow(ax, (heun_xs[i] + 0.012, y_traj),
                  (heun_xs[i+1] - 0.012, y_traj),
                  color=C_ACCENT, lw=1.6, mutation_scale=10)
    ax.text(0.5, y_mid - 0.05,
            T('Heun:  4 steps,  8 NFE  (2 NFE / step,  Euler predict $\\rightarrow$ midpoint corrector)',
              'Heun: 4 步, 8 NFE（2 NFE/步, Euler 预测 $\\rightarrow$ 中点校正）'),
            ha='center', fontsize=7, color=C_ACCENT, style='italic')

    sub_ax = ax.inset_axes([0.06, 0.08, 0.40, 0.26])
    t = np.linspace(0, 1, 100)
    sub_ax.plot(t, t, color=C_NOISE, lw=1.2, ls='--', label='linear')
    sigma_shift = t * np.exp(3.0 * (1 - t)) / (t * np.exp(3.0 * (1 - t)) + (1 - t))
    sub_ax.plot(t, sigma_shift, color=C_PURPLE, lw=1.8, label='shifted (s=3)')
    sub_ax.set_xlabel('t', fontsize=6.5)
    sub_ax.set_ylabel(r'$\sigma(t)$', fontsize=6.5)
    sub_ax.legend(fontsize=6, loc='upper left', framealpha=0.9)
    sub_ax.set_title(T('Shifted schedule', 'Shifted 调度'), fontsize=7.5, pad=2)
    sub_ax.tick_params(labelsize=6)

    box = FancyBboxPatch((0.56, 0.10), 0.38, 0.24,
                         boxstyle='round,pad=0.01', fc=C_LIGHT, ec=C_TEAL, lw=1.0)
    ax.add_patch(box)
    ax.text(0.75, 0.30, 'AdaLN-Zero', ha='center', fontsize=7.5,
            fontweight='bold', color=C_TEAL)
    ax.text(0.75, 0.245, r'$h = (1+\gamma) x + \beta$',
            ha='center', fontsize=7.5, color='#374151')
    ax.text(0.75, 0.19, r'$\gamma \sim \mathcal{N}(0, 10^{-5})$  init $\rightarrow$ 0',
            ha='center', fontsize=6.8, color='#6B7280')
    ax.text(0.75, 0.135,
            T('zero-init residual path', '零初始化残差路径'),
            ha='center', fontsize=6.8, color=C_TEAL, style='italic')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


# ════════════════════════════════════════════════════════════
# Panel (c): DPM-Solver++ (RF multistep)
# ════════════════════════════════════════════════════════════
def panel_dpm_solver(ax):
    style_ax(ax,
             T('(c) DPM-Solver++  (RF multistep)',
               '(c) DPM-Solver++（RF 多步法）'),
             T('iso-quality,  NFE -37%', '同质, NFE -37%'), C_POS)

    y_axis = 0.50
    ax.annotate('', xy=(0.92, y_axis), xytext=(0.08, y_axis),
                arrowprops=dict(arrowstyle='-|>', color='#374151', lw=1.0))
    ax.text(0.93, y_axis, r'$t$', ha='left', va='center', fontsize=8, color='#374151')
    ax.text(0.92, y_axis + 0.05,
            T(r'$1$ (noise)', r'$1$（噪声）'),
            ha='center', fontsize=7, color=C_NOISE)
    ax.text(0.08, y_axis + 0.05,
            T(r'$0$ (data)', r'$0$（数据）'),
            ha='center', fontsize=7, color=C_DATA)

    steps = np.linspace(1.0, 0.0, 7)
    step_xs = 0.08 + 0.84 * (1 - steps)
    for i, (s, sx) in enumerate(zip(steps, step_xs)):
        col = C_DATA if i == len(steps)-1 else (C_NOISE if i == 0 else '#9CA3AF')
        ax.add_patch(Circle((sx, y_axis), 0.016, fc=col, ec='white', lw=0.8, zorder=4))
        ax.text(sx, y_axis - 0.05, r'$t_{{{}}}$'.format(i), ha='center',
                fontsize=6.5, color='#6B7280')

    y_hist_top = 0.92
    y_hist_bot = 0.72
    x0_pred_vals = y_hist_bot + (y_hist_top - y_hist_bot) * (0.3 + 0.4 * np.sin(np.pi * (1 - steps)) ** 0.5)
    fine_t = np.linspace(0, 1, 200)
    fine_x = np.interp(fine_t, np.linspace(0, 1, len(step_xs)), step_xs)
    fine_y = np.interp(fine_t, np.linspace(0, 1, len(x0_pred_vals)), x0_pred_vals)
    ax.plot(fine_x, fine_y, color=C_ACCENT, lw=1.6, alpha=0.45, zorder=2)
    for sx, hy in zip(step_xs, x0_pred_vals):
        ax.add_patch(Circle((sx, hy), 0.013, fc=C_ACCENT, ec='white', lw=0.6, zorder=5))
    ax.text(0.5, y_hist_top + 0.04,
            T(r'$\hat{x}_0^{\mathrm{pred}}(t)$  history  $\rightarrow$  polynomial interpolation',
              r'$\hat{x}_0^{\mathrm{pred}}(t)$  历史预测  $\rightarrow$  多项式插值'),
            ha='center', fontsize=7, color=C_ACCENT, style='italic')

    n_idx = 3
    ax.annotate('', xy=(step_xs[n_idx], x0_pred_vals[n_idx]),
                xytext=(step_xs[n_idx-1], x0_pred_vals[n_idx-1]),
                arrowprops=dict(arrowstyle='<->', color=C_POS, lw=1.4))
    ax.text((step_xs[n_idx]+step_xs[n_idx-1])/2,
            (x0_pred_vals[n_idx]+x0_pred_vals[n_idx-1])/2 + 0.025,
            T('order=2 pair', 'order=2 配对'),
            ha='center', fontsize=6.5, color=C_POS, fontweight='bold')

    ax.text(0.5, 0.32,
            r'$x_{n+1} = \frac{t_{n+1}}{t_n} x_n + \left(1 - \frac{t_{n+1}}{t_n}\right) \hat{x}_n + \phi_1 D_1$',
            ha='center', fontsize=9, color='#374151')
    ax.text(0.5, 0.22,
            T('semi-linear exact  +  $t$-space polynomial correction',
              '半线性精确积分  +  $t$ 空间多项式校正'),
            ha='center', fontsize=7, color=C_TEAL, style='italic')
    ax.text(0.5, 0.12,
            T('4 steps (5 NFE):  0.746 = Heun (8 NFE)  $\\rightarrow$  same mAP,  37% fewer NFE',
              '4 步 (5 NFE): 0.746 = Heun (8 NFE)  $\\rightarrow$  同等 mAP, NFE 降 37%'),
            ha='center', fontsize=7.2, color=C_POS, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='#ECFDF5', ec=C_POS, lw=0.8))
    ax.text(0.5, 0.04,
            T('6 steps (7-8 NFE):  0.747,  +0.001  (marginal,  NFE-aligned)',
              '6 步 (7-8 NFE): 0.747, +0.001（边际, NFE 对齐）'),
            ha='center', fontsize=6.8, color='#6B7280', style='italic')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


# ════════════════════════════════════════════════════════════
# Panel (d): Hard OT Coupling
# ════════════════════════════════════════════════════════════
def panel_hard_ot(ax):
    style_ax(ax, '(d) Hard OT Coupling', '+0.001', C_POS)

    y_data = 0.76
    n = 5
    xs_data = np.linspace(0.15, 0.85, n)
    for x in xs_data:
        ax.add_patch(Circle((x, y_data), 0.028, fc=C_DATA, ec='white', lw=1.0, zorder=4))
    ax.text(0.07, y_data, r'$x_0$', fontsize=10, color=C_DATA, fontweight='bold', va='center')
    ax.text(0.07, y_data - 0.06, T('data', '数据'), fontsize=6.5, color=C_DATA)

    y_noise = 0.38
    perm = [2, 0, 4, 1, 3]
    xs_noise = np.linspace(0.15, 0.85, n)[perm]
    for x in xs_noise:
        ax.add_patch(Circle((x, y_noise), 0.028, fc=C_NOISE, ec='white', lw=1.0, zorder=4))
    ax.text(0.07, y_noise, r'$x_1$', fontsize=10, color=C_NOISE, fontweight='bold', va='center')
    ax.text(0.07, y_noise - 0.06, T('noise', '噪声'), fontsize=6.5, color=C_NOISE)

    for i in range(n):
        add_arrow(ax, (xs_data[i], y_data - 0.03),
                  (xs_noise[i], y_noise + 0.03),
                  color=C_ACCENT, lw=1.3, mutation_scale=10)

    ax.text(0.5, 0.20, r'$\min_{\pi \in \Pi} \sum_i c(x_0^{(i)}, x_1^{(\pi_i)})$',
            ha='center', fontsize=10, color='#374151',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#D1D5DB', lw=0.6))
    ax.text(0.5, 0.08,
            T('bijective (hard) assignment   $\\cdot$   minibatch OT',
              '双射（硬）配对   $\\cdot$   minibatch OT'),
            ha='center', fontsize=7.5, color=C_ACCENT, style='italic')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


# ════════════════════════════════════════════════════════════
# Panel (e): Sinkhorn Stochastic OT
# ════════════════════════════════════════════════════════════
def panel_sinkhorn(ax):
    style_ax(ax,
             T('(e) Sinkhorn Stochastic OT  (eps=5.0)',
               '(e) Sinkhorn Stochastic OT（ε=5.0）'),
             '+0.002', C_POS)

    sub = ax.inset_axes([0.08, 0.18, 0.40, 0.62])
    n = 6
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P[i, j] = np.exp(-((i - j) ** 2) / 1.8)
    P = P / P.sum(axis=1, keepdims=True)
    sub.imshow(P, cmap='YlOrBr', aspect='auto', vmin=0, vmax=P.max())
    sub.set_xlabel(r'$x_1$ index', fontsize=6.5)
    sub.set_ylabel(r'$x_0$ index', fontsize=6.5)
    sub.set_title(T('coupling  $P \\in \\Sigma$', '耦合矩阵  $P \\in \\Sigma$'),
                  fontsize=7.5, pad=2)
    sub.tick_params(labelsize=6)
    rng = np.random.default_rng(11)
    for i in range(n):
        j = rng.choice(n, p=P[i])
        sub.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, ec=C_POS, lw=1.4))

    y_data = 0.78
    y_noise = 0.28
    xs_data = np.linspace(0.62, 0.92, 4)
    xs_noise = np.linspace(0.62, 0.92, 4)
    for x in xs_data:
        ax.add_patch(Circle((x, y_data), 0.022, fc=C_DATA, ec='white', lw=0.8, zorder=4))
    for x in xs_noise:
        ax.add_patch(Circle((x, y_noise), 0.022, fc=C_NOISE, ec='white', lw=0.8, zorder=4))
    ax.text(0.585, y_data, r'$x_0$', fontsize=9, color=C_DATA, fontweight='bold', va='center')
    ax.text(0.585, y_noise, r'$x_1$', fontsize=9, color=C_NOISE, fontweight='bold', va='center')

    pairs = [(0, 0, 0.5), (0, 1, 0.3), (1, 1, 0.4), (1, 2, 0.4),
             (2, 2, 0.5), (2, 3, 0.3), (3, 3, 0.6)]
    for i, j, w in pairs:
        add_arrow(ax, (xs_data[i], y_data - 0.025),
                  (xs_noise[j], y_noise + 0.025),
                  color=C_POS, lw=0.5 + 2.0 * w, mutation_scale=8, alpha=0.6 + 0.4 * w)

    ax.text(0.77, 0.17,
            T('stochastic samples  ~  $P$', '随机采样  ~  $P$'),
            ha='center', fontsize=6.8, color=C_POS, style='italic')
    ax.text(0.77, 0.08, r'$\min \langle P, C \rangle - \varepsilon H(P)$',
            ha='center', fontsize=8, color='#374151')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


# ════════════════════════════════════════════════════════════
# Panel (f): Focal Loss gamma=3
# ════════════════════════════════════════════════════════════
def panel_focal(ax):
    style_ax(ax, '(f) Focal Loss  $\\gamma=3$', '+0.004', C_POS)

    sub = ax.inset_axes([0.14, 0.18, 0.72, 0.62])
    p = np.linspace(0.01, 0.99, 200)

    def focal(p, gamma, alpha=0.25):
        return -alpha * (1 - p) ** gamma * np.log(p)

    gammas = [(1, C_NOISE, T('$\\gamma=1$  (CE)', '$\\gamma=1$（CE）')),
              (2, C_DATA, T('$\\gamma=2$  (default)', '$\\gamma=2$（默认）')),
              (3, C_ACCENT, T('$\\gamma=3$  (ours)', '$\\gamma=3$（本文）'))]
    for g, c, lbl in gammas:
        lw = 2.4 if g == 3 else 1.3
        ls = '-' if g == 3 else '--'
        sub.plot(p, focal(p, g), color=c, lw=lw, ls=ls, label=lbl)

    sub.set_xlabel(T('predicted probability  $p$  (for positive class)',
                      '预测概率  $p$（正类）'),
                    fontsize=7)
    sub.set_ylabel('focal loss', fontsize=7)
    sub.legend(fontsize=6.5, loc='center right', framealpha=0.95)
    sub.set_title(T('Hard-example weighting  $(1-p)^{\\gamma}$',
                    '难样本加权  $(1-p)^{\\gamma}$'),
                  fontsize=7.5, pad=2)
    sub.tick_params(labelsize=6.5)
    sub.grid(alpha=0.25, lw=0.5)
    sub.set_ylim(0, 2.2)

    sub.annotate(T('$\\gamma=3$ amplifies\nhard (low-p) examples',
                    '$\\gamma=3$ 放大\n难样本（低 $p$）权重'),
                 xy=(0.15, focal(0.15, 3)), xytext=(0.40, 1.55),
                 fontsize=6.5, color=C_ACCENT,
                 arrowprops=dict(arrowstyle='->', color=C_ACCENT, lw=0.9))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


# ════════════════════════════════════════════════════════════
# Panel (g): OT Flow Coupling  [was (h), renumbered after removing Box Refine Net]
# ════════════════════════════════════════════════════════════
def panel_ot_flow(ax):
    style_ax(ax, '(g) OT Flow Coupling', '+0.005', C_POS)

    y_base = 0.68
    t = np.linspace(0, 1, 100)
    x_large = 0.15 + 0.7 * (1 - t)
    x_small = 0.15 + 0.7 * (1 - t) + 0.05 * np.sin(np.pi * (1 - t))

    ax.plot(x_large, y_base + 0.0 * t, color=C_DATA, lw=2.0,
            label=T('large obj  ($\\kappa \\approx 1$)', '大目标（$\\kappa \\approx 1$）'))
    ax.plot(x_small, y_base - 0.10 + 0.0 * t, color=C_ACCENT, lw=2.0,
            label=T('small obj  ($\\kappa > 1$)', '小目标（$\\kappa > 1$）'))

    ax.add_patch(Circle((0.85, y_base), 0.022, fc=C_NOISE, ec='white', lw=0.8, zorder=4))
    ax.add_patch(Circle((0.15, y_base), 0.022, fc=C_DATA, ec='white', lw=0.8, zorder=4))
    ax.add_patch(Circle((0.85, y_base - 0.10), 0.022, fc=C_NOISE, ec='white', lw=0.8, zorder=4))
    ax.add_patch(Circle((0.15, y_base - 0.10), 0.022, fc=C_ACCENT, ec='white', lw=0.8, zorder=4))
    ax.text(0.85, y_base + 0.05, r'$x_1$', ha='center', fontsize=7, color=C_NOISE)
    ax.text(0.13, y_base + 0.05, r'$x_0$', ha='center', fontsize=7, color=C_DATA)

    ax.legend(fontsize=6.5, loc='lower center', ncol=2, framealpha=0.95,
              bbox_to_anchor=(0.55, 0.42))

    sub = ax.inset_axes([0.10, 0.08, 0.42, 0.26])
    s = np.linspace(0.02, 0.15, 100)
    lam = 0.3
    kappa = 1 + lam * (s.max() - s) / s.max()
    sub.plot(s, kappa, color=C_PURPLE, lw=1.8)
    sub.set_xlabel(T('object scale  s', '目标尺度  s'), fontsize=6.5)
    sub.set_ylabel(r'$\kappa(s)$', fontsize=6.5)
    sub.set_title(T('Scale-conditioned $\\kappa(s)$', '尺度调制  $\\kappa(s)$'),
                  fontsize=7.5, pad=2)
    sub.tick_params(labelsize=6)
    sub.axhline(1.0, color=C_NOISE, ls=':', lw=0.8)
    sub.fill_between(s, 1.0, kappa, alpha=0.15, color=C_PURPLE)

    box = FancyBboxPatch((0.60, 0.10), 0.34, 0.24,
                         boxstyle='round,pad=0.01', fc=C_LIGHT, ec=C_POS, lw=1.0)
    ax.add_patch(box)
    ax.text(0.77, 0.30, 'OT Flow coupling', ha='center', fontsize=7.5,
            fontweight='bold', color=C_POS)
    ax.text(0.77, 0.235, r'$\sigma(t, s) = t^{\kappa(s)}$',
            ha='center', fontsize=8, color='#374151')
    ax.text(0.77, 0.175, T('Sinkhorn OT pairing', 'Sinkhorn OT 配对'),
            ha='center', fontsize=7, color='#374151')
    ax.text(0.77, 0.13, T('per-batch transport plan', '逐 batch 传输方案'),
            ha='center', fontsize=6.8, color=C_POS, style='italic')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


# ════════════════════════════════════════════════════════════
# Main: combined grid + individual panels, EN + ZH
# ════════════════════════════════════════════════════════════
PANELS = [
    ('panel_a_ddpm',     panel_ddpm),
    ('panel_b_rf_heun',  panel_rf_heun),
    ('panel_c_dpm_solver', panel_dpm_solver),
    ('panel_d_hard_ot',  panel_hard_ot),
    ('panel_e_sinkhorn', panel_sinkhorn),
    ('panel_f_focal',    panel_focal),
    ('panel_g_ot_flow',  panel_ot_flow),
]


def generate(lang: str, out_dir: str):
    setup_rc(lang)
    suffix = f'_{lang}' if lang != 'en' else ''

    # ── Combined 2x4 grid (7 panels, 8th hidden) ──
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    fig.subplots_adjust(left=0.03, right=0.985, top=0.92, bottom=0.05,
                        wspace=0.10, hspace=0.20)
    for idx, (name, pfunc) in enumerate(PANELS):
        ax = axes.flat[idx]
        ax.set_facecolor('white')
        pfunc(ax)
    # Hide the 8th (unused) axes
    axes.flat[7].set_visible(False)

    fig.suptitle(
        T('Experiment Lineage — Algorithmic Schematics of Each Improvement',
          '实验脉络 — 各改进的底层算法示意图'),
        fontsize=13, fontweight='bold', y=0.975)
    fig.text(0.5, 0.012,
             T('Baseline: DiffusionDet DDPM = 0.729 ± 0.003  ->  RF+Heun+AdaLN = 0.746 ± 0.001  '
               '(chromo dataset, DiffusionDet default aug, 3 seeds)',
               '基线: DiffusionDet DDPM = 0.729 ± 0.003  ->  RF+Heun+AdaLN = 0.746 ± 0.001  '
               '（chromo 数据集, DiffusionDet 默认 aug, 3 seeds）'),
             ha='center', fontsize=8, color='#6B7280', style='italic')
    combined_path = os.path.join(out_dir, f'experiment_lineage_schematics{suffix}.png')
    fig.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'[saved] {combined_path}')
    plt.close(fig)

    # ── Individual panels ──
    for name, pfunc in PANELS:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.set_facecolor('white')
        pfunc(ax)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        path = os.path.join(out_dir, f'{name}{suffix}.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'[saved] {path}')
        plt.close(fig)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    # Remove old Box Refine Net files (panel_g_box_refine was the old panel g)
    old_g = os.path.join(out_dir, 'panel_g_box_refine.png')
    if os.path.exists(old_g):
        os.remove(old_g)
        print(f'[removed] {old_g}')
    old_g_zh = os.path.join(out_dir, 'panel_g_box_refine_zh.png')
    if os.path.exists(old_g_zh):
        os.remove(old_g_zh)
        print(f'[removed] {old_g_zh}')

    generate('en', out_dir)
    generate('zh', out_dir)


if __name__ == '__main__':
    main()
