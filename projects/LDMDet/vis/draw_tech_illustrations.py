import os

import matplotlib.pyplot as plt
import numpy as np


def save_fig(name):
    path = f'projects/LDMDet/vis/{name}.png'
    plt.savefig(path, bbox_inches='tight', dpi=150)
    print(f'Saved {path}')
    plt.close()


def draw_rf():
    """1. Rectified Flow: Linear vs Curved path"""
    plt.figure(figsize=(6, 5))

    # Points
    p_noise = np.array([0.8, 0.8])
    p_data = np.array([0.2, 0.2])

    # RF Path (Straight)
    plt.plot([p_noise[0], p_data[0]], [p_noise[1], p_data[1]],
             'r-',
             linewidth=3,
             label='Rectified Flow (Linear)')
    plt.arrow(0.65, 0.65, -0.1, -0.1, head_width=0.03, color='r')

    # Diffusion Path (Curved)
    t = np.linspace(0, 1, 100)
    curved_x = p_data[0] + (p_noise[0] - p_data[0]) * t + 0.15 * np.sin(
        np.pi * t)
    curved_y = p_data[1] + (p_noise[1] - p_data[1]) * t - 0.1 * np.sin(
        np.pi * t)
    plt.plot(
        curved_x,
        curved_y,
        'b--',
        alpha=0.6,
        label='Traditional Diffusion (Curved)')

    plt.scatter(*p_noise, color='black', s=100, zorder=5)
    plt.text(
        p_noise[0] + 0.02,
        p_noise[1],
        'Noise (t=1)',
        fontsize=12,
        fontweight='bold')
    plt.scatter(*p_data, color='green', s=100, zorder=5)
    plt.text(
        p_data[0] - 0.15,
        p_data[1] - 0.05,
        'Data (t=0)',
        fontsize=12,
        fontweight='bold',
        color='green')

    plt.title('Rectified Flow: The Shortest Path', fontsize=14)
    plt.xlabel('Latent Space X')
    plt.ylabel('Latent Space Y')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    save_fig('rf_vs_diffusion')


def draw_heun():
    """2. Heun Solver: Euler vs Heun steps"""
    plt.figure(figsize=(6, 5))

    # True solution (a curve)
    t = np.linspace(0, 1, 100)
    true_path = 0.8 - 0.6 * t**2
    plt.plot(t, true_path, 'k-', alpha=0.3, label='True ODE Trajectory')

    # Euler step
    t_curr, t_next = 0.2, 0.6
    y_curr = 0.8 - 0.6 * t_curr**2
    slope_curr = -1.2 * t_curr
    y_euler = y_curr + slope_curr * (t_next - t_curr)
    plt.plot([t_curr, t_next], [y_curr, y_euler],
             'b--',
             marker='o',
             label='Euler Step (High Error)')

    # Heun step
    slope_next = -1.2 * t_next  # Simplified: assuming we know the next slope
    y_heun = y_curr + 0.5 * (slope_curr + slope_next) * (t_next - t_curr)
    plt.plot([t_curr, t_next], [y_curr, y_heun],
             'r-',
             marker='s',
             linewidth=2,
             label='Heun Step (2nd Order Correction)')

    plt.annotate(
        'Predictor',
        xy=(t_next, y_euler),
        xytext=(t_next + 0.05, y_euler + 0.05),
        arrowprops=dict(arrowstyle='->'))
    plt.annotate(
        'Corrector',
        xy=(t_next, y_heun),
        xytext=(t_next + 0.05, y_heun - 0.05),
        arrowprops=dict(arrowstyle='->'))

    plt.title('Heun Solver: Predictor-Corrector', fontsize=14)
    plt.xlabel('Time t')
    plt.ylabel('State x')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    save_fig('euler_vs_heun')


def draw_logit_coupling():
    """3. Logit-Velocity Coupling"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # BBox Evolution
    ax1.set_title('BBox Space Evolution', fontsize=12)
    ax1.add_patch(
        plt.Rectangle((0.2, 0.2),
                      0.3,
                      0.5,
                      fill=False,
                      edgecolor='blue',
                      alpha=0.3,
                      linestyle='--'))
    ax1.add_patch(
        plt.Rectangle((0.4, 0.3),
                      0.3,
                      0.5,
                      fill=False,
                      edgecolor='red',
                      linewidth=2))
    ax1.annotate(
        '',
        xy=(0.55, 0.55),
        xytext=(0.35, 0.45),
        arrowprops=dict(arrowstyle='->', color='red'))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Logit Evolution
    ax2.set_title('Logit (Score) Space', fontsize=12)
    classes = ['Chr1', 'Chr2', 'Chr3']
    scores_t = [0.2, 0.1, 0.05]
    scores_next = [0.8, 0.05, 0.02]

    x = np.arange(len(classes))
    ax2.bar(
        x - 0.15,
        scores_t,
        0.3,
        label='Score at t_curr',
        color='blue',
        alpha=0.3)
    ax2.bar(x + 0.15, scores_next, 0.3, label='Score at t_next', color='red')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.set_ylabel('Confidence')
    ax2.legend()

    plt.suptitle(
        'Logit-Velocity Coupling: Spatial-Semantic Alignment',
        fontsize=14,
        y=1.05)
    save_fig('logit_velocity_coupling')


def draw_shifted_schedule():
    """4. Shifted Schedule: t vs t_shifted"""
    plt.figure(figsize=(6, 5))

    t = np.linspace(0, 1, 100)
    shifts = [1.0, 3.0, 5.0]
    colors = ['gray', 'red', 'blue']
    styles = ['--', '-', ':']

    for s, c, st in zip(shifts, colors, styles):
        t_shifted = (s * t) / (1 + (s - 1) * t)
        label = f'Shift s={s}' + (' (Default)' if s == 1.0 else '')
        plt.plot(t, t_shifted, color=c, linestyle=st, linewidth=2, label=label)

        # Draw sampling points for s=3.0
        if s == 3.0:
            sample_t = np.linspace(0, 1, 6)
            sample_t_shifted = (s * sample_t) / (1 + (s - 1) * sample_t)
            plt.scatter(
                sample_t, sample_t_shifted, color='red', s=30, zorder=5)
            for st_val in sample_t_shifted:
                plt.axhline(y=st_val, color='red', alpha=0.1, xmin=0, xmax=1)

    plt.title('Shifted Schedule: Focusing on Data End', fontsize=14)
    plt.xlabel('Original Time Step (Uniform)')
    plt.ylabel('Shifted Time Step (Actual Sampling)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)

    plt.annotate(
        'More steps near Data (t=0)',
        xy=(0.2, 0.4),
        xytext=(0.4, 0.2),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

    save_fig('shifted_schedule')


if __name__ == '__main__':
    os.makedirs('projects/LDMDet/vis', exist_ok=True)
    draw_rf()
    draw_heun()
    draw_logit_coupling()
    draw_shifted_schedule()
