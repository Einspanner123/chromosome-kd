import matplotlib.pyplot as plt
import numpy as np


def plot_trajectories():
    # 设置基础风格
    plt.rcParams['axes.linewidth'] = 1.5

    plt.figure(figsize=(8, 8))

    # 设置起点(数据)和终点(噪声)
    start = np.array([0.2, 0.2])
    end = np.array([0.8, 0.8])

    # 1. Rectified Flow (直线)
    plt.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color='#3498db',
        lw=5,
        label='Rectified Flow (Straight ODE)',
        zorder=2,
    )

    # 2. DDPM/DDIM (弯曲路径)
    t = np.linspace(0, 1, 100)
    # 构造一条具有代表性的弯曲曲线
    curve_x = start[0] + (end[0] - start[0]) * t
    curve_y = start[1] + (end[1] - start[1]) * t**2 + 0.12 * np.sin(t * np.pi)
    plt.plot(
        curve_x,
        curve_y,
        color='#95a5a6',
        lw=3,
        linestyle='--',
        label='DDPM/DDIM (Curved ODE)',
        zorder=1,
    )

    # 绘制采样点 (RF)
    steps = [0, 0.25, 0.5, 0.75, 1.0]
    for s in steps:
        pt_rf = start + (end - start) * s
        plt.scatter(
            pt_rf[0],
            pt_rf[1],
            color='#3498db',
            s=150,
            edgecolors='black',
            linewidth=1.5,
            zorder=3,
        )

    # 添加标注
    plt.annotate(
        r'Data ($\mathbf{x}_0$)',
        xy=start,
        xytext=(-40, -30),
        textcoords='offset points',
        fontsize=14,
        fontweight='bold',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
    )

    plt.annotate(
        r'Noise ($\mathbf{x}_1$)',
        xy=end,
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=14,
        fontweight='bold',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
    )

    plt.title(
        'Probability Flow Trajectory Comparison',
        fontsize=18,
        fontweight='bold',
        pad=20,
    )
    plt.axis('equal')
    plt.axis('off')
    plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)

    plt.savefig(
        '/home/linkst/workplace/chromo/chromosome-kd/projects/LDMDet/scripts/vis_paper/trajectory_comparison.png',
        dpi=300,
        bbox_inches='tight',
    )
    print('Exquisite trajectory plot saved successfully.')


if __name__ == '__main__':
    plot_trajectories()
