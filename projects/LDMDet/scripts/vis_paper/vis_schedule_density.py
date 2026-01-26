import matplotlib.pyplot as plt
import numpy as np


def shifted_schedule(t, s):
    # 修正逻辑：
    # 当 s > 1 时，我们希望均匀的 t 被映射到更靠近 0 的地方。
    # 原公式 (s*t)/(1+(s-1)*t) 会让 t 变大（靠近 1）。
    # 修正公式为：t_shifted = t / (t + s*(1-t))
    if s == 1.0:
        return t
    return t / (t + s * (1 - t))


def plot_density():
    plt.rcParams["axes.linewidth"] = 1.5

    # 我们定义 x 轴：0 是数据 (Data), 1 是噪声 (Noise)
    t = np.linspace(0, 1, 1000)
    shifts = [1.0, 3.0, 5.0]
    colors = ["#95a5a6", "#3498db", "#e74c3c"]

    # 1. 映射曲线 (我们希望 t_shifted 在靠近 0 的地方更陡峭)
    plt.figure(figsize=(10, 6))
    for s, color in zip(shifts, colors):
        # 修正逻辑：我们 shift 的是“离数据的距离”
        # 这样 s > 1 时，采样点会更靠近 0
        t_shifted = shifted_schedule(t, s)
        plt.plot(t, t_shifted, label=f"Shift $s={s:.1f}$", color=color, lw=3)

    plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    plt.title("Corrected Time-Shifted Schedule", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Inference Step Index (Normalized 0 to 1)", fontsize=14)
    plt.ylabel("Effective Time Step $t$ (0: Data, 1: Noise)", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(
        "/home/linkst/workplace/chromo/chromosome-kd/projects/LDMDet/scripts/vis_paper/schedule_mapping.png",
        dpi=300,
        bbox_inches="tight",
    )

    # 2. 采样密度图 (关键修正)
    plt.figure(figsize=(10, 4))
    # 模拟从 1 (噪声) 到 0 (数据) 的 4 步均匀采样下标
    # 均匀下标: [1.0, 0.75, 0.5, 0.25, 0.0]
    u = np.linspace(1, 0, 5)

    for i, (s, color) in enumerate(zip(shifts, colors)):
        # 应用 Shift 公式
        sampled_t = shifted_schedule(u, s)

        plt.scatter(
            sampled_t,
            np.ones_like(sampled_t) * i,
            color=color,
            s=150,
            edgecolors="black",
            linewidth=1.5,
            zorder=3,
        )
        plt.hlines(i, -0.05, 1.05, colors=color, alpha=0.2, lw=2)

    plt.yticks(range(len(shifts)), [f"$s={s}$" for s in shifts], fontsize=12)
    plt.title(
        "Sampling Density (Concentrated towards Data $t=0$)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Effective Time Step $t$ (0: Data $\leftarrow$ 1: Noise)", fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.gca().invert_xaxis()  # 翻转坐标轴，符合从噪声到数据的直觉
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "/home/linkst/workplace/chromo/chromosome-kd/projects/LDMDet/scripts/vis_paper/sampling_density.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("Fixed density plots saved successfully.")


if __name__ == "__main__":
    plot_density()
