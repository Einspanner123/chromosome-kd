import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_per_class_bars(file_path):
    # 读取 Excel
    df = pd.read_excel(file_path)

    # 提取类别列 (A1 到 Y)
    classes = [
        'A1',
        'A2',
        'A3',
        'B4',
        'B5',
        'C6',
        'C7',
        'C8',
        'C9',
        'C10',
        'C11',
        'C12',
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

    baseline_vals = df.iloc[0][classes].values
    rf_vals = df.iloc[1][classes].values

    # 设置画布
    plt.figure(figsize=(20, 8))
    x = np.arange(len(classes))
    width = 0.35

    # 绘制柱状图
    plt.bar(
        x - width / 2,
        baseline_vals,
        width,
        label='Baseline (DDPM)',
        color='#95a5a6',
        edgecolor='black',
        alpha=0.8,
    )
    plt.bar(
        x + width / 2,
        rf_vals,
        width,
        label='Ours (RF-Shifted)',
        color='#3498db',
        edgecolor='black',
    )

    # 添加细节
    plt.title(
        'Per-class Detection Performance (mAP) Comparison',
        fontsize=22,
        fontweight='bold',
        pad=25,
    )
    plt.xlabel('Chromosome Categories', fontsize=16)
    plt.ylabel('mAP', fontsize=16)
    plt.xticks(x, classes, fontsize=14)
    plt.yticks(np.arange(0, 1.0, 0.1), fontsize=12)
    plt.ylim(0, 0.95)

    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # 添加提升标记 (在柱状图上方标注差值和相对百分比)
    for i in range(len(classes)):
        diff = rf_vals[i] - baseline_vals[i]
        rel_diff = (
            (diff / baseline_vals[i]) * 100 if baseline_vals[i] > 0 else 0
        )

        # 标注绝对提升和相对提升
        plt.text(
            i,
            max(rf_vals[i], baseline_vals[i]) + 0.01,
            f'+{diff:.2f}\n({rel_diff:+.1f}%)',
            ha='center',
            va='bottom',
            fontsize=9,
            color='#e74c3c',
            fontweight='bold',
        )

    plt.legend(
        fontsize=14,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        shadow=True,
    )
    plt.tight_layout()

    # 保存图片
    output_path = '/home/linkst/workplace/chromo/chromosome-kd/projects/LDMDet/scripts/vis_paper/per_class_comparison_bar.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Per-class bar chart saved to {output_path}')


if __name__ == '__main__':
    file_path = (
        '/home/linkst/workplace/chromo/chromosome-kd/compare_class.XLSX'
    )
    plot_per_class_bars(file_path)
