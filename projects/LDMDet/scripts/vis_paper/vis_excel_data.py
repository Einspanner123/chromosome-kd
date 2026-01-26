import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

def plot_class_radar(df):
    # 提取类别列 (A1 到 Y)
    classes = ['A1', 'A2', 'A3', 'B4', 'B5', 'C6', 'C7', 'C8', 'C9', 'C10', 
               'C11', 'C12', 'D13', 'D14', 'D15', 'E16', 'E17', 'E18', 
               'F19', 'F20', 'G21', 'G22', 'X', 'Y']
    
    num_vars = len(classes)
    
    # 计算角度
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)
    
    # 设置基础风格
    plt.xticks(angles[:-1], classes, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.6, 0.7, 0.8], ["0.6", "0.7", "0.8"], color="grey", size=8)
    plt.ylim(0.5, 0.85)
    
    # 绘制 Baseline
    values_baseline = df.iloc[0][classes].values.flatten().tolist()
    values_baseline += values_baseline[:1]
    ax.plot(angles, values_baseline, linewidth=2, linestyle='solid', label="Baseline (DDPM)", color='#95a5a6')
    ax.fill(angles, values_baseline, '#95a5a6', alpha=0.1)
    
    # 绘制 RF
    values_rf = df.iloc[1][classes].values.flatten().tolist()
    values_rf += values_rf[:1]
    ax.plot(angles, values_rf, linewidth=2, linestyle='solid', label="Ours (RF-Shifted)", color='#3498db')
    ax.fill(angles, values_rf, '#3498db', alpha=0.2)
    
    plt.title("Per-class mAP Performance Comparison", size=20, fontweight='bold', pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.savefig("/home/linkst/workplace/chromo/chromosome-kd/projects/LDMDet/scripts/vis_paper/class_radar.png", dpi=300, bbox_inches='tight')
    print("Class radar chart saved.")

def plot_scale_bars(df):
    scales = ['mAP_s', 'mAP_m', 'mAP_l']
    labels = ['Small', 'Medium', 'Large']
    
    baseline_vals = df.iloc[0][scales].values
    rf_vals = df.iloc[1][scales].values
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 7))
    rects1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#95a5a6', edgecolor='black', alpha=0.8)
    rects2 = ax.bar(x + width/2, rf_vals, width, label='Ours (RF-Shifted)', color='#3498db', edgecolor='black')
    
    ax.set_ylabel('mAP', fontsize=14)
    ax.set_title('Performance Comparison by Scale', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)
    
    # 添加数值标注
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    # 添加提升百分比
    for i in range(len(labels)):
        diff = rf_vals[i] - baseline_vals[i]
        improvement = (diff / baseline_vals[i]) * 100
        ax.text(i, max(rf_vals[i], baseline_vals[i]) + 0.03, f'+{improvement:.1f}%', 
                ha='center', color='#e74c3c', fontweight='bold', fontsize=12)

    plt.ylim(0, 0.85)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("/home/linkst/workplace/chromo/chromosome-kd/projects/LDMDet/scripts/vis_paper/scale_comparison.png", dpi=300, bbox_inches='tight')
    print("Scale comparison bar chart saved.")

if __name__ == "__main__":
    file_path = "/home/linkst/workplace/chromo/chromosome-kd/compare_class.XLSX"
    df = pd.read_excel(file_path)
    plot_class_radar(df)
    plot_scale_bars(df)
