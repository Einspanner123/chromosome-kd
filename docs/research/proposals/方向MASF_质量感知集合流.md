# MASF：质量感知集合流（Mass-Aware Set Flow）

> 状态：Phase-0 结构诊断通过；进入最小机制实现  
> 优先数据：Chromosome20240904 / Dataset 1  
> 诊断基座：A4 RF + DPM-Solver++ seed42

## 1. 问题定义

现有检测器把 $N=500$ 个 proposal 表示成等质量经验测度

$$
\mu_t=\sum_{i=1}^{N}\delta_{x_i(t)}.
$$

RF 只学习坐标速度 $\dot x_i=v_\theta(x_i,t)$，因而沿轨迹隐式保持每个粒子的单位
质量和总质量 $N$。但一张染色体图像的目标测度只有约 $M=47$ 个原子。当前训练通过
有放回地把 $M$ 个 GT 重复到 $N$ 个端点来规避基数不一致；当前推理再依赖分类阈值
和 NMS 删除重复框。这不是集合质量的生成模型，而是“等质量坐标流 + 事后删重”。

MASF 将经验测度扩展为

$$
\mu_t=\sum_i m_i(t)\delta_{x_i(t)},\qquad m_i(t)\ge0,
$$

并采用非平衡连续性方程

$$
\partial_t\mu_t+\nabla\!\cdot(\mu_t v_t)=g_t\mu_t.
$$

沿特征线有

$$
\dot x_i=v_t(x_i),\qquad \frac{d}{dt}\log m_i=g_t(x_i),
$$

允许 proposal 在坐标运输的同时产生或消亡，因此总质量不再被错误地固定为 500。

## 2. Phase-0 实测

诊断程序：`experiments/analysis/mass_flow_phase0.py`。它读取未经过 NMS 的四步轨迹，
统计每步最大类别概率之和、高置信 proposal 数、正确类别匹配的重复数及 NMS 删除量。

来源文件：

- `work_dirs/diagnosis/mass_flow_phase0_chr2024_seed42.json`
- `work_dirs/diagnosis/mass_flow_phase0_chr2024_seed42_score50.json`
- `work_dirs/diagnosis/mass_flow_phase0_chr2024_seed42_iou75.json`

100 张均匀抽样验证图像的结果如下。

| 条件 | 选中 proposals | NMS 删除率 | GT 覆盖率 | 每 GT 额外重复数 |
|---|---:|---:|---:|---:|
| score≥0.05，IoU≥0.50 | 321.52 | 79.64% | 95.33% | 5.06 |
| score≥0.50，IoU≥0.50 | 146.36 | 69.67% | 90.85% | 2.12 |
| score≥0.05，IoU≥0.75 | 321.26 | 79.41% | 89.20% | 4.61 |

平均 GT 数为 47。四个求解步的 proposal 置信质量
$\sum_i\max_c p_{ic}$ 分别为 144.39、148.78、148.56、151.04，即从 GT 总质量的
3.07 倍上升到 3.22 倍，没有随去噪收缩。即使把阈值提高到 0.5，仍有约 146 个
proposal，NMS 需要删除近 70%。高 IoU=0.75 条件下仍存在每 GT 4.61 个额外副本，
因此该现象不能由低分噪声 proposal 单独解释。

诊断证明了“重复质量存在且量级很大”，但尚未证明学习质量动力学一定提高 AP；NMS
可能已经吸收大部分冗余。因此下一阶段仍采用可证伪门控，而不直接进行长周期全量训练。

## 3. 最小可验证实现

### 3.1 质量目标

对 matcher 分给 GT $g$ 的正 proposals 集合 $I_g$，定义数据端质量

$$
m_i^0=\begin{cases}
1/|I_g|,&i\in I_g,\\
0,&i\text{ 为未匹配 proposal}.
\end{cases}
$$

于是每个已覆盖 GT 的总质量严格为 1，$\sum_i m_i^0$ 等于已覆盖 GT 数，而不是动态
匹配正样本数。这与普通 objectness 的关键区别是：普通分类把同一 GT 的多个正样本
都监督为 1，MASF 对它们施加守恒的份额监督。

### 3.2 质量 RF

噪声端令 $m_i^1=1$，采用与坐标 RF 相同的线性条件路径

$$
m_i(t)=(1-t)m_i^0+t m_i^1.
$$

网络预测数据端质量 $\hat m_i^0$。从 $t_n$ 到 $t_{n+1}$ 的 data-prediction 更新为

$$
m_{n+1}=\frac{t_{n+1}}{t_n}m_n+
\left(1-\frac{t_{n+1}}{t_n}\right)\hat m_i^0.
$$

当 $\hat m_i^0$ 沿轨迹恒定时，该更新精确积分线性质量路径。质量状态通过
$\log(m_i+\epsilon)$ 的小型 embedding 注入 proposal feature，使坐标流与质量流真正
耦合，而不是仅在最终 NMS 前增加一个分数头。

### 3.3 损失与推理

第一版使用逐 proposal 质量回归与全局守恒项：

$$
\mathcal L_{mass}=\frac1N\sum_i|\hat m_i^0-m_i^0|
+\lambda_c\frac{(\sum_i\hat m_i^0-M_{covered})^2}{M_{covered}+\epsilon}.
$$

最终排序采用 $p_{ic}\hat m_i^0$；renewal 只在 Phase-1B 才读取质量状态。Phase-1A
先保持坐标、类别、solver 和 renewal 不变，仅验证守恒质量是否能减少重复排序错误。

## 4. 预注册推进标准

1. Phase-1A 先在 Dataset 1 seed42 冻结 A4，仅训练质量分支 12 epoch；
2. 相对 A4 mAP 至少 +0.002，且 NMS 删除率下降至少 10 个百分点，才进入质量状态
   solver 耦合；
3. 若 mAP 不提升但重复率明显下降，只能作为效率/校准支线，不作为精度创新；
4. 若两项均未达到，停止 MASF，不通过调整 NMS 阈值追逐结果；
5. Phase-1B 通过后再做 seed123 和 mini-COCO，不提前占用通用检测资源。
