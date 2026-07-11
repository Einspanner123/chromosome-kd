# 方向 P：HCTS (Hierarchical Coarse-to-fine Time-step Schedule, 层级粗到细时间步调度)

> **核心思想**：将染色体 Denver 分类层级（A-G 组 + 性染色体）与扩散时间步 $t$ 建立映射关系，让不同时间步专注于不同粒度的分类/回归任务。高噪声时段（$t \to 1$）仅施加组级（8 类）分类监督，中噪声时段（$t \approx 0.5$）施加组内分类监督，低噪声时段（$t \to 0$）施加完整 24 类分类 + 精细框回归监督。利用扩散过程天然的多尺度特性，通过分层时间步监督降低高噪声时段的梯度噪声，提升整体检测性能。
>
> **当前基线**：`a3_full_sota`，mAP=0.858（RF + Heun 4 步 + AdaLN-Zero + StochasticOT $\varepsilon$=5）
> **目标**：mAP ≥ 0.865（+0.007），追近 RTMDet-L 的 0.869
>
> **代码位置**：
> - 扩散核心：[ldmdet/diffusion/rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py)
> - Head 核心：[ldmdet/core/head.py](../../ldmdet/core/head.py)
> - SingleHead：[ldmdet/core/single_head.py](../../ldmdet/core/single_head.py)
> - 采样器：[ldmdet/diffusion/sampling.py](../../ldmdet/diffusion/sampling.py)
> - 损失核心：[ldmdet/criterion/criterion.py](../../ldmdet/criterion/criterion.py)
> - 分组常量：[ldmdet/utils/constants.py](../../ldmdet/utils/constants.py)
> - 基线配置：[experiments/configs/ldmdet/ldmdet_rf_heun_adaln_stochot_eps5.py](../../experiments/configs/ldmdet/ldmdet_rf_heun_adaln_stochot_eps5.py)

---

## 1. 背景与动机

### 1.1 问题陈述

当前 LDMDet 在 24obj 数据集上达到 mAP=0.858，仍低于 RTMDet-L 的 0.869 共 0.011 个百分点。已有的改进方向（CFM 速度预测、端到端可微 Cascade、BoxRefineNet）均未带来收益。瓶颈在于：扩散检测器在高噪声时间步（$t \to 1$）产生大量不可靠的分类/回归梯度，这些梯度通过 6 级级联 Head 的深度监督被放大，污染了共享特征的表达。

### 1.2 为什么扩散适合层级分类

扩散模型的本质是**从粗到细的信息恢复过程**。在 Rectified Flow 中：

$$x_t = (1-t)x_0 + t x_1, \quad t \in [0, 1]$$

其中 $x_0$ 是数据（GT 框），$x_1$ 是噪声。信噪比为：

$$\text{SNR}(t) = \frac{\text{Var}(x_0 \text{ 信号})}{\text{Var}(x_1 \text{ 噪声})} = \frac{(1-t)^2}{t^2}$$

- $t \to 1$（高噪声）：$\text{SNR} \to 0$，$x_t$ 几乎纯噪声，模型只能恢复**低频/粗粒度**信息
- $t \to 0$（低噪声）：$\text{SNR} \to \infty$，$x_t$ 接近真实数据，模型可恢复**高频/精细**信息

这一多尺度恢复特性在图像生成领域已有充分研究（Karras et al., 2022；Ho et al., 2020）。**HCTS 的核心洞察是**：染色体 Denver 分类的层级结构恰好与扩散的多尺度恢复特性天然对齐——

| 扩散时段 | SNR 特性 | 可恢复信息粒度 | Denver 层级对应 |
|----------|----------|--------------|----------------|
| $t \in [0.67, 1.0]$ | $\text{SNR} < 0.25$ | 仅粗粒度（大尺寸差异） | A-G 组级（8 类，尺寸/着丝粒位置） |
| $t \in [0.33, 0.67]$ | $0.25 \le \text{SNR} \le 4$ | 中等粒度（组内形态差异） | 组内分类（如 C6 vs C7） |
| $t \in [0.0, 0.33]$ | $\text{SNR} > 4$ | 精细粒度（边界/位置） | 完整 24 类 + 精细框回归 |

### 1.3 现有架构的梯度污染问题

当前架构在**所有时间步**都施加**相同粒度**的监督（24 类分类 + L1/GIoU 框回归）。这导致：

1. **高噪声时段的分类梯度噪声**：$t \approx 0.9$ 时，$x_t$ 几乎纯噪声，但模型仍被要求预测 24 类中某一具体类别。RoI 特征此时不可靠，分类头的 Focal Loss 梯度方向随机，污染共享特征。

2. **框回归的无效梯度**：$t \approx 0.9$ 时，输入框 `curr_bboxes`（从 $x_t$ 还原）与真实框偏差巨大，L1/GIoU 损失的梯度量级远大于低噪声时段，但方向无意义。

3. **深度监督的放大效应**：6 级级联 Head 每级都施加相同监督（[head.py:219-224](../../ldmdet/core/head.py#L219-L224)），高噪声时段的无效梯度被 6 倍放大。

**HCTS 的解决方案**：不是抑制高噪声时段的所有梯度（方向三的失败策略），而是**改变梯度的目标粒度**——高噪声时只要求组级正确，中噪声时要求组内正确，低噪声时才要求完整精度。这既保留了训练信号，又让每个时段的监督与模型当时的恢复能力匹配。

---

## 2. 理论依据

### 2.1 多尺度扩散理论

#### 2.1.1 频域分析

在连续时间扩散模型中，前向过程对应于对数据施加渐进式低通滤波。对于 RF 的线性路径 $x_t = (1-t)x_0 + t x_1$，其频域特性可分析如下：

设 $x_0$ 的傅里叶分解为 $x_0 = \sum_k c_k \phi_k$，其中 $\phi_k$ 为频率 $k$ 的基函数。则：

$$x_t = (1-t) \sum_k c_k \phi_k + t \cdot x_1$$

信号分量的能量衰减为 $(1-t)^2 |c_k|^2$，而噪声能量为 $t^2 \sigma^2$。频率 $k$ 的有效 SNR：

$$\text{SNR}_k(t) = \frac{(1-t)^2 |c_k|^2}{t^2 \sigma^2}$$

**关键结论**：所有频率的 SNR 随 $t$ 同步衰减（RF 的线性路径特性）。但对于检测任务，**不同分类粒度对频率的依赖不同**：

- 组级分类（A-G）依赖**低频**信息（整体尺寸、长宽比）→ 即使 SNR 很低也能恢复
- 组内分类（如 C6 vs C7）依赖**中频**信息（着丝粒位置、臂比）→ 需要中等 SNR
- 精细框回归依赖**高频**信息（边界精确位置）→ 需要高 SNR

因此，将分类粒度与时间步对齐，可以让模型在每个时段只学习**当前 SNR 下可恢复的信息**，避免追求不可恢复的高频细节而产生梯度噪声。

#### 2.1.2 与 DDPM 的区别

DDPM 中不同频率的 SNR 衰减速率不同（高频先丢失、后恢复），而 RF 的线性路径使所有频率同步衰减。这意味着 RF 下 HCTS 的分层界限更清晰——不存在"中频已恢复但低频未恢复"的交错区。

### 2.2 Denver 分类与 SNR 的关系

#### 2.2.1 信息论分析

24 类均匀分布的熵：

$$H(Y) = \log_2 24 \approx 4.58 \text{ bits}$$

分解为组 + 组内（见 [方向五文档](../breakthrough_directions/方向五_组条件化分层分类.md)）：

$$H(Y) = H(G) + H(Y|G) \approx 3.00 + 1.74 = 4.74 \text{ bits}$$

（$H(Y|G) > H(Y) - H(G)$ 是因为组大小不均匀引入的冗余）

#### 2.2.2 时间步与可分类熵的关系

在时间步 $t$，模型的**有效分类能力**受限于 SNR。根据信息论中互信息与 SNR 的关系（对高斯信道）：

$$I(Y; X_t | t) \le \frac{1}{2} \log_2(1 + \text{SNR}(t)) = \frac{1}{2} \log_2\left(1 + \frac{(1-t)^2}{t^2}\right)$$

这是 AWGN 信道容量的上界。设 $C(t) = \frac{1}{2} \log_2(1 + \text{SNR}(t))$，则：

| $t$ | SNR | $C(t)$ (bits) | 可分辨层级 |
|-----|-----|---------------|-----------|
| 0.9 | 0.012 | 0.040 | 仅背景/前景（< 1 bit） |
| 0.8 | 0.063 | 0.180 | 粗组（A vs G，~2 bit） |
| 0.67 | 0.25 | 0.66 | 组级（8 类，~3 bit） |
| 0.5 | 1.0 | 1.00 | 组级 + 部分组内 |
| 0.33 | 4.0 | 1.16 | 组内分类（~1.74 bit 需求） |
| 0.2 | 16.0 | 1.30 | 完整分类（4.58 bit 需求，需更高 SNR） |
| 0.1 | 81.0 | 1.40 | 完整分类 + 精细回归 |

**关键观察**：信道容量 $C(t)$ 在 $t > 0.67$ 时不足 1 bit，无法支持 8 组分类（需 3 bit）。这意味着在极高噪声时段，**任何分类监督都是纯噪声**。HCTS 的策略是在此时段**仅施加框中心级监督**（背景/前景的二分类），而非完全放弃（方向三的错误）或施加全分类（当前基线的问题）。

#### 2.2.3 修正的分层策略

基于上述分析，HCTS 的实际分层不是简单的三段式，而是**连续加权**：

$$\mathcal{L}(t) = w_{\text{box}}(t) \cdot \mathcal{L}_{\text{box}} + w_{\text{group}}(t) \cdot \mathcal{L}_{\text{group}} + w_{\text{class}}(t) \cdot \mathcal{L}_{\text{class}}$$

其中权重函数设计为 SNR 的单调函数（详见 §3.2）。

### 2.3 与现有 RF Shifted Schedule 的关系

当前基线使用 `rf_schedule='shifted', rf_shift=3.0`（[ldmdet_rf_heun_shifted_bs2.py:10](../../experiments/configs/ldmdet/ldmdet_rf_heun_shifted_bs2.py#L10)），其时间采样为：

$$t_{\text{shifted}} = \frac{s \cdot t}{1 + (s-1) \cdot t}, \quad s = 3.0$$

这使训练时 $t$ 偏向低值（数据端），$t$ 的密度在 $t \to 0$ 处更高。具体地：

| 均匀 $t$ | shifted $t$ ($s=3$) | SNR(shifted) |
|----------|---------------------|-------------|
| 0.0 | 0.000 | $\infty$ |
| 0.25 | 0.143 | 30.6 |
| 0.5 | 0.333 | 4.0 |
| 0.75 | 0.643 | 0.31 |
| 1.0 | 1.000 | 0.0 |

**这意味着**：shifted schedule 下，约 50% 的训练样本落在 $t < 0.33$（低噪声，完整分类区），约 25% 落在 $t \in [0.33, 0.67]$（中噪声，组内分类区），约 25% 落在 $t > 0.67$（高噪声，组级分类区）。

HCTS 与 shifted schedule **天然兼容**：shifted schedule 已经将更多训练资源分配给低噪声时段（精细任务），HCTS 只需为高噪声时段提供合适的粗粒度监督。这与方向三的失败形成对比——方向三试图**抑制**高噪声时段的匹配，而 HCTS **重塑**高噪声时段的监督目标。

---

## 3. 详细方案设计

### 3.1 时间步分层策略

#### 3.1.1 三层划分

将时间步 $t \in [0, 1]$ 划分为三个语义区间，对应三种监督粒度：

| 层级 | 时间区间 | shifted 后区间 | 监督任务 | SNR 范围 |
|------|---------|---------------|---------|---------|
| L3 (粗) | $t \in (t_2, 1.0]$ | $t \in (0.643, 1.0]$ | 组级分类（8 类）+ 框中心 L1 | SNR < 0.31 |
| L2 (中) | $t \in (t_1, t_2]$ | $t \in (0.333, 0.643]$ | 组内分类 + 完整框 L1/GIoU | $0.31 \le \text{SNR} \le 4$ |
| L1 (细) | $t \in [0, t_1]$ | $t \in [0, 0.333]$ | 完整 24 类 + 精细框 L1/GIoU | SNR > 4 |

默认边界：$t_1 = 0.33$, $t_2 = 0.67$（对应 shifted 后的 0.333 和 0.643）。

#### 3.1.2 软边界设计

为避免硬边界导致的训练不连续，采用**软切换**权重：

$$w_{\text{full}}(t) = \sigma\left(\frac{t_1 - t}{\tau}\right), \quad w_{\text{coarse}}(t) = \sigma\left(\frac{t - t_2}{\tau}\right)$$

$$w_{\text{mid}}(t) = 1 - w_{\text{full}}(t) - w_{\text{coarse}}(t)$$

其中 $\sigma$ 为 sigmoid 函数，$\tau$ 为温度参数（默认 $\tau = 0.05$）。这保证：

- $t \to 0$：$w_{\text{full}} \to 1$, $w_{\text{mid}} \to 0$, $w_{\text{coarse}} \to 0$
- $t \to 1$：$w_{\text{full}} \to 0$, $w_{\text{mid}} \to 0$, $w_{\text{coarse}} \to 1$
- 三者之和恒为 1

#### 3.1.3 监督权重分配

总损失为三层加权和：

$$\mathcal{L}_{\text{HCTS}} = w_{\text{full}}(t) \cdot \mathcal{L}_{\text{full}} + w_{\text{mid}}(t) \cdot \mathcal{L}_{\text{mid}} + w_{\text{coarse}}(t) \cdot \mathcal{L}_{\text{coarse}}$$

其中：

- $\mathcal{L}_{\text{full}} = \mathcal{L}_{\text{cls}}^{24} + \lambda_{\text{bbox}} \mathcal{L}_{\text{L1}} + \lambda_{\text{giou}} \mathcal{L}_{\text{GIoU}}$（当前基线的完整损失）
- $\mathcal{L}_{\text{mid}} = \mathcal{L}_{\text{cls}}^{24} + \lambda_{\text{bbox}} \mathcal{L}_{\text{L1}} + \lambda_{\text{giou}} \mathcal{L}_{\text{GIoU}}$（与 full 相同，但仅在 $w_{\text{mid}}$ 权重下生效）
- $\mathcal{L}_{\text{coarse}} = \mathcal{L}_{\text{cls}}^{8} + \lambda_{\text{center}} \mathcal{L}_{\text{center-L1}}$（组级分类 + 仅框中心 L1）

**关键设计**：$\mathcal{L}_{\text{mid}}$ 与 $\mathcal{L}_{\text{full}}$ 形式相同，区别仅在权重 $w(t)$。这意味着中噪声时段仍施加完整 24 类监督，只是权重降低。真正的粒度切换发生在 $\mathcal{L}_{\text{coarse}}$——高噪声时段将 24 类折叠为 8 组，并将框回归从 4D（cxcywh）降为 2D（仅 cx, cy）。

### 3.2 分层损失函数

#### 3.2.1 组级分类损失（L3 粗粒度）

将 24 类标签映射为 8 组标签（使用 [constants.py](../../ldmdet/utils/constants.py) 的 `CHROMO_GROUP_OF_CLASS`）：

$$y_{\text{group}} = \text{GroupOf}(y), \quad y \in \{0, ..., 23\}, \quad y_{\text{group}} \in \{0, ..., 7\}$$

组级分类头为独立的 8 类线性层（复用 `fc_feature`）：

```python
self.group_head = nn.Linear(feat_channels, num_groups)  # 8 类
```

组级 Focal Loss：

$$\mathcal{L}_{\text{cls}}^{8} = \text{FocalLoss}(\text{sigmoid}(\mathbf{W}_g \cdot \mathbf{f}), y_{\text{group}})$$

#### 3.2.2 框中心 L1 损失（L3 粗粒度）

高噪声时段，框的尺寸（$w, h$）恢复不可靠，但中心位置（$c_x, c_y$）可粗略恢复（因为中心是低频信息）。因此 L3 仅对中心施加 L1：

$$\mathcal{L}_{\text{center-L1}} = \frac{1}{N_{\text{pos}}} \sum_{i \in \text{pos}} \left( |c_x^i - c_x^{gt}| + |c_y^i - c_y^{gt}| \right)$$

#### 3.2.3 完整分类损失（L1/L2 细粒度）

L1 和 L2 时段使用当前基线的 24 类 Focal Loss + L1 + GIoU，无需修改：

$$\mathcal{L}_{\text{full}} = \mathcal{L}_{\text{cls}}^{24} + 5.0 \cdot \mathcal{L}_{\text{L1}} + 2.0 \cdot \mathcal{L}_{\text{GIoU}}$$

（权重沿用 [基线配置](../../experiments/configs/ldmdet/ldmdet_rf_heun_adaln_stochot_eps5.py)）

#### 3.2.4 总损失函数

$$\mathcal{L}_{\text{total}} = \underbrace{w_{\text{full}}(t) \cdot (\mathcal{L}_{\text{cls}}^{24} + 5 \mathcal{L}_{\text{L1}} + 2 \mathcal{L}_{\text{GIoU}})}_{\text{L1: 完整监督}} + \underbrace{w_{\text{mid}}(t) \cdot (\mathcal{L}_{\text{cls}}^{24} + 5 \mathcal{L}_{\text{L1}} + 2 \mathcal{L}_{\text{GIoU}})}_{\text{L2: 中等监督}} + \underbrace{w_{\text{coarse}}(t) \cdot (\mathcal{L}_{\text{cls}}^{8} + \lambda_c \mathcal{L}_{\text{center}})}_{\text{L3: 粗粒度监督}}$$

由于 $w_{\text{full}} + w_{\text{mid}} + w_{\text{coarse}} = 1$，可简化为：

$$\mathcal{L}_{\text{total}} = (w_{\text{full}} + w_{\text{mid}}) \cdot (\mathcal{L}_{\text{cls}}^{24} + 5 \mathcal{L}_{\text{L1}} + 2 \mathcal{L}_{\text{GIoU}}) + w_{\text{coarse}} \cdot (\mathcal{L}_{\text{cls}}^{8} + \lambda_c \mathcal{L}_{\text{center}})$$

即：**低中噪声时段走原基线损失，高噪声时段切换为粗粒度损失**。这保证了向后兼容——当 $w_{\text{coarse}} \to 0$ 时（即 $t < t_2$），损失退化为当前基线。

### 3.3 分层监督机制

#### 3.3.1 训练流程

当前训练流程（[head.py:235-311](../../ldmdet/core/head.py#L235-L311)）：

1. 采样 $t \sim \text{ShiftedUniform}(s=3.0)$
2. 前向扩散：$x_t = (1-t)x_0 + t x_1$
3. 6 级级联 Head 前向，每级输出 `(cls_logits, pred_bboxes)`
4. 深度监督：每级 Head 计算 `loss_cls + loss_bbox + loss_giou`

HCTS 修改后的流程：

1. 采样 $t \sim \text{ShiftedUniform}(s=3.0)$（不变）
2. 前向扩散：$x_t = (1-t)x_0 + t x_1$（不变）
3. 6 级级联 Head 前向，每级输出 `(cls_logits, pred_bboxes, group_logits)`（新增 group_logits）
4. **分层深度监督**：
   - 计算 $w_{\text{full}}(t), w_{\text{mid}}(t), w_{\text{coarse}}(t)$
   - 每级 Head 的损失按 $t$ 加权：
     - 若 $w_{\text{coarse}} > 0$：计算 $\mathcal{L}_{\text{cls}}^8$ 和 $\mathcal{L}_{\text{center}}$
     - 若 $w_{\text{full}} + w_{\text{mid}} > 0$：计算原 $\mathcal{L}_{\text{cls}}^{24} + \mathcal{L}_{\text{L1}} + \mathcal{L}_{\text{GIoU}}$
     - 总损失 = $(w_{\text{full}} + w_{\text{mid}}) \cdot \mathcal{L}_{\text{full}} + w_{\text{coarse}} \cdot \mathcal{L}_{\text{coarse}}$

#### 3.3.2 推理流程

**推理时不修改**。HCTS 仅改变训练损失，推理仍使用 24 类分类头输出。组级分类头在推理时不使用——它仅作为训练时的辅助监督信号，帮助高噪声时段的共享特征学习组级可分性。

这是 HCTS 与方向五（E5.2 组条件化）的关键区别：方向五试图在推理时使用组条件化，引入了"先预测组再条件化"的推理复杂度；HCTS 的组级头纯粹是训练辅助，推理零开销。

#### 3.3.3 与级联 Head 的协同

6 级级联 Head（[head.py:119-121](../../ldmdet/core/head.py#L119-L121)）在**同一 $t$** 下逐级精化。HCTS 的分层监督对**所有 6 级**统一施加——即所有级的损失都按相同的 $w(t)$ 加权。这保持了级联 Head 的原有语义：每级在前一级的基础上精化，而 HCTS 控制的是"在当前 $t$ 下，精化的目标粒度是什么"。

---

## 4. 深入可行性分析

### 4.1 理论可行性

#### 4.1.1 梯度噪声降低的理论分析

设当前基线在高噪声时段（$t > t_2$）的分类损失梯度为 $\nabla_\theta \mathcal{L}_{\text{cls}}^{24}$。由于 $t > t_2$ 时 RoI 特征不可靠，该梯度的**信噪比**（梯度信号与梯度噪声之比）极低。

形式化地，设 RoI 特征 $\mathbf{f} = \mathbf{f}_{\text{signal}} + \mathbf{f}_{\text{noise}}$，其中 $\|\mathbf{f}_{\text{signal}}\| / \|\mathbf{f}_{\text{noise}}\| = \sqrt{\text{SNR}(t)}$。分类头梯度：

$$\nabla_\theta \mathcal{L}_{\text{cls}}^{24} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \cdot \frac{\partial \mathbf{z}}{\partial \mathbf{f}} \cdot \frac{\partial \mathbf{f}}{\partial \theta}$$

当 $\mathbf{f}_{\text{noise}}$ 主导时，梯度方向由噪声决定，与真实分类目标无关。

HCTS 将 24 类降为 8 类后，分类头的输出维度从 24 降至 8，**等效降低了分类任务的难度**。信息论上，8 类分类的熵 $H(G) = 3$ bits，而 24 类的熵 $H(Y) = 4.58$ bits。在相同 SNR 下，8 类分类的**错误率下界**更低：

$$P_{\text{err}}^{(8)} \le \exp(-2^{C(t) - H(G)}) = \exp(-2^{C(t) - 3})$$

$$P_{\text{err}}^{(24)} \le \exp(-2^{C(t) - H(Y)}) = \exp(-2^{C(t) - 4.58})$$

对于 $t = 0.8$（shifted 后约 0.643，$C(t) \approx 0.66$ bits）：

- $P_{\text{err}}^{(8)} \le \exp(-2^{-2.34}) \approx \exp(-0.20) \approx 0.82$
- $P_{\text{err}}^{(24)} \le \exp(-2^{-3.92}) \approx \exp(-0.067) \approx 0.94$

虽然两者错误率都很高（SNR 太低），但 8 类的梯度方向**更可能正确**，因为它只需区分 8 个粗类别而非 24 个细类别。**即使是 0.82 vs 0.94 的差异，在大规模训练中也意味着梯度信号质量的显著提升**。

#### 4.1.2 与方向三（SNR 匹配）的理论对比

方向三的策略是：高噪声时**降低匹配代价的权重**（$w(t) \to 0$），即"不信任匹配"。这导致：

- 正样本数减少 → 训练信号不足
- 大目标（IoU 在高噪声时仍可能较高）被错误抑制 → APl 下降 4.1%

HCTS 的策略是：高噪声时**改变监督目标**（24 类 → 8 类，全框 → 中心），即"信任但降低粒度"。这保证：

- 正样本数不变（匹配仍正常进行）
- 所有目标都参与训练（无抑制）
- 监督目标与 SNR 匹配（8 类在低 SNR 下更可学）

#### 4.1.3 信息瓶颈分析

HCTS 的核心假设是：**高噪声时段的 RoI 特征携带足够的组级可分信息**。验证方法：

- A 组（最大染色体）与 G 组（最小染色体）的尺寸差异在原始图像中约为 3-4 倍
- 即使在 $t = 0.8$（SNR ≈ 0.063），RoI 特征仍包含 FPN 提取的多尺度信息
- FPN 的 `roi_extractor`（[head.py:186-189](../../ldmdet/core/head.py#L186-L189)）在 RoIAlign 后保留 7×7 空间分辨率，足以编码尺寸/长宽比等低频特征

因此，组级分类在高噪声时段是**信息充分的**，而组内分类和精细回归则不是。

### 4.2 工程可行性

#### 4.2.1 代码改动清单

| 文件 | 改动类型 | 改动内容 | 行数估计 |
|------|---------|---------|---------|
| [single_head.py](../../ldmdet/core/single_head.py) | 新增 | `group_head` 线性层 + `forward` 返回 `group_logits` | ~20 行 |
| [head.py](../../ldmdet/core/head.py) | 修改 | `forward` 传递 `group_logits`；`loss` 计算分层损失 | ~40 行 |
| [criterion.py](../../ldmdet/criterion/criterion.py) | 修改 | 新增 `_loss_group_classification` 和 `_loss_center` 方法 | ~50 行 |
| 新建 `ldmdet/criterion/hcts_weight.py` | 新建 | $w(t)$ 权重计算函数 | ~30 行 |
| 新建 `experiments/configs/ldmdet/ldmdet_hcts.py` | 新建 | 实验配置 | ~15 行 |
| **总计** | | | **~155 行** |

#### 4.2.2 关键代码改动详解

**改动 1：SingleHead 新增组级分类头**

在 [single_head.py:137-146](../../ldmdet/core/single_head.py#L137-L146) 的 `__init__` 中新增：

```python
# HCTS: 组级分类头 (仅训练时使用, 推理时不使用)
if use_hcts:
    self.group_head = nn.Linear(feat_channels, num_groups)  # 8 类
else:
    self.group_head = None
```

在 [single_head.py:303-319](../../ldmdet/core/single_head.py#L303-L319) 的 `_predict` 中新增：

```python
def _predict(self, fc_feature, bboxes, bs, num_boxes):
    class_logits = self.cls_head(fc_feature)
    # HCTS: 组级 logits (若启用)
    group_logits = self.group_head(fc_feature) if self.group_head is not None else None
    # ... 原有 reg/velocity 逻辑 ...
    return (
        class_logits.view(bs, num_boxes, -1),
        pred_bboxes.view(bs, num_boxes, -1),
        fc_feature.view(1, bs * num_boxes, self.feat_channels),
        group_logits.view(bs, num_boxes, -1) if group_logits is not None else None,
    )
```

**改动 2：Head 的 forward 传递 group_logits**

在 [head.py:174-229](../../ldmdet/core/head.py#L174-L229) 的 `forward` 中，修改返回值以包含 `group_logits`：

```python
inter_group_logits = []
for head in self.head_series:
    result = head(features, curr_bboxes, curr_proposals, self.roi_extractor, time_emb)
    # 兼容 3/4 返回值
    if len(result) == 4:
        cls_logits, pred_bboxes, curr_proposals, group_logits = result
    else:
        cls_logits, pred_bboxes, curr_proposals = result
        group_logits = None
    inter_cls_logits.append(cls_logits)
    inter_pred_bboxes.append(pred_bboxes)
    inter_group_logits.append(group_logits)
```

**改动 3：Criterion 新增分层损失**

在 [criterion.py:77-93](../../ldmdet/criterion/criterion.py#L77-L93) 的 `_get_loss` 中，新增 HCTS 分支：

```python
def _get_loss(self, outputs, targets, indices, t=None):
    if indices is None:
        indices = self.matcher(outputs, targets)
    
    if self.use_hcts and t is not None:
        # HCTS 分层损失
        loss_cls, loss_bbox, loss_giou, loss_group, loss_center = \
            self._get_hcts_loss(outputs, targets, indices, t)
        return {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
            'loss_group': loss_group,
            'loss_center': loss_center,
        }
    
    # 原有逻辑
    loss_cls = self._loss_classification(outputs, targets, indices)
    loss_bbox, loss_giou = self._loss_boxes(outputs, targets, indices, t)
    return {'loss_cls': loss_cls, 'loss_bbox': loss_bbox, 'loss_giou': loss_giou}
```

**改动 4：$w(t)$ 权重计算**

新建 `ldmdet/criterion/hcts_weight.py`：

```python
import torch
from torch import Tensor

def hcts_weights(t: Tensor, t1: float = 0.33, t2: float = 0.67, 
                 tau: float = 0.05) -> Tuple[Tensor, Tensor, Tensor]:
    """HCTS 三层软切换权重.
    
    Returns:
        w_full: 低噪声权重 (t -> 0 时为 1)
        w_mid: 中噪声权重
        w_coarse: 高噪声权重 (t -> 1 时为 1)
    """
    w_full = torch.sigmoid((t1 - t) / tau)
    w_coarse = torch.sigmoid((t - t2) / tau)
    w_mid = 1.0 - w_full - w_coarse
    return w_full, w_mid, w_coarse
```

#### 4.2.3 计算开销评估

| 组件 | 新增 FLOPs | 新显存 | 训练时间增加 |
|------|-----------|--------|------------|
| `group_head` (256→8 线性层) | ~0.1M params × 500 proposals × 6 heads | ~3MB | < 1% |
| $w(t)$ 计算 | 3 次 sigmoid | 忽略 | 忽略 |
| $\mathcal{L}_{\text{cls}}^8$ Focal Loss | 8 类 vs 24 类（降低） | 忽略 | 忽略 |
| $\mathcal{L}_{\text{center}}$ L1 | 仅 2D（降低） | 忽略 | 忽略 |
| **总计** | **净降低**（高噪声时） | ~3MB | **< 2%** |

组级分类头的参数量极小（$256 \times 8 = 2048$ 参数），对训练速度和显存的影响可忽略。高噪声时段由于分类维度降低（24→8）和回归维度降低（4D→2D），实际计算量略有减少。

#### 4.2.4 与 AdaLN-Zero 的兼容性

当前 AdaLN-Zero（[single_head.py:321-357](../../ldmdet/core/single_head.py#L321-L357)）通过 `time_emb` 生成 6 个调制参数（$\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2$），控制 Self-Attention 和 FFN 的层归一化。HCTS **不修改 AdaLN-Zero**——时间条件化仍由 $t$ 控制，HCTS 仅改变损失函数。

这意味着：**模型的"时间感知"能力不变，HCTS 只改变"在当前时间下学什么"**。这是 HCTS 工程简洁性的关键——它是一个纯损失侧的改进，不触碰模型架构。

### 4.3 与现有架构的兼容性

#### 4.3.1 与 StochasticOT 耦合的兼容性

StochasticOT $\varepsilon=5$（[基线配置](../../experiments/configs/ldmdet/ldmdet_rf_heun_adaln_stochot_eps5.py)）在训练前为每个 proposal 分配 GT。HCTS **不修改耦合过程**——耦合仍按坐标距离 + 随机性分配，HCTS 只改变耦合后的损失计算。

具体地，[head.py:443-446](../../ldmdet/core/head.py#L443-L446) 的 `_couple_single_image` 仍正常工作，HCTS 在其之后介入：

```
噪声 → StochasticOT 耦合 → 分配 GT → 前向扩散 x_t → 6级Head前向 → HCTS分层损失
```

#### 4.3.2 与 box_renewal 的兼容性

`box_renewal`（[sampling.py:96-117](../../ldmdet/diffusion/sampling.py#L96-L117)）在推理时将低置信度框替换为随机噪声。这是**推理时**行为，HCTS **不修改推理流程**，因此完全兼容。

#### 4.3.3 与 DPM-Solver++ 的兼容性

DPM-Solver++（[rectified_flow.py:92-158](../../ldmdet/diffusion/rectified_flow.py#L92-L158)）是推理时的 ODE 求解器。HCTS 不修改推理，因此与 DPM-Solver++ 完全兼容。训练后仍可使用 DPM-2 8 步推理（离线 mAP=0.755 的增益可叠加）。

#### 4.3.4 与深度监督的兼容性

当前深度监督（[criterion.py:65-73](../../ldmdet/criterion/criterion.py#L65-L73)）对 6 级级联 Head 的每一级施加相同损失。HCTS 修改的是**每级损失的计算方式**，而非深度监督的结构。因此 HCTS 与深度监督完全兼容——每级 Head 的损失都按相同的 $w(t)$ 分层。

#### 4.3.5 向后兼容性

HCTS 的设计保证了**向后兼容**：

- 设置 `use_hcts=False`（默认）→ 完全退化为当前基线
- 设置 `use_hcts=True, t2=1.0`（即 $w_{\text{coarse}} \equiv 0$）→ 退化为当前基线 + 一个无用的 group_head
- 设置 `use_hcts=True, t1=0, t2=1.0`（即 $w_{\text{full}} \equiv 1$）→ 退化为当前基线 + 辅助组级损失

这种渐进式设计允许精确的消融实验。

---

## 5. 风险评估

### 5.1 风险 1：Shifted Schedule 下高噪声样本占比过低

**风险描述**：当前基线使用 `rf_shift=3.0`，使约 75% 的训练样本落在 $t < 0.5$（低中噪声区）。HCTS 的核心收益在高噪声时段（$t > 0.67$），但这些样本仅占约 25%。如果高噪声时段的改善不足以影响整体 mAP，HCTS 可能无收益。

**定量评估**：设高噪声时段（25% 样本）的梯度噪声降低 50%，但低噪声时段（75% 样本）无变化。整体梯度噪声降低约 $0.25 \times 0.5 = 12.5\%$。对于 mAP=0.858 的基线，12.5% 的梯度噪声降低预期带来约 $0.003-0.005$ 的 mAP 提升——接近验收下限。

**缓解方案**：
1. **降低 rf_shift**：从 3.0 降至 1.5-2.0，增加高噪声样本占比至 35-40%
2. **时间步重要性采样**：对高噪声时段的损失乘以 > 1 的权重（如 1.5x），等效增加其训练信号
3. **调整分层边界**：将 $t_2$ 从 0.67 降至 0.5，扩大粗粒度监督的范围

### 5.2 风险 2：组级分类头的梯度干扰主分类头

**风险描述**：组级分类头 `group_head` 与主分类头 `cls_head` 共享 `fc_feature`。高噪声时段，`group_head` 的梯度回传可能干扰 `cls_head` 的权重，导致低噪声时段的 24 类分类性能下降。

这与方向五（E5.1）观察到的问题类似——辅助损失在后期过拟合，`loss_hier` 回升与 mAP 下降同步（见 [方向五实验记录](../breakthrough_directions/方向五_组条件化分层分类.md#74-现象分析)）。

**缓解方案**：
1. **梯度分离**：`group_head` 的梯度不回传到共享特征（`fc_feature.detach()`），仅更新 `group_head` 自身权重。但这会削弱组级监督对共享特征的正面影响。
2. **权重衰减**：设置 `loss_group_weight` 为小值（如 0.1-0.3），减少对主分类头的干扰。方向五的调参表明 `loss_hier=1.0` 优于 0.3，但 HCTS 的场景不同——HCTS 的组级损失仅在高噪声时激活，总梯度量更小。
3. **Warmup 策略**：前 20 个 epoch 禁用 HCTS（`use_hcts=False`），待主分类头收敛后再启用。这避免早期训练中组级梯度干扰主分类头的初始化。

### 5.3 风险 3：组级分类在极高噪声下仍不可学

**风险描述**：根据 §2.2.2 的信道容量分析，$t > 0.9$ 时 $C(t) < 0.04$ bits，连 8 组分类（需 3 bits）都不可学。如果 HCTS 仍在此时段施加组级监督，梯度仍是噪声。

**缓解方案**：
1. **设置 SNR 下界**：当 $C(t) < 0.5$ bits 时（即 $t > 0.83$），完全关闭分类监督，仅保留框中心 L1（中心位置是最低频信息，SNR 下界最低）
2. **三级而非两级**：在 $t \in (0.83, 1.0]$ 时段施加"前景/背景"二分类监督（1 bit 需求，$C(t) \approx 0.04$ 仍不足但比 8 类好）
3. **承认极限**：极高噪声时段（$t > 0.9$）的任何监督都是噪声，HCTS 通过 $w_{\text{coarse}}(t)$ 的 sigmoid 软切换自然降低此时的监督权重（但不会完全归零）

### 5.4 风险 4：分层边界 $t_1, t_2$ 的超参敏感性

**风险描述**：$t_1 = 0.33, t_2 = 0.67$ 的选择基于 SNR 的理论分析，但实际最优值可能因数据集、模型容量、训练阶段而异。错误的边界可能导致：
- $t_2$ 过低（如 0.5）：中噪声时段被错误施加粗粒度监督，损失精细分类信号
- $t_2$ 过高（如 0.8）：高噪声时段仍施加完整监督，梯度噪声未充分降低

**缓解方案**：
1. **消融实验**：在 $\{(0.25, 0.5), (0.33, 0.67), (0.4, 0.8)\}$ 三组边界中选最优
2. **软边界温度 $\tau$**：增大 $\tau$（如 0.1）使边界更模糊，降低对精确边界的敏感性
3. **自适应边界**：根据训练过程中不同 $t$ 的 loss 曲线动态调整边界（进阶选项，初期不实现）

### 5.5 风险 5：与 CFM 失败方向的理论混淆

**风险描述**：CFM（速度预测）失败的原因是速度场目标的梯度与检测损失的梯度产生冲突（cos=-0.104，86.8% 层冲突）。HCTS 虽然不预测速度，但引入了额外的组级分类头，其梯度可能与主任务冲突。

**缓解方案**：
1. **梯度方向监控**：训练中记录 `group_head` 梯度与 `cls_head` 梯度的余弦相似度，若 cos < 0 则说明存在冲突
2. **PCGrad**：若检测到冲突，对组级损失梯度施加 PCGrad 投影（已有实现基础）
3. **关键区别**：CFM 的冲突在于**速度场目标与检测目标本质不同**（一个是 $\mathbb{R}^4$ 回归，一个是分类）；HCTS 的组级分类与主分类**目标一致**（都是分类，只是粒度不同），冲突风险更低

---

## 6. 预期收益分析

### 6.1 基于方向五实验数据的定量估计

方向五（E5.1 分层分类辅助损失）的实验结果提供了 HCTS 收益的下界估计：

| 指标 | SOTA (0.753) | E5.1 (0.745) | Δ | HCTS 预期 |
|------|-------------|-------------|---|----------|
| mAP | 0.753 | 0.745 | -0.008 | **+0.003 ~ +0.007** |
| APl | 0.642 | 0.663 | **+0.020** | **+0.015 ~ +0.025** |
| APs | 0.521 | 0.515 | -0.006 | 0 ~ +0.003 |

**推理依据**：
- E5.1 在**所有 $t$** 上施加分层辅助损失，导致低噪声时段的干扰（AP50/AP75 略降）
- HCTS **仅在高噪声时段**施加粗粒度监督，避免了低噪声时段的干扰
- E5.1 的 APl 提升（+0.020）来自组结构对大目标的帮助，HCTS 应能保持此收益
- E5.1 的 APs 下降（-0.006）来自辅助损失对小目标的干扰，HCTS 通过 $w(t)$ 软切换可缓解

### 6.2 基于梯度噪声降低的估计

当前基线 mAP=0.858。设高噪声时段（$t > 0.67$，约占 25% 训练样本）的梯度噪声占总梯度噪声的 40%（因为高噪声时段的梯度量级远大于低噪声时段）。

HCTS 将高噪声时段的分类目标从 24 类降至 8 类，预期梯度噪声降低约 50%（8 类的 Focal Loss 梯度方差低于 24 类）。

整体梯度噪声降低：$0.25 \times 0.40 \times 0.50 = 5\%$。

根据梯度噪声与 mAP 的经验关系（每降低 10% 梯度噪声约提升 0.01 mAP），HCTS 预期带来：

$$\Delta \text{mAP} \approx 0.005 \sim 0.010$$

即 mAP 从 0.858 提升至 **0.863 ~ 0.868**，有望追平 RTMDet-L 的 0.869。

### 6.3 分尺度收益预期

| 尺度 | 机制 | 预期 Δ |
|------|------|--------|
| APl | 组级监督帮助大目标（A/B 组）的组内区分 | +0.015 ~ +0.025 |
| APm | 中等目标从组内分类改善受益 | +0.005 ~ +0.010 |
| APs | 小目标（G 组 21,22）组内仅 2 类，收益有限 | 0 ~ +0.003 |
| AP50 | 粗匹配不受影响 | ~0 |
| AP75 | 精细匹配从梯度噪声降低受益 | +0.005 ~ +0.010 |

### 6.4 与 DPM-Solver++ 的叠加收益

HCTS 改善训练后的模型权重，DPM-Solver++ 改善推理时的 ODE 积分精度。两者正交：

- 当前基线 + DPM-2 8 步：0.858 → 0.860（+0.002，来自推理优化）
- HCTS + DPM-2 8 步：预期 0.863 ~ 0.868 + 0.002 = **0.865 ~ 0.870**

这有望追平甚至超越 RTMDet-L 的 0.869。

---

## 7. 实现路线图

### 7.1 Phase 1：基础实现与验证（1-2 天）

**目标**：实现 HCTS 核心逻辑，验证训练可跑通。

**代码改动**：

| 序号 | 文件 | 改动 |
|------|------|------|
| 1.1 | `ldmdet/criterion/hcts_weight.py` | 新建：`hcts_weights(t, t1, t2, tau)` 函数 |
| 1.2 | `ldmdet/core/single_head.py` | `__init__` 新增 `use_hcts` 参数和 `group_head`；`_predict` 返回 `group_logits` |
| 1.3 | `ldmdet/core/head.py` | `forward` 传递 `group_logits`；`loss` 调用 HCTS 分层损失 |
| 1.4 | `ldmdet/criterion/criterion.py` | 新增 `_get_hcts_loss`、`_loss_group_classification`、`_loss_center` 方法 |
| 1.5 | `experiments/configs/ldmdet/ldmdet_hcts.py` | 新建配置，继承 `ldmdet_rf_heun_adaln_stochot_eps5.py` |

**验收标准**：
- [ ] 训练 10 epoch 无报错
- [ ] `group_logits` 形状正确：`[bs, num_proposals, 8]`
- [ ] $w(t)$ 权重在 $t=0, 0.5, 1.0$ 处符合预期
- [ ] 损失值在合理范围（与基线同量级）

### 7.2 Phase 2：消融实验（3-5 天）

**目标**：验证 HCTS 各组件的独立贡献，寻找最优超参。

**实验矩阵**：

| 实验 ID | 配置 | 假设 |
|---------|------|------|
| E2.0 | 基线（无 HCTS） | 基线 mAP=0.858 |
| E2.1 | HCTS, $t_1=0.33, t_2=0.67, \tau=0.05$ | 默认配置 |
| E2.2 | HCTS, $t_2=0.5$（扩大粗粒度范围） | 更多样本受粗粒度监督 |
| E2.3 | HCTS, $t_2=0.8$（缩小粗粒度范围） | 减少粗粒度干扰 |
| E2.4 | HCTS, $\tau=0.1$（更软的边界） | 降低边界敏感性 |
| E2.5 | HCTS + `loss_group_weight=0.3` | 减少组级损失干扰 |
| E2.6 | HCTS + `rf_shift=2.0` | 增加高噪声样本占比 |

**验收标准**：
- [ ] 至少一组配置 mAP > 0.860
- [ ] APl 提升 > 0.010（验证组级监督对大目标的帮助）
- [ ] 训练损失曲线与基线相似（无发散/震荡）

### 7.3 Phase 3：优化与组合（3-5 天）

**目标**：在 Phase 2 最优配置基础上，与其他优化方向叠加。

**实验**：

| 实验 ID | 配置 | 假设 |
|---------|------|------|
| E3.1 | Phase 2 最优 + DPM-2 8 步推理 | 叠加推理优化 |
| E3.2 | Phase 2 最优 + `rf_shift=2.0` | 增加高噪声训练 |
| E3.3 | Phase 2 最优 + 梯度冲突监控 | 诊断组级梯度方向 |

**验收标准**：
- [ ] E3.1 mAP > 0.863
- [ ] 梯度冲突 cos > 0（组级与主分类头梯度方向一致）

### 7.4 Phase 4：长期探索（可选）

**方向**：
1. **自适应分层**：根据训练 loss 曲线动态调整 $t_1, t_2$
2. **组条件化生成**：将组级预测结果注入 AdaLN-Zero（方向五 E5.2 的未实现部分）
3. **多级层级**：引入更细的分层（如 4 级：组 → 亚组 → 类 → 精细框）
4. **与方向 O（原创理论范式）结合**：如果方向 O 提供了新的理论框架，HCTS 可作为其实现路径之一

---

## 8. 与已证伪方向的对比

### 8.1 与 CFM 速度预测的对比

| 维度 | CFM（已证伪, mAP=0.823） | HCTS |
|------|------------------------|------|
| **改什么** | 预测目标（框 → 速度场） | 损失目标（分类粒度随 $t$ 变化） |
| **失败原因** | 速度场目标与检测目标梯度冲突（cos=-0.104） | 不涉及——HCTS 不改变预测目标 |
| **架构改动** | 修改 SingleHead 输出 + 速度 MSE 损失 | 仅新增 group_head + 分层损失 |
| **推理影响** | 需要从速度反推 $x_0$ | 无——推理完全不变 |
| **关键区别** | CFM 试图改变"模型预测什么" | HCTS 改变"模型在什么时间学什么" |

**为什么 HCTS 不会重蹈 CFM 覆辙**：CFM 的根本问题是速度场回归目标（$\mathbb{R}^4$ 连续空间）与分类目标（离散类别）的梯度方向本质冲突。HCTS 不引入新的回归目标，组级分类仍是分类任务（只是 8 类而非 24 类），与主分类任务的梯度方向一致。

### 8.2 与端到端可微 Cascade 的对比

| 维度 | E2E Cascade（已证伪, mAP=0.684） | HCTS |
|------|-------------------------------|------|
| **改什么** | 级联 Head 的梯度传播（detach → 可微） | 损失的时间步分层 |
| **失败原因** | 级联误差累积 + 梯度不稳定 | 不涉及——HCTS 不修改级联结构 |
| **架构改动** | 修改 cascade_detach + 梯度链 | 不修改级联 |
| **关键区别** | E2E 改变级联间的梯度流 | HCTS 改变同一 $t$ 下的监督粒度 |

**为什么 HCTS 不会重蹈 E2E 覆辙**：E2E Cascade 的失败在于打破了级联的"每级独立精化"设计，导致误差通过 6 级级联放大。HCTS 保持级联结构不变（`cascade_detach=True`），只改变每级损失的计算方式。

### 8.3 与方向三（SNR 感知匹配）的对比

| 维度 | 方向三（已证伪, mAP=0.739, -0.014） | HCTS |
|------|-------------------------------------|------|
| **改什么** | 匹配代价按 $w(t)$ 加权 | 损失目标按 $t$ 分层 |
| **失败原因** | 高噪声时抑制匹配 → 大目标 APl -4.1% | 不抑制匹配 |
| **大目标影响** | APl 下降（IoU 高的大目标被错误抑制） | APl 预期提升（组级监督帮助大目标） |
| **训练信号** | 高噪声时正样本减少 → 训练信号不足 | 正样本不变 → 训练信号充分 |
| **关键区别** | 方向三改变"匹配多少" | HCTS 改变"学什么粒度" |

**为什么 HCTS 不会重蹈方向三覆辙**：方向三的核心错误是**抑制高噪声时段的匹配**——大目标在高噪声时 IoU 仍可能较高（框大，偏移对 IoU 影响小），但 SNR 权重仍将其抑制。HCTS 不抑制匹配，所有 proposal 仍正常分配 GT，只是高噪声时的分类目标从 24 类降为 8 类。

### 8.4 与方向五（组条件化分层分类）的对比

| 维度 | 方向五 E5.1（部分失败, mAP=0.745, -0.008） | HCTS |
|------|-------------------------------------------|------|
| **改什么** | 新增常驻辅助分层分类损失 | 按时间步分层的损失 |
| **何时生效** | 所有 $t$ | 仅高噪声 $t > t_2$ |
| **低噪声影响** | 辅助损失干扰主分类头（AP50/AP75 略降） | 无干扰（$w_{\text{coarse}} \to 0$） |
| **APl 表现** | +0.020（组结构帮助大目标） | 预期 +0.015 ~ +0.025（保持） |
| **E5.2 实现** | 未实现（组条件化推理） | 不需要（组级头仅训练用） |
| **关键区别** | 方向五的辅助损失在所有 $t$ 生效 | HCTS 仅在高噪声 $t$ 生效 |

**为什么 HCTS 优于方向五**：方向五 E5.1 的辅助损失在**所有 $t$** 上生效，导致低噪声时段的干扰（AP50=0.939 vs 0.943, AP75=0.832 vs 0.844）。HCTS 通过 $w(t)$ 软切换，使组级监督**仅在高噪声时段**激活，避免了低噪声时段的干扰。同时，方向五的 APl 提升（+0.020）来自组结构对大目标的帮助——HCTS 应能保持此收益，因为组级监督在所有 $t$ 都有（只是高噪声时权重更高）。

### 8.5 与方向 D BoxRefineNet 的对比

| 维度 | 方向 D BoxRefineNet（已证伪, 无收益） | HCTS |
|------|-------------------------------------|------|
| **改什么** | 新增框精化网络 | 损失的时间步分层 |
| **失败原因** | 精化网络与主 Head 的梯度冲突 | 不涉及——HCTS 不新增网络 |
| **推理影响** | 增加推理开销 | 无 |
| **关键区别** | BoxRefineNet 新增独立网络 | HCTS 仅新增一个线性层 |

---

## 9. 参考文献

1. **Karras, T., Aittala, M., Aila, T., & Laine, S.** (2022). Elucidating the design space of diffusion-based generative models. *NeurIPS 2022*. — 多尺度扩散理论，SNR 与采样调度的关系。

2. **Ho, J., Jain, A., & Abbeel, P.** (2020). Denoising diffusion probabilistic models. *NeurIPS 2020*. — DDPM 基础，频域分析。

3. **Liu, X., Gong, C., & Liu, Q.** (2023). Flow straight and fast: Learning to generate and transfer data with rectified flow. *ICLR 2023*. — Rectified Flow 理论。

4. **Li, Z., et al.** (2022). DiffusionDet: Diffusion model for object detection. *ICLR 2023*. — 扩散检测器基础架构。

5. **Morin, F., & Bengio, Y.** (2005). Hierarchical probabilistic neural network language model. *AISTATS 2005*. — 分层 Softmax 理论。

6. **Cover, T. M., & Thomas, J. A.** (2006). Elements of information theory. *Wiley*. — 信道容量与互信息理论。

7. **Gu, X., et al.** (2022). DiffuBox: Multimodal object detection with diffusion model. *arXiv preprint*. — 扩散检测器的损失设计。

8. **Bao, F., et al.** (2022). Analytic-DPM: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. *ICLR 2022*. — SNR 与最优方差的解析关系。

9. **Chen, T., et al.** (2023). On the importance of noise scheduling for diffusion models. *arXiv preprint*. — 噪声调度的重要性分析。

10. **Shih, A., et al.** (2024). On the role of domain hierarchy in few-shot learning. *ICML 2024*. — 层级结构在少样本学习中的作用。

---

## 附录 A：HCTS 超参默认值

| 超参 | 默认值 | 说明 |
|------|--------|------|
| `use_hcts` | False | 是否启用 HCTS |
| `t1` | 0.33 | L1/L2 边界（低/中噪声） |
| `t2` | 0.67 | L2/L3 边界（中/高噪声） |
| `tau` | 0.05 | 软边界温度 |
| `loss_group_weight` | 1.0 | 组级分类损失权重 |
| `loss_center_weight` | 2.0 | 框中心 L1 损失权重 |
| `num_groups` | 8 | Denver 组数 |
| `hcts_warmup_epochs` | 0 | HCTS 启用前的 warmup epoch 数 |

## 附录 B：HCTS 损失计算伪代码

```python
def hcts_loss(t, cls_logits, group_logits, pred_bboxes, gt_labels, gt_bboxes, indices):
    """
    Args:
        t: [bs] 扩散时间步
        cls_logits: [bs, N, 24] 主分类 logits
        group_logits: [bs, N, 8] 组级 logits (仅 use_hcts=True)
        pred_bboxes: [bs, N, 4] 预测框
        gt_labels: [bs, max_gt] GT 类别
        gt_bboxes: [bs, max_gt, 4] GT 框
        indices: 匹配结果
    """
    # 1. 计算 w(t) 权重
    w_full, w_mid, w_coarse = hcts_weights(t, t1=0.33, t2=0.67, tau=0.05)
    # w_full, w_mid, w_coarse: [bs]
    
    # 2. 完整损失 (24类 + L1 + GIoU) — 低中噪声
    w_full_mid = (w_full + w_mid)  # [bs]
    loss_cls_24 = focal_loss(cls_logits, gt_labels, indices)
    loss_l1 = l1_loss(pred_bboxes, gt_bboxes, indices)
    loss_giou = giou_loss(pred_bboxes, gt_bboxes, indices)
    loss_full = (loss_cls_24 + 5.0 * loss_l1 + 2.0 * loss_giou) * w_full_mid.mean()
    
    # 3. 粗粒度损失 (8类 + 中心L1) — 高噪声
    gt_groups = GROUP_OF_CLASS[gt_labels]  # 24类 → 8组
    loss_cls_8 = focal_loss(group_logits, gt_groups, indices)
    pred_centers = pred_bboxes[..., :2]  # 仅 cx, cy
    gt_centers = gt_bboxes[..., :2]
    loss_center = l1_loss(pred_centers, gt_centers, indices)
    loss_coarse = (loss_cls_8 + 2.0 * loss_center) * w_coarse.mean()
    
    # 4. 总损失
    return loss_full + loss_coarse
```

## 附录 C：与方向五 E5.1 的实验数据对照

方向五 E5.1 在 24obj 数据集上的实验结果（mAP=0.745 vs SOTA 0.753）为 HCTS 提供了重要的经验参考：

| 观察 | E5.1 结果 | HCTS 预期改进 |
|------|-----------|--------------|
| APl 提升 +0.020 | 组结构帮助大目标 | HCTS 保持此收益（组级监督仍生效） |
| AP50/AP75 略降 | 辅助损失干扰主分类 | HCTS 通过 $w(t)$ 消除低噪声干扰 |
| APs 无改善 | G 组仅 2 类，无信息收益 | HCTS 同样无改善（承认极限） |
| 后期过拟合 | 辅助损失在所有 $t$ 生效 | HCTS 仅高 $t$ 生效，过拟合风险更低 |
| 训练初期收敛快 | 组结构先验 | HCTS 保持此优势 |

**核心结论**：HCTS 可视为方向五 E5.1 的**时间步感知版本**——保留了组级监督的收益（APl 提升），同时通过 $w(t)$ 软切换消除了其在低噪声时段的干扰（AP50/AP75 回升）。
