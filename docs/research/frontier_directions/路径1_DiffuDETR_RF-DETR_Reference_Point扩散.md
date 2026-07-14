# 路径 1：DiffuDETR 路线 — RF 扩散生成 RF-DETR Query Reference Points

> **方向类型**：架构迁移方向（范式转换，可独立投稿）
> **目标会议**：ICLR 2027 / CVPR 2027 / IEEE TMI
> **预期增益**：mAP ≥ 0.858 (LDMDet a3_full_sota)，理想 mAP ≥ 0.870
> **状态**：方案设计阶段（未开始实验）
> **前置依赖**：RF-DETR 代码库 (roboflow/rf-detr)、DINOv2 预训练权重
> **文献基础**：DiffuDETR (ICLR 2026)、RF-DETR (ICLR 2026)、DeFloMat (arXiv 2025)、DN-DETR (CVPR 2022)、DINO (ICLR 2023)

---

## 1. 摘要

本方案提出将 LDMDet 已验证的 Rectified Flow (RF) + Heun 采样 + AdaLN-Zero + box_renewal 理论体系，迁移至 RF-DETR 的 query reference points 生成任务。核心思想是：保留 RF-DETR 的 DINOv2 backbone 与 cross-attention decoder 主干拓扑，但对 decoder 进行实质性改造——(a) 在每个 decoder layer 的 self-attn / cross-attn / FFN 之后插入 AdaLN-Zero 时间条件化门控；(b) 将 decoder 嵌入 Heun 多步采样循环（训练时单步随机 $t$，推理时 4 步 ODE 积分），每次采样步均调用一次 decoder 前向；(c) 将 query 的 4D reference points 从静态投影（`sigmoid(MLP(Q_content))`）替换为 RF 扩散生成（从高斯先验经 ODE 积分至 reference point 分布），decoder 被嵌入多步采样循环并增加时间条件化；(d) 在采样步之间插入 reference point renewal 机制。该路线由 DiffuDETR (ICLR 2026) 在 COCO 上验证可行（+1.0 mAP），并由 DeFloMat 证明 Rectified Flow 在检测中 3 步即可在 MRE 临床数据集上超越 DiffusionDet 4 步收敛上限。与 LDMDet 已证伪的 CFM 速度预测、PD-RF 蒸馏、SC-RF 自条件化方向不同，本方案在 DETR cross-attention 架构下操作 4D reference points，规避了级联 RoIAlign 架构的梯度冲突与信息截断问题。

> **工程框架说明**：RF-DETR 基于 PyTorch Lightning + Pydantic Config 构建而非 mmdet，因此本方案在 RF-DETR 官方代码库 (roboflow/rf-detr) 的 fork 上实现，不复用 chromosome-kd 项目的 mmdet 桥接。LDMDet 的 RF 核心代码（`rectified_flow.py`、`sampling.py`）需手动移植到 RF-DETR 代码库中。

---

## 2. 问题动机

### 2.1 当前 LDMDet 在 24obj 数据集的 SOTA

LDMDet 在 24obj 染色体数据集（24 类：A1-A3, B4-B5, C6-C12, D13-D15, E16-E18, F19-F20, G21-G22, X, Y）上的主路线 SOTA 为 **a3_full_sota**，配置组合为：

| 组件 | 配置 |
|------|------|
| 扩散类型 | Rectified Flow (RF)，直线路径 $x_t = (1-t)x_0 + t x_1$ |
| 采样器 | Heun 4 步（二阶，8 NFE） |
| 时间条件化 | AdaLN-Zero |
| 耦合策略 | Stochastic OT，Sinkhorn ε=5，multinomial sampling |
| 级联结构 | 6 × SingleDiffusionDetHead，cascade_detach=True |
| box_renewal | 启用，低置信度 proposal 替换为随机噪声 |
| **mAP** | **0.858** |

配置文件：`experiments/configs/ldmdet/directions/mainline_ablation_24obj/a3_full_sota_24obj.py`，基线继承自 `ldmdet_rf_heun_adaln_stochot_eps5.py`。

### 2.2 LDMDet 的局限性

尽管 a3_full_sota 达到 0.858 mAP，LDMDet 架构存在以下结构性局限：

1. **4 步采样延迟**：Heun 4 步需要 8 NFE（每步 2 次前向），每次前向需经过 6 级 cascade head，推理延迟较高。在临床核型分析中，一个病例含数十张显微图像，累计推理时间影响工作流效率。

2. **RoIAlign 框敏感性**：LDMDet 沿用 DiffusionDet 范式，通过 RoIAlign 从 FPN 特征图中裁剪 proposal 区域特征。RoIAlign 对框坐标精度敏感——当 proposal 偏离目标时，裁剪特征质量急剧下降，形成"框不准→特征差→框更不准"的负反馈。这在染色体密集场景（重叠、交叉）中尤为严重。

3. **500 proposal 浪费**：LDMDet 默认使用 500 个 proposal，而 24obj 数据集每张图像平均仅 ~20-46 个染色体对象。大量 proposal 被分配为背景，计算资源浪费且增加假阳性风险。box_renewal 机制虽部分缓解（低置信度 proposal 替换为噪声重采样），但本质未解决 proposal 利用率问题。

4. **级联架构的梯度隔离**：cascade_detach=True 阻断了级联间的梯度传播（已由方向 N 端到端可微 Cascade 证伪：mAP=0.684，-0.172），这限制了端到端联合优化的可能性。

### 2.3 RF-DETR 的优势

RF-DETR (Roboflow, ICLR 2026) 是首个在 COCO 上突破 60 AP 的实时检测 Transformer，其核心优势包括：

| 特性 | RF-DETR | LDMDet |
|------|---------|--------|
| **推理延迟** | RF-DETR-L: 6.8ms (T4, TensorRT FP16) | Heun 4步 × 6级cascade ≈ 55ms+ |
| **Backbone** | DINOv2 ViT（互联网规模自监督预训练） | ResNet50/Swin（ImageNet 监督） |
| **域泛化** | DINOv2 特征对 OOD 数据泛化强 | ResNet50 对染色体域泛化有限 |
| **Query 机制** | learnable embedding + IoU-aware selection + reference points | 500 random proposals + RoIAlign |
| **NAS** | 权重共享 NAS，硬件感知 Pareto 搜索 | 固定架构，无 NAS |
| **特征交互** | cross-attention（全局感受野） | RoIAlign（局部裁剪） |

RF-DETR 的 **cross-attention decoder** 相比 RoIAlign 的关键优势：reference points 作为 deformable attention 的采样锚点，即使初始位置偏差，cross-attention 仍能通过全局特征交互校正——这从根本上规避了 RoIAlign 的框敏感性问题。

### 2.4 为什么选择 DiffuDETR 路线

选择将 LDMDet 的 RF 理论迁移到 RF-DETR 的 reference point 生成，基于以下三重支撑：

**文献支撑**：
- **DiffuDETR (ICLR 2026 Poster)**：已在 COCO/LVIS/V3Det 上验证扩散生成 reference points 的有效性。DiffuDINO 在 COCO 上达 51.9 mAP（+1.0 vs DINO 50.9），LVIS +2.4，V3Det +2.2。该方法从高斯先验扩散生成 query reference points，仅需多次 decoder 前向即可采样，无需额外 RoI 特征提取。
- **DeFloMat (arXiv:2512.22406)**：证明 Rectified Flow（直线路径 + Conditional OT）在检测中 3 步即超越 DiffusionDet 4 步收敛上限（43.32% vs 31.03% AP，**注：该数据为 MRE 临床医学数据集上的结果，非 COCO**），且在少步区间的 Recall 和稳定性显著优于扩散。
- **FlowDet (arXiv:2512.16771)**：CFM 重构检测，COCO 上 **up to** +3.6% AP vs DiffusionDet（注：原文使用 "up to" 限定词，增益依赖于具体配置与对比基线）。（注：方向 H 实验发现 FlowDet 实际预测端点 $\hat{x}_1$ 而非速度，这与 LDMDet 的 $x_0$ 预测范式兼容。）

**理论可行性**：
- LDMDet 的 RF 理论体系（AdaLN-Zero 的时间条件化、box_renewal 的动态重采样）已在 24obj 数据集验证有效，且这些理论是 **目标无关的**——它们操作的是 4D 框/reference point 坐标空间，与具体的特征提取方式（RoIAlign vs cross-attention）解耦。LDMDet 同时验证了 Stochastic OT ε=5 仅匹配 random baseline（不带来正向增益），因此本方案直接采用 random coupling（见 §3.3 决策）。
- DiffuDETR 已证明扩散生成 reference points 在 DETR 架构下可行，本方案进一步将 DDPM 替换为 RF（更直路径、更少步数），并引入 LDMDet 验证的 AdaLN-Zero 时间条件化。

**工程可行性**：
- RF-DETR 代码库开源 (Apache 2.0 for N/S/M/L)，DINOv2 权重开源，可直接 fine-tune。
- reference point 是 4D 向量（cx, cy, w, h 归一化坐标），与 LDMDet 的 box 空间维度一致，RF 扩散机制可直接迁移。

---

## 3. 核心理论

### 3.1 RF-DETR 的 Query 机制详解

RF-DETR 的 query 机制继承自 DINO/Deformable DETR 范式，包含三个核心组件：

**1. Learnable Content Embeddings**：一组可学习的 query embedding $\mathbf{Q}_{content} \in \mathbb{R}^{N_q \times d}$，编码 query 的"内容"信息（关注什么类别的什么特征）。这部分在训练中学习，推理时固定。

**2. IoU-aware Query Selection**：RF-DETR 从 backbone 输出的 encoder memory 中，根据预测的 classification logits 和 IoU 分数选择 top-K encoder features 作为 query 的初始 content embedding：

$$\mathbf{Q}_{content}^{init} = \text{TopK}(\sigma(\text{cls logits}) \cdot \text{IoU score}, \mathbf{M}_{enc})$$

其中 $\mathbf{M}_{enc}$ 是 encoder memory。这使得 query 内容与图像内容自适应匹配。

**3. Reference Points**：每个 query 关联一个 4D reference point $\mathbf{r} \in \mathbb{R}^4$（归一化的 cx, cy, w, h），作为 deformable cross-attention 的采样锚点。在标准 RF-DETR 中，reference points 通过线性投影从 selected encoder features 生成：

$$\mathbf{r}^{init} = \text{sigmoid}(\text{MLP}(\mathbf{Q}_{content}^{init}))$$

**关键观察**：reference points 决定了 cross-attention 的采样位置，直接影响检测质量。在密集场景（如染色体重叠）中，静态初始化的 reference points 可能远离目标，导致 attention 采样到错误区域。

### 3.2 RF 扩散生成 Reference Points 的数学 Formulation

本方案将 reference points 的生成从静态投影替换为 RF 扩散过程。

**前向扩散（训练时）**：给定 GT reference points $\mathbf{r}_0$（从 GT boxes 转换的归一化坐标，位于 $[0,1]^4$），高斯噪声 $\mathbf{r}_1 \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$，RF 前向路径为：

$$\mathbf{r}_t = (1-t) \mathbf{r}_0 + t \mathbf{r}_1, \quad t \in [0, 1]$$

其中 $t=0$ 对应纯数据（GT reference points），$t=1$ 对应纯噪声。这是直线路径，目标速度场为常数：

$$\mathbf{u}_t = \mathbf{r}_1 - \mathbf{r}_0$$

**模型预测**：decoder 接收 noisy reference points $\mathbf{r}_t$、时间步 $t$、image features $\mathbf{M}_{enc}$，预测 $\hat{\mathbf{r}}_0$：

$$\hat{\mathbf{r}}_0 = f_\theta(\mathbf{r}_t, t, \mathbf{Q}_{content}, \mathbf{M}_{enc})$$

训练损失使用标准检测损失（与 LDMDet 一致的 $x_0$ 预测范式）：

$$\mathcal{L}_{det} = \mathcal{L}_{cls}(\hat{\mathbf{r}}_0 \to \text{boxes}, \text{labels}) + \mathcal{L}_{L1}(\hat{\mathbf{r}}_0, \mathbf{r}_0) + \mathcal{L}_{GIoU}(\hat{\mathbf{r}}_0, \mathbf{r}_0)$$

**反向采样（推理时）**：从 $\mathbf{r}_1 \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ 出发，经 ODE 积分至 $\mathbf{r}_0$：

$$\mathbf{r}_{t-\Delta t} = \mathbf{r}_t + \Delta t \cdot \mathbf{v}_t, \quad \mathbf{v}_t = \frac{\mathbf{r}_t - \hat{\mathbf{r}}_0}{t}$$

使用 Heun 4 步采样（二阶）：

$$\mathbf{r}_{next} = \mathbf{r}_t + \frac{\Delta t}{2}(\mathbf{v}_t + \mathbf{v}_{next})$$

其中 $\mathbf{v}_{next}$ 在 Euler 预测点 $\mathbf{r}_{t+\Delta t}^{Euler}$ 处重新评估。

**坐标空间匹配（关键工程问题）**：reference points 位于 $[0,1]^4$ 归一化空间，而 LDMDet 默认使用 $\sigma = 2.0$（即 $\mathbf{r}_1 \sim \mathcal{N}(0, 4\mathbf{I})$），噪声分布的标准差远大于 reference points 的有效范围。这会导致中间时间步（如 $t=0.5$）的 $\mathbf{r}_t = 0.5 \mathbf{r}_0 + 0.5 \mathbf{r}_1$ 落在 $[-3, 3]$ 之外的无效区域，宽高甚至可能为负，破坏 decoder 对 reference points 的消费（deformable attention 采样位置 + box 解码）。

LDMDet 通过显式重映射解决该问题（`ldmdet/models/detectors/ldmdet.py`）：

$$\mathbf{r}_0^{diff} = (\text{norm\_gt\_cxcywh} \times 2 - 1) \times \text{snr\_scale}$$

即将 $[0,1]$ 归一化坐标重映射到 $[-\text{snr\_scale}, +\text{snr\_scale}]$ 空间，使 GT 分布与噪声 $\mathcal{N}(0, \sigma^2 \mathbf{I})$ 在数值尺度上匹配。本方案必须采用类似的重映射，提供三种可选方案：

- **(a) 推荐：snr_scale 重映射（与 LDMDet 一致）**：训练时将 GT reference points 经 $\mathbf{r}_0^{diff} = (\mathbf{r}_0 \times 2 - 1) \times \text{snr\_scale}$ 映射到扩散空间，采样后再以 $\hat{\mathbf{r}}_0 = \text{sigmoid}(\hat{\mathbf{r}}_0^{diff} / \text{snr\_scale})$ 反映射回 $[0,1]$。snr_scale=2.0 与 LDMDet 默认值一致，数值稳定已验证。
- **(b) 降低 $\sigma$ 到 1.0**：使 $\mathbf{r}_1 \sim \mathcal{N}(0, \mathbf{I})$，与 $[0,1]$ 空间尺度更接近，但需重新评估 RF 路径直线性（噪声与数据尺度匹配度影响 RF 直线度），且与 LDMDet 验证配置偏离，需额外数值稳定性分析。
- **(c) logit 空间操作**：将 reference points 经 logit 变换映射到 $\mathbb{R}^4$，在 logit 空间执行 RF 扩散，最后 sigmoid 回 $[0,1]$。理论最干净，但 logit 在边界附近数值不稳定，且与 LDMDet 实现不兼容。

**本方案选择 (a)**：保持与 LDMDet 实现完全一致，便于直接迁移 `rectified_flow.py`，且 snr_scale=2.0 已在 24obj 验证稳定。后续若发现 RF-DETR decoder 对扩散空间坐标敏感，可回退到 (b) 或 (c)。

**与 LDMDet 的数学一致性**：上述 formulation 与 LDMDet 的 `rectified_flow.py` 完全一致——`q_sample` 实现 $x_t = (1-t)x_0 + t x_1$，`get_velocity` 实现 $\mathbf{v}_t = (x_t - x_0^{pred})/t$，`heun_step` 实现二阶 Heun 积分。唯一区别是 $x_0$ 从"GT boxes"变为"GT reference points"（经 snr_scale 重映射后）。

**Reflow 讨论（M1）**：RF 理论中，单次 RF coupling（"single-step RF"）得到的传输路径并非严格直线，而是近似直线；要获得更直的路径（更少步数即可收敛），需进行 reflow（用已训练模型生成 $(\hat{x}_0, \hat{x}_1)$ 配对，重新训练）。LDMDet a3_full_sota 使用的是 **single-step RF coupling（非 reflowed）**，依赖 Heun 4 步二阶积分弥补路径弯曲。本方案沿用该策略：不进行 reflow，理由是 (i) reflow 需要额外的两阶段训练，工作量翻倍；(ii) Heun 4 步已能在 LDMDet 上达到 0.858 mAP，证明 single-step RF coupling 足够；(iii) reflow 的收益主要体现在 1-2 步极速采样，而本方案目标步数为 4 步。若阶段 5 后希望压缩到 1-2 步，可再考虑 reflow。

### 3.3 Coupling 策略：Random Coupling（而非 Stochastic OT）

**问题**：RF 训练需要将每个 GT reference point $\mathbf{r}_0^{(i)}$ 与一个噪声样本 $\mathbf{r}_1^{(j)}$ 配对。随机配对（random coupling）是基线，OT 配对理论上可以生成更直的传输路径。

**LDMDet 的发现（已验证，决定性证据）**：
- Hard OT（匈牙利匹配）：mAP=0.735，**负增益**（-1.6% vs random 0.751）。原因：OT Diversity Collapse——在低维空间（d=4），Hard OT 过度约束配对多样性，导致训练分布坍缩。
- Stochastic OT ε=5（Sinkhorn + multinomial sampling）：mAP=0.751，**匹配 random baseline**，且通过 ε 控制多样性。

**CAM 定理**（已验证）：Argmax 操作在 Sinkhorn transport matrix 上消除了 ε 依赖的多样性控制。Stochastic sampling（`torch.multinomial(row_probs, 1)` 替代 argmax）恢复了 ε-diversity 单调性（Theorem 4.3: $H_{stoch}(V|X_t; \varepsilon)$ 随 $\varepsilon$ 单调递增）。

**本方案的决策**：reference points 是 4D 向量，与 LDMDet 的 box 空间维度一致（d=4），OT Diversity Collapse 理论直接适用。由于 LDMDet 实验已证明 **Stochastic OT ε=5 仅匹配 random baseline，不带来正向增益**，且 OT 迁移需额外实现 Sinkhorn 迭代（增加工程复杂度），本方案**直接采用 random coupling**（随机配对），不迁移 Stochastic OT 模块：

$$\text{Coupling} = \text{Random}(\mathbf{R}_0, \mathbf{R}_1)$$

即每个 GT reference point $\mathbf{r}_0^{(i)}$ 与从 $\mathbf{R}_1$ 中均匀随机采样的噪声样本 $\mathbf{r}_1^{(j)}$ 配对。这省去了 Sinkhorn 迭代的计算开销和工程实现工作量，且不损失 mAP（基于 LDMDet 实验结论）。

### 3.4 AdaLN-Zero 时间嵌入

**AdaLN-Zero**（Adaptive Layer Normalization with Zero Initialization）是 DiT (Peebles & Xie, NeurIPS 2023) 提出的时间条件化机制，在 LDMDet 中已验证有效（a3_full_sota 的 single_head 使用 `time_conditioning='adaln_zero'`）。

**机制（DiT 原版，含残差门控 α）**：对 decoder 的每一层，时间嵌入 $\mathbf{e}_t = \text{MLP}(\text{Sinusoidal}(t))$ 通过自适应仿射变换注入。DiT 原版 AdaLN-Zero 同时预测三个参数 $\gamma, \beta, \alpha$，并对 SubLayer（self-attn / cross-attn / FFN）的输出做带残差门控的调制：

$$\mathbf{h}_{out} = \mathbf{h} + \alpha(\mathbf{e}_t) \cdot \text{SubLayer}\big(\gamma(\mathbf{e}_t) \cdot \text{LayerNorm}(\mathbf{h}) + \beta(\mathbf{e}_t)\big)$$

其中 $\gamma, \beta, \alpha$ 均由线性层从 $\mathbf{e}_t$ 预测。**Zero Initialization**：**$\alpha$ 的输出权重初始化为 0**（而非 $\gamma / \beta$ 初始化为 0），使得训练初期 SubLayer 输出被门控为零，整个 layer 退化为纯残差连接 $\mathbf{h}_{out} = \mathbf{h}$，保证训练稳定性。随着训练进行，$\alpha$ 渐进学习非零值，时间条件化信号逐步引入。

> **实现说明**：本方案遵循 DiT 原版的 $\alpha$ 残差门控 zero-init。若实现简化版（仅 $\gamma / \beta$，无 $\alpha$，并将 $\gamma / \beta$ 初始化为 0/1），需在代码注释中明确标注"非 DiT 原版 AdaLN-Zero"，因简化版的训练动态与原版不同（$\gamma / \beta$ zero-init 使 AdaLN 退化为 LayerNorm 而非恒等，SubLayer 输出未被门控，可能引入额外训练噪声）。LDMDet single_head 的实现以 `single_head.py` 实际代码为准，本方案迁移时需核对 zero-init 的具体对象。

**注入位置**：在 RF-DETR decoder 的每个 cross-attention layer 和 FFN 之前注入 AdaLN-Zero（对 SubLayer 输入做 $\gamma \cdot \text{LN} + \beta$ 调制，对 SubLayer 输出做 $\alpha$ 残差门控），使 decoder 能根据时间步 $t$ 自适应调整特征处理方式（高噪声时关注全局结构，低噪声时关注局部精修）。

### 3.5 Reference Point Renewal 机制（box_renewal 的改造）

**LDMDet 的 box_renewal**（已验证）：在多步采样中，每步结束后将低置信度 proposal（$\text{score} < \text{threshold}$）替换为随机噪声，保留至少 `min_keep` 个高置信度 proposal。这实现了动态 proposal 重采样——低质量 proposal 被注入噪声后，模型在后续采样步中重新预测粗略 $\hat{x}_0$ 并继续精修，而非在错误位置上持续累积误差。

```python
# LDMDet sampling.py: apply_box_renewal 的核心逻辑
scores = torch.sigmoid(cls_logits).max(-1)[0]
keep = scores > score_thr  # 默认 0.05
if keep.sum() < min_keep:  # 默认 10
    keep[topk_idx] = True
x_raw_new[~keep] = torch.randn_like(x_raw[~keep])  # 注入噪声 (t_next 时刻)
```

**本方案的 reference point renewal**：将相同逻辑应用于 reference points。在 Heun 采样的每步之间：

1. decoder 前向得到 class logits 和 reference point 预测 $\hat{\mathbf{r}}_0$
2. 计算每个 query 的置信度 score = max(sigmoid(cls_logits))
3. 低置信度 query 的 reference points 在**当前时间步 $t_{next}$** 注入噪声（即重置为与 $t_{next}$ 匹配的噪声样本 $\mathbf{r}_{t_{next}}^{new}$），而非回退到 $t=1$ 重新开始整个扩散轨迹
4. 保留至少 `min_keep` 个高置信度 query 不被替换
5. 在下一步采样中，decoder 对被替换的 query 重新前向，预测粗略 $\hat{\mathbf{r}}_0$，后续采样步继续精修

> **关键澄清（修订 S3）**：renewal **不是**"重新从 $t=1$ 开始扩散"。采样的时间步调度（$t: 1.0 \to 0.0$）对所有 query 统一推进，renewal 只是在当前 $t_{next}$ 时刻把低置信度 query 的 $\mathbf{r}_{t_{next}}$ 替换为新的噪声样本，让模型在剩余的采样步里重新从该噪声样本预测 $\hat{\mathbf{r}}_0$ 并精修。这与 LDMDet `sampling.py` 的实际实现一致——`apply_box_renewal` 在 Heun step 之后调用，替换后的 `x_raw` 直接进入下一个 `t_next` 的采样循环，不会重置时间步调度。

**关键区别**：与 LDMDet 不同，RF-DETR 的 query content embedding 是 learnable 的（不被 renewal），只有 reference points 被 renewal。这保留了 query 的语义信息，仅重置空间锚点。

### 3.6 与 LDMDet 的理论映射

| LDMDet 概念 | 本方案对应 | 映射关系 |
|-------------|-----------|----------|
| Proposal (500 random boxes) | Query reference points (N_q 个) | 数量更少，质量更高 |
| RoIAlign (局部特征裁剪) | Cross-attention (全局特征交互) | 框不敏感，全局感受野 |
| box_renewal (低置信度 box→noise) | ref point renewal (低置信度 ref→noise) | 仅重置空间锚点，保留语义 |
| Cascade 6 × SingleHead | Decoder layers (6 层) | 层间特征传递方式不同 |
| GT boxes (xyxy) | GT reference points (cxcywh 归一化) | 坐标空间转换 |
| Stochastic OT ε=5 (noise, GT box) | Random coupling (noise, GT ref) | LDMDet 实验证明 OT 仅匹配 random baseline，本方案直接用 random coupling |
| AdaLN-Zero (single_head) | AdaLN-Zero (decoder layer) | 时间条件化机制一致 |
| Heun 4 步采样 | Heun 4 步采样 | ODE 积分器一致 |

**核心理论不变量（弱化声明）**：RF 直线路径 $x_t = (1-t)x_0 + t x_1$、AdaLN-Zero 的 zero-init 稳定性、Heun 的二阶精度——这些理论在 4D reference point 空间与 4D box 空间**在数学形式上一致**，因为它们操作的都是 $\mathbb{R}^4$ 中的向量，ODE 积分、时间条件化的数学结构可原样迁移。

> **重要限定（修订 S4）**：上述"形式一致"**不等于**"完全等价"。在 LDMDet 中，box 坐标仅作为扩散目标，RoIAlign 对 box 坐标的消费是"裁剪"操作，box 与特征之间的耦合是单向的（box → 裁剪区域 → 特征）。而在 RF-DETR 中，reference point 具有**双重角色**：(1) 扩散目标（与 LDMDet 的 box 角色一致）；(2) deformable cross-attention 的采样锚点（新增角色）。这意味着 reference point 的精度同时影响 attention 采样位置，进而影响 decoder 提取的特征质量，形成 **reference point ↔ attention 特征 ↔ reference point 预测** 的双向耦合。因此，RF 扩散在 reference point 空间的误差传播特性与 LDMDet 的 box 空间不完全等价——中间时间步 $t$ 的 noisy reference point 不仅数值上偏离 GT，还会导致 attention 采样到错误位置，引入 LDMDet 中不存在的额外误差源。该耦合的正面影响是 cross-attention 的全局感受野可校正初始偏差（见 §2.3），负面影响是扩散噪声可能通过 attention 机制被放大。实际等价性需由实验验证，本方案在阶段 2-3 中专门检验这一耦合。

---

## 4. 与已证伪方向的区分

本方案必须明确区分于 LDMDet 已证伪的 5 个方向，并论证为何这些方向的失败原因在本方案中不适用。

### 4.1 与 CFM (方向 H) 的区别

**已证伪结果**：CFM 速度预测，mAP=0.823 < 基线 0.856，Delta=-0.033。

**失败根因**：
1. 级联架构 Head 1-5 的输入 ≈ $x_0$（前一 head 去噪后的框），丢失 $x_{noise}$ 信息，无法计算速度目标 $\mathbf{v} = x_{noise} - x_{start}$。
2. 速度损失 MSE 收敛到 $\text{Var}(x_{noise}) = 4$（因 Head 1-5 输入不含噪声，速度目标不可学），注入梯度噪声而非有效监督。
3. FlowDet 论文实际预测端点 $\hat{x}_1$ 而非速度，用标准检测损失而非速度 MSE。

**本方案的区别**（三个独立规避点）：
- **(1) $x_0$ 预测规避速度损失**：与 LDMDet a3_full_sota 一致，本方案预测 $x_0$（reference points），用标准检测损失（L1 + GIoU + Focal），不引入速度 MSE 损失。RF 的速度仅在采样时从 $x_0^{pred}$ 反推（$\mathbf{v}_t = (\mathbf{r}_t - \hat{\mathbf{r}}_0)/t$），与 LDMDet 的 `get_velocity` 完全一致，仅用于 ODE 积分，不参与训练损失。
- **(2) 无级联架构规避信息截断**：RF-DETR decoder 是单链式 cross-attention layers（非级联），每层输入包含完整的 $\mathbf{r}_t$（带噪声的 reference points），不存在"输入 ≈ $x_0$"的信息截断问题。
- **(3) Cross-attention 规避框敏感性**：RF-DETR 使用 cross-attention（全局特征交互）而非 RoIAlign（局部裁剪），reference points 作为 deformable attention 的采样锚点而非硬裁剪区域，即使初始位置偏差也能通过全局特征交互校正，规避了 RoIAlign"框不准→特征差→框更不准"的负反馈。

**结论**：CFM 的失败根因可分解为三个独立问题——速度预测损失、级联信息截断、RoIAlign 框敏感性。本方案通过 "$x_0$ 预测 + 单链 decoder + cross-attention" 分别独立规避这三个问题，三者互不依赖。

### 4.2 与 PD-RF 的区别

**已证伪结果**：PD-RF 4→1 蒸馏，v1-v4 全失败，best mAP=0.851（零增益），v4 灾难性崩塌至 0.252。

**失败根因**：
1. 1 步 Euler 无法逼近 4 步 DPM-Solver++ 预测（gap~1.0 不收敛）。
2. 蒸馏梯度（raw 空间 MSE）与检测梯度（SimOTA + Focal + L1 + GIoU）严重冲突，grad_norm 持续 150-200。

**本方案的区别**：
- **本方案不做蒸馏**：本方案保留 Heun 4 步采样，不压缩到 1 步。推理时使用与训练一致的多步 ODE 积分。
- **无蒸馏梯度冲突**：训练损失仅有检测损失（L1 + GIoU + Focal + 匈牙利匹配），不引入蒸馏 MSE 损失，不存在梯度冲突问题。
- **步数作为 NAS 维度**：RF-DETR 的 NAS 框架天然支持采样步数作为搜索维度，不同硬件可选择不同步数（1/2/4步），而非通过蒸馏压缩。

**结论**：PD-RF 的失败是"步数压缩 + 蒸馏梯度冲突"的问题。本方案不压缩步数、不蒸馏，完全规避。

### 4.3 与 SC-RF 的区别

**已证伪结果**：SC-RF 自条件化，mAP=0.860 < A4 baseline 0.862，Delta=-0.002。

**失败根因**：
1. 训练时 $\hat{X}_0 = f_\theta(X_t)$ 是 $X_t$ 的确定函数，$I(X_0; \hat{X}_0 | X_t) = 0$，条件化不提供额外信息。
2. 推理时的跨时间步信息增益不足以抵消训练时的残差学习偏差。
3. 在级联架构下，自条件化输入与 cascade 的去噪逻辑冲突。

**本方案的区别**：
- **本方案不引入自条件化**：decoder 仅接收 $(\mathbf{r}_t, t, \mathbf{Q}_{content}, \mathbf{M}_{enc})$，不接收上一步的 $\hat{\mathbf{r}}_0^{prev}$。
- **无额外输入通道**：不修改 decoder 的输入维度，不引入零初始化的过渡机制。
- **reference point renewal 已提供"重启动"能力**：低置信度 query 的 reference points 被替换为噪声重新扩散，这比自条件化更直接地利用了多步采样的迭代精炼能力。

**结论**：SC-RF 的失败是"自条件化在级联架构下信息增益不足"的问题。本方案不使用自条件化，用 renewal 机制替代迭代精炼。

### 4.4 与 ScaleConditionedRF 的区别

**已证伪结果**：ScaleConditionedRF，mAP=0.741 < 0.746，Delta=-0.005。

**失败根因**：尺度条件化引入了额外的条件变量，在低维空间（d=4）中增加了训练难度，且尺度信息已被 GT box 的 (w, h) 隐式编码，显式条件化冗余。

**本方案的区别**：本方案不引入尺度条件化。reference points 的 (w, h) 维度已包含尺度信息，decoder 通过 cross-attention 自然学习多尺度特征。

### 4.5 与端到端可微 Cascade (方向 N) 的区别

**已证伪结果**：端到端可微 Cascade，mAP=0.684，Delta=-0.172。

**失败根因**：去除 cascade_detach 后，6 级 head 的链式梯度传播严重不稳定，mAP 震荡 0.35~0.68。

**本方案的区别**：
- **本方案无级联结构**：RF-DETR decoder 是标准的 Transformer decoder layers（self-attention + cross-attention + FFN），梯度通过残差连接自然传播，不存在级联 detach 问题。
- **DETR 的梯度传播已被验证稳定**：DN-DETR、DINO 等已证明 Transformer decoder 的端到端训练是稳定的，梯度通过 cross-attention 和残差连接高效传播。

**结论**：方向 N 的失败是"级联架构去 detach 后链式梯度不稳定"的问题。本方案使用标准 Transformer decoder，梯度传播路径短且稳定。

### 4.6 总结：为何已证伪方向的失败原因不适用

| 已证伪方向 | 失败根因 | 本方案为何规避 |
|-----------|---------|---------------|
| CFM (H) | 速度预测 + 级联信息截断 | 用 $x_0$ 预测，无级联 |
| PD-RF | 蒸馏梯度冲突 + 步数压缩 | 不蒸馏，保留多步 |
| SC-RF | 自条件化信息增益不足 | 不自条件化，用 renewal |
| ScaleCondRF | 尺度条件化冗余 | 不引入尺度条件化 |
| E2E Cascade (N) | 链式梯度不稳定 | 标准 Transformer decoder |

**核心论点**：LDMDet 已证伪方向的失败根因可分为两类——(1) 级联 RoIAlign 架构的结构性问题（CFM、E2E Cascade）；(2) 训练目标的梯度冲突（PD-RF 蒸馏、CFM 速度 MSE）。本方案通过迁移到 RF-DETR 的标准 Transformer decoder 架构，消除了第一类问题；通过坚持 $x_0$ 预测 + 标准检测损失（无蒸馏、无速度 MSE），消除了第二类问题。

---

## 5. 架构设计

> **实现框架**：本方案在 RF-DETR 官方代码库 (roboflow/rf-detr, 基于 PyTorch Lightning + Pydantic Config) 的 fork 上实现，不复用 chromosome-kd 项目的 mmdet 桥接——RF-DETR 与 mmdet 框架不兼容。LDMDet 的 RF 核心代码（`rectified_flow.py`、`sampling.py`）需手动移植到 RF-DETR 代码库中，移植时需适配 PyTorch Lightning 的训练循环和 Pydantic 的配置系统。

### 5.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Image                              │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              DINOv2 Backbone (ViT, frozen/finetuned)            │
│              多尺度特征: {P3, P4, P5} (encoder memory)          │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Transformer Encoder (可选, 继承 RF-DETR)            │
│              增强 encoder memory 的全局交互                      │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  ┌─────────────────────────────────────────────────────────┐    │
│  │        IoU-aware Query Selection (继承 RF-DETR)         │    │
│  │  从 encoder memory 选 top-K 作为 content embedding      │    │
│  └──────────────────────┬──────────────────────────────────┘    │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │     RFReferencePointGenerator (新增模块)                 │    │
│  │                                                         │    │
│  │  训练: GT ref points + RandomCoupling → r_t = (1-t)r_0 │    │
│  │        + t*r_1, 注入 AdaLN-Zero(t) 到 decoder           │    │
│  │  推理: r_1 ~ N(0,σ²I) → Heun 4步 ODE → r_0              │    │
│  │        + ref point renewal 每步重置低置信度 query        │    │
│  └──────────────────────┬──────────────────────────────────┘    │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │     RF-DETR Decoder (修改: + AdaLN-Zero time cond)      │    │
│  │                                                         │    │
│  │  Layer 1: Self-Attn → AdaLN-Zero(t) → Cross-Attn → FFN │    │
│  │  Layer 2: Self-Attn → AdaLN-Zero(t) → Cross-Attn → FFN │    │
│  │  ...                                                    │    │
│  │  Layer 6: Self-Attn → AdaLN-Zero(t) → Cross-Attn → FFN │    │
│  │                                                         │    │
│  │  Input: (Q_content, r_t, t, M_enc)                      │    │
│  │  Output: (cls_logits, r_0_pred)                         │    │
│  └──────────────────────┬──────────────────────────────────┘    │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │     Prediction Head (继承 RF-DETR)                      │    │
│  │  cls_logits → class probabilities                       │    │
│  │  r_0_pred → bounding boxes (xyxy)                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Hungarian Matching + Loss (训练)                    │
│              NMS / Post-processing (推理)                        │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 RF-DETR Baseline 修改点

相对标准 RF-DETR，本方案的修改点：

| 组件 | 标准 RF-DETR | 本方案 |
|------|-------------|--------|
| Query reference points | 静态投影 `sigmoid(MLP(Q_content))` | RF 扩散生成（从高斯先验 ODE 积分） |
| Decoder 时间条件化 | 无 | AdaLN-Zero(t) 注入每层 |
| (noise, GT) 配对 | 无 | Random coupling（随机配对，见 §3.3 决策） |
| 推理采样 | 单次 decoder 前向 | Heun 4 步（4 次 decoder 前向） |
| 低置信度 query 处理 | 无 | ref point renewal（替换为噪声重采样） |
| CDN 去噪训练 | 继承（可选保留） | 与 RF 扩散训练共存（见风险评估） |

### 5.3 新增模块：RFReferencePointGenerator

```python
class RFReferencePointGenerator(nn.Module):
    """RF 扩散生成 query reference points。

    训练: 从 GT ref points 前向扩散, decoder 预测 x_0
    推理: 从高斯噪声反向 ODE 积分, Heun 4步
    """

    def __init__(
        self,
        num_queries: int = 300,
        feat_channels: int = 256,
        snr_scale: float = 2.0,
        sampling_timesteps: int = 4,
        solver_type: str = 'heun',
        ref_renewal: bool = True,
        score_thr: float = 0.05,
        min_keep: int = 10,
    ):
        super().__init__()
        self.rf = RectifiedFlow(snr_scale=snr_scale)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(feat_channels),
            nn.Linear(feat_channels, feat_channels * 4),
            nn.SiLU(),
            nn.Linear(feat_channels * 4, feat_channels * 4),
        )
        self.solver_type = solver_type
        self.sampling_timesteps = sampling_timesteps
        self.ref_renewal = ref_renewal
        # ... (renewal params)

    def get_train_queries(
        self, gt_ref_points: Tensor, num_queries: int
    ) -> Tuple[Tensor, Tensor]:
        """训练时: Random coupling 配对 + 前向扩散 → r_t"""
        # 1. Random coupling 配对 (noise, GT) — 见 §3.3 决策
        r_noise = torch.randn_like(gt_ref_points)
        # 随机打乱 noise 顺序, 实现 random coupling
        perm = torch.randperm(r_noise.shape[0], device=r_noise.device)
        r_paired_noise = r_noise[perm]

        # 2. 随机时间步
        t = torch.rand((gt_ref_points.shape[0],), device=device)

        # 3. RF 前向: r_t = (1-t)*r_0 + t*r_1
        r_t, velocity = self.rf.q_sample(
            gt_ref_points, r_paired_noise, t
        )
        return r_t, t

    def get_inference_queries(
        self, num_queries: int, decoder_fn
    ) -> Tensor:
        """推理时: Heun 4步 ODE 积分 r_1 → r_0"""
        r = torch.randn(num_queries, 4)  # r_1 ~ N(0, σ²I)
        timesteps = torch.linspace(1.0, 0.0, self.sampling_timesteps + 1)

        for i in range(self.sampling_timesteps):
            t_curr, t_next = timesteps[i], timesteps[i + 1]
            cls_logits, r0_pred = decoder_fn(r, t_curr)
            # heun_step 的 model_fn 要求返回 (x_0_pred_next, _) 元组
            r = self.rf.heun_step(
                r, r0_pred, t_curr, t_next,
                lambda x, t: (decoder_fn(x, t)[1], None)
            )
            if self.ref_renewal and t_next > 0:
                r = self.apply_ref_renewal(r, cls_logits)
        return r
```

### 5.4 AdaLN-Zero 时间嵌入的注入位置

AdaLN-Zero 在 RF-DETR decoder 的每个 layer 中，应用于 self-attention、cross-attention 和 FFN 的输出：

```
Decoder Layer (modified):
  h = Q_content + SelfAttn(Q_content)
  h = h + AdaLN-Zero(h, t)        ← 新增: 时间条件化 self-attn 输出
  h = h + CrossAttn(h, M_enc, ref=r_t)
  h = h + AdaLN-Zero(h, t)        ← 新增: 时间条件化 cross-attn 输出
  h = h + FFN(h)
  h = h + AdaLN-Zero(h, t)        ← 新增: 时间条件化 FFN 输出
```

AdaLN-Zero 的实现（DiT 原版，含 α 残差门控 zero-init，与 §3.4 公式一致）：

```python
class AdaLNZero(nn.Module):
    """DiT 原版 AdaLN-Zero: 同时预测 gamma, beta, alpha,
    alpha 的输出权重初始化为 0, 使训练初期 SubLayer 输出被门控为零,
    整个 layer 退化为纯残差连接, 保证训练稳定性。"""

    def __init__(self, dim):
        self.norm = nn.LayerNorm(dim)
        # 预测 (gamma, beta, alpha) 三组参数
        self.emb = nn.Linear(dim * 4, dim * 3)
        nn.init.zeros_(self.emb.weight)  # zero init (含 alpha)
        nn.init.zeros_(self.emb.bias)

    def forward(self, x, t_emb, sublayer_out):
        """x: 残差输入, t_emb: 时间嵌入, sublayer_out: SubLayer(self-attn/cross-attn/FFN)的输出"""
        gamma, beta, alpha = self.emb(t_emb).chunk(3, dim=-1)
        h = gamma * self.norm(x) + beta
        # alpha 残差门控: 训练初期 alpha=0, 退化为纯残差 h_out = x
        return x + alpha * sublayer_out(h)
```

**Zero Initialization 的关键对象**：DiT 原版 AdaLN-Zero 的 zero-init 针对的是 **α 的输出权重**（使 α 初始为 0），而非 γ/β。这使得训练初期整个 layer 退化为纯残差连接 `h_out = x`（而非退化为 LayerNorm），保证训练稳定性。随着训练进行，α 渐进学习非零值，时间条件化信号逐步引入。

> **实现说明**：若为简化实现而省略 α 残差门控（仅保留 γ/β 并将 γ 初始化为 1、β 初始化为 0），需在代码注释中明确标注"受 AdaLN-Zero 启发的简化变体，非 DiT 原版"——因简化版的训练动态与原版不同（γ/β zero-init 使 AdaLN 退化为 LayerNorm 而非恒等残差，SubLayer 输出未被门控，可能引入额外训练噪声）。本方案推荐使用 DiT 原版含 α 的实现。LDMDet single_head 的实现以 `single_head.py` 实际代码为准，迁移时需核对 zero-init 的具体对象。

### 5.5 Random Coupling 配对的实现

Random coupling 配对在 `RFReferencePointGenerator.get_train_queries` 中实现（见 §3.3 决策：不迁移 Stochastic OT，直接用 random coupling）：

1. **生成噪声**：为每个 GT reference point 生成一个高斯噪声样本 $\mathbf{r}_1^{(i)} \sim \mathcal{N}(0, \mathbf{I})$
2. **随机打乱**：对噪声样本的索引做随机排列（`torch.randperm`），实现 random coupling
3. **配对**：每个 GT ref point 与打乱后的噪声样本配对

```python
# Random coupling (见 §3.3: LDMDet 实验证明 Stochastic OT ε=5 仅匹配 random baseline)
r_noise = torch.randn_like(gt_ref_points)
perm = torch.randperm(r_noise.shape[0], device=r_noise.device)
r_paired_noise = r_noise[perm]  # random coupling
```

**不迁移 Stochastic OT 的理由**（见 §3.3）：LDMDet 实验已证明 Stochastic OT ε=5 在 d=4 空间仅匹配 random baseline（mAP=0.751 vs random 0.751），不带来正向增益。直接使用 random coupling 省去了 Sinkhorn 迭代（20 iters）的计算开销和 `coupling/ot_flow.py` 的工程迁移工作量。

### 5.6 Reference Point Renewal 的触发条件

ref point renewal 在推理采样的每步之间触发：

```python
def apply_ref_renewal(
    self, r: Tensor, cls_logits: Tensor
) -> Tensor:
    """低置信度 reference points 替换为噪声 (在扩散空间中, 见 §3.2 snr_scale 重映射)"""
    scores = torch.sigmoid(cls_logits).max(-1)[0]
    keep = scores > self.score_thr  # 默认 0.05
    if keep.sum() < self.min_keep:  # 默认 10
        _, topk_idx = scores.topk(self.min_keep)
        keep[topk_idx] = True
    r_new = r.clone()
    # r 处于扩散空间 (snr_scale 重映射后, 见 §3.2), r_1 ~ N(0, I) 是该空间的纯噪声
    r_new[~keep] = torch.randn_like(r[~keep])  # r_1 ~ N(0, I), 扩散空间
    return r_new
```

> **空间一致性说明**：上述 `torch.randn_like` 生成的是扩散空间（snr_scale 重映射后）的纯噪声 r_1 ~ N(0, I)，与 §3.2 选择方案 (a) 的 snr_scale 重映射一致。**若未采用 snr_scale 重映射**（即 r 仍在 [0,1] 归一化空间），则 `torch.randn_like` 会产生负值，破坏 reference points 的有效性——此时需改用 `sigmoid(torch.randn_like(r[~keep]) * σ)` 或 `torch.rand_like(r[~keep])`（均匀分布 [0,1]）生成 [0,1] 空间的噪声。本方案默认采用 snr_scale 重映射，故 `torch.randn_like` 是正确的。

**触发条件**：
1. 仅在推理时触发（训练时不触发，因训练时 $\mathbf{r}_t$ 由前向扩散确定）
2. 每步采样后触发（Heun 4 步则触发 3 次，最后一步不触发）
3. 仅替换 reference points，不替换 content embedding

### 5.7 训练 vs 推理流程对比

**训练流程**：
```
1. DINOv2 backbone → encoder memory M_enc
2. IoU-aware query selection → Q_content
3. GT boxes → GT reference points R_0 (cxcywh 归一化)
4. Random coupling: (R_0, R_noise) → paired R_1 (见 §3.3)
5. 随机时间步 t ~ U[0, 1]
6. RF 前向: R_t = (1-t)*R_0 + t*R_1
7. Decoder 前向 (with AdaLN-Zero(t)):
   (Q_content, R_t, t, M_enc) → (cls_logits, R_0_pred)
8. 匈牙利匹配: (R_0_pred, cls_logits) ↔ (GT boxes, labels)
9. 检测损失: L_cls (Focal) + L_L1 + L_GIoU
10. (可选) CDN 去噪损失: 与 RF 扩散损失共存
```

**推理流程**：
```
1. DINOv2 backbone → encoder memory M_enc
2. IoU-aware query selection → Q_content
3. 初始化: R_1 ~ N(0, σ²I)
4. Heun 4步采样:
   for i in [0, 1, 2, 3]:
     t_curr, t_next = timesteps[i], timesteps[i+1]
     cls_logits, R_0_pred = Decoder(Q_content, R_t, t_curr, M_enc)
     R_t = HeunStep(R_t, R_0_pred, t_curr, t_next)
     if t_next > 0:
       R_t = apply_ref_renewal(R_t, cls_logits)
5. 最终预测: (cls_logits, R_0_pred) → NMS →检测结果
```

---

## 6. 实验设计

### 6.1 阶段 1：RF-DETR Baseline Fine-tune

**目标**：建立 RF-DETR 在 24obj 数据集的对照基线。

**配置**：
- 模型：RF-DETR-S（32.1M 参数，512×512 分辨率）
- Backbone：DINOv2 ViT-S（预训练权重）
- 训练：fine-tune 150 epochs，bs=8，lr=1e-4（RF-DETR 默认）
- 数据增强：标准 Mosaic + MixUp + RandomFlip

**消融变量**：无（纯 baseline）

**预期 mAP**（条件式：前提为 DINOv2 特征成功适配 24obj 域 + 数据增强有效抑制过拟合）：0.82-0.85。该预期基于 RF-DETR 在 COCO 上 53.0 AP 的参考，但 24obj 是小数据集 + 细粒度分类，实际 mAP 受域迁移难度影响，可能偏低。**注意**：此预期缺乏严格理论依据，仅为经验估计，实际值需由实验确定。

**对照**：LDMDet a0_baseline（无 RF，无 Heun，无 OT）= ?

### 6.2 阶段 2：引入 RF 扩散生成 Reference Points（单步 Euler）

**目标**：验证 RF 扩散生成 reference points 的基本可行性。

**配置**：
- 在阶段 1 基础上，将 query reference points 从静态投影替换为 RF 扩散生成
- 采样器：Euler 1 步（最小验证）
- 无 Stochastic OT（random coupling）
- 无 AdaLN-Zero（decoder 无时间条件化）
- 无 ref point renewal

**消融变量**：RF 扩散生成 vs 静态投影

**预期 mAP**（条件式：前提为 1 步 Euler 截断误差可接受 + snr_scale 重映射正确 + reference point 双重角色耦合可控）：0.80-0.83。单步 Euler 截断误差较大，预期略低于 baseline。**注意**：该预期假设扩散生成机制基本可行，实际值受 reference point 与 attention 的耦合特性影响，无严格理论保证。

**关键观察**：验证扩散生成 reference points 是否能收敛。若 mAP < 0.75，说明 1 步 Euler 不够（截断误差过大），需排查 snr_scale 重映射、时间步调度等工程问题，**不否定方案本身**——后续阶段 3 引入 Heun 4 步 + AdaLN-Zero 后再评估。

### 6.3 阶段 3：引入 Heun 4 步 + AdaLN-Zero

**目标**：引入 LDMDet a3_full_sota 的核心理论体系（Heun 多步采样 + AdaLN-Zero 时间条件化）。

**配置**：
- 采样器：Heun 4 步（二阶，8 NFE）
- 耦合：Random coupling（见 §3.3 决策，不迁移 Stochastic OT）
- 时间条件化：AdaLN-Zero（DiT 原版含 α 残差门控 zero-init）
- 无 ref point renewal

**消融变量**：
- Heun 4 步 vs Euler 1 步（验证多步采样增益）
- AdaLN-Zero vs 无时间条件化（验证时间条件化增益）

**预期 mAP**（条件式：前提为 Heun 4 步有效降低截断误差 + AdaLN-Zero 时间条件化被 decoder 正确利用 + reference point 双重角色耦合不显著恶化）：0.85-0.87。该预期参考 LDMDet a3_full_sota 0.858 mAP，但 RF-DETR 的 cross-attention 架构与 LDMDet 的 RoIAlign 不同，reference point 的双重角色（扩散目标 + attention 锚点）可能引入额外误差，实际 mAP 无严格理论保证。

### 6.4 阶段 4：引入 Reference Point Renewal

**目标**：验证 ref point renewal 的增益。

**配置**：
- 在阶段 3 基础上，启用 ref point renewal
- score_thr=0.05，min_keep=10

**消融变量**：ref point renewal on/off

**预期 mAP**（条件式：前提为阶段 3 已达到预期 + renewal 机制在 cross-attention 架构下的增益与 LDMDet RoIAlign 架构下相当）：0.855-0.875。该预期参考 renewal 在 LDMDet 中 +0.005~0.010 的增益，但 RF-DETR 的 query content embedding 是 learnable 的（不被 renewal），renewal 效果可能不同。

### 6.5 阶段 5（可选）：DPM-Solver++ 6 步对比

**目标**：对比 Heun 4 步与 DPM-Solver++ 6 步的精度-效率权衡。

**配置**：
- 采样器：DPM-Solver++ 2 阶 6 步（7 NFE，比 Heun 4 步 8 NFE 少 12.5%）
- 其他配置同阶段 4

**消融变量**：DPM-Solver++ 6 步 vs Heun 4 步

**预期 mAP**（条件式：前提为 DPM-Solver++ 在 reference point 空间的收敛行为与 LDMDet box 空间类似）：与 Heun 4 步持平或略高。该预期参考 LDMDet 离线对比（DPM-2 8 步 0.755 > Heun 4 步 0.753，+0.002），但 reference point 空间的 ODE 路径特性可能不同。

### 6.6 实验汇总表

| 阶段 | 配置 | 预期 mAP | 消融变量 |
|------|------|---------|---------|
| 1 | RF-DETR baseline | 0.82-0.85 | 无（对照基线） |
| 2 | + RF 扩散生成 (Euler 1步) | 0.80-0.83 | RF vs 静态投影 |
| 3 | + Heun 4步 + AdaLN-Zero | 0.85-0.87 | 多步/时间条件化 |
| 4 | + ref point renewal | 0.855-0.875 | renewal on/off |
| 5 | (可选) DPM-Solver++ 6步 | ≥阶段4 | DPM vs Heun |

---

## 7. 风险评估

### 7.1 DINOv2 Backbone 特征是否适合扩散生成

**风险**：DINOv2 是自监督预训练的 ViT，其特征空间与 ImageNet 监督预训练的 ResNet 不同。扩散生成需要 encoder memory 对"带噪声的 reference points"提供稳定的条件特征，DINOv2 特征是否支持这种条件化尚不确定。

**缓解**：
- 阶段 1 的 baseline fine-tune 会调整 DINOv2 特征到 24obj 域
- RF-DETR 已验证 DINOv2 特征在 COCO 检测中有效（56.5 AP）
- 若 DINOv2 特征不适合，可回退到 ResNet50/Swin backbone + RF-DETR decoder

**风险等级**：中

### 7.2 CDN 去噪训练与 RF 扩散训练的梯度冲突

**风险**：DiffuDETR 的 DiffuDINO 变体基于 DINO 的 CDN（Contrastive Denoising Queries）。CDN 在训练时向 GT reference points 添加高斯噪声并学习去噪，这与 RF 扩散训练的"前向加噪 + 预测 $x_0$"机制在形式上相似但目标不同：
- CDN：固定噪声尺度（DN-DETR 的噪声调度），单步去噪，辅助训练信号
- RF 扩散：随机时间步 $t \sim U[0,1]$，多步 ODE 采样，主训练信号

两者同时使用可能导致梯度冲突——CDN 的去噪梯度与 RF 扩散的 $x_0$ 预测梯度在 decoder 参数上竞争。

**缓解**：
- 优先使用 DiffuDETR 变体（基于 Deformable DETR，无 CDN）而非 DiffuDINO
- 若使用 DiffuDINO，将 CDN 作为辅助损失（weight=0.5），RF 扩散作为主损失
- 监控梯度冲突：测量 CDN 梯度与 RF 扩散梯度的余弦相似度
- **可操作判据**：若梯度冲突严重（余弦相似度持续 < 0 或 grad_norm 持续 > 100），则**关闭 CDN**（设置 CDN weight=0），仅保留 RF 扩散训练信号；若冲突可控（余弦相似度 > 0.3），则保留 CDN 作为辅助信号

**风险等级**：中高

### 7.3 多步采样对 RF-DETR 实时性的影响

**风险**：RF-DETR-L baseline 单次前向 6.8ms。Heun 4 步需要 4 次 decoder 前向（8 NFE），若 decoder 占总延迟的 ~50%（3.4ms），则 4 步采样 ≈ 6.8 + 3×3.4 = 17.0ms。这仍在 30ms 约束内，但丧失了 RF-DETR 的实时优势。

**量化分析**：
| 配置 | 延迟估计 | 是否满足 ≤30ms |
|------|---------|---------------|
| RF-DETR-L baseline (1步) | 6.8ms | ✅ |
| + Heun 4 步 | ~17-22ms | ✅ |
| + DPM-Solver++ 6 步 | ~20-25ms | ✅ |
| + ref point renewal (额外开销小) | +0.5ms | ✅ |

**缓解**：
- NAS 搜索采样步数维度：在目标硬件上搜索 1/2/4 步的精度-延迟 Pareto 前沿
- 若 4 步延迟过高，可使用 DPM-Solver++ 6 步（7 NFE vs Heun 8 NFE）
- ref point renewal 的额外开销可忽略（仅阈值过滤 + 噪声替换）

**风险等级**：低（30ms 约束内可满足）

### 7.4 小数据集 (24obj) 过拟合风险

**风险**：24obj 数据集规模较小（估计 ~1000-5000 张图像），DINOv2 backbone（33M+ 参数）+ RF 扩散生成（额外参数）可能导致过拟合。

**缓解**：
- 冻结 DINOv2 backbone 前几层，仅 fine-tune 后几层 + decoder
- 使用强数据增强（Mosaic + MixUp + RandomFlip + ColorJitter）
- Early stopping（patience=30，与 LDMDet 一致）
- 监控 train/val mAP gap，若 >0.05 则增加正则化（dropout, weight decay）

**风险等级**：中

### 7.5 NAS 搜索空间需扩展

**风险**：RF-DETR 的 NAS 搜索空间包括 image resolution、patch size、decoder layers、query tokens，但不包括采样步数。引入 RF 扩散后，采样步数成为新的精度-延迟权衡维度，需扩展 NAS 搜索空间。

**缓解**：
- 将采样步数 {1, 2, 4} 加入 NAS 搜索维度
- 在 base model 训练完成后，用 grid search 评估不同步数的子网
- 这是 RF-DETR NAS 框架的自然扩展，不需要修改训练流程

**风险等级**：低

### 7.6 OT Diversity Collapse（已规避）

**风险**：LDMDet 的理论分析表明，在低维空间（d=4），Hard OT 会导致 OT Diversity Collapse（ΔH/H_rand ∝ 1/d）。Stochastic OT ε=5 虽可匹配 random baseline，但不带来正向增益。

**缓解（已实施）**：本方案**直接采用 random coupling**（见 §3.3 决策），不迁移 Stochastic OT 模块，从设计层面规避了 OT Diversity Collapse 风险。LDMDet 实验已证明 random coupling 与 Stochastic OT ε=5 在 d=4 空间下 mAP 相同（0.751），因此该决策不损失性能。

**风险等级**：已规避（通过设计决策消除）

### 7.7 训练-推理 Gap（Train-Test Discrepancy）

**风险**：RF 扩散的多步采样在训练时仅使用单步前向（随机 $t \sim U[0,1]$），但推理时使用 Heun 4 步 ODE 积分。这种训练-推理 gap 可能导致：(1) 推理时多步采样的误差累积未在训练中被优化；(2) 中间时间步的 noisy reference points 分布与训练时不同（训练时 $t$ 均匀采样，推理时 $t$ 按离散时间步调度）。LDMDet 的 DPM-Solver++ 训练实验（前车之鉴）已证明：直接用 DPM-Solver++ 多步推理而训练时仅单步，会导致 mAP 下降（离线对比 DPM-2 8步 0.755 vs Heun 4步 0.753，差距小但存在），说明 train-test gap 的影响虽可控但不可忽略。

**缓解**：
- 训练时模拟推理调度：以推理时间步调度（$t \in \{1.0, 0.75, 0.5, 0.25\}$）的概率分布采样 $t$，而非均匀 $U[0,1]$，使训练分布与推理分布对齐
- 监控 train (1步) vs inference (4步) 的 mAP gap，若 >0.02 则考虑引入 consistency loss 或多步训练
- 若 gap 严重，可参考 LDMDet 的 DPM-Solver++ 训练实验策略：在训练后期加入多步采样微调

**风险等级**：中

---

## 8. 成功判据

### 8.1 精度判据

| 判据 | 阈值 | 说明 |
|------|------|------|
| **最低目标** | mAP ≥ 0.858 | 匹配 LDMDet a3_full_sota |
| **理想目标** | mAP ≥ 0.870 | +0.012 增益，证明 DETR 架构优势 |
| **可接受** | mAP ≥ 0.850 | 低于 SOTA 但证明方案可行，可继续优化 |

### 8.2 速度判据

| 判据 | 阈值 | 说明 |
|------|------|------|
| **推理延迟** | ≤ 30ms | RF-DETR-L baseline 6.8ms × 4步 + overhead |
| **目标延迟** | ≤ 20ms | 保持近实时性能 |
| **NFE** | ≤ 8 | Heun 4步 = 8 NFE，DPM-Solver++ 6步 = 7 NFE |

### 8.3 判据汇总

- **成功**：mAP ≥ 0.858 且 延迟 ≤ 30ms → 方案可行，继续优化
- **部分成功**：mAP ≥ 0.850 且 延迟 ≤ 30ms → 方案可行，需进一步调优
- **失败**：mAP < 0.850 或 延迟 > 30ms → 方案不可行，归档

---

## 9. 实施计划

### 9.1 代码改动范围估计

| 模块 | 改动类型 | 工作量 | 说明 |
|------|---------|--------|------|
| 环境搭建 + RF-DETR 代码库熟悉 | 新建 | 2 周 | fork RF-DETR 代码库、PyTorch Lightning + Pydantic 配置系统学习、依赖安装 |
| 24obj 数据集适配 + baseline 训练 (阶段 1) | 新建+修改 | 2-3 周 | COCO 格式转换、数据增强配置、RF-DETR-S baseline 跑通与调优 |
| RF 扩散核心移植 (rectified_flow.py + snr_scale 重映射) | 新建 | 1-2 周 | 从 LDMDet 手动移植到 RF-DETR 代码库，适配 PyTorch Lightning |
| AdaLN-Zero 时间条件化 (decoder 修改) | 修改 | 1-2 周 | DiT 原版含 α 残差门控实现，注入 self-attn/cross-attn/FFN |
| Heun 多步采样 + 训练/推理 loop | 修改 | 2-3 周 | 训练 loop 添加 RF 扩散逻辑、推理 loop 添加 Heun 4步 ODE 积分 |
| 阶段 2 实验 (Euler 1步验证) | 实验 | 1 周 | RF 扩散生成基本可行性验证 |
| 阶段 3 实验 (Heun+AdaLN) | 实验 | 1-2 周 | 完整 SOTA 配置实验与调优 |
| ref point renewal 实现 + 阶段 4 实验 | 新建+实验 | 1-2 周 | renewal 机制实现与消融实验 |
| 阶段 5 (可选 DPM-Solver++) | 实验 | 1 周 | 可选对比实验 |
| 评估、文档更新、决策 | 文档 | 1 周 | 成功/失败判定、文档更新 |
| **总计** | | **~13-19 周** | 1 人全职 |

### 9.2 依赖项

| 依赖 | 来源 | 说明 |
|------|------|------|
| RF-DETR 代码库 | github.com/roboflow/rf-detr | Apache 2.0 (N/S/M/L) |
| DINOv2 权重 | github.com/facebookresearch/dinov2 | ViT-S/B/L 预训练权重 |
| LDMDet RF 核心代码 | 本项目 `ldmdet/diffusion/rectified_flow.py` | RF + Heun + DPM-Solver++ |
| 24obj 数据集 | `data/24_chromosomes_object/coco/` | COCO 格式，24 类 |

> **注**：本方案不迁移 LDMDet 的 `coupling/ot_flow.py`（Stochastic OT 模块），直接使用 random coupling（见 §3.3 决策），减少一项依赖。

### 9.3 计算资源需求

| 阶段 | GPU | 预计时间 | 说明 |
|------|-----|---------|------|
| 阶段 1 (baseline) | 1× A100 80G | ~3-5 天 | 150 epochs, bs=8, RF-DETR-S |
| 阶段 2 (RF Euler) | 1× A100 80G | ~3-5 天 | 同上 |
| 阶段 3 (Heun+AdaLN) | 1× A100 80G | ~5-7 天 | 多步采样训练较慢 |
| 阶段 4 (renewal) | 1× A100 80G | ~5-7 天 | 同上 |
| 阶段 5 (DPM-Solver++) | 1× A100 80G | ~3-5 天 | 可选 |
| **总计** | | **~3-4 周** | 含调试和重跑 |

> **GPU 资源说明**：RF-DETR-L 官方训练使用 80GB A100，本方案以 RF-DETR-S（参数量更小）为主，仍建议使用 1× A100 80G 确保充足显存。若仅有 40GB A100，可使用 2× A100 40G (DDP) 替代。
>
> **16GB GPU 可行性评估**：项目实际硬件为 16GB GPU（如 RTX 4080/V100 16G）。RF-DETR-S 在 512×512 分辨率下，DINOv2 ViT-S backbone + decoder 的训练显存占用估计 ~12-14GB（bs=2, 混合精度），理论上可在 16GB GPU 上运行，但 batch size 受限（bs=2-4 vs 推荐 bs=8），训练时间相应增加 2-4 倍。建议：(1) 阶段 1 baseline 在 16GB GPU 上验证可行性；(2) 若显存不足，启用梯度累积（gradient accumulation）模拟更大 batch size；(3) 阶段 3-4 的多步采样训练显存开销更大（需存储中间状态），可能需要 24GB+ GPU。

### 9.4 里程碑

| 里程碑 | 预计完成 | 交付物 |
|--------|---------|--------|
| M0: 环境搭建 + 代码库熟悉 | 第 2 周 | RF-DETR fork 运行环境就绪 |
| M1: RF-DETR baseline 跑通 | 第 4-5 周 | 阶段 1 mAP 结果 |
| M2: RF 扩散生成跑通 | 第 7-9 周 | 阶段 2 mAP 结果 |
| M3: 完整 SOTA 配置 (Heun+AdaLN) | 第 10-12 周 | 阶段 3 mAP 结果 |
| M4: ref point renewal + 阶段 4 | 第 12-15 周 | 阶段 4 mAP 结果 |
| M5: 评估与决策 | 第 13-19 周 | 成功/失败判定 + 文档更新 |

---

## 10. 参考文献

1. **DiffuDETR**: Nawar, Y., Badran, M., & Torki, M. (2026). DiffuDETR: Rethinking Detection Transformers with Denoising Diffusion Process. *ICLR 2026 Poster*. OpenReview: nkp4LdWDOr. Project: https://mbadran2000.github.io/DiffuDETR

2. **RF-DETR**: Robinson, I., Robicheaux, P., Popov, M., Ramanan, D., & Peri, N. (2026). RF-DETR: Neural Architecture Search for Real-Time Detection Transformers. *ICLR 2026*. arXiv:2511.09554. Code: https://github.com/roboflow/rf-detr

3. **DeFloMat**: Lee, H., Lee, C., Seo, N., Lim, J.S., & Hong, H. (2025). DeFloMat: Detection with Flow Matching for Stable and Efficient Generative Object Localization. arXiv:2512.22406.

4. **FlowDet**: Baty et al. (2025). FlowDet: Unifying Object Detection and Generative Transport Flows. arXiv:2512.16771. (CVPR 2026 LoViF Workshop)

5. **DN-DETR**: Li, F., Zhang, H., Liu, S., Guo, J., Ni, L.M., & Zhang, L. (2022). DN-DETR: Accelerate DETR Training by Introducing Query Denoising. *CVPR 2022*. arXiv:2203.01305.

6. **DINO**: Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L.M., & Shum, H.Y. (2023). DINO: DETR with Improved Denoising Anchor Boxes for End-to-End Object Detection. *ICLR 2023*.

7. **Rectified Flow**: Liu, X., Gong, C., & Liu, Q. (2023). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. *ICLR 2023*. arXiv:2209.03003.

8. **Flow Matching**: Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. *ICLR 2023*.

9. **DiffusionDet**: Chen, S., Sun, P., Song, Y., & Luo, P. (2023). DiffusionDet: Diffusion Model for Object Detection. *ICCV 2023*.

10. **DiT (AdaLN-Zero)**: Peebles, W., & Xie, S. (2023). Scalable Diffusion Models with Transformers. *NeurIPS 2023*. (AdaLN-Zero 时间条件化机制)

11. **DINOv2**: Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning Robust Visual Features without Supervision. *TMLR 2024*.

12. **Deformable DETR**: Zhu, X., Su, W., Lu, L., Li, B., Wang, X., & Dai, J. (2021). Deformable DETR: Deformable Transformers for End-to-End Object Detection. *ICLR 2021*.

13. **LDMDet 项目实验日志**: chromosome-kd 项目 `.trae/skills/ldmdet-experiment-log`，包含 RF + Heun + Stochastic OT + AdaLN-Zero 的完整实验记录与理论推导。

14. **OT Diversity Collapse 理论**: chromosome-kd 项目 `docs/theory/THEORY_FRAMEWORK.md`，包含 CAM 定理、Stochastic Coupling 的 ε-diversity 单调性证明。

---

> **文档版本**：v2.0 (2026-07-13) — 基于三轮审查反馈修订
> **作者**：chromosome-kd 研究团队
> **审核状态**：待审核
> **下一步**：审核通过后启动阶段 1 实验
> **修订记录**：v2.0 修复 AdaLN-Zero α 残差门控公式、heun_step lambda 返回值、apply_ref_renewal 噪声空间一致性、RF-DETR/mmdet 框架不兼容声明、GPU 资源与工作量估计、CFM 区分论证逻辑、训练-推理 gap 风险、CDN 可操作缓解、预期 mAP 条件式表述、Stage 2 失败判据；删除 Stochastic OT 模块迁移（改用 random coupling）
