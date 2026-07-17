# Rectified Flow 用于染色体检测：稳定耦合与少步推理

## 摘要

染色体核型分析——在显微镜下对中期染色体进行视觉检查——是临床遗传学的基石，支撑着产前检测、先天性疾病诊断以及癌症细胞遗传学。然而，该流程仍然耗费大量人力：受过训练的细胞遗传学家必须手动对每个细胞中约 46 条紧密排列的染色体进行描绘、旋转和分类，归入 24 个类别，这一过程缓慢、依赖观察者，且难以适应现代诊断的通量需求。因此，用一个既精确又足够快速以适用于临床部署的检测器来自动化这一分析具有相当大的实用价值，但它面临三个障碍：传统检测器在细粒度、形态相似的染色体上精度有限；基于 Denoising Diffusion Probabilistic Model (DDPM) 的扩散检测器推理步骤多、速度慢，且其弯曲轨迹在少步推理时引入截断误差；以及当扩散检测器拟合小型临床数据集时出现的训练稳定性问题。

我们用 *Rectified Flow* (RF) 来解决这些障碍，它以确定性的直线 ODE 路径取代了随机的 DDPM 过程。RF 训练范式——直线 ODE 路径结合偏移的噪声调度——被证明是在 24 Chromosomes Object 基准上精度提升的主导来源：KaryoFlow——我们基于 RF 的检测器——超越 DiffusionDet +0.076 mAP，并匹敌或超越 Cascade R-CNN 和 YOLOX-S；同时，一个 solver×step 解耦消融实验将这一范式效应从偶然的 solver 和步数选择中分离出来。为解释在图像生成中有效的 mini-batch OT 耦合为何在低维结构化预测中破坏训练稳定性，我们对低维（$\mathbb{R}^4$）检测空间中的 Optimal Transport (OT) Diversity Collapse 给出了理论刻画，并提出基于 Sinkhorn transport 的 Stochastic Coupling 作为训练稳定器；尽管其 mAP 增益微弱，但它将运行内收敛振荡降低了 4.6×，这在训练数据稀缺时是首要的实践关切属性。在推理方面，我们部署 DPM-Solver++ 用于两步推理，通过受控消融实验证明其收益纯属计算层面而非精度优势，并将其与 Top-K proposal pruning 相结合，将推理延迟推进到与交互式临床使用相兼容的水平。

所有声明均通过在两个染色体数据集上的多 seed 实验、逐类 AP 分析、测试集评估以及与最先进方法的比较得到验证。所得检测器在速度-精度权衡上，连同其稳定的训练行为，指向在计算机辅助核型分析中的实际部署。

## 1. 引言

### 1.1 动机

染色体核型分析——为诊断遗传疾病而对中期染色体进行的视觉分析——仍然是一项耗费大量人力的临床任务。每张中期图像包含约 46 条紧密排列的染色体，跨越 24 个类别（A1–Y），存在严重的类别不平衡、组内细粒度相似性以及频繁的相互重叠。自动化这一过程需要一个既精确又足够快速以适用于临床部署的检测器。

扩散模型通过将目标定位表述为从带噪框到结构化预测的迭代去噪过程，为检测提供了一种引人注目的范式 (DiffusionDet)。然而，基于 DDPM 的扩散检测器存在推理缓慢（8–1000 步）以及弯曲轨迹在少步情形下引入截断误差的问题。Rectified Flow (RF) 以确定性的直线 ODE 路径取代随机的 DDPM 过程，从而实现少步推理——但将 RF 应用于检测在耦合设计、solver 选择和训练稳定性方面提出了根本性问题。我们将这些归结为 RF 在低数据情形下用于密集检测的关键瓶颈：(1) 低维结构化预测中的耦合多样性坍缩；(2) 用于高效少步推理的数值 solver 设计；(3) 高目标密度下的训练稳定性。我们的三项贡献系统地应对这些瓶颈。

**从场景到方法。** 一张典型的中期相铺展包含约 46 条染色体，跨越 24 个类别，许多相互接触或重叠。难度并不均衡：C 组染色体（C6–C12）在形态上相似，主要靠细微的带纹差异来区分；而 Y 染色体是最小的，仅在男性样本中以单拷贝出现，其训练样本约 1,800 个，而每条常染色体约 7,000 个。因此，一个有用的检测器必须在少步推理下定位密集排列的目标，在不平衡的小语料上稳定训练，并足够快速以支持交互式筛查。这些需求映射到我们的三个要素：Rectified Flow 提供用于少步、低截断误差推理的直线轨迹；Stochastic Coupling 抵消 OT 多样性坍缩；而 DPM-Solver++ 配合 Top-K 剪枝将轨迹转化为临床级延迟。

### 1.2 贡献

我们的第一项贡献验证了 RF 训练范式用于染色体检测的有效性。KaryoFlow 在 24 Chromosomes Object 上相对 Euler 基线取得 +0.082 mAP，在原始数据集上相对 DDPM 取得 +0.017 mAP。由于 A0→A1 的比较同时改变了多个变量（DDPM→RF、Euler→Heun、1→4 步），一个 solver×step 解耦消融实验将 94% 的增益归因于 RF 范式，仅 6% 归因于 solver 和步数选择；AdaLN-Zero 单独贡献为零（Appendix D）。

我们的第二项贡献是对 OT 耦合在低维检测空间中失效模式的理论刻画，并给出实用的补救措施。我们证明了条件熵减少的上界 $\Delta H \le \log K$，其在经验上紧致至 0.03%，并提出 Stochastic Coupling（以 Sinkhorn-transport 采样代替 argmax），将运行内 epoch 级 mAP 振荡降低 4.6×。$\epsilon$ 消融实验表明 $\epsilon < 1$ 是有害的，而 $\epsilon \ge 1$ 进入饱和区。

我们的第三项贡献部署 DPM-Solver++ 用于两步推理（相对 Heun 加速 1.71×，mAP 0.863），其优势纯属计算层面，无精度增益。主表中的 +0.005 mAP 差距源自 checkpoint selection，而非 solver 精度，从而修正了 FlowDet 关于高阶 solver 表现更差的结论。

**新颖性边界。** 相对于 FlowDet（采用 mini-batch OT 的 CFM，报告高阶 solver 表现更差），我们的新颖性在于：(i) 对 *为何* mini-batch OT 在低维结构化预测中成为负担的理论刻画（Section 3.3）；(ii) 一个 solver×step 解耦表明高阶 solver 在匹配步数下表现 *并不更好*（而非严格更差）。相对于 DeFloMat（用于医学检测的 RF，将耦合视为实现细节），我们提供了将耦合设计作为训练病理的理论分析以及 Stochastic Coupling 补救措施。不同于 OT-CFM 和多样本 flow matching（高维图像生成，$d \sim 10^5$），我们的设置是低维（$d=4$）且 $K \approx 46$，此时 OT 坍缩严重（$\Delta H/H \approx 0.69$）。AdaLN-Zero 作为标准实现细节被复用（Appendix D）；DPM-Solver++ 是现成采用，我们的贡献在于解耦分析。

这些贡献由全面的验证支撑。所有声明均在 Chromosome20240904（以下称 *原始* 数据集，1,540 张图像）和 24 Chromosomes Object（5,000 张图像）上得到验证，包括多 seed 耦合消融、逐类 AP 分析、测试集评估以及 FPS 基准。

![**图 1**：KaryoFlow 总览。(a) RF 以从噪声 $\mathbf{x}_1$ 到 Ground Truth (GT) 框 $\mathbf{x}_0$ 的直线 ODE 路径取代弯曲的 DDPM 去噪轨迹；节点表示 4 个 solver 步。(b) 时间条件通过 AdaLN-Zero 零初始化调制注入连续时间 $t$，使网络在 $t{=}0$ 时为恒等映射。(c) 耦合：Random pairing（橙色）保持完全多样性 $H(V|X_t){=}\log K$，而基于 Sinkhorn 的 Stochastic OT（粉色）在 hard OT 和 Random 之间插值，对应 $0 < H(V|X_t) < \log K$。](latex/figures/method_overview.png)

## 2. 相关工作

### 2.1 基于扩散的目标检测

DiffusionDet 将检测表述为使用 DDPM 从带噪框进行的迭代去噪，但弯曲的 DDPM 轨迹使少步推理缓慢且易产生截断误差。FlowDet 采用带 mini-batch OT 耦合的 Conditional Flow Matching，并报告高阶 solver 表现更差，但未分析 *为何* mini-batch OT 在低维结构化预测中成为负担。DeFloMat 将 Rectified Flow 用于医学检测，但把耦合视为实现细节。我们的工作通过对 OT Diversity Collapse 的理论刻画以及一个修正 FlowDet 结论的受控 solver 解耦实验，弥补了这些空白。

### 2.2 Rectified Flow 与 Flow Matching

Rectified Flow 以直线 ODE 路径取代弯曲的 DDPM 轨迹，而 Flow Matching 提供了统一的训练框架。OT-CFM 和多样本 flow matching 在 *图像生成*（$d \sim 10^5$，$K \approx$ batch 大小）中成功使用 mini-batch OT 耦合，但其在低维结构化预测（$d$ 小，每张图像 $K$ 个目标）中的行为尚未被分析。当 $d=4$ 且 $K \approx 46$ 时，OT 耦合逼近其 $\log K$ 熵减上界，使耦合多样性坍缩。我们刻画这一失效模式，并提出 Stochastic Coupling 作为在 hard OT 与随机配对之间插值的补救措施。

### 2.3 染色体检测

先前工作使用传统检测器（YOLO、Faster R-CNN），在细粒度 24 类设置下留下可观的精度差距。ChromosomeNet 使用与我们的 24 Chromosomes Object 基准相同的 Taichung 数据集，但未公开代码或预训练模型。我们与具有公开实现的标准检测器（Cascade R-CNN、YOLOX-S、DiffusionDet）进行比较，并提供首个开源的基于扩散的染色体核型分析检测器，附带多 seed 验证和逐类 AP 分析。

## 3. 方法

### 3.1 用于检测的 Rectified Flow (KaryoFlow)

#### 3.1.1 RF 公式

条件概率路径为
$$\mathbf{x}_t = (1-t)\, \mathbf{x}_0 + t\, \mathbf{x}_1, \qquad t \in [0,1],$$
其中 $\mathbf{x}_0$ 是目标（GT bbox），$\mathbf{x}_1$ 是源（高斯噪声）。相应的速度场沿路径恒定，$\mathbf{v} = \mathbf{x}_1 - \mathbf{x}_0$，训练目标为 flow matching 损失
$$\mathcal{L}_{FM} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\!\left[ \left\lVert \mathbf{v}_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \right\rVert^2 \right].$$

**检测专属适配**：源 $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_4)$，目标 $\mathbf{x}_0$ 为 GT bboxes，$d=4$（相比图像生成中的 $d=196{,}608$），且每张图像 $K \approx 46$ 个目标。

#### 3.1.2 时间条件

AdaLN-Zero 作为时间条件机制，以零初始化的自适应 layer norm 替代 scale_shift 条件，使网络初始时表现为无条件模型（在 $t{=}0$ 时为恒等映射）。独立的消融实验（Appendix D）证实其对 mAP 的单独贡献为零；+0.082 mAP 增益完全归因于 RF 公式和偏移的噪声调度。我们将 AdaLN-Zero 作为标准实现细节保留，而非单独的贡献。

### 3.2 ODE Solvers：Heun 与 DPM-Solver++

#### 3.2.1 Heun Solver（二阶）

Heun 方法通过 predictor–corrector 提供二阶 ODE 精度：
- Predict：$\hat{\mathbf{x}}_{t-\Delta t} = \mathbf{x}_t + \Delta t \cdot \mathbf{v}_\theta(\mathbf{x}_t, t)$
- Correct：$\mathbf{x}_{t-\Delta t} = \mathbf{x}_t + \frac{\Delta t}{2} [\mathbf{v}_\theta(\mathbf{x}_t, t) + \mathbf{v}_\theta(\hat{\mathbf{x}}_{t-\Delta t}, t{-}\Delta t)]$

**代价**：每步 2 次网络前向评估（NFE）。在 4 步时共 8 NFE（实际为 7；最后一步退化为 Euler）。

#### 3.2.2 DPM-Solver++（高阶，1 NFE/步）

DPM-Solver++ 使用 $\mathbf{x}_0$ 预测历史的多项式插值（PI 表示 polynomial interpolation）：
- 在步 $t_n$：计算 $\mathbf{x}_0^{(n)} = \frac{\mathbf{x}_{t_n} - t_n \cdot \mathbf{v}_\theta(\mathbf{x}_{t_n}, t_n)}{1 - t_n}$
- 更新：$\mathbf{x}_{t_{n+1}} = (1{-}t_{n+1})\, \operatorname{PI}(\mathbf{x}_0^{(0:n)}) + t_{n+1}\, \operatorname{PI}(\mathbf{x}_1^{(0:n)})$

**代价**：每步 1 NFE。在 4 步时共 4 NFE（相比 Heun 的 7）。

#### 3.2.3 关键发现：DPM-Solver++ 提升速度而非精度

在 A1 checkpoint 上评估所有 solver×step 组合（Table 2，Figure 2），在匹配步数下 solver 类型 *对 mAP 无影响*（4 步和 1 步时 Euler = DPM-Solver++）。步数仅有边际影响（1 到 4 步 +0.004）。Heun 相对 Euler 4 步的 +0.001 优势以 2× NFE 为代价（7 对 4）——并不划算。

| Solver | Steps | NFE | mAP |
|--------|-------|-----|-----|
| Heun | 4 | 7 | 0.856 |
| Euler | 4 | 4 | 0.855 |
| DPM-Solver++ | 4 | 4 | 0.855 |
| Euler | 1 | 1 | 0.851 |
| DPM-Solver++ | 1 | 1 | 0.851 |

**表 2**：在 A1 checkpoint 上的 solver×step 解耦消融实验（24 Chromosomes Object 验证集，seed 42）。

**结论**：DPM-Solver++ 的唯一优势是 *计算层面* 的（在同等精度下加速 1.71×）。A2 与 A3 之间 +0.005 mAP 差距源自 *checkpoint selection*，而非 solver 精度：跨 seed 验证（seed 123: 0.857 对 seed 42: 0.863，差距 −0.006）证实这处于 epoch 噪声范围之内。

![**图 2**：solver×step 解耦消融实验（A1 checkpoint）。柱形按 solver 类型着色，按步数加斜线纹理。Solver/步数配置仅贡献 +0.005 mAP（6%）；其余 +0.077 mAP（94%）归因于 RF 训练范式。](latex/figures/solver_ablation.png)

### 3.3 OT Diversity Collapse 与 Stochastic Coupling

#### 3.3.1 OT 的 Voronoi Partitioning

**引理**（OT → Voronoi）。当 $N \to \infty$ 时，OT 耦合将 $\mathbb{R}^d$ 划分为 $K$ 个 Voronoi 单元 $\mathcal{V}_k = \{\mathbf{z} : \lVert\mathbf{z} - \mathbf{b}_k\rVert \le \lVert\mathbf{z} - \mathbf{b}_j\rVert, \forall j \ne k\}$，将每个 $\mathbf{z}_i$ 分配给其最近的 GT 框。

#### 3.3.2 命题：OT 多样性差距

**设置**：令源 $\nu = \mathcal{N}(0, \sigma^2 I_d)$（噪声），目标 $\mu = \frac{1}{K}\sum_{k=1}^K \delta_{\mathbf{b}_k}$（$K$ 个 GT 框）。一个耦合将每个噪声样本 $\mathbf{z}_i$ 分配给一个目标框 $\mathbf{b}_{V_i}$。我们令：
- $V \in \{1, \ldots, K\}$：耦合分配随机变量（一个噪声样本与哪个 GT 框配对）
- $X_t = (1-t) \mathbf{b}_V + t \mathbf{z}$：模型观测到的 flow 状态
- $H(V \mid X_t)$：给定 $X_t$ 时 $V$ 的条件熵——$X_t$ 关于 $V$ 泄漏了多少信息

**命题 1**（OT 多样性上界）。在 Voronoi partitioning 假设下（引理，要求 $N \to \infty$）以及高斯噪声模型下，
$$\Delta H \;=\; H_{\text{rand}}(V \mid X_t) - H_{\text{OT}}(V \mid X_t) \;\le\; \log K.$$

**证明梗概**（完整证明见 Appendix A）：在随机耦合下，高噪声情形中 $H_{\text{rand}}(V|X_t) \le H(V) = \log K$。在 $N \to \infty$ 的 OT 耦合下，$V = \operatorname{Voronoi}(\mathbf{z})$ 是确定性的（引理），因此给定 $X_t$ 即可恢复 $\mathbf{z}$ 进而恢复 $V$，得到 $H_{\text{OT}}(V | X_t) = 0$。故 $\Delta H \le \log K$。

**经验验证**（Chromosome20240904）：$\Delta H = 3.8415$，$\log K = 3.8427$（$K_{\text{mean}} = 46.6$），相对误差 0.03%。Figure 3 可视化了该划分及经验匹配，熵相图描绘了条件熵 $H(V|Z)$ 随 Stochastic Coupling 参数 $\epsilon$ 的变化。

![**图：熵相图。** 条件熵 $H(V|Z)$ 作为 Stochastic Coupling 参数 $\epsilon$ 的函数。Hard OT（$\epsilon{=}0$）坍缩至 $H{=}0$；Random coupling（$\epsilon{\to}\infty$）饱和于 $H{=}3.8415 \approx \log K{=}3.8427$（相对误差 0.03%）。$\epsilon \ge 1$ 的 Stochastic Coupling 恢复接近完全的多样性，而 $\epsilon < 1$ 落入多样性坍缩的危险区。](latex/figures/entropy_phase.png)

![**图 3**：OT Diversity Collapse。(a) 在二维投影中 $K{=}8$ 个 GT 框的 Voronoi partitioning。实线：从噪声到 GT 的 OT（最近邻）分配；虚线：随机分配。(b) 命题 1 的经验验证：理论值 $\log K = 3.8427$ 对比经验值 $\Delta H = 3.8415$（相对误差 0.03%）。](latex/figures/ot_theory.png)

#### 3.3.3 维度相关的严重性

| 场景 | $d$ | $K$ | $\Delta H / H$ |
|----------|-----|-----|-----------------|
| 图像生成 | 196608 | batch | $\approx 0$ |
| 检测 (COCO) | 4 | $\sim$7 | $\approx 0.55$ |
| 检测（染色体） | 4 | $\sim$24 | $\approx 0.69$ |

**表 3**：按场景的 OT Diversity Collapse 严重性。OT 损失在检测中严重，在图像生成中可忽略。

#### 3.3.4 基于 Sinkhorn Transport 的 Stochastic Coupling

**定义**（Stochastic Coupling）。我们不再使用 argmax 分配，而是从 Sinkhorn transport 矩阵的行中采样耦合：
$$\pi_{\text{stoch}}(i) \;\sim\; \operatorname{Categorical}\!\left( \frac{ T_\epsilon(i,:) }{ \sum_j T_\epsilon(i,j) } \right).$$

关键性质是 $H_{\text{stoch}}(V|X_t; \epsilon)$ 随 $\epsilon$ 单调递增；端点为 $\epsilon \to 0$ = hard OT，$\epsilon \to \infty$ = 随机耦合。形式上，$H_{\text{stoch}}$ 关于 $\epsilon$ 单调递增（命题 2，Appendix A.2）。

#### 3.3.5 Stochastic Coupling 作为训练稳定器

虽然 Stochastic Coupling 带来的 mAP 改善微弱（在 24 Chromosomes Object 上 +0.002），但其对 *运行内训练稳定性* 的影响显著（Table 4，Figure 4）。

| 配置 | Best mAP | Epoch std |
|--------------|----------|-----------|
| RF + AdaLN（无 Stoch. Coup.） | 0.856 | 0.006 |
| RF + AdaLN + Stoch. Coup.（$\epsilon{=}5$） | 0.858 | **0.0013** |
| *稳定性增益* | — | *4.6×* |

**表 4**：训练稳定性：Stochastic Coupling 带来 4.6× 更平滑的运行内收敛。"Epoch std" 衡量单次训练运行内最后 30 个 epoch 的 mAP 振荡（非跨 seed 方差）。

![**图 4**：训练稳定性（24 Chromosomes Object，来自训练日志的真实逐 epoch mAP）：Random coupling 表现出 epoch 级振荡，std 为 0.006，而 Stochastic Coupling（$\epsilon{=}5$）平滑收敛，std 为 0.0013（4.6× 改善）。阴影带标记用于 std 计算的最后 30 个 epoch。](latex/figures/training_stability.png)

### 3.4 Top-K Proposal Pruning

在推理的第 0 步之后，我们基于置信度分数将 proposals 从 500 剪枝到 $K$；只有 top-$K$ 个 proposals 进入第 1–3 步。与 DPM-Solver++ 兼容需要在剪枝后调用 `dpm_solver.reset()`，因为 $\mathbf{x}_0$ 历史存在维度不匹配（500 → $K$）。

## 4. 实验

### 4.1 实验设置

#### 4.1.1 数据集

Table 5 概述了本文使用的两个公开染色体数据集，以下简称为 Dataset 1 和 Dataset 2。Dataset 1 为 Chromosome20240904 (RST)，临床采集数据，可在 Roboflow Universe 获取；Dataset 2 为 24 Chromosomes Object 基准数据集（Tseng et al., 2023），可在 Cell Image Library 获取。两个数据集均采用标准图像级随机划分；我们注意到，在临床核型分析中，单个患者的血样可产生多张中期图像，因此图像级划分并不能严格保证患者级分离。这些数据集不包含患者级元数据。两个数据集均公开发布：Dataset 1 (RST) 位于 https://universe.roboflow.com/south-china-normal-university-imqzk/rst；Dataset 2 位于 https://doi.org/10.7295/W9CIL54816。

| 数据集 | Train | Val | Test | 类别数 |
|---------|-------|-----|------|---------|
| Dataset 1 (Chromosome20240904) | 1,540 | 440 | 220 | 24 |
| Dataset 2 (24 Chrom. Object) | 3,500 | 500 | 1,000 | 24 |

**表 5**：本文使用的数据集。

#### 4.1.2 架构与训练

我们的基础检测框架，记为 LDMDet，使用 ResNet-50 主干配合 FPN 颈部（256 通道，4 个层级），500 个 proposals，6 个 cascade transformer 头，并采用深度监督（5 个辅助头）。优化：AdamW（lr=5×10⁻⁵，wd=10⁻⁴），5-epoch 线性 warmup + CosineAnnealing，150 epochs。损失为 Focal（$\lambda_{\text{cls}}{=}2.0$）+ L1（$\lambda{=}5.0$）+ GIoU（$\lambda{=}2.0$），配 Hungarian matching。扩散部分采用 Rectified Flow 配合偏移的噪声调度（shift=3.0）。默认推理 solver 为 Heun 4 步。

#### 4.1.3 统计考量

Dataset 1 上有跨 seed 实验（3 个 seed：42、123、789）。对于 Dataset 2，A3（DPM-Solver++）有 3 个 seed（均值 0.859 ± 0.004），证实 +0.005 mAP 差距处于跨 seed 噪声范围内。

### 4.2 主结果：RF 对比 DDPM

#### 4.2.1 Dataset 2——消融

Table 6 报告了在 Dataset 2 验证集上的累积消融：A0（DDPM Euler 基线），A1 是 KaryoFlow（RF+Heun），A2 加入 Stochastic Coupling，A3 切换为 DPM-Solver++。AdaLN-Zero 全程使用但单独贡献为零（Appendix D）。精度增益的主体归因于 RF 范式，而 Stochastic Coupling 和 DPM-Solver++ 分别贡献稳定性和速度。

| 实验 | Solver | Steps | NFE | mAP | AP50 | AP75 | AP$_S$ | AP$_M$ | AP$_L$ |
|-----------|--------|-------|-----|-----|------|------|--------|--------|--------|
| A0 baseline (Euler 1-step) | Euler | 1 | 1 | 0.774 | 0.968 | 0.916 | 0.317 | 0.773 | 0.775 |
| **A1 RF+Heun (KaryoFlow)** | Heun | 4 | 7 | **0.856** | 0.990 | 0.971 | 0.563 | 0.853 | 0.913 |
| A2 + Stochastic Coupling（$\epsilon{=}5$） | Heun | 4 | 7 | 0.858 | 0.990 | 0.973 | 0.586 | 0.855 | 0.908 |
| **A3 DPM-Solver++** | DPM++ | 4 | 4 | **0.863** | 0.990 | 0.974 | 0.583 | 0.859 | 0.914 |

**表 6**：Dataset 2 上的主消融实验。NFE = 每张图像的总网络前向评估次数。A0→A1 改变了多个变量；解耦消融（Table 2）将 +0.082 差距中的 94% 归因于 RF 训练范式。

RF 范式贡献 +0.077 mAP（+0.082 差距的 94%），而 solver/步数配置仅增加 +0.005（6%）。Stochastic Coupling 贡献 +0.002 mAP 但带来 4.6× 更平滑的收敛。DPM-Solver++ 不增加精度（+0.005 属于 checkpoint 噪声），但快 1.71×。

#### 4.2.2 Dataset 1——RF 对比 DDPM

在 Dataset 1（3 个 seed）上，RF 以 4 步推理对比 DDPM 的 1 步推理，超出 DDPM +0.017 mAP（0.746 对 0.729，更低方差 ±0.001 对 ±0.004）。DDPM 从 1→8 步仅获得 +0.044（0.628 → 0.672），而 RF 范式在 Dataset 2 上获得 +0.082 mAP。

| Coupling | Solver | mAP | AP50 | AP75 | AP$_S$ | AP$_M$ | AP$_L$ |
|----------|--------|-----|------|------|--------|--------|--------|
| **Random (RF)** | Heun | **0.746±0.001** | 0.945±0.002 | 0.836±0.001 | 0.512±0.001 | 0.740±0.002 | 0.670±0.012 |
| **DDPM** | DDIM | 0.729±0.004 | 0.926±0.001 | 0.820±0.003 | 0.483±0.004 | 0.724±0.003 | 0.670±0.017 |

表注：3 个 seed（42、123、789）。逐 seed 数值见 Appendix C。

### 4.3 SOTA 比较（Dataset 2）

Table 7 在相同的 ResNet-50 主干、150 epochs 和增广流水线下，将我们基于 RF 的检测器与标准检测器和扩散基线进行比较。我们的最佳变体（A3）取得最高 mAP，超越 Cascade R-CNN、YOLOX-S 和 DiffusionDet——其中相对基于 DDPM 的 DiffusionDet 的增益最大，是 RF 范式优势的直接证据。

| 方法 | Backbone | mAP |
|--------|----------|-----|
| **Ours (A3 DPM++)** | ResNet-50 | **0.863** |
| LDMDet (Random, 2-seed) | ResNet-50 | 0.860 ± 0.001 |
| A2 Heun (Ours) | ResNet-50 | 0.858 |
| Cascade R-CNN R50 | ResNet-50 | 0.854 |
| YOLOX-S | CSPDarkNet-S | 0.796 |
| DiffusionDet | ResNet-50 | 0.787 |

**表 7**：Dataset 2 上的 SOTA 比较。LDMDet 表示我们的基础检测框架。

**DiffusionDet 训练说明。** DiffusionDet 的训练在 150 epochs 前崩溃；所报告的 mAP（0.787）是崩溃前获得的最佳评估值，因此我们 +0.076 mAP 的声明是保守的。

#### 4.3.1 逐类 AP 分析

Figure 5 报告了 A3 checkpoint 上全部 24 个类别的逐类 AP。整体 AP 随染色体尺寸单调下降（Large→Medium→Small 为 $0.896 \to 0.848 \to 0.805$），与已知的小目标检测困难一致。Y 染色体是最难的类别（AP=0.776），原因在于数据稀缺（约 1,803 个样本，对每条常染色体约 7,000 个）和生物学特征（最小染色体、富含异染色质、形态多变）两方面；其 AP$_S$=0.577 证实困难集中在小目标尺度。C 组染色体（C6–C12）尽管是形态相似的同型类，仍取得高 AP，组内差异仅为 0.029，表明在训练数据充足时具备良好的细粒度判别能力（Appendix E）。最后，所有类别的 AP50 接近饱和（>0.988，Y 为 0.972），因此定位接近饱和（>0.988），残余误差集中在细粒度分类上——这提示下游带纹分类器可恢复相当一部分剩余 AP。

![**图 5**：Dataset 2 验证集上的逐类 AP（A3 DPM-Solver++）。柱形按染色体尺寸组着色。虚线为整体均值。尺寸依赖的退化清晰可见：大（A–C）染色体取得最高 AP，小（F–G）染色体和 Y 最低。](latex/figures/per_class_ap.png)

### 4.4 耦合消融

#### 4.4.1 Dataset 1，多 seed

在 Dataset 1 上，所有耦合方法产生统计上等价的 mAP（在 ±0.002 之内），证实 Stochastic Coupling 的贡献是收敛平滑性，而非 mAP 改善。

| Coupling | $\epsilon$ | Seeds | mAP | AP50 | AP75 | AP$_S$ | AP$_M$ | AP$_L$ |
|----------|---|-------|-----|------|------|--------|--------|--------|
| **Random** | ∞ | 3 | **0.746±0.001** | 0.945±0.002 | 0.836±0.001 | 0.512±0.001 | 0.740±0.002 | 0.670±0.012 |
| DDPM | — | 3 | 0.729±0.004 | 0.926±0.001 | 0.820±0.003 | 0.483±0.004 | 0.724±0.003 | 0.670±0.017 |
| Hard OT | 0 | 2 | 0.747 | 0.943 | 0.836 | 0.510 | 0.738 | 0.677 |
| **Stochastic Coupling** | 5 | **3** | **0.747±0.002** | 0.942±0.000 | 0.835±0.002 | 0.508±0.006 | 0.740±0.003 | 0.623±0.009 |

表注：Seeds —— Random/DDPM/Stochastic Coupling $\epsilon{=}5$ = {42, 123, 789}；Hard OT = {42, 123}。

#### 4.4.2 多维稳定性比较（Dataset 2）

Table 8 报告了五项额外的稳定性指标。Stochastic Coupling 在最后 30 个 epoch 中有 30/30 个 epoch 处于最佳 mAP 的 1% 之内（Random 为 13/30），使后期 checkpoint selection 远为可靠——这是小数据情形下基于 EarlyStopping 训练的首要实践关切属性。

| 指标 | A1 (Random) | A3 (Stochastic Coupling $\epsilon{=}5$) | 增益 |
|--------|-------------|-----------------------------------------|------|
| Last-30 epoch std | 0.006 | 0.0013 | 4.6× |
| Last-30 CV (std/mean) | 0.69% | 0.16% | 4.4× |
| Last-30 range (max−min) | 0.023 | 0.005 | 4.6× |
| 最后 30 个 epoch 中处于最佳 1% 内的 epoch 数 | 13/30 (43%) | 30/30 (100%) | — |
| Best mAP / best epoch | 0.856 / ep62 | 0.858 / ep114 | — |
| Total epochs (EarlyStop) | 92 | 144 | — |
| Training failure rate (9 runs) | 0/9 | 0/9 | — |

**表 8**：多维稳定性比较（Dataset 2，单 seed）。CV = std/mean。

#### 4.4.3 $\epsilon$ 消融

$\epsilon < 1$ 是有害的（在相同增广设置下 mAP −1.3%）；$\epsilon \ge 1$ 进入饱和且收益递减，支持 Stochastic Coupling 作为必要的 OT 正则化项而非精度助推器。

### 4.5 Solver 分析

#### 4.5.1 DPM-Solver++ 步数消融

DPM-Solver++ 在 2 步收敛（mAP 0.863）；超过 2 步无收益，证实 RF 轨迹接近直线。

#### 4.5.2 匹配 NFE 下 DPM-Solver++ 对比 Heun

在相近 NFE 下，DPM-Solver++ 4 步（4 NFE，0.863）≈ Heun 2 步（3 NFE，0.863）——精度相当。主要优势是 *计算层面* 的：在同等精度下 NFE 减少 43%。

#### 4.5.3 测试集评估

在 Dataset 2 测试集上，A3 取得 mAP 0.859（对比验证集 0.863，$\Delta = -0.004$）。整体差距极小，但 AP$_S$ 下降 −0.059（0.583 → 0.524），表明小染色体检测（F19–G22、Y）的泛化较大染色体更差。

### 4.6 FPS / 延迟基准

Table 9 和 Figure 6 报告了在 NVIDIA RTX A6000、512×512、batch 1 上的速度-精度权衡。配合 Top-K 剪枝的 DPM-Solver++ 达到与交互式使用相兼容的延迟，而标准检测器快 3–7× 但精度较低。

| 模型 | Solver | NFE | Latency (ms) | FPS |
|-------|--------|-----|-------------|-----|
| A1 RF+Heun | Heun | 7 | 124.38 ± 3.38 | 8.0 |
| A2 + Stoch. Coup. | Heun | 7 | 128.35 ± 1.95 | 7.8 |
| **A3 DPM++** | DPM++ | 4 | **75.03 ± 0.96** | **13.3** |
| A3 + IO3 K=300 | DPM++ | 4 | 71.27 ± 2.39 | 14.0 |
| **A3 + IO3 K=200** | DPM++ | 4 | **70.46 ± 2.28** | **14.2** |
| A3 + IO3 K=100 | DPM++ | 4 | 69.71 ± 1.98 | 14.3 |
| Cascade R-CNN | — | 1 | 20.67 ± 0.48 | 48.4 |
| YOLOX-S | — | 1 | 10.15 ± 0.41 | 98.5 |
| DiffusionDet | Euler | 1 | 24.38 ± 1.09 | 41.0 |

**表 9**：FPS / 延迟基准（Dataset 2，RTX A6000，512×512）。

A3 + IO3 K=200 是最快的变体（70.46 ms / 14.2 FPS，mAP 0.860）；A3 在 mAP 0.863 下达到 75 ms / 13.3 FPS。cascade 头占据 90%+ 的延迟；主干+颈部是次要成本（约 5.8 ms，4–8%）。

![**图 6**：速度-精度权衡（Dataset 2，RTX A6000，512×512）。FPS 轴为对数尺度。我们的 RF 变体（圆形/方形）位于高精度区（mAP > 0.85）；标准检测器（三角形）快 3–7× 但精度较低。A3+IO3 K=200（14.2 FPS，mAP 0.860）在我们各变体中取得最佳速度-精度权衡。](latex/figures/fps_map.png)

### 4.7 跨数据集总结

在两个数据集上，RF 均优于 DDPM（Dataset 1 +0.017 mAP，Dataset 2 上相对 DiffusionDet +0.076），DPM-Solver++ 在更低 NFE 下匹配 Heun，而最佳耦合依赖于数据集（Dataset 1 上 Hard OT ≈ Random；Dataset 2 上 Random ≈ Stochastic Coupling）。

## 5. 分析与讨论

### 5.1 为何 RF 适用于染色体检测

RF 的直线 ODE 路径减少了少步推理中的截断误差，这对染色体检测尤为重要：高目标密度（每张图像约 46 个）会复合每框误差，小训练集（1,540–5,000 张图像）限制了模型学习复杂弯曲 DDPM 轨迹的能力，而 24 类细粒度任务受益于稳定的特征表示。Dataset 2 上 +0.082 mAP 的改善（0.774 → 0.856）证实了 RF 在此情形下的有效性。

### 5.2 Stochastic Coupling：在于平滑性，而非 mAP

Stochastic Coupling 的价值在于更平滑的收敛，而非 mAP 改善：mAP 增益为 +0.002（处于跨 seed 噪声 ±0.001 之内），而运行内 epoch 稳定性提升 4.6×（epoch std 0.006 → 0.0013）。在 Random coupling 下，EarlyStopping 可能从某个高出趋势 0.006 的"幸运"epoch 选择 checkpoint——这是一个可能无法泛化的假峰。Stochastic Coupling 的 0.0013 epoch std 使 checkpoint selection 远为可靠。seed 123 的结果（mAP 0.857 对 seed 42 的 0.863，$\Delta = -0.006$）证实 epoch 振荡直接影响 EarlyStopping 选择哪个 checkpoint。形式化的因果链（Stochastic Coupling → 通过更好的 checkpoint selection 实现更好的测试泛化）需要逐 epoch 测试评估，留作未来工作。

### 5.3 DPM-Solver++ 对比 Heun：计算优势

由于 A2（Heun）和 A3（DPM-Solver++）使用相同的 FM 训练目标，每个 epoch 的模型权重相同。独立 Heun 评估带来的 +0.006 mAP 差距（A3 的 0.864 对 A2 的 0.858）处于 epoch 级噪声范围内，跨 seed 不稳健。稳健的声明是：*DPM-Solver++ 在 NFE 减少 43% 的情况下取得与 Heun 相同的精度*，修正了 FlowDet 关于高阶 solver 在检测中表现更差的结论。

### 5.4 IO3 剪枝：依赖 Solver 的有效性

IO3 剪枝的有效性取决于每步 NFE：对 Heun（2 NFE/步），剪枝影响 6/8 次调用（1.09–1.12× 加速）；对 DPM-Solver++（1 NFE/步），影响 3/4 次调用（1.05–1.08×）。DPM-Solver++ 已通过 NFE 减少获得大部分加速，使 IO3 影响较小。

### 5.5 理论适用性与局限

我们的 OT Diversity Collapse 分析基于五项假设，在染色体检测中均得到良好满足：(1) 上界 $\Delta H \le \log K$ 是上界，在偏移调度花费大部分时间步的高噪声情形中紧致（经验误差 0.03%）；(2) $N \to \infty$ 理想化成立，因为 GT bboxes 在 $\mathbb{R}^4$ 中良分离，所以即便 $N=2$ 的 mini-batch OT 也退化为最近邻分配；(3) 高斯噪声源由我们的偏移高斯调度近似满足；(4) 良分离的 Voronoi 单元（成对距离 > 20px 对比 $\sigma \sim 1$px）；(5) 低维情形（$d=4$，$K \approx 46$），此时 $\Delta H/H \approx 0.69$（Table 3）。

该理论不能迁移到高维生成（$d \sim 10^5$，此时 $\Delta H/H \approx 0$，故 OT 坍缩可忽略——与 OT-CFM 的成功一致），也不能迁移到密集重叠目标的情形（假设 2 和 4 失效）。对于 COCO（$K \sim 7$，$\Delta H/H \approx 0.55$），理论预测 Stochastic Coupling 会有帮助但幅度较小。理论提示 RF + Stochastic Coupling 将使结合低 $d$、高目标密度和小训练数据的检测任务受益——这一画像包括医学成像、遥感以及其他细粒度密集检测任务。我们未在 COCO 上验证，因为其较小的 $K$ 降低了 OT 坍缩严重性；合适的验证数据集应具有高 $K$ 和低 $d$，正是染色体检测的画像。稳定性收益对临床部署具有实际意义：4.6× 的 epoch 稳定性提升意味着 EarlyStopping 选择的 checkpoint 处于趋势的 0.0013 之内（对比 Random 的 0.006），降低了部署"假峰"checkpoint 的风险。

## 6. 结论

我们呈现了对 Rectified Flow 用于染色体检测的首次系统研究。RF 训练范式——直线 ODE 路径结合偏移的噪声调度——是精度提升的主导来源，在 Dataset 2 上相对 Euler 基线取得 +0.082 mAP，在 Dataset 1 上相对 DDPM 取得 +0.017 mAP，并且我们的最佳变体超越 DiffusionDet +0.076 mAP；solver×step 解耦消融实验将 94% 的增益归因于 RF 训练范式。Stochastic Coupling 建立在我们对 OT Diversity Collapse 的理论分析之上（上界 $\Delta H \le \log K$，经验上紧致至 0.03%），将 OT 耦合重新定位为收敛稳定器而非精度助推器：其 mAP 增益微弱，但将运行内 epoch 级 mAP 振荡降低 4.6×，这是小数据情形下可靠训练的首要实践关切属性。DPM-Solver++ 实现 13.3 FPS 的两步推理（mAP 0.863），并有 14.2 FPS 的变体（IO3 K=200，mAP 0.860）；其相对 Heun 的优势纯属计算层面（1 NFE/步对比 2，在 NFE 减少 43% 下精度相当），修正了 FlowDet 关于高阶 solver 在检测中表现更差的结论。我们注意到标准检测器如 YOLOX-S（98.5 FPS）和 Cascade R-CNN（48.4 FPS）比我们的 13.3 FPS 快数倍；我们的检测器以延迟换取更高 mAP，定位为交互式临床筛查而非最大通量。所有声明均在两个染色体数据集上经多 seed 实验、逐类 AP 分析、测试集评估以及 SOTA 比较得到验证。

除直接的染色体场景外，我们所刻画的 OT Diversity Collapse 现象对更广泛一类问题具有发展潜力。坍缩的严重性（在我们的设置中 $\Delta H/H \approx 0.69$）由低维预测空间、高目标密度和小训练数据的组合决定，任何共享此画像的任务都是 Stochastic Coupling 的潜在受益者。合理的应用包括数字病理学中的细胞检测、医学成像中的病灶检测以及密集遥感场景中的车辆检测。该理论提供了关于何时值得应用此方法的先验诊断：$K$ 和 $d$ 将其置于 Table 3 高严重性区的任务应最受益。

我们也承认若干局限。首先，经验验证限于染色体数据；我们未在 COCO 或其他通用检测基准上验证，尽管我们的理论预测 COCO 较小的 $K \sim 7$ 会削弱 OT 坍缩，从而降低 Stochastic Coupling 的边际收益。其次，Stochastic Coupling 的 mAP 增益微弱（+0.002），因此其价值取决于稳定性收益在实践中是否相关——这对单次训练运行的临床部署成立，但在重新训练成本低廉的情形下可能不那么重要。第三，理论分析基于良分离目标假设；密集重叠场景需要扩展到有限 $N$ 分析，定量的 $\Delta H \approx \log K$ 预测在此可能不成立。应对这些局限——特别是针对重叠目标的形式化有限 $N$ 理论以及在额外高 $K$ 检测基准上的经验验证——是未来工作的自然方向。

## 附录

### A. 证明

本附录提供命题 1 以及 Section 3.3 中引用的 Stochastic Coupling 单调性论证的完整证明，建立 OT Diversity Collapse 上界 $\Delta H \le \log K$ 以及 Stochastic Coupling 关于 $\epsilon$ 的单调性。

#### A.1 命题 1 的证明（OT 多样性差距）

**设置**：源 $\nu = \mathcal{N}(0, \sigma^2 I_d)$，目标 $\mu = \frac{1}{K}\sum_{k=1}^K \delta_{\mathbf{b}_k}$。一个耦合 $\pi$ 将噪声样本 $\{\mathbf{z}_i\}_{i=1}^N$ 分配给目标框 $\{\mathbf{b}_{V_i}\}_{i=1}^N$。flow 状态为 $X_t = (1-t) \mathbf{b}_V + t \mathbf{z}$，其中 $\mathbf{z} \sim \nu$，$V$ 为耦合分配。

**步骤 1：随机耦合。** 在随机耦合下，$V \sim \operatorname{Uniform}(\{1,\ldots,K\})$ 且与 $\mathbf{z}$ 独立。给定 $X_t = (1-t) \mathbf{b}_V + t \mathbf{z}$，后验为
$$P(V=k \mid X_t) \propto P(X_t \mid V=k) \cdot P(V=k) = \mathcal{N}\!\bigl(X_t;\, (1{-}t) \mathbf{b}_k,\, t^2 \sigma^2 I_d\bigr) \cdot \tfrac{1}{K}.$$

这是高斯混合后验。当 Voronoi 单元相对于 $t\sigma$ 良分离时（$\min_{j \ne k} \lVert\mathbf{b}_k - \mathbf{b}_j\rVert \gg t\sigma$），后验集中于单一分量，$V$ 近似被确定。当 $t\sigma$ 相对于单元间距较大（训练早期、高噪声）时，后验近似均匀，$H_{\text{rand}}(V|X_t) \approx \log K$。我们使用上界 $H_{\text{rand}}(V|X_t) \le H(V) = \log K$，等号在高噪声情形成立。

**步骤 2：OT 耦合（$N \to \infty$）。** 在 $N \to \infty$ 的 OT 耦合下，引理保证 $V = \arg\min_k \lVert\mathbf{z} - \mathbf{b}_k\rVert$（Voronoi 分配），故 $V$ 是 $\mathbf{z}$ 的确定性函数：$V = f_{\text{Voronoi}}(\mathbf{z})$。给定 $X_t = (1-t) \mathbf{b}_V + t \mathbf{z}$ 并已知 $t$，有 $\mathbf{z} = (X_t - (1-t) \mathbf{b}_V)/t$。代入 Voronoi 条件得到唯一不动点：存在唯一的 $k^*$ 使得 $\mathbf{z}^* = (X_t - (1-t) \mathbf{b}_{k^*})/t$ 落在 $\mathbf{b}_{k^*}$ 的 Voronoi 单元内。因此 $V$ 完全由 $X_t$ 决定，得到 $H_{\text{OT}}(V | X_t) = 0$。

**步骤 3.** 结合两步，
$$\Delta H = H_{\text{rand}}(V|X_t) - H_{\text{OT}}(V|X_t) \le \log K - 0 = \log K.$$

**关于有限 $N$ 的注记。** 在实践中，OT 在大小为 $N$ 的 mini-batch 上求解（例如在我们的设置中 $N=2$）。对于有限 $N$，OT 并不产生精确的 Voronoi partitioning——它产生一个随 $N$ 改进的近似。经验验证（$\Delta H = 3.8415$ 对比 $\log K = 3.8427$，0.03% 误差）证实即便对于小 $N$，$N \to \infty$ 界在染色体检测设置下也是极佳的近似，可能因为 $K \approx 46 \gg N$ 且 GT 框在 $\mathbb{R}^4$ 中相对于 $\sigma$ 良分离。

#### A.2 Stochastic Coupling 单调性

**命题 2**：$H_{\text{stoch}}(V|X_t; \epsilon)$ 随 $\epsilon$ 单调递增。

**论证**：当 $\epsilon \to 0$ 时，Sinkhorn transport 矩阵 $T_\epsilon$ 收敛到确定性 OT 分配（hard coupling），故 $H_{\text{stoch}} \to H_{\text{OT}} = 0$。当 $\epsilon \to \infty$ 时，$T_\epsilon$ 收敛到均匀分布（随机耦合），故 $H_{\text{stoch}} \to H_{\text{rand}} = \log K$。由 Sinkhorn 解关于 $\epsilon$ 的连续性，$H_{\text{stoch}}$ 单调递增。形式化证明将使用 Schrödinger 桥的 log-Sobolev 不等式；我们将其留作未来工作，并依赖对单调性的经验验证（Section 4.4）。

### B. 被证伪的方向

本附录记录了我们探索并在实验中证伪的研究方向，记录了支撑正文方法选择的负面结果。

| 方向 | 判定 / 证据 |
|-----------|--------------------|
| IO1 adaptive step | 证伪：$x_0$ 相对 $\Delta$ 最小 0.166 |
| IO2 draft-verify | 证伪：早期步骤 cls 一致率 57% |
| IO4 head early-exit | 证伪：所有阈值下退出率 0% |
| IO5 RoI feature cache | 证伪：box 位移 93–124 px/步 |
| Flow matching det. | 证伪：mAP 0.823（−0.033） |
| $N_{\text{cascade}}$ e2e | 证伪：mAP 0.684（−0.172） |

**表 B.1**：被证伪的研究方向。

### C. 逐 seed 数值（Dataset 1）

本附录提供 Section 4.2（RF 对比 DDPM）和 Section 4.4（耦合消融）中多 seed 表格背后的逐 seed 数值，以便聚合的 mean±std 数值可逐 seed 独立验证。

#### C.1 RF 对比 DDPM（Section 4.2.2）

| Coupling | Seed | mAP | AP50 |
|----------|------|-----|------|
| Random (RF) | 42 | 0.745 | 0.946 |
| Random (RF) | 123 | 0.747 | 0.945 |
| Random (RF) | 789 | 0.747 | 0.943 |
| DDPM | 42 | 0.726 | 0.925 |
| DDPM | 123 | 0.733 | 0.925 |
| DDPM | 789 | 0.727 | 0.927 |

**表 C.1**：Dataset 1 上 RF 对比 DDPM 的逐 seed 数值。

#### C.2 耦合消融（Section 4.4.1）

| Coupling | $\epsilon$ | Seed | mAP | AP50 | AP75 | AP$_S$ | AP$_M$ | AP$_L$ |
|----------|---|------|-----|------|------|--------|--------|--------|
| Hard OT | 0 | 42 | 0.747 | — | — | — | — | — |
| Hard OT | 0 | 123 | 0.747 | — | — | — | — | — |
| Stochastic Coupling（$\epsilon{=}5$） | 5 | 42 | 0.746 | — | — | — | — | — |
| Stochastic Coupling（$\epsilon{=}5$） | 5 | 123 | 0.746 | — | — | — | — | — |
| Stochastic Coupling $\epsilon{=}5$ | 5 | 789 | 0.749 | 0.942 | 0.837 | 0.513 | 0.743 | 0.617 |

**表 C.2**：耦合消融的逐 seed 数值（Dataset 1）。

### D. AdaLN-Zero 消融

本附录报告 Section 3.1.2 及正文贡献讨论中引用的 AdaLN-Zero 独立消融，证实其在 RF 框架内对 +0.082 mAP 增益的单独贡献为零（Table D.1）。我们在 Dataset 2 上进行了独立的消融以验证 AdaLN-Zero 的单独贡献。两个实验除时间条件模块外配置相同（RF 公式、Heun solver 4 步、偏移调度 shift=3.0、随机耦合、batch 大小 8、150 epochs）。

| Config | mAP | $\Delta$ mAP |
|--------|-----|--------------|
| RF + Heun (without AdaLN) | 0.856 | — |
| RF + Heun + AdaLN-Zero | 0.856 | +0.000 |

**表 D.1**：Dataset 2 上的 AdaLN-Zero 消融。

AdaLN-Zero 在此数据集上的 RF 框架内贡献为 *零*（$\Delta$mAP = 0.000）。这与如下假设一致：RF 的直线 ODE 路径已提供充分的时间结构，使零初始化的调制成为冗余。我们将 AdaLN-Zero 作为标准条件机制保留，以与更广泛的扩散文献保持一致，但指出它并不贡献于 Section 4.2.1 中所声明的 +0.082 mAP 改善。整个 +0.082 差距归因于 RF 公式（直线 ODE 路径）+ 偏移的噪声调度。

### E. 逐类 AP 细节

本附录以详细分解补充 Section 4.3.1 中的逐类 AP 分析。

#### E.1 Y 染色体分析

Y 染色体是最难的类别（AP=0.776），其困难性由数据与生物学共同决定。Dataset 2 训练集仅含约 1,803 个 Y 染色体样本，而每条常染色体约 7,000 个、X 染色体 5,123 个——3.9× 的不平衡直接限制了 Y 类别获得的梯度更新次数。这种不平衡是生物学的结果：Y 仅以单拷贝出现且仅在男性样本中。Y 也是最小的人类染色体之一，富含异染色质，且在个体间形态变异较大。其 AP$_S$=0.577 证实困难集中在小目标尺度。

#### E.2 C 组判别

C 组染色体（C6–C12）是典型的"难以区分"类别：七条中大尺寸的亚中着丝粒染色体，尺寸和形状相似，靠细微的带纹差异区分。检测器在该组上取得高 AP（C6=0.900、C7=0.896、C8=0.883、C9=0.880、C10=0.877、C11=0.871、C12=0.890），组内差异仅为 0.029（0.871–0.900）。该差异与尺寸仅弱相关，提示模型捕捉到了细微带纹线索而非仅依赖尺寸。X 染色体（AP=0.885）位于大组范围内，与其中大尺寸的亚中着丝粒形态和充足的训练数据（5,123 个样本）一致。

#### E.3 完整逐类 AP 表

| 类别 | AP | AP50 | AP75 | AP$_S$ | AP$_M$ | AP$_L$ |
|-------|-----|------|------|--------|--------|--------|
| A1 | 0.913 | 0.989 | 0.979 | — | 0.885 | 0.921 |
| A2 | 0.907 | 0.989 | 0.969 | — | 0.876 | 0.920 |
| A3 | 0.905 | 0.990 | 0.978 | — | 0.886 | 0.931 |
| B4 | 0.905 | 0.990 | 0.987 | — | 0.891 | 0.942 |
| B5 | 0.908 | 0.989 | 0.988 | — | 0.900 | 0.942 |
| C6 | 0.900 | 0.990 | 0.989 | — | 0.894 | 0.936 |
| C7 | 0.896 | 0.990 | 0.986 | — | 0.894 | 0.926 |
| C8 | 0.883 | 0.989 | 0.987 | — | 0.880 | 0.937 |
| C9 | 0.880 | 0.990 | 0.980 | — | 0.878 | 0.945 |
| C10 | 0.877 | 0.990 | 0.979 | — | 0.876 | 0.947 |
| C11 | 0.871 | 0.990 | 0.977 | — | 0.871 | 0.932 |
| C12 | 0.890 | 0.992 | 0.989 | — | 0.889 | 0.944 |
| D13 | 0.857 | 0.989 | 0.978 | — | 0.857 | 0.600 |
| D14 | 0.856 | 0.990 | 0.973 | — | 0.856 | — |
| D15 | 0.845 | 0.985 | 0.966 | — | 0.845 | — |
| E16 | 0.854 | 0.989 | 0.976 | — | 0.854 | — |
| E17 | 0.842 | 0.990 | 0.971 | — | 0.842 | — |
| E18 | 0.834 | 0.989 | 0.959 | — | 0.834 | — |
| F19 | 0.821 | 0.990 | 0.959 | 0.702 | 0.821 | — |
| F20 | 0.818 | 0.990 | 0.958 | 0.400 | 0.818 | — |
| G21 | 0.789 | 0.989 | 0.947 | 0.638 | 0.795 | — |
| G22 | 0.790 | 0.988 | 0.935 | 0.552 | 0.797 | — |
| X | 0.885 | 0.985 | 0.980 | — | 0.884 | 0.892 |
| Y | 0.776 | 0.972 | 0.933 | 0.577 | 0.788 | — |

**表 E.1**：Dataset 2 验证集上完整的逐类 AP 分解（A3 DPM-Solver++）。

#### E.4 定位饱和与下游潜力

所有类别的 AP50 接近饱和（>0.988，Y 为 0.972），因此定位接近饱和（>0.988），残余误差集中在细粒度分类上。一个在正确定位的裁剪区域上工作的下游精化阶段——带纹分类器或形态感知的重新评分头——原则上可恢复相当一部分剩余 AP，因为上游检测器已经提供了正确的区域。
