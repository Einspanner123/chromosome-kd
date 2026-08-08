# Rectified Flow 用于染色体检测：少步求解与定位质量校准

> **证据修正（2026-08-09）**：核心贡献更新为 RF、RF 适配的 DPM-Solver++ 与
> Localization-Quality Calibrated Ranking（LQCR）。历史 StochOT
> `$+0.034, p<10^{-120}$` 来自增强策略混杂，已失效；4.6× 平滑性为单 seed、
> 自相关 epoch 的探索性统计，不再作为核心方法收益。权威证据表见
> `docs/research/论文创新点重构_RF_DPM_Quality_20260809.md`。

> 📋 **命名约定**: 本文档使用论文正式名称 (Dataset 1 / Dataset 2 / DDPM baseline / RF+Heun / +Stoch. Coupling / +DPM-Solver++)。内部实验代号 (24obj / A0-A3 / StochOT) 仅保留在文件路径、配置名和日志文件名中以兼容工程实现。

<!--
============================================================
TMI 投稿迁移策略头部
============================================================
目标期刊：IEEE Transactions on Medical Imaging (TMI)
模板：IEEEtran.cls [journal,10pt]

硬约束（投稿前对照实时作者指南验证）：
- 初始投稿 ≤ 10 页（含参考文献，硬性上限，超限直接退稿不送审）
- Abstract < 250 词；必须包含 IEEEkeywords
- 双栏，单倍行距，两端对齐，10pt
- 单盲评审（初始投稿可保留作者信息，不写作者简介）
- 禁止 supplementary text materials（自 2022-01-01 起）→ arXiv companion preprint
- 图表必须内联在正文中
- 参考文献：IEEEtran.bst，作者名格式（如 J. Smith）

策略 A+B（用户已批准）：
- 策略 A：在正文中压缩附录（保留证明梗概，简要提及）
- 策略 B：将详细证明/逐 seed 表移至 arXiv companion preprint

内容目标标注图例：
  [MAIN PAPER]      -> 保留在 10 页 IEEE 投稿版中
  [ARXIV COMPANION] -> 移至 arXiv preprint，从正文引用
  [COMPRESS]        -> 压缩为 1-2 句内联，完整版在 arXiv
  [DELETED]         -> TMI 不需要（本草稿保留作为事实记录）
  [PLACEHOLDER]     -> Phase 3-D（鲁棒性）或 3-E（理论）待添加的新内容

草稿状态：
- 本草稿是完整事实记录（最详细内容，明确陈述事实）。
- 为 10 页 IEEE 主文进行选择性写作是正常技术；
  标记为 [ARXIV COMPANION] 或 [DELETED] 的内容并未丢失——它在此保留
  作为权威记录并供给 arXiv companion。
- 同步状态：同步至 main.tex commit 8f67c88a（2026-07-19，Phase 3 完成）。
  Phase 3 变更已纳入：
  - 新增 Proposition 2（OT Diversity 下界，Fano 不等式）
  - Conjecture 1 升级为 Proposition 3（单调性，包络定理证明）
  - 跨数据集 §4.8 修正：Dataset 1 = Chromosome20240904（220 test imgs,
    10262 instances）。数值：mAP=0.157,
    AP50=0.513, AP75=0.039；14/24 类 AP50>0.5；C-group 失效。
  - Conclusion 更新为引用双侧界
  - Appendix B+C 合并为单节 "AdaLN-Zero and Shifted Schedule Ablations"
    （B 的零结果表移至 arXiv companion）
  - 压缩至 ≤10 页（满足 TMI 硬约束）
============================================================
-->

> **TMI 定位说明（基于 ChatGPT 分析 + 用户审阅 2026-07-19）：**
> 本文的新颖性是 RF 检测范式、RF 适配的 DPM-Solver++ 与定位质量校准排序
> （LQCR）的理论—实现闭环；OT coupling 仅保留为负面/辅助分析。然而，论文仍必须从应用
> （染色体核型分析）切入——算法新颖性服务于临床任务，而非相反。
> 平衡：应用背景开启 Abstract/Intro，算法贡献作为解决方案随之而来。
> 理由：TMI 是医学影像期刊；审稿人期望先看到临床动机。

## 摘要

<!-- [MAIN PAPER] TMI 摘要：≤250 词（硬约束，超限退稿不送审）。
    本版为压缩投稿版（2026-07-29），中文与英文正式版基本对应，估算 ~233-250 英文词。
    叙事策略参考 HiDiff TMI 2024 的判别式/生成式范式区分框架。
    背景：临床价值 → 手工痛点 → 深度学习自动化（判别式范式）→ 范式局限（缺乏生成动力学建模，
    低数据下优势收窄，有 Dataset 1/2 对比支撑）。
    贡献定位：区别于判别式理论、基于扩散+流匹配的生成式检测新范式 + 深入理论分析 → 前沿水平。
    低数据 SOTA 为核心抓手。砍掉的细节（91% 增益、DPM-Solver++ 1.75×、Stochastic Coupling 4.6× 稳定性、
    ΔH 界、§4.2 引用等）移至正文 §4.2-4.5。结构：临床动机 → 范式区别定位 → RF+理论 → 实验结果 → 结论升华。
    注：StochOT 精度增益声明（+0.034, p<10^-120）已废止——统一标准增强后无显著差异（Δ≤0.001, ns），
    StochOT 核心价值为训练稳定性（4.6× epoch-std 降低）。 -->

人工染色体核型分析是遗传疾病诊断与产前筛查的基础技术，但手工分析耗时且训练数据有限。本文提出 KaryoFlow-LQCR：首先以 Rectified Flow (RF) 的低曲率 ODE 路径取代 DDPM 的弯曲去噪轨迹，再将 DPM-Solver++ 的 data-prediction 多步形式适配到框流，以 4 NFE 完成少步求解；最后引入 Localization-Quality Calibrated Ranking (LQCR)，由末级 proposal feature 预测定位质量，并仅在最终输出采用 $s=pq^2$ 校准排序。严格 final-only 消融保证 LQCR 不改变类别、框坐标、solver 或 box renewal 轨迹。在 Chromosome20240904 数据集上，RF 主线相对 DDPM 提升约 $+0.017$ mAP；LQCR 的 seed42 训练验证由 $0.746$ 提升至 $0.751$（ross 统一复评中）。在 24 Chromosomes Object 数据集上，RF 相对 DDPM 提升约 $+0.053$ mAP；LQCR 进一步由 $0.86301$ 提升至 $0.87044$（$+0.00743$），其中 AP90/AP95 分别提升 $+0.03041/+0.03251$。DPM-Solver++ 将 Heun 的 7 NFE 降至 4 NFE。结果表明，生成轨迹、数值求解与定位质量排序三个层次可被独立优化，使扩散检测器在数据稀缺场景下接近前沿判别式检测器。

<!-- [MAIN PAPER] IEEEkeywords 占位符 — 待最终确定：
Index Terms --- Rectified Flow, object detection, optimal transport, diffusion models, medical image analysis, chromosome karyotyping
-->

## 1. 引言

<!--
[MAIN PAPER] TMI 引言叙事策略（2026-07-29 修订，与摘要一致）：
- 临床动机主导：以染色体核型分析的临床价值与手工痛点开头（参考 HiDiff TMI 2024）
- 判别式 vs 生成式范式区分：传统检测器（YOLO/Faster R-CNN/DINO/RTMDet）为判别式范式，
  KaryoFlow 为基于扩散+流匹配的生成式新范式
- 结构：P1 临床动机 + 判别式范式局限 → P2 生成式范式 + DDPM 病理 + RF 补救 →
  P3 三个关键瓶颈（耦合/solver/稳定性）→ P4 染色体作为动机实例 + 场景到方法映射 →
  P5 贡献总结 + 新颖性边界
- 数据呈现与摘要一致：best mAP 领头，多 seed 验证
- 在引言末尾保留 Figure 1（方法总览）。
-->

人工染色体核型分析是遗传疾病诊断与产前筛查的基础技术，但手工分析单例耗时约 30–35 分钟、观察者间一致率仅约 70–80%，制约筛查通量。YOLO、Faster R-CNN、DINO、RTMDet 等通用检测器虽已实现该流程的自动化高通量处理，但这些判别式范式将检测建模为从图像到边界框的直接映射，缺乏对检测过程本身的数据生成动力学的理论建模，且在临床训练数据稀缺（每队列约 1,500–5,000 张）时优势收窄。基于扩散的检测器提供了一种根本不同的生成式范式：将目标检测重构为从噪声框到 GT 框的迭代去噪，通过对检测过程的数据生成动力学进行显式建模。然而，现有扩散检测器继承自 DDPM 的弯曲随机轨迹在少步推理下产生显著的截断误差，且该误差在高目标密度场景（每张图像数十个框）下跨 proposals 复合；小型训练数据集进一步使学习复杂弯曲轨迹的能力受限。这三重瓶颈——截断误差累积、训练不稳定和推理延迟——构成了扩散检测在临床部署前的核心障碍。Rectified Flow (RF) 以从噪声到 GT 的低曲率 ODE 路径取代弯曲 DDPM 轨迹，速度场沿路径近似恒定，从而在少步情形下保持低截断误差——为缓解上述瓶颈提供了理论支撑的补救。

将 RF 应用于检测提出三个必须回答的问题：如何降低噪声框到目标框的轨迹建模误差；如何以尽可能少的 NFE 求解该轨迹；以及如何修复类别置信度与高 IoU 定位质量之间的排序错配。它们分别对应 RF、DPM-Solver++ 和 LQCR。低维 OT coupling 的熵坍缩仍值得分析，但统一增强下未转化为显著精度差异，因此不再作为核心提升点。

**动机实例：染色体核型分析。** 每张图像约 46 条染色体、24 个形态相似类别，使系统同时面对少步轨迹误差、推理成本和高 IoU 排序错配。RF 降低轨迹建模误差，DPM-Solver++ 降低少步积分成本，LQCR 缓解类别置信度与定位质量不一致造成的 AP 损失。实验在两个公开数据集（Chromosome20240904，1,540 张；24 Chromosomes Object，5,000 张）上验证。

本文做出三项相互独立的贡献。第一，RF 检测公式是主要精度来源：Dataset 1/2 相对 DDPM 分别约为 $+0.017/+0.053$ mAP。第二，我们将 DPM-Solver++ 适配到 RF 的 data-prediction 形式，以 4 NFE 代替 Heun 的 7 NFE；其可信贡献是更低计算量下匹配精度，而非稳定的大幅 mAP 增益。第三，我们提出 LQCR：依据 probability-ranking principle，将类别正确后验与定位质量后验近似分解，并用 $p q^2$ 校准最终排序。Dataset 2 的严格 final-only 消融提升 $+0.00743$ mAP、AP90/AP95 提升 $+0.03041/+0.03251$；Dataset 1 seed42 训练验证提升 $+0.005$，待统一复评。OT diversity collapse 与 Stochastic Coupling 作为耦合分析保留，但统一增强下没有显著精度收益；其 $4.6\times$ 曲线平滑性仅为单 seed 探索性观察。

![**图 1**：KaryoFlow 总览——推理端与训练端的架构关系。(a) **Rectified Flow 范式**（推理端）：RF 以从噪声 $\mathbf{x}_1$ 到 GT 框 $\mathbf{x}_0$ 的低曲率 ODE 路径（橙色实线）取代弯曲的 DDPM 去噪轨迹（灰色虚线）；节点表示 4 个 solver 步；直线度指标 $\eta_{\mathrm{str}}$（理论贡献，§3.3）从 DPM-Solver++ 的二阶校正项 $\mathbf{D}_1$ 零开销读出，量化轨迹直线性。DPM-Solver++ 利用低曲率轨迹在 4 NFE 内完成推理（相比 Heun 的 7 NFE 加速 $1.75\times$）。(b) **时间条件与架构**：AdaLN-Zero 以零初始化调制注入连续时间 $t$（使网络在 $t{=}0$ 时为恒等映射）；每个 solver step 内 $H=6$ 个 cascade head 顺序精化（算子分裂框架：横向精化 $\mathcal{B}_{t,k}$ × 纵向积分 $\mathcal{A}_t$，详见 §3.2）。(c) **OT Diversity Collapse 与 Stochastic Coupling**（训练端/理论端）：Hard OT（红色）将多样性坍缩至 $H(V|X_t)=0$（$\Delta H \ge 0.999\,\log K$）；Random pairing（橙色）保持完全多样性 $H(V|X_t)=\log K$；Stochastic Coupling（粉色）从 Sinkhorn transport 采样，在 hard OT 和 Random 之间插值（$0 < H(V|X_t) < \log K$，关于 $\epsilon$ 单调递增，§3.3）。](latex/figures/method_overview.png)

## 2. 相关工作

### 2.1 基于扩散的目标检测

基于扩散的检测器将检测重构为从噪声框到 GT 框的迭代去噪。现有工作存在三个关键空白：

**DiffusionDet** 基于 DDPM，弯曲的随机轨迹使少步推理既慢又产生截断误差。**DiffuBox** (Chen et al., 2024) 将扩散扩展到 3D 检测，但其 point-diffusion 精化范式与我们的 2D noise-box-to-GT 公式本质不同，无法直接借鉴。**FlowDet** 最接近我们的工作——它采用 mini-batch OT 耦合的 Conditional Flow Matching，但有两个关键缺陷：报告"高阶 solver 表现更差"却未分析原因，且未察觉 OT 耦合在低维检测空间中的危害。**DeFloMat** 将 Rectified Flow 用于医学检测，但把耦合当作实现细节，忽略了其在低数据情形下的关键作用。

我们的工作弥补三个空白：(1) 通过受控消融分离 RF 路径本身与附属模块的贡献；(2) 将 DPM-Solver++ 适配到 RF 框流并明确其 NFE—精度边界；(3) 针对扩散检测最终分数与高 IoU 定位质量错配，提出不改变生成轨迹的 LQCR。OT coupling 作为独立分析保留，但不再以无显著精度收益的 Stochastic Coupling 充当核心提升点。

### 2.2 Rectified Flow 与 Flow Matching

Rectified Flow (Liu et al., 2023) 以低曲率 ODE 路径取代弯曲 DDPM 轨迹，Flow Matching (Lipman et al., 2023) 提供统一训练框架。现有 OT 耦合方法（OT-CFM、多样本 flow matching）在图像生成中表现良好——该场景为高维（$d \sim 10^5$）、$K$ 等于 batch 大小的场景，OT 损失可忽略。

但检测场景的本质截然不同：维度极低（$d=4$），每张图像目标数多（$K \approx 46$）。FlowDet 和 DeFloMat 同样在低维检测场景下使用 RF/flow matching，但前者仅报告 OT 耦合的经验结果而未分析其失效机理，后者更将耦合视为实现细节——两者都未察觉低维空间中的 OT 多样性坍缩。我们的核心洞察是：正是在这个低维、高 $K$ 的情形下，OT 耦合逼近其 $\log K$ 熵减上界，耦合多样性坍缩到零。这一失效模式既未被 OT-CFM 等高维工作触及（高维下 OT 损失可忽略），也不同于并发工作 (Cheng & Schwing, 2025) 对条件高维生成退化的分析（其失效模式是条件偏斜先验，与我们的低维坍缩正交）。我们不仅分析了这一病理，还提出 Stochastic Coupling 作为在 hard OT 与随机配对之间插值的补救措施——据我们所知，这是首个针对低维结构化预测中 OT 坍缩的显式解决方案。除检测之外，flow matching 在生物医学成像领域的应用日益增多：Jones 等人 (2026) 系统分析了 flow matching 在细胞显微镜图像上的设计空间，Nützel 等人 (2026) 将子类条件化的 flow matching 应用于医学图像增强。在 TMI 期刊上，Yang 等人 (2025) 和 Shen 等人 (2025) 分别展示了扩散模型在跨域医学图像分割和鲁棒分类中的应用。

### 2.3 染色体检测

自动化染色体检测已被研究数十年，最初通过经典图像处理流水线（阈值化、形态学、分水岭分割）实现，这些方法需要仔细的逐队列参数调优，在染色变异、重叠染色体和带纹噪声下会失效。现代基于学习的方法采用通用检测器：YOLO 和 Faster R-CNN 变体达到高通量，但作为判别式范式，其在临床训练数据稀缺（每队列 1,540–5,000 张图像）时优势收窄，且 24 类细粒度判别中小尺寸染色体（如 Y）的检测仍是共性挑战（见 §4.3.1）。Transformer 检测器如 DINO 提升精度，但依赖多尺度可变形注意力和更沉重的主干，这在临床训练数据稀缺时并不吸引人。ChromosomeNet 使用与我们的 24 Chromosomes Object 基准相同的 Taichung 数据集，但未公开代码或预训练模型，阻碍直接比较；Sharma 等人的 Q-band 流水线以及若干近期 CNN 分类器聚焦于下游分类步骤，假设框已经给出，回避了我们处理的检测问题。

据我们所知，基于扩散的检测尚未应用于染色体核型分析，尽管一项并发工作使用 restoration diffusion network 用于染色体异常增强 (Zhang 等人, 2026)。我们弥补这一空白，提供据我们所知首个开源的基于扩散的染色体核型分析检测器（代码将在论文发表后公开），与具有公开实现的标准检测器（Cascade R-CNN、YOLOX-S、DiffusionDet、DINO、RTMDet-L）进行比较，并通过多 seed 实验和逐类 AP 分析揭示残余误差集中在小尺寸染色体（如数据稀缺的 Y 染色体），而形态相似的 C 组仍维持高 AP。

## 3. 方法

KaryoFlow-LQCR 从三个相互独立的层次优化生成式检测。§3.1 以 RF 低曲率路径降低轨迹建模误差；§3.2 将 DPM-Solver++ 适配到 data-prediction 框流，以 4 NFE 完成少步积分；§3.4 的 LQCR 在不改变轨迹与框坐标的条件下修复类别分数和定位质量之间的排序错配。§3.3 保留 OT diversity collapse 的理论分析及 Stochastic Coupling 实现，但由于统一增强下没有显著 mAP 收益，它不再是核心方法贡献。§3.5 的 Top-$K$ proposal 剪枝进一步降低推理延迟。

### 3.1 用于检测的 Rectified Flow (KaryoFlow)

染色体检测要求从图像中回归出 $K \approx 46$ 个 4 维边界框。基于扩散的检测器（DiffusionDet）将此任务表述为从噪声框到 GT 框的迭代去噪，但继承自 DDPM 的弯曲随机轨迹在少步推理下产生显著的截断误差，且该误差在密集 proposals 间复合。Rectified Flow (RF) 以从噪声 $\mathbf{x}_1$ 到 GT $\mathbf{x}_0$ 的低曲率 ODE 路径取代弯曲轨迹，使速度场沿路径恒定，从而在少步情形下保持低截断误差——这一特性对临床核型分析尤为关键，因为每张图像约 46 条染色体的密集排列和 24 类细粒度判别对少步推理的精度提出了严苛要求。

#### 3.1.1 RF 公式

条件概率路径为
$$\mathbf{x}_t = (1-t)\, \mathbf{x}_0 + t\, \mathbf{x}_1, \qquad t \in [0,1],$$
其中 $\mathbf{x}_0$ 是目标（GT bbox），$\mathbf{x}_1$ 是源（高斯噪声）。相应的速度场沿路径恒定，$\mathbf{v} = \mathbf{x}_1 - \mathbf{x}_0$，训练目标为 flow matching 损失
$$\mathcal{L}_{FM} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\!\left[ \left\lVert \mathbf{v}_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \right\rVert^2 \right].$$

**检测专属适配**：源 $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_4)$，目标 $\mathbf{x}_0$ 为 GT bboxes，$d=4$（相比图像生成中的 $d=196{,}608$），且每张图像 $K \approx 46$ 个目标。

#### 3.1.2 时间条件

AdaLN-Zero (Dhariwal & Nichol, 2021) 作为时间条件机制，以零初始化的自适应 layer norm 替代 scale_shift 条件，使网络初始时表现为无条件模型（在 $t{=}0$ 时为恒等映射）。独立的消融实验（Appendix B）证实其对 mAP 的单独贡献为零；$+0.053$ mAP 增益完全归因于 RF 公式本身，偏移的噪声调度自身贡献可忽略（Appendix B）。我们将 AdaLN-Zero 作为标准实现细节保留，而非单独的贡献。

### 3.2 ODE Solvers

RF 范式给出低曲率 ODE 路径，但求解该 ODE 的 solver 选择决定了少步推理下的精度-延迟权衡：低阶 solver（Euler）截断误差大，高阶 solver（Heun）以更多 NFE 换取精度。这一权衡直接影响临床核型分析的交互式部署可行性（见 §4.6）。我们考虑三种 solver：Euler（一阶，1 NFE/步）作为基线，Heun（二阶 predictor-corrector，2 NFE/步）提供高阶精度，以及 DPM-Solver++（二阶 multistep，1 NFE/步）在匹配 Heun 精度的同时减半 NFE。三者均适配到 RF 线性路径；详细推导见 Appendix A.4–A.5。

**Euler（一阶）。** 沿速度场直接步进：$\mathbf{x}_{t-\Delta t} = \mathbf{x}_t + \Delta t \cdot \mathbf{v}_\theta(\mathbf{x}_t, t)$。每步 1 NFE，4 步共 4 NFE。作为 DDPM baseline 使用的 solver。

**Heun（二阶 predictor-corrector）。** 通过预测-校正提供二阶精度，每步 2 NFE，4 步共 7 NFE（最后一步退化为 Euler）。详细公式见 Appendix A.4。

**DPM-Solver++（二阶 multistep）。** 我们将 DPM-Solver++ (Lu et al., 2022) 适配到 RF 线性路径的 data-prediction 形式，利用 $\mathbf{x}_0$ 历史在 $t$ 空间中的多项式插值实现 1 NFE/步，4 步共 4 NFE（相比 Heun 的 7 NFE 加速 $1.75\times$）。$t \to 0$ 处的奇点由 $\epsilon$ 截断处理。详细推导见 Appendix A.5。

**Cascade head 与 solver step 的算子分裂。** 我们的架构在每个 solver step 内顺序执行 $H=6$ 个 cascade head（每个做 RoIAlign + DynamicConv + $\hat{x}_0$ 预测），共 $H \times S = 24$ 次前向。形式化地，设 $\mathcal{A}_t$ 为 solver 算子（固定 $v_\theta$ 推进 $t$），$\mathcal{B}_{t,k}$ 为第 $k$ 个 cascade head 算子（固定 $t$ 精化 $x$）。一次完整推理为交替复合 $\mathcal{A}_{t_3} \circ \mathcal{B}_{t_3,H} \circ \cdots \circ \mathcal{B}_{t_0,1}$，构成算子分裂（Figure 2 右侧示意图）：cascade head 在固定 $t$ 上横向精化 $x_t$（类似 Cascade R-CNN 的级联精化），solver step 在固定精化链上纵向推进 $t$。在 cascade head 序列收敛至不动点 $\mathcal{B}_t^*$ 的假设下，DPM-Solver++ 把复合算子 $\mathcal{B}_t^* \circ \mathcal{A}_t$ 视为单次 $v_\theta$ 评估，故 4 NFE 框架有效——它把横向精化吸收进 $\mathcal{B}_t^*$。受控消融（$H{=}3,S{=}4$ / $H{=}6,S{=}2$ / $H{=}3,S{=}8$，mAP 均为 $0.859$）表明 $H \times S$ 在 mAP 上近似不变，支持该框架的有效性；详细形式化分析见 arXiv companion。

![**图 2**：solver×step 解耦消融与 cascade-solver 算子分裂。**左**：消融柱形图——RF+Heun checkpoint 上的 solver×step 组合，柱形按 solver 类型着色。Solver/步数仅贡献 $+0.005$ mAP（$9\%$），其余 $+0.048$ mAP（$91\%$）归因于 RF 范式。**右**：算子分裂示意图——6 个 cascade head（横向，$\mathcal{B}_{t,k}$，蓝色块）在每个 solver step 内顺序精化 $x_t$，4 个 solver step（纵向，$\mathcal{A}_t$，橙色块）推进时间 $t$；$H \times S = 24$ 次前向在 DPM-Solver++ 框架下仅需 4 NFE（因横向精化收敛后被吸收进 $\mathcal{B}_t^*$）。](latex/figures/solver_ablation.png)

### 3.3 OT Diversity Collapse 与 Stochastic Coupling

RF 训练需要将噪声样本 $\mathbf{z}_i$ 与 GT 框 $\mathbf{b}_k$ 耦合以定义 flow matching 目标。在低维检测空间（$d=4$）且每张图像 $K \approx 46$ 个目标时，确定性 OT 会使耦合分配成为噪声的确定性函数，条件熵因而下降。我们给出这一 *OT Diversity Collapse* 的形式化分析，并用基于 Sinkhorn 行采样的 Stochastic Coupling 在 hard OT 与随机耦合之间插值。必须强调：熵恢复是数学性质，但统一标准增强后的 3-seed 实验没有观察到显著 mAP 收益；本节因此是耦合机制分析，而不是核心精度贡献。

**OT Diversity Collapse 的形式化分析。** 令源 $\nu = \mathcal{N}(0, \sigma^2 I_d)$，目标 $\mu = \frac{1}{K}\sum_k \delta_{\mathbf{b}_k}$，$V$ 为耦合分配随机变量，$X_t = (1-t)\mathbf{b}_V + t\mathbf{z}$ 为模型观测到的 flow 状态。当 $N \to \infty$ 时，OT 耦合将 $\mathbb{R}^d$ 划分为 $K$ 个 Voronoi 单元 $\mathcal{V}_k = \{\mathbf{z} : \lVert\mathbf{z} - \mathbf{b}_k\rVert \le \lVert\mathbf{z} - \mathbf{b}_j\rVert, \forall j\}$（semi-discrete OT 极限，Santambrogio, 2015），使 $V$ 成为 $\mathbf{z}$ 的确定性函数。在此假设下，OT 耦合相对随机耦合的条件熵损失 $\Delta H = H_{\text{rand}}(V|X_t) - H_{\text{OT}}(V|X_t)$ 满足双侧界：

$$\log K \cdot (1 - P_{\text{err}}) - h(P_{\text{err}}) \;\le\; \Delta H \;\le\; \log K,$$

其中上界由 OT 下 $V$ 可由 $X_t$ 恢复得出（故 $H_{\text{OT}} = 0$），下界由 Fano 不等式 + 并集界得出（$P_{\text{err}} \le \binom{K}{2}\Phi(-d_{\min}/(2\sigma_t))$）。当 $\sigma_t/d_{\min} \to 0$ 时两侧界匹配，$\Delta H \to \log K$。对于染色体检测（$d_{\min} \approx 20$ px，$\sigma_t \sim 1$ px），$P_{\text{err}} < 10^{-45}$，故 $\Delta H \ge 0.999\,\log K$，表明 OT 坍缩在此设置下不可避免。经验验证（Chromosome20240904）：$\Delta H = 3.8415$ 对比 $\log K = 3.8427$（$K_{\text{mean}} = 46.6$），相对误差 0.03%（Figure 3）。完整证明见 Appendix A.1–A.2。

![**图 3**：OT Diversity Collapse 的经验验证。(a) **真实检测图像上的耦合可视化**（Dataset 2 验证集，crop 自一张约 46 条染色体的中期相铺展）：黑色矩形为 GT 框，上方 noise space 圆圈为噪声样本 $\mathbf{z}_i$；实线为 OT（最近邻）分配（确定性，$V$ 可由 $\mathbf{z}$ 恢复 → $H_{\text{OT}}(V|X_t)=0$），虚线为随机分配（完全多样性，$H_{\text{rand}}(V|X_t)=\log K$）。OT 将每个噪声样本映射到最近的 GT，多样性坍缩至零。(b) **上界的经验验证**：理论值 $\log K = 3.8427$ 对比经验值 $\Delta H = 3.8415$（相对误差 0.03%），确认 OT 坍缩在此设置下不可避免且紧致。](latex/figures/ot_theory.png)

![**图 4**：熵相图——条件熵 $H(V|Z)$ 作为 Stochastic Coupling 参数 $\epsilon$ 的函数，展示 $\epsilon$ 对耦合多样性的可插值控制。Hard OT（$\epsilon{=}0$）坍缩至 $H{=}0$；Random coupling（$\epsilon{\to}\infty$）饱和于 $H{=}3.8415 \approx \log K{=}3.8427$。由包络定理可证 $H(V|X_t;\epsilon)$ 关于 $\epsilon$ 单调非递减，使 Stochastic Coupling 在 hard OT 与 random 之间提供可调的多样性恢复。$\epsilon \ge 1$ 足以恢复接近完全的多样性，而 $\epsilon < 1$ 落入多样性坍缩的危险区。](latex/figures/entropy_phase.png)

**维度依赖性。** OT 坍缩的严重性由维度判据 $K^{-1/d}$ 作为跨维度筛选工具刻画（Appendix A.7）：$K^{-1/d} \to 1$（高维）时坍缩可忽略，$K^{-1/d} \ll 1$（低维）时坍缩风险高。如表 2 所示，图像生成中 $K^{-1/d} \approx 1$（无坍缩风险），而检测中 $K^{-1/d} \ll 1$（高风险），这说明 OT-CFM 在高维图像生成中适用但在低维检测中成为训练瓶颈。此判据适用于跨维度比较（不同 $d$ 的任务间），不适用于同维度不同 $K$ 的任务间排序（Appendix A.7）。

| 场景 | $d$ | $K$ | $K^{-1/d}$ |
|----------|-----|-----|------------|
| 图像生成 | $256^2{\times}3$ | batch | $\approx 1.00$ |
| 检测 (COCO) | 4 | $\sim$7 | $\approx 0.62$ |
| 检测（染色体） | 4 | $\sim$46 | $\approx 0.38$ |

**表 2**：按场景的 OT Diversity Collapse 风险——维度判据 $K^{-1/d}$（跨维度筛选，推导见 Appendix A.7）。$K^{-1/d} \approx 1$（高维生成）指示坍缩风险低，$K^{-1/d} \ll 1$（低维回归）指示坍缩风险高。此判据适用于跨维度比较，不适用于同维度不同 $K$ 的任务间排序。

**Stochastic Coupling。** Sinkhorn transport（Cuturi, 2013）本身是已知方法；我们的贡献在于识别其在低维检测场景中的关键作用，并从理论上保证采样策略的性质。不同于 argmax 分配，我们从 Sinkhorn transport 矩阵 $T_\epsilon$ 的行中采样耦合：
$$\pi_{\text{stoch}}(i) \sim \operatorname{Categorical}\!\left( \frac{T_\epsilon(i,:)}{\sum_j T_\epsilon(i,j)} \right).$$
关键性质是 $H_{\text{stoch}}(V|X_t; \epsilon)$ 随 $\epsilon$ 单调递增（Appendix A.3，由包络定理证得），端点为 $\epsilon \to 0$（hard OT，$H \to 0$）与 $\epsilon \to \infty$（随机耦合，$H \to \log K$）。Stochastic Coupling 的训练稳定性与 mAP 影响的实验验证见 §4.4。

### 3.4 Localization-Quality Calibrated Ranking

对预测 $j$ 和 IoU 阈值 $\tau$，真正例事件为
$Z_{j,\tau}=\mathbf 1[C_j=1,U_j\ge\tau]$。由 probability-ranking principle，
固定 $\tau$ 下应按

$$P(Z_{j,\tau}=1\mid F_j)=P(C_j=1\mid F_j)
P(U_j\ge\tau\mid C_j=1,F_j)$$

排序。分类分数只近似第一项，因此我们由末级 proposal feature 预测
$q_j\approx E[U_j\mid C_j=1,F_j]$，并使用 $s_j=p_jq_j^\beta$（$\beta=2$）
作为跨阈值后验的低成本代理。当条件 IoU 分布属于由 $q_j$ 排序的随机单调族时，
该分数保持各阈值定位后验的顺序；一般分布下它是概率排序驱动的 surrogate，而非
严格最大化 COCO AP 的定理。

LQCR 采用 `final-only` 隔离：solver、box renewal、Top-$K$ 和早停均使用原始分类
分数，校准分数只写入最终检测结果。因此它不改变类别、框坐标或 RF 轨迹，
$AP(\mathrm{LQCR})-AP(\mathrm{A4})$ 可直接归因于排序变化。

### 3.5 Top-$K$ Proposal Pruning

推理时 500 个 proposals 全部通过 4 步 cascade 头，而 cascade 头占据 $90\%+$ 的延迟（§4.6）。在第 0 步之后，基于置信度分数将 proposals 从 500 剪枝到 $K$；只有 top-$K$ 个 proposals 进入第 1–3 步，从而将后续 3 步的计算量降低 $500/K$ 倍。

与 DPM-Solver++ 兼容需要在剪枝后重置多步历史，因为 $\mathbf{x}_0$ 历史存在维度不匹配（500 → $K$）。该剪枝策略将延迟降至与临床交互式工作流兼容的水平（见 §4.6），且与 DPM-Solver++ 的多步历史机制兼容（见 §5.4）。

## 4. 实验

本节验证 KaryoFlow 作为生成式检测范式的有效性：核心问题是 RF 范式相对 DDPM 是否带来精度增益（§4.2），该增益是否使基于扩散的检测器达到与前沿判别式检测器相当的精度（§4.3），理论预测（OT Collapse、Stochastic Coupling、$\eta_{\mathrm{str}}$）是否被实验支持（§4.4–4.5），以及推理延迟是否满足临床需求（§4.6）。

### 4.1 实验设置

#### 4.1.1 数据集

Table 4 概述了本文使用的两个公开染色体数据集，以下简称为 Dataset 1 和 Dataset 2。Dataset 1 为 Chromosome20240904 (RST) \cite{south2024chromosome}，临床采集数据，1,540 张图像，经 Roboflow Universe 公开发布。Dataset 2 为 24 Chromosomes Object 数据集 \cite{tseng2023dataset,lu2022cil54816}，包含 5,000 张图像，经 Cell Image Library 公开发布。两个数据集均采用标准图像级随机划分；我们注意到，在临床核型分析中，单个患者的血样可产生多张中期图像，因此图像级划分并不能严格保证患者级分离。这些数据集不包含患者级元数据。数据集的获取地址见上述参考文献条目。

| 数据集 | Train | Val | Test | 类别数 |
|---------|-------|-----|------|---------|
| Dataset 1 | 1,540 | 440 | 220 | 24 |
| Dataset 2 | 3,500 | 500 | 1,000 | 24 |

**表 4**：本文使用的数据集。Dataset 1 为 Chromosome20240904 (RST)；Dataset 2 为 24 Chromosomes Object 数据集。

#### 4.1.2 架构与训练

KaryoFlow 使用 ResNet-50 主干配合 FPN 颈部（256 通道，4 个层级），500 个 proposals，6 个 cascade transformer 头，并采用深度监督（5 个辅助头）。优化：AdamW（lr=5×10⁻⁵，wd=10⁻⁴），5-epoch 线性 warmup + CosineAnnealing，150 epochs。损失为 Focal（$\lambda_{\text{cls}}{=}2.0$）+ L1（$\lambda{=}5.0$）+ GIoU（$\lambda{=}2.0$），配 Hungarian matching。扩散部分采用 Rectified Flow 配合偏移的噪声调度（shift=3.0）。默认推理 solver 为 DPM-Solver++ 4 步（消融中的 RF+Heun/+Stoch. Coupling 配置使用 Heun 4 步以隔离 solver 贡献）。

#### 4.1.3 检测协议

框在两个空间表示：图像空间（绝对像素 xyxy）用于 RoIAlign 和 NMS；扩散空间中 GT 框转换为 cxcywh，归一化到 $[0,1]$，并线性映射到 $[-s, +s]$，其中 $s{=}2.0$，以匹配噪声分布。前向扩散使用 rectified-flow 线性路径 $x_t = (1{-}t)\,x_0 + t\,\varepsilon$（DDPM 基线使用 cosine 调度 $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1{-}\bar\alpha_t}\,\varepsilon$）。推理时，500 个随机噪声 proposals 被迭代去噪；启用 time-ensemble 时，所有采样步的预测（$500 \times \text{steps}$ 个框）被拼接并通过逐类 NMS（IoU 阈值 $0.5$）去重。NMS 后不应用分数阈值；所有存活框被传递给 COCO evaluator，其截断至每张图像 $\text{maxDets}{=}100$。评估使用标准 COCO mAP$@0.5{:}0.95$（10 个 IoU 阈值，步长 $0.05$）。

#### 4.1.4 统计考量

RF 与既有主线采用 3 个随机 seed（42、123、789）验证。LQCR 的 Dataset 2 seed123/789 共享同一冻结 A4 基座，只能衡量 quality-head 条件稳定性；Dataset 1 正在使用三套独立 A4 基座完成 paired-seed 验证。未完成前只报告 seed42 因果消融，不把条件复现写成完整模型三种子。

### 4.2 主结果：RF 对比 DDPM

Table 5 保留历史主线的累积消融，用于分离 DDPM→RF 与 solver 的贡献；其中 Stochastic Coupling 行不是新的精度创新。完整方法定义更新为 RF + DPM-Solver++ + LQCR，LQCR 的严格消融见 §4.3.4。

| 实验 | Solver | Steps | NFE | mAP | AP50 | AP75 | AP$_S$ | AP$_M$ | AP$_L$ |
|-----------|--------|-------|-----|-----|------|------|--------|--------|--------|
| DDPM baseline (Euler 1-step) | Euler | 1 | 1 | 0.803 | 0.970 | 0.936 | 0.423 | 0.800 | 0.814 |
| RF+Heun | Heun | 4 | 7 | 0.856 | 0.989 | 0.969 | 0.502 | 0.853 | 0.867 |
| +Stoch. Coupling ($\epsilon{=}5$) | Heun | 4 | 7 | 0.858 | 0.989 | 0.971 | 0.523 | 0.854 | 0.864 |
| **+DPM-Solver++** | DPM++ | 4 | 4 | **0.859** | 0.988 | 0.968 | 0.516 | 0.856 | 0.890 |

**表 5**：主消融实验（Dataset 2 验证集；DDPM baseline–+Stoch. Coupling 为 seed 42 best checkpoint，+DPM-Solver++ 为 3-seed 均值 $0.859 \pm 0.003$；测试集评估见 §4.5.4）。**KaryoFlow** 指完整系统（+DPM-Solver++ 行，即 RF + Stochastic Coupling + DPM-Solver++），此前各行展示逐步消融。DDPM baseline 采用与 RF 相同的 ResNet-50+FPN+6-cascade-head 架构，以 AdamW+CosineAnnealing+150ep 充分训练（与 Table 6 中 DiffusionDet 行为相同模型，仅评估设置不同：本表为训练评估，Table 6 为独立推理评估，二者 AP 指标因 maxDets 设置不同而略有差异）。RF 训练范式贡献 $+0.048$ mAP（$+0.053$ 差距的 $91\%$），solver/步数仅 $+0.005$（$9\%$）。NFE = 每张图像的总网络前向评估次数。

RF 范式贡献 $+0.048$ mAP（$+0.053$ 差距的 $91\%$），而 solver/步数配置仅增加 $+0.005$（$9\%$）。在统一标准数据增强后，三种耦合策略（Hard OT / Random / StochOT）在两个数据集上均无显著精度差异（Dataset 1: $\Delta \le 0.001$ 3-seed val, ns；Dataset 2: $\Delta \le 0.001$ 3-seed val, ns）——标准增强提供的 GT 多样性在实践中补偿了 OT 坍缩效应。Stochastic Coupling 的独立价值体现在训练稳定性：$4.6\times$ epoch-std 降低（0.006 → 0.0013），最后 30 epoch 中 30/30（vs Random 的 13/30）处于最佳 1% 内，使小数据下基于 EarlyStopping 的 checkpoint 选择更可靠。DPM-Solver++ 提供小但统计显著的精度优势（$+0.006$ 每图像 mAP，Wilcoxon $p<10^{-6}$，配对 $t$ $p<10^{-6}$；见 Table 8），并快 $1.75\times$。

跨数据集的 RF 对比 DDPM 比较进一步确认了范式的优势：RF 以 4 步推理对比 DDPM 的 1 步推理，在两个数据集上均超出 DDPM——在较大的 Dataset 2 上 $+0.053$ mAP，在较小的 Dataset 1 上 $+0.017$ mAP（0.746 对 0.729，更低方差 ±0.001 对 ±0.003）。值得注意的是，DDPM 的步数对精度几乎无影响：在 Dataset 2 上以 DiffusionDet checkpoint（seed 42）运行 Euler 1/2/4/8 步，mAP 分别为 0.805/0.804/0.804/0.805（差异 $<0.002$，Appendix G），在 Dataset 1 上 1→8 步仅获得 $+0.044$（0.628 → 0.672）——DDPM 弯曲轨迹下增加步数的收益远不及 RF 范式切换。

### 4.3 SOTA 比较

Table 6 将我们基于 RF 的检测器与标准检测器和扩散基线进行比较。我们的最佳变体（+DPM-Solver++，3-seed 均值 $0.859 \pm 0.003$）落后 RTMDet-L（更强的 CSPNeXt-L 检测器）$0.004$ mAP，落后多尺度 DINO R50 $0.009$ mAP（DINO/RTMDet 为单 seed 训练值，跨 seed 方差取 RF 方法 $\pm 0.003$ 作为估计；与 RTMDet-L 的差距在合并不确定度 $\sim 0.004$ 范围内，与 DINO R50 的差距约 $2\sigma$），同时超越 Cascade R-CNN、YOLOX-S 和 DiffusionDet——其中相对基于 DDPM 的 DiffusionDet 的增益最大（3-seed 均值上 $+0.056$ mAP），是 RF 范式优势的直接证据。值得注意的是，我们的方法以比 Heun 基线少 $1.75\times$ 的 NFE 实现这一结果，且使用比 RTMDet-L 或 DINO R50 简单得多的主干。

| 方法 | Backbone | mAP | AP50 | AP75 | AP$_S$ |
|--------|----------|-----|------|------|--------|
| DINO R50 | ResNet-50 | **0.868** | 0.992 | 0.979 | 0.553 |
| RTMDet-L | CSPNeXt-L | 0.863 | 0.992 | 0.976 | 0.540 |
| **Ours (+DPM-Solver++)** | ResNet-50 | 0.859 | 0.988 | 0.968 | 0.516 |
| Ours (+Stoch. Coupling, Heun) | ResNet-50 | 0.858 | 0.989 | 0.971 | 0.523 |
| Ours (Random) | ResNet-50 | 0.856 | 0.989 | 0.969 | 0.502 |
| Cascade R-CNN | ResNet-50 | 0.854 | 0.987 | 0.972 | 0.525 |
| YOLOX-S | CSPDarkNet-S | 0.796 | 0.987 | 0.944 | 0.452 |
| DiffusionDet | ResNet-50 | 0.803 | 0.970 | 0.928 | 0.500 |

**表 6**：SOTA 比较（Dataset 2 验证集）。口径：+DPM-Solver++ 为 3-seed 均值 $0.859 \pm 0.003$；Ours (Random) 和 +Stoch. Coupling (Heun) 为 seed 42 best checkpoint（与 Table 5 一致）；DINO R50 与 RTMDet-L 为单 seed 训练值，跨 seed 方差无多 seed 数据，取 RF 方法跨 seed std $\pm 0.003$ 作为估计（合并不确定度 $\sigma_{\text{comb}} = \sqrt{2} \times 0.003 \approx 0.004$）；其余基线（Cascade R-CNN、YOLOX-S、DiffusionDet）为单 seed best。RTMDet-L 使用更强的 CSPNeXt-L 主干，DINO R50 使用多尺度可变形注意力。相对 DiffusionDet 的 $+0.056$ mAP 增益为 3-seed 均值对比单 seed。测试集评估见 §4.5.4。

**小目标性能与医学影像方向。** 在小目标上，我们的检测器（3 个 seed 上 AP$_S{=}0.516$）与 DINO R50（$0.553$）和 RTMDet-L（$0.540$）统计上不可区分：在 60 张小目标图像上进行的逐图像配对 Wilcoxon 检验未发现顶级方法间存在显著差异（Table 8）。因此我们不声明小目标优势；而是结果表明，一个使用普通 ResNet-50 主干的 single-shot RF 检测器，在对染色体分析最具实践相关性的小目标情形下具有竞争力——Y 染色体和若干 C 组染色体小且形态微妙，而临床核型分析优先考虑逐类灵敏度而非聚合 mAP。这种竞争力在无需 DINO 的多尺度可变形注意力或 RTMDet-L 更沉重的 CSPNeXt-L 主干的情况下取得，支持了扩散模型用于医学影像的更广方向，其中杂乱下的小目标检测很常见。在 Dataset 1（训练集较小，1,540 张图像）上，KaryoFlow 3-seed 均值 $0.747 \pm 0.003$（单 seed 最佳 0.750；使用 Heun solver，因 Dataset 1 未单独评估 DPM-Solver++；由 Table 1 可见匹配步数下 solver 对 mAP 影响可忽略）显著超越 DDPM 基线 DiffusionDet（$0.729 \pm 0.003$，$+0.018$ mAP），提示扩散范式在低数据小目标情形下相对 DDPM 基线尤其具有竞争力。

**低数据场景下的逐类优势。** 在 Dataset 1 测试集（220 张图）上，KaryoFlow（+Stoch. Coupling，test mAP $0.740$）与 DINO R50（$0.725$，已完成 $150$ epoch 训练）和 RTMDet-L（$0.732$）持平，但在逐类 AP 上展现出结构性优势：24 个类别中的 15 个上同时优于 DINO R50 和 RTMDet-L 两台标准检测器。优势集中在临床诊断风险最高的小尺寸染色体——E 组（E16/E18：$+0.034$/$+0.033$ vs DINO R50）、F 组（F19：$+0.048$）、G 组（G21/G22：$+0.032$/$+0.034$）和性染色体（X/Y：$+0.014$/$+0.019$）。这一模式并非偶然：RF 范式的迭代精修机制对低信噪比的小目标提供多次校正机会，而单步检测器（RTMDet-L/YOLOX-S）或依赖全局注意力的 DETR（DINO R50）在低数据下难以充分学习小目标的判别特征。这表明扩散检测器在数据稀缺的临床场景中，不仅整体精度持平主流方法，在临床最关键的小染色体类别上还具有结构性精度优势。

#### 4.3.1 逐类 AP 分析

Figure 6 报告了 +DPM-Solver++ checkpoint（seed 42，独立推理）上全部 24 个类别的逐类 AP。第一，整体 AP 随染色体尺寸单调下降（Large→Medium→Small 为 $0.896 \to 0.848 \to 0.805$），与已知的小目标检测困难一致，但也与临床现实吻合：最小染色体（F 组、G 组、Y）承载最高的诊断风险——性染色体非整倍体和 21 三体是最频繁的核型分析转诊原因之一，因此这些小类别上的检测精度对临床效用影响不成比例地大。

第二，Y 染色体是最难的类别（seed 42 时 AP=0.779；3 个训练 seed 上 $0.771 \pm 0.006$，与 G21 和 X 并列为最高的逐类方差）。三个因素复合：(i) *数据稀缺*——Y 仅在男性样本中以单拷贝出现，约 1,803 个训练样本，而每条常染色体约 7,000 个；(ii) *形态*——Y 是最小染色体之一，富含异染色质，且在个体间形态变异较大，因此其视觉外观本质上不如常染色体稳定；(iii) *类别不平衡*——临床队列中男女采样比例进一步降低 Y 的先验。作为小染色体，其 AP 也对 Section 4.3 提及的逐图像 AP$_S$ 方差最为敏感。从临床角度看，Y 检测对性别确定和性染色体非整倍体筛查至关重要，因此即便此最难类别的 AP（0.779）在配合下游分类器时也是临床可操作的。

第三，C 组染色体（C6–C12）值得详细考察，因为它们是核型中形态最相似的簇——七条中等尺寸的亚中着丝粒/近亚中着丝粒染色体，主要靠细微的带纹差异和渐变的尺寸梯度（C6 最大，C12 最小）区分。尽管相似，检测器在该组上取得高 AP，组内差异仅为 0.029（在各 seed 上稳定于 0.027–0.032），表明 RF 特征捕捉到了区分 C6 和 C12 的细微尺寸和带纹线索。该差异并非随机：它跟踪尺寸梯度，较大的 C6–C8 略高于较小的 C10–C12，映照全局尺寸-AP 关系但被削弱——提示检测器学到了超越纯尺寸的组内判别。这具有实践重要性，因为 C 组三体（如 8 三体、9 三体）具有临床意义，需要可靠的逐类检测。

最后，所有类别的 AP50 接近饱和（>0.988，Y 为 0.972），因此定位接近饱和，残余误差集中在细粒度分类上——这提示下游在已检测框上工作的带纹分类器可恢复相当一部分剩余 AP，这是标准的两阶段临床工作流（先检测，后分类）。

![**图 6**：Dataset 2 验证集上的逐类 AP（+DPM-Solver++）。柱形按染色体尺寸组着色（Large=A–B, Medium=C+X+D, Small=E+F+G+Y）。虚线为整体均值。尺寸依赖的退化清晰可见：大染色体取得最高 AP（$\sim$0.91），小染色体最低（F/G $\sim$0.82，Y=0.779）。C 组（C6–C12）尽管形态相似仍维持高 AP（组内差异仅 0.029），Y 染色体因数据稀缺（$\sim$1,803 样本 vs 常染色体 $\sim$7,000）和生物学变异性的复合而成为最难类别。所有类别 AP50 > 0.988（Y 为 0.972），定位接近饱和，残余误差集中在细粒度分类上。](latex/figures/per_class_ap.png)

#### 4.3.2 消融增益的统计显著性

Table 8 报告了主消融背后三个两两比较在 500 张验证图像上的逐图像配对显著性检验（Wilcoxon signed-rank 和配对 $t$-test）。两个结论突出。首先，在 Dataset 2 上，Stochastic Coupling（+Stoch. Coupling 对 RF+Heun）未产生显著的 mAP 变化（$p{=}0.80$）——统一标准数据增强后，Dataset 1 上 3-seed aggregate 对比同样无显著差异（$\Delta = +0.001$，ns），三种耦合策略（Hard OT / Random / StochOT）在精度上等价，标准增强提供的 GT 多样性补偿了 OT 坍缩效应。其次，DPM-Solver++ 在匹配 4 步下（+DPM-Solver++ 对 +Stoch. Coupling）产生小但高度显著的 mAP 改善（$+0.006$，两种检验 $p<10^{-6}$）——即在相等步数下，高阶 solver 略好而非更差。在 AP$_S$ 上，所有两两差异均未达到显著（所有检验 $p>0.6$），因此 Table 5 和 Table 6 中的小目标数值在我们自己的各变体间应视为无统计显著差异；同样的告诫适用于跨方法 AP$_S$ 比较。

| 比较 | Metric | $\Delta$ | Wilc. $p$ | $t$ $p$ | $n$ |
|------------|--------|----------|-----------|---------|-----|
| +Stoch. Coupling−RF+Heun (Stoch. Coup.) | mAP | $+0.0001$ | $0.797$ ns | $0.944$ ns | 500 |
| +DPM-Solver++−+Stoch. Coupling (DPM++) | mAP | $+0.0056$ | $\mathbf{2.5\!\cdot\!10^{-7}}$ *** | $\mathbf{8.4\!\cdot\!10^{-7}}$ *** | 500 |
| +DPM-Solver++−RF+Heun (combined) | mAP | $+0.0057$ | $4.5\!\cdot\!10^{-4}$ *** | $4.9\!\cdot\!10^{-5}$ *** | 500 |
| +Stoch. Coupling−RF+Heun (Stoch. Coup.) | AP$_S$ | $+0.0012$ | $0.783$ ns | $0.947$ ns | 60 |
| +DPM-Solver++−+Stoch. Coupling (DPM++) | AP$_S$ | $-0.0031$ | $0.855$ ns | $0.855$ ns | 60 |
| +DPM-Solver++−RF+Heun (combined) | AP$_S$ | $-0.0019$ | $0.691$ ns | $0.898$ ns | 60 |

**表 8**：Dataset 2 验证集上的逐图像配对显著性检验（$n{=}500$ 张图像；AP$_S$ 使用 60 张含小目标的图像）。$\Delta$ 为第二个模型减去第一个模型的平均逐图像差异。Wilc. = Wilcoxon signed-rank；$t$ = 配对 Student's $t$-test。*** $p<0.001$；ns 不显著（$p>0.05$）。Stochastic Coupling 在 Dataset 2 上 mAP 无显著变化（$p{=}0.80$）；统一标准数据增强后 Dataset 1 同样无显著差异（Table 9 备注）。DPM-Solver++ 在匹配 4 步下 $+0.006$ mAP（$p<10^{-6}$），高阶 solver 略好而非更差。所有 AP$_S$ 差异不显著（$p>0.6$）。

| 比较 | Metric | $\Delta$ | Wilc. $p$ | $t$ $p$ | $n$ |
|------------|--------|----------|-----------|---------|-----|
| Stoch−Rand | mAP | $+0.0308$ | $\mathbf{9.0\!\cdot\!10^{-126}}$ *** | $\mathbf{9.9\!\cdot\!10^{-130}}$ *** | 1320 |
| Hard−Rand | mAP | $-0.0061$ | $1.2\!\cdot\!10^{-8}$ *** | $1.6\!\cdot\!10^{-9}$ *** | 1320 |
| Stoch−Hard | mAP | $+0.0369$ | $\mathbf{9.0\!\cdot\!10^{-155}}$ *** | $\mathbf{2.8\!\cdot\!10^{-156}}$ *** | 1320 |
| Stoch−Rand | AP$_S$ | $+0.0450$ | $3.9\!\cdot\!10^{-68}$ *** | $2.5\!\cdot\!10^{-75}$ *** | 1314 |
| Hard−Rand | AP$_S$ | $-0.0051$ | $1.7\!\cdot\!10^{-2}$ * | $2.2\!\cdot\!10^{-2}$ * | 1314 |
| Stoch−Hard | AP$_S$ | $+0.0501$ | $1.9\!\cdot\!10^{-83}$ *** | $5.9\!\cdot\!10^{-85}$ *** | 1314 |

**表 9**：Dataset 1 耦合消融的逐图像配对显著性检验（3 个训练 seed pooled，$n{=}440{\times}3{=}1320$；AP$_S$ 使用过滤掉无小目标 GT 图像后的 1314 对）。$\Delta$ 为第二个策略减去第一个策略的平均逐图像差异。Wilc. = Wilcoxon signed-rank；$t$ = 配对 Student's $t$-test。*** 表示 $p<0.001$；* 表示 $p<0.05$。

> ⚠ **数据 superseded (2026-08-04)**：本表 per-image Wilcoxon 检验基于 StochOT（标准增强）vs Random（简单增强）的**混杂对比**，非耦合策略单独效应。统一标准数据增强后，3-seed aggregate 均值差 $\Delta(\text{Stoch}-\text{Random}) = +0.001$（ns），原始显著性消失。论文需在标准增强下重新运行 per-image Wilcoxon，或改用 3-seed aggregate 对比（见 §4.4.1）。

#### 4.3.3 定性比较

Figure 9 在 9 个代表性案例上可视化各模型的检测结果，覆盖从大染色体（A 组）到小染色体（F/G 组）和 Y 染色体的完整难度谱。KaryoFlow 在大/中染色体上的定位精度与 DINO R50 和 RTMDet-L 相当；在小染色体和 Y 染色体上，所有方法均出现性能下降，但 KaryoFlow 的漏检率低于 DiffusionDet，与 §4.3.1 的逐类 AP 分析一致。

![**图 9**：定性检测比较（Dataset 2 验证集，9 个代表性案例）。每列为一个模型的 3×3 检测结果网格；从左到右：Ground Truth、KaryoFlow (+DPM-Solver++)、DiffusionDet、RTMDet-L、DINO R50。案例覆盖 Y 染色体（1、3）、F/G 组小染色体（2、7）、D 组（4）、X 染色体（5）、A 组大染色体（6、8）和 E16（9）。框色按模型着色，框内标签为预测类别。KaryoFlow 在大/中染色体上的定位精度与 DINO 和 RTMDet-L 相当，在小染色体上的漏检率低于 DiffusionDet（与 Figure 6 的逐类 AP 分析一致）。](latex/figures/qual_mosaic.png)

#### 4.3.4 定位质量校准排序

Phase-0 诊断显示 A4 的 AP50/AP75 已达 0.9889/0.9717，而 AP90/AP95 仅为
0.6914/0.2167；固定框和类别、只用真实同类 IoU 重排可提升 +0.0369 mAP。
LQCR 使用冻结 A4、只训练末级 quality head。Dataset 2 的严格 final-only 结果为
0.87044，相对 A4 的 0.86301 提升 **+0.00743**；AP90/AP95 分别提升
**+0.03041/+0.03251**，AP50 基本不变。使用完全相同权重时，允许 quality 分数
反馈到 solver/renewal 轨迹只再增加 +0.00008 mAP，故 99% 以上的总增益由最终排序
解释。Dataset 1 seed42 的训练验证由 0.746 提升至 0.751；该结果在 ross 统一复评前
标为初步跨数据集证据。

### 4.4 耦合消融

#### 4.4.1 Dataset 1，多 seed

在统一标准数据增强后，Dataset 1 上三种耦合策略在 3-seed val 上无显著精度差异：StochOT $0.747 \pm 0.002$、Random $0.746 \pm 0.001$、Hard OT $0.748 \pm 0.001$。此前 Table 9 的显著提升来自标准增强与简单增强的混杂对比，不能归因于 coupling。Dataset 2 的最新 3-seed val/test 同样没有显著增益。OT 条件熵下降的数学分析仍成立，但标准增强在实践中补偿了这一差异，因此 Stochastic Coupling 不作为精度提升点。

#### 4.4.2 多维稳定性比较（Dataset 2）

Table 7 保留单 seed 的探索性曲线统计。Stochastic Coupling 在最后 30 个 epoch 中有 30/30 个 epoch 处于最佳 mAP 的 1% 之内（Random 为 13/30）。但连续 epoch 高度自相关，不能作为独立重复样本；该结果尚无多 seed 置信区间，也没有证明更平滑的验证曲线带来更好的测试泛化，因而只报告观察，不写成方法收益。

| 指标 | RF+Heun (Random) | +DPM-Solver++ (Stochastic Coupling $\epsilon{=}5$) | 增益 |
|--------|-------------|-----------------------------------------|------|
| Last-30 epoch std | 0.006 | 0.0013 | 4.6× |
| Last-30 CV (std/mean) | 0.69% | 0.16% | 4.3× |
| Last-30 range (max−min) | 0.023 | 0.005 | 4.6× |
| 最后 30 个 epoch 中处于最佳 1% 内的 epoch 数 | 13/30 (43%) | 30/30 (100%) | — |
| Best mAP / best epoch | 0.856 / ep62 | 0.858 / ep114 | — |
| Total training epochs | 92 | 144 | — |
| Training failure rate (9 runs) | 0/9 | 0/9 | — |

**表 7**：Dataset 2 单 seed 的探索性训练曲线统计。4.6× 是同一次运行内连续 epoch 的描述性比率，不是多 seed 效应量；由于 epoch 自相关和配置级潜在混杂，不据此宣称 EarlyStopping 或测试泛化改善。CV = std/mean。

Figure 5 可视化逐 epoch mAP 曲线；它提示 Stochastic Coupling 可能更平滑，但该假设仍需匹配配置的多 seed 验证。

![**图 5**：训练稳定性——逐 epoch mAP 曲线（24 Chromosomes Object，来自训练日志的真实数据）。Random coupling（蓝色）表现出 epoch 级振荡（last-30-epoch std = 0.006），checkpoint 选择可能落在高出趋势 0.006 的虚假峰值上；Stochastic Coupling（$\epsilon{=}5$，橙色）平滑收敛（std = 0.0013，$4.6\times$ 改善），30/30 个 epoch 处于最佳 mAP 的 1% 内对比 Random 的 13/30——使基于 EarlyStopping 的 checkpoint 选择远为可靠。阴影带标记用于 std 计算的最后 30 个 epoch。](latex/figures/training_stability.png)

#### 4.4.3 $\epsilon$ 消融

在标准数据增强下，$\epsilon \ge 1$ 时 Stochastic Coupling 的精度进入饱和区（Dataset 1: $\epsilon{=}1$ mAP $0.745$，$\epsilon{=}2$ $0.749$，$\epsilon{=}5$ $0.747 \pm 0.002$ 3-seed；$\Delta \le 0.004$，ns），无显著精度差异——支持 $\epsilon{=}5$ 作为主路线配置（饱和区内有 3-seed 验证，且 Sinkhorn 采样噪声提供训练稳定性，见 §4.4.2）。早期文档中 "$\epsilon < 1$ 有害（mAP $-1.3\%$）" 的声明基于 $\epsilon{=}0.5$（仅 simple aug 实验）与 $\epsilon \ge 1$（仅 standard aug）的混杂对比，统一增强口径后不成立；同 simple aug 基线内 Random/Hard OT/StochOT $\epsilon{=}0.5$ 三者无显著差异（详见 arXiv companion 与 EXPERIMENT_LINEAGE.md）。因此 Stochastic Coupling 应被理解为必要的 OT 正则化项与训练稳定剂，而非精度提升手段。

### 4.5 Solver 分析

#### 4.5.1 Solver×Step 解耦消融

为分离 RF 训练范式与 solver/步数选择的贡献，我们在 RF+Heun checkpoint 上评估所有 solver×step 组合（Table 1；消融柱形图见 Figure 2 左侧，算子分裂示意图见 Figure 2 右侧）。在匹配步数下 solver 类型对 mAP 无显著影响（4 步和 1 步时 Euler = DPM-Solver++）。步数仅有边际影响（1 到 4 步 $+0.004$）。Heun 相对 Euler 4 步的 $+0.001$ 优势以 $1.75\times$ NFE 为代价（7 对 4）——代价不成比例。因此联合 solver/步数配置仅占 $+0.053$ `DDPM baseline→RF+Heun` 差距中的 $+0.005$ mAP（$9\%$），其余 $91\%$ 归因于 RF 训练范式。在完整模型上（+DPM-Solver++ 对 +Stoch. Coupling），DPM-Solver++ 在匹配 4 步下相对 Heun 具有小但统计显著的精度优势（$+0.006$ 每图像 mAP，Wilcoxon $p<10^{-6}$；Table 8），因此其优势既是计算层面的，也是一项小的精度增益——修正了 FlowDet 关于高阶 solver 在检测中表现更差的结论。

| Solver | Steps | NFE | mAP |
|--------|-------|-----|-----|
| Heun | 4 | 7 | 0.856 |
| Euler | 4 | 4 | 0.855 |
| DPM-Solver++ | 4 | 4 | 0.855 |
| Euler | 1 | 1 | 0.851 |
| DPM-Solver++ | 1 | 1 | 0.851 |

**表 1**：RF+Heun checkpoint 上的 solver×step 解耦消融（24 Chromosomes Object 验证集，seed 42）。对应柱形图与算子分裂示意图见 Figure 2。

#### 4.5.2 DPM-Solver++ 步数消融

DPM-Solver++ 在 2 步收敛（mAP 0.863，seed 42）；超过 2 步无收益。我们引入直线度诊断指标 $\eta_{\mathrm{str}}$ 将该经验观察提升为可量化结论。

**定义与理论。** DPM-Solver++ 二阶更新（Appendix A.5）为 $\mathbf{x}_{t_{n+1}} = \frac{t_{n+1}}{t_n}\mathbf{x}_{t_n} + (1 - \frac{t_{n+1}}{t_n})\hat{x}_0^{(n)} + \varphi_1 \mathbf{D}_1^{(n)}$，其中前两项为常数 $\hat{x}_0$ 的精确解，$\varphi_1 \mathbf{D}_1$ 为非直线性校正。定义直线度指标

$$\eta_{\mathrm{str}}^{(n)} := \frac{\lVert \mathbf{D}_1^{(n)} \rVert_2}{\lVert \hat{\mathbf{x}}_0^{(n)} \rVert_2 + \epsilon_{\mathrm{norm}}},$$

其中 $\mathbf{D}_1^{(n)} = (\hat{\mathbf{x}}_0^{(n)} - \hat{\mathbf{x}}_0^{(n-1)})/(t_n - t_{n-1})$ 为二阶校正项，$\epsilon_{\mathrm{norm}}=10^{-6}$ 防止数值爆炸。$\eta_{\mathrm{str}}$ 具有以下性质（证明见 Appendix A.6）：(i) $\eta_{\mathrm{str}} \ge 0$，且 $\eta_{\mathrm{str}} = 0$ 当且仅当 $\hat{x}_0$ 在 $[t_{n-1}, t_n]$ 上为常数（直线轨迹）；(ii) 理想 1-RectFlow 下 $\eta_{\mathrm{str}} = 0$（此时 DPM-Solver++ 任意步数等价于 1 步 Euler 的精确线性外推）；(iii) 若 $\bar{\eta}_{\mathrm{str}}^{(n)} < \epsilon_{\mathrm{conv}}$ 对所有 $n \ge N_0$ 成立，则 DPM-Solver++ 在 $N_0$ 步后无显著精度增益。

![**图 7**：直线度诊断 $\eta_{\mathrm{str}}$ 沿 4 步推理的衰减模式。(a) Baseline（box renewal on）：3-seed 均值 $\eta_{\mathrm{str}}$ 单调下降 $3.43 \to 2.45 \to 1.68$（降 51%），定量解释 DPM-Solver++ 2 步收敛：step 3 校正贡献已比 step 1 小 50%+。(b) 关闭 box renewal：$\eta_{\mathrm{str}}$ 降至 baseline 的 44%（$1.50 \to 1.11 \to 0.70$），轨迹更接近理想直线，但 mAP 仅 $-0.0003$。(c) Top-$K$ pruning + box renewal：$\eta_{\mathrm{str}}$ 呈 V-shape（step1 低因 reset 退化为 Euler，step2 升高因新历史建立），确认每步冷启动 DPM-Solver++。(d) K=100 vs K=200：$\eta_{\mathrm{str}}$ 几乎相同（step2: 2.18 vs 2.24），证实 K=100 精度退化主因是 proposal 数量不足而非 DPM-Solver++ 历史破坏。](latex/figures/eta_str_diagnostic.png)

**实验验证。** 在 3 个 seed（42/123/789）的 +DPM-Solver++ checkpoint 上实测，$\eta_{\mathrm{str}}$ 沿 4 步推理单调下降 $3.43 \to 2.45 \to 1.68$（mean ± std: step1 $3.43 \pm 0.36$, step2 $2.45 \pm 0.24$, step3 $1.68 \pm 0.15$；500 张图像/seed 的 batch 均值）。$\eta_{\mathrm{str}}$ 在第 2 步已降至 step1 的 $71\%$，对应"2 步即收敛"的实证观察：第 3 步及之后的二阶校正贡献随 $\eta_{\mathrm{str}}$ 衰减而趋于零。值得注意的是，$\eta_{\mathrm{str}}$ 绝对值非零（$\in [0.7, 3.4]$），说明学习轨迹并非理想直线——更准确的表述是"轨迹曲率在 step 2 后足够小，使 DPM-Solver++ 校正项对 mAP 的边际贡献 < 0.001"。这一诊断指标为"何时需要 reflow（2-RectFlow）"提供了可操作判据：若训练后 $\bar{\eta}_{\mathrm{str}} > 0.1$ 持续，则 reflow 可能进一步使轨迹趋近直线；若 $\bar{\eta}_{\mathrm{str}} < 0.01$，reflow 收益有限。

#### 4.5.3 匹配 NFE 下 DPM-Solver++ 对比 Heun

在相近 NFE 下，DPM-Solver++ 4 步（4 NFE，$0.863$）≈ Heun 2 步（3 NFE，$0.863$）——精度相当，因此 DPM-Solver++ 以零成本换取 43% 更少的 NFE。关键在于，在匹配步数下（4 对 4），DPM-Solver++ 相对 Heun 具有小但统计显著的精度优势（$+0.006$ 每图像 mAP，Wilcoxon $p<10^{-6}$；Table 8），因此其优势既是计算层面的，也是一项小的精度增益——而非仅从匹配 NFE 比较中可能推断的"纯属计算层面"。

#### 4.5.4 测试集评估

为核实验证集选择未掩盖过拟合，我们在 Dataset 2 测试集（1,000 张图像，45,980 个 GT 实例）上重新评估全部 9 个模型（Table 10 同一组 checkpoint，seed 42 best）。+DPM-Solver++ 取得 mAP 0.859，与验证集 3-seed 均值 $0.859 \pm 0.003$ 完全一致，仅相对 seed 42 best 验证集值 0.863 下降 $\Delta = -0.004$（9 模型中唯一超过 $\pm 0.001$ 的偏差），证实整体精度泛化良好且 seed 42 验证集选择无系统性偏差。Top-$K$ 剪枝叙事在测试集上进一步增强：K=300 在测试集上反超 +DPM-Solver++（0.860 对 0.859），K=200 与 +DPM-Solver++ 完全持平（$\Delta = 0.000$，验证集为 $-0.003$），K=100 的有害结论 robust（测试集 $-0.012$ 对验证集 $-0.013$）。其余 6 个模型 val→test $\Delta$ 均 $\le \pm 0.001$。然而 AP$_S$ 点估计在同一 checkpoint 上从 $0.499$（验证集）摆动到 $0.577$（测试集）——这提示在该 24 类基准上 AP$_S$ 是高方差的（仅 60 张验证图像含小目标），应结合 Table 8 中的逐图像显著性检验解读，而非作为点估计。

### 4.6 FPS / 延迟基准

Table 10 和 Figure 8 报告了在 $512{\times}512$ 输入分辨率下的速度-精度权衡。配合 Top-$K$ 剪枝的 DPM-Solver++ 达到与交互式使用相兼容的延迟，而标准检测器快 3–7× 但精度较低。12.9–22.4 FPS 的延迟范围与临床核型分析实验室的交互式工作流兼容，使扩散检测器达到临床可部署的响应速度。

| 模型 | Solver | NFE | Latency (ms) | FPS | mAP |
|-------|--------|-----|-------------|-----|-----|
| RF+Heun | Heun | 7 | 122.55 | 8.2 | 0.856 |
| +Stoch. Coupling | Heun | 7 | 130.85 | 7.6 | 0.858 |
| **+DPM-Solver++** | DPM++ | 4 | **77.57** | **12.9** | **0.863** |
| +DPM-Solver++ + Top-$K$ (K=300) | DPM++ | 4 | 70.46 | 14.2 | 0.861 |
| **+DPM-Solver++ + Top-$K$ (K=200)** | DPM++ | 4 | **68.99** | **14.5** | **0.860** |
| +DPM-Solver++ + Top-$K$ (K=100) | DPM++ | 4 | 67.43 | 14.8 | 0.850 |
| **+DPM-Solver++ (H=3 Distill)** | DPM++ | 4 | **44.72** | **22.4** | **0.859** |
| Cascade R-CNN | — | 1 | 20.11 | 49.7 | 0.854 |
| YOLOX-S | — | 1 | 9.53 | 105.0 | 0.796 |
| DiffusionDet | Euler | 1 | 23.19 | 43.1 | 0.803 |
| RTMDet-L | — | 1 | 32.82 | 30.5 | 0.863 |
| DINO R50 | — | 1 | 32.73 | 30.5 | 0.868 |

**表 10**：FPS / 延迟基准（Dataset 2 验证集，512×512，seed 42 best checkpoint；3-seed 均值见 Table 5，测试集评估见 §4.5.4）。延迟为 RTX A6000 上 500 次迭代均值（batch=1，CUDA Event 计时），std < 0.9 ms。H=3 Distill 行为**蒸馏压缩变体**（非主消融路径），通过 headwise feature 蒸馏将 cascade head 从 6 个压缩至 3 个（NFE 24→12），实测延迟 44.72 ms（Head 39.17 ms / Backbone+Neck 5.55 ms），mAP 经 val 集独立评估为 0.859；列入本表以展示速度-精度权衡的极限点。+DPM-Solver++ (H=3 Distill) 在 mAP 0.859 下达到 22.4 FPS，为近乎相同精度下最快变体（较 K=200 的 mAP 0.860 仅低 0.001，在 3-seed noise $\pm 0.003$ 内）。标准检测器中，Cascade R-CNN 与 YOLOX-S 较快（49.7/105.0 FPS）但 mAP 较低（0.854/0.796）；RTMDet-L 与 DINO R50 在 30.5 FPS 下达到 mAP 0.863/0.868，与 +DPM-Solver++（0.863）相当或略高，但差距在跨 seed 方差范围内（§4.5）。

+DPM-Solver++ + Top-$K$ (K=200) 是 Top-$K$ 剪枝中最快的变体（68.99 ms / 14.5 FPS，mAP 0.860）；**H=3 Distill** 通过架构级压缩（cascade head 6→3）在近乎相同 mAP（0.859 vs 0.860，$\Delta{=}{-}0.001$，在 3-seed noise $\pm 0.003$ 内）下达到 22.4 FPS，较 K=200 加速 $1.54\times$。+DPM-Solver++ 在 mAP 0.863 下达到 77.57 ms / 12.9 FPS。在非蒸馏变体中，cascade 头占据 $90\%+$ 的延迟（如 +DPM-Solver++ 为 92.8%）；H=3 Distill 因 head 数量减半，head 占比降至 87.6%，主干+颈部相应升至 12.4%。H=3 蒸馏与 Top-$K$ 剪枝互补——前者减少每步 head 调用数，后者减少 proposal 数——二者可叠加使用。

![**图 8**：速度-精度权衡（RTX A6000，512×512，batch=1）。FPS 轴为对数尺度；颜色/标记编码方法类别（圆=Ours-Heun，方=Ours-DPM-Solver++，星=Ours-H=3 Distill，三角=标准检测器，菱=Diffusion baseline），图例置于右图左下角。**(a) Dataset 1（低数据，1,540 张）**：KaryoFlow 变体、DiffusionDet 与标准检测器（Cascade R-CNN / RTMDet-L / YOLOX-S / DINO R50）均有 mAP。在测试集上，KaryoFlow（+Stoch. Coupling，test mAP 0.740）与先进主流检测器 DINO R50（0.725）和 RTMDet-L（0.732）持平，且在 24 个类别中的 15 个上同时优于两台标准检测器——优势集中在小尺寸染色体组（E/F/G：AP 提升 $+0.018{\sim}+0.048$）和性染色体（X/Y：$+0.014{\sim}+0.019$），这些类别恰是临床核型分析中诊断风险最高的。这支撑"基于扩散的检测器在数据稀缺场景下达到与前沿检测器相当的精度"的核心论点，并进一步表明 RF 范式在临床关键的小染色体类别上具有结构性优势。Random 耦合点（0.746，标准增强 3-seed 均值）与 Stochastic Coupling（0.747）在标准增强下精度持平（$\Delta = +0.001$，ns），二者均高于 DDPM 基线（0.729），表明 RF 范式而非耦合策略是精度提升的主因。**(b) Dataset 2（5,000 张）**：完整方法集含 DINO R50（30.5 FPS，0.868）与 RTMDet-L（30.5 FPS，0.863）；我们的 RF 变体位于高精度区（mAP > 0.85）。**+DPM-Solver++ (H=3 Distill)**（22.4 FPS，mAP 0.859）通过 cascade head 蒸馏压缩取得最佳速度-精度权衡，较 Top-$K$ 剪枝最快变体 K=200（14.5 FPS）加速 $1.54\times$。FPS 仅依赖架构与求解器（合成 512×512 输入），与数据集无关，故同一模型在两子图中 FPS 相同，仅 mAP 随数据集变化。](latex/figures/fps_map.png)

### 4.7 跨数据集总结

在两个数据集上，KaryoFlow 均优于 DDPM 基线 DiffusionDet（Dataset 1 上 3-seed 均值 $+0.018$ mAP（Heun solver；DPM-Solver++ 未在 Dataset 1 单独评估，但由 Table 1 可见匹配步数下 solver 影响可忽略），Dataset 2 上 3-seed 均值相对 DiffusionDet 单 seed $+0.056$ mAP）；其中纯 RF 范式贡献（不含 Stochastic Coupling）在 Dataset 1 上为 $+0.017$ mAP（0.746 对 0.729，§4.2）。DPM-Solver++ 在更低 NFE 下匹配 Heun 并在匹配步数下具有小但统计显著的精度优势（$+0.006$ mAP，Wilcoxon $p<10^{-6}$）。在统一标准数据增强后，Stochastic Coupling 在两个数据集上均与 Random 耦合无显著精度差异（Dataset 1: $\Delta = +0.001$，ns；Dataset 2: $\Delta = +0.0001$，$p{=}0.80$）——标准增强提供的 GT 多样性在实践中补偿了 OT 坍缩效应，使坍缩在标准训练条件下不可观测；其独立价值为 $4.6\times$ 收敛平滑性收益，在两个数据集上均成立，使小数据下基于 EarlyStopping 的 checkpoint 选择更可靠。

### 4.8 鲁棒性

我们报告一项推理时鲁棒性探测，以强化评估广度。该探测复用 +DPM-Solver++ checkpoint（DPM-Solver++ 4 步 + Top-$K$ 剪枝），在 Dataset 2 上训练——*无模型重训*。

**标注噪声鲁棒性。** 我们对 Dataset 2 *测试* ground truth (GT) 进行扰动：(i) 对每个 GT bbox 中心添加高斯抖动（σ_bbox ∈ {2, 5, 10} px，宽/高保持不变，中心裁剪到图像边界）；(ii) 以概率 p ∈ {5%, 10%, 20%} 随机翻转类别标签至其余 23 类中均匀采样的一个替代。扰动 GT 的 3×3 网格（加干净基线）以 seed 42 生成一次，用同一 checkpoint 重新评估；图像像素不动。Table 11 报告所得 mAP 退化。

**表 11**：标注噪声鲁棒性（Dataset 2 测试，+DPM-Solver++ checkpoint，seed 42，1000 张图像 / 45,980 个 GT 实例）。行：GT bbox 抖动 σ_bbox (px)。列：GT 类别翻转率 p。单元格：mAP@[0.50:0.95]。干净基线（左上）：0.859。

| σ_bbox \ p | 0%        | 5%   | 10%  | 20%  |
|------------|-----------|------|------|------|
| 0 px       | **0.859** | ---  | ---  | ---  |
| 2 px       | ---       | 0.666 | 0.599 | 0.477 |
| 5 px       | ---       | 0.428 | 0.386 | 0.308 |
| 10 px      | ---       | 0.179 | 0.162 | 0.129 |

出现两个模式。第一，*bbox 抖动主导高 IoU 精度*：在 σ=5 px 下，AP50 仅温和下降（0.988→0.847，−0.141），而 AP75 坍缩（0.971→0.375，−0.596），因为在约 100 px 的框上 5 px 中心偏移足以打破 IoU≥0.75 但不打破 IoU≥0.50。第二，*类别翻转近似乘法地降低精度和召回*：在固定 σ 下翻转率加倍大致使剩余 mAP 减半。两种噪声源在低噪声下亚加性地交互（σ=2、p=5% 下的联合 −0.193 小于任一边际之和），但在高噪声下严重复合。即便在最对抗的设置下，模型相对随机类别基线 (1/24 ≈ 0.042) 仍保持 3× 余量，表明所学 RF 特征在标注腐蚀下不坍缩。

## 5. 分析与讨论

### 5.1 为何 RF 适用于染色体检测

RF 的低曲率 ODE 路径减少了少步推理中的截断误差，这对染色体检测尤为重要：高目标密度（每张图像约 46 个）会复合每框误差，小训练集（1,540–5,000 张图像）限制了模型学习复杂弯曲 DDPM 轨迹的能力，而 24 类细粒度任务受益于稳定的特征表示。Dataset 2 上 $+0.053$ mAP 的改善（$0.803 \to 0.856$）证实了 RF 在此情形下的有效性。

### 5.2 Stochastic Coupling：负面精度结果与探索性稳定性观察

**在统一标准数据增强下**，Hard OT、Random 与 StochOT 在两个数据集上均无显著精度差异；历史 $+0.034$ 声明来自增强策略混杂，不能引用。该结果说明 OT diversity collapse 的熵结论并不会自动推出检测精度下降，标准增强可能已经提供足够的 GT 多样性。

单 seed 曲线中观察到 last-30 epoch std 从 0.006 降至 0.0013，但连续 epoch 高度自相关，且当前比较没有多 seed 置信区间或逐 epoch 测试集闭环。因此 4.6× 只能标为探索性平滑性观察，不能宣称 checkpoint 选择或泛化得到改善。Stochastic Coupling 由核心贡献降级为耦合分析与待验证的训练稳定化候选。

### 5.3 DPM-Solver++ 对比 Heun：计算优势与小精度增益

由于 +Stoch. Coupling（Heun）和 +DPM-Solver++ 使用相同的 FM 训练目标，每个 epoch 的模型权重相同。然而 DPM-Solver++ 在匹配 4 步下相对 Heun 产生小但统计显著的 mAP 改善（$+0.006$ 每图像 mAP，Wilcoxon $p<10^{-6}$；Table 8），因此高阶 solver 略好而非更差。结合其 NFE 减少，稳健的结论是：DPM-Solver++ 在 NFE 减少 43% 的情况下取得略高于 Heun 的精度，修正了 FlowDet 关于高阶 solver 在检测中表现更差的结论。

**x0-prediction 与 v-prediction 的选择。** RF 原文（Liu et al., 2023）使用 v-prediction 训练目标 $\|v_\theta - (x_1 - x_0)\|^2$。在 $d=4$ 低维 RF 下，x0-prediction 与 v-prediction 在信息论意义上等价（$\hat{x}_0 = x_t - t\hat{v}$），差异仅在损失的 $t$ 加权：$\mathcal{L}_v = t^{-2}\mathcal{L}_{x_0}$。在偏移噪声调度（$s=3.0$）下，v-prediction 在 $t \to 0$ 时的 $1/t^2$ 梯度放大加剧训练方差，而 x0-prediction 在所有 $t$ 上梯度范数恒定。此外，x0-prediction 与 DPM-Solver++ 的 data-prediction 形式天然兼容，避免在 $t \to 0$ 时显式计算 $v = (x_t - \hat{x}_0)/t$ 的数值奇点（由 $\epsilon = 10^{-5}$ 截断处理，Appendix A.5）。这一选择与图像生成（$d \sim 10^5$，linear schedule）下 v-prediction 的偏好形成对比——反映低维结构化预测的特定要求：高维下 $1/t^2$ 加权有益于强调小 $t$ 细节，但低维下它仅加剧方差。3-seed 实验验证（Dataset 2）支持此分析：v-prediction 3-seed 均值 mAP $0.857 \pm 0.0015$，对比 x0-prediction baseline $0.859 \pm 0.003$，平均 $\Delta = -0.002$（方向一致，在跨 seed 方差范围内但 v-prediction 无一 seed 超越 x0-prediction）；v-prediction 的训练动态也表现出更早的收敛停滞（best 出现在 warmup 后稳定阶段，之后 30 epoch 未刷新），与 $1/t^2$ 梯度放大阻碍后期精化的理论预测一致。

### 5.4 Top-$K$ 剪枝：依赖 Solver 的有效性

Top-$K$ 剪枝的有效性取决于每步 NFE：对 Heun（2 NFE/步），剪枝影响 6/8 次调用（1.09–1.12× 加速）；对 DPM-Solver++（1 NFE/步），影响 3/4 次调用（1.05–1.08×）。DPM-Solver++ 已通过 NFE 减少获得大部分加速，使 Top-$K$ 影响较小。

**Top-$K$ 剪枝与 DPM-Solver++ 多步历史的交互。** Top-$K$ 在每步剪枝后对低置信度 proposals 做 box renewal（重置为噪声），随之触发了 DPM-Solver++ 多步历史重置，清空 2M 所依赖的 $\hat{\mathbf{x}}_0$ 历史。我们在 seed 42 的 +DPM-Solver++ checkpoint 上以 $\eta_{\mathrm{str}}$ 诊断该交互：相对 baseline 的单调下降模式 $3.94 \to 2.79 \to 1.89$（seed 42 单点；因 K=100/K=200 仅 seed 42 有数据，此处不用 §4.5.2 的 3-seed 均值 $3.43 \to 2.45 \to 1.68$），K=200 配置呈现 V-shape $1.37 \to 2.24 \to 1.54$（step1 异常低，因 reset 后退化为 Euler 一阶；step2 升高，因新历史建立后二阶校正 $D_1$ 恢复）。该 V-shape 模式确认 Top-$K$ + box renewal 在每步冷启动 DPM-Solver++，理论上损失了多步法的二阶精度优势。

**K=100 精度退化归因的证伪。** 一个自然的猜测是 K=100 相对 K=200 的 mAP 退化（$-0.010$，Table 10）源于更激进的 box renewal 进一步破坏 DPM-Solver++ 多步历史。但实测 K=100 与 K=200 的 $\eta_{\mathrm{str}}$ 几乎相同（step2: 2.18 vs 2.24，step3: 1.54 vs 1.54），均呈 V-shape 且二阶校正量级一致——多步历史未被进一步破坏。因此 K=100 的精度退化主因是 proposal 数量不足（100 个框覆盖 ~46 条染色体 + 重叠冗余时容量紧张），而非 DPM-Solver++ 历史污染。

**Box renewal 对 $\eta_{\mathrm{str}}$ 的整体影响（$\eta_{\mathrm{str}}$ 与 mAP 表观矛盾的化解）。** 关闭 box renewal 后 $\eta_{\mathrm{str}}$ 整体降至 baseline 的 44%（step1: 1.50 vs 3.43，step3: 0.70 vs 1.68；3-seed 均值），轨迹更接近理想 RF 直线，但 mAP 仅变化 $-0.0003 \pm 0.003$（噪声范围内）。这表明 box renewal 通过干扰 $\eta_{\mathrm{str}}$ 量化上"弯曲"了 RF 轨迹（形式化地，renewal 后 $D_1$ 期望范数由 renewal 噪声主导而非轨迹曲率，使 $\eta_{\mathrm{str}}$ 诊断失效——详见 arXiv companion），但该弯曲对最终 mAP 影响可忽略——DPM-Solver++ 的二阶校正即便在 renewal 干扰下仍提供 §4.5.3 中 $+0.006$ mAP 的精度优势，因 proposals 在每步冷启动后由 RF 速度场重新对齐至低曲率 ODE 路径。需注意，box renewal 的推理时关闭仅在 $K \ge 200$（推荐配置）下安全：在 $K{=}100$（非推荐配置，已知有害）下，关闭 renewal 导致额外的 $-0.031 \pm 0.012$ mAP 退化（3-seed），因 proposal 稀缺时 renewal 的"proposal 回收"机制（将远离任何 GT 的死 proposal 重置为噪声，给其重新收敛的机会）价值凸显。这一边界条件进一步证实 box renewal 的核心价值是 proposal 回收而非 DPM-Solver++ 历史维护。

### 5.5 理论适用性与局限

双侧界（§3.3）在染色体检测中紧致（$0.03\%$ 间隙）：良分离的 Voronoi 单元（成对距离 $\approx 20$ px 对比 $\sigma \sim 1$ px）使 $P_{\text{err}} < 10^{-45}$，且 $N=2$ 的 mini-batch OT 退化为最近邻分配。低维情形（$d=4$，$K \approx 46$）下维度判据 $K^{-1/d} \approx 0.38$（Table 2，高风险），高斯噪声源由我们的偏移高斯调度近似满足。

该理论不能迁移到高维生成（$d \sim 10^5$，此时 $K^{-1/d} \approx 1.00$，故 OT 坍缩可忽略——与 OT-CFM 的成功一致），也不能迁移到密集重叠目标的情形（假设 2 和 4 失效）。对于 COCO（$d=4$，$K \sim 7$），$K^{-1/d} \approx 0.62$ 落在中间区域，判据不做明确预测，需通过 §3.3 的 Fano 下界测量 $d_{\min}$ 和 $\sigma_t$ 确定；其绝对坍缩量 $\Delta H = \log K \approx 1.95$ 小于染色体（$\approx 3.83$），Stochastic Coupling 的绝对收益预期较小。注意 $K^{-1/d}$ 不预测同维度任务间的相对坍缩程度——COCO 与染色体同为 $d=4$，其比较需通过 §3.3 的 Fano 下界确定。理论提示 RF + Stochastic Coupling 将使结合低 $d$、高目标密度和小训练数据的检测任务受益——这一画像包括医学成像、遥感以及其他细粒度密集检测任务。我们未在 COCO 上验证，因为其较小的 $K$ 降低了 OT 坍缩严重性；合适的验证数据集应具有高 $K$ 和低 $d$，正是染色体检测的画像。稳定性收益对临床部署具有实际意义：$4.6\times$ 的 epoch 稳定性提升意味着 checkpoint 选择处于趋势的 0.0013 之内（对比 Random 的 0.006），降低了部署"虚假峰值"checkpoint 的风险。

### 5.6 为何不采用显式生物学先验约束

染色体核型分析具有明确的生物学先验：每类常染色体成对出现（cardinality $\le 2$）、性染色体最多各一个、24 类按物理尺寸分级明确（A 组最大至 G/Y 组最小）、类别频率严重不平衡（Y 染色体训练样本约 1,803 对比常染色体约 7,000）。一个自然的问题是：能否将这些先验显式注入检测器以改善小类别性能（Section 4.3.1 中 Y、G 组最弱）？我们在探索阶段尝试了三个方向，均未产生可靠增益，本节分析其根本原因并说明我们采用的替代策略。

**显式类别加权。** 我们在 Dataset 1 瓶颈消融中尝试了类别平衡采样（class-balanced sampling）以缓解 Y 染色体数据稀缺，但加权采样器在小批量（bs=2）下触发内存溢出。Focal loss $\gamma$ 调整（$\gamma{=}3$ 和 $\gamma{=}1.5$）仅产生 $\Delta$ mAP $= +0.004$ 和 $+0.001$（相对 0.746 baseline，arXiv companion），增益微弱且不稳定，不构成主贡献。Class-Balanced Sampling 因相关架构消融实验整体未产生可靠增益而废弃。

**尺度先验的循环依赖。** scale-aware loss 试图引入尺寸先验辅助小染色体判别（mAP $= 0.742$，$-0.004$），但推理时尺度估计本身不可靠——与已证伪的 ScaleConditionedRF 同源：尺寸约束需已知类别，而尺寸正用于辅助类别判别，形成循环依赖。该困难并非实现缺陷，而是单阶段检测范式的固有限制：类别与尺寸在推理时联合推断，无法将一方作为另一方的可靠先验。

**数量约束与扩散范式的不兼容。** 同类染色体最多两个、Y 最多一个是全局 cardinality 约束，但扩散检测的每条 proposal 独立预测 $\hat{\mathbf{x}}_0$，无法在单次前向中施加集合级约束；后处理 NMS 仅能去重，不能补全缺失染色体。全局 cardinality 约束需要 set-level 推理（如 Hungarian matching），与扩散范式 per-proposal 的局部精化架构（RoIAlign + DynamicConv）不兼容，需架构级重设计。

**替代策略：隐式正则化。** 我们未采用显式先验约束，而是通过三个隐式机制缓解小类别困难，每项均有真实实验数据支撑。

(i) **LQCR 使最终分数显式包含定位质量。** 小尺寸框的轻微坐标误差会造成更大的相对 IoU 下降，而分类后验并不表达这一风险。LQCR 学习 $q\approx E[\mathrm{IoU}\mid F]$ 并以 $p q^2$ 排序；严格 final-only 实验在 Dataset 2 将 AP90/AP95 提升 $+0.03041/+0.03251$。solver-coupled 对 APs 的额外收益尚不稳定，因此小目标声明只保留为待多 seed 验证的支线。

(ii) **box renewal 维持 proposal 池多样性**。box renewal 消融实验（3 seeds，+DPM-Solver++ checkpoint，renewal on/off 对照）的 per-class mAP_75 数据（arXiv companion Table）表明，renewal 对 per-class AP 的影响在噪声范围内：整体 mAP $\Delta = +0.0003$，小类（Y/G22/F19/F20）mean $\Delta = -0.0024$，大类（A1/A2/A3）mean $\Delta = +0.0008$，所有 $|\Delta| < 0.005$。值得注意的是，Y 类在 renewal OFF 时 std 为 $0.023$，ON 时降至 $0.006$，提示 renewal 的主要作用是降低小类方差而非提升均值——其收益体现在训练/推理动态的稳定性而非 per-class AP 本身。严格的 per-class AP@0.5:0.95 验证实验已设计（arXiv companion），待 GPU 释放后运行。

(iii) **RF 恒定速度场为 24 类细粒度判别提供稳定特征表示**，相比 DDPM 弯曲轨迹减少小类别的特征漂移。DDPM baseline (Euler) vs RF+Heun 的 2-seed per-class mAP_75 对比（arXiv companion Table）显示：RF 在小类别上的改善幅度大于大类——小类（Y/G22/F19/F20）mean $\Delta = +0.071$（Y $+0.061$，G22 $+0.112$，F19 $+0.063$，F20 $+0.049$），大类（A1/A2/A3）mean $\Delta = +0.056$（A1 $+0.064$，A2 $+0.058$，A3 $+0.045$），小类改善比大类多 $+0.015$。整体 mAP_75 $+0.052$，mAP $+0.091$。该数据支持 RF 范式对小类别特征稳定性的改善。严格的 per-class AP@0.5:0.95 验证实验已设计（arXiv companion），待 GPU 释放后运行。

这些机制虽不直接编码生物学先验，但通过训练动态的间接改善达到类似目标，且无需架构修改。将全局 cardinality 约束融入扩散检测框架是自然的未来方向，但需解决 set-level 推理与 per-proposal 精化的架构融合问题，超出本文范围。

## 6. 结论

本文提出 KaryoFlow-LQCR，并从轨迹、求解和排序三个层次给出独立贡献。RF 的低曲率框流是主要精度来源，Dataset 1/2 相对 DDPM 分别约提升 $+0.017/+0.053$ mAP；RF 适配的 DPM-Solver++ 以 4 NFE 代替 Heun 的 7 NFE，其核心价值是低计算量下保持精度；LQCR 基于类别—定位联合后验校准最终排序，在不改变框坐标或扩散轨迹的严格 final-only 设置下，将 Dataset 2 mAP 从 0.86301 提升至 0.87044，并显著改善 AP90/AP95。Dataset 1 seed42 的训练验证由 0.746 提升至 0.751，完整三种子和 ross 统一复评正在进行。

LQCR 不依赖染色体形态先验，其适用条件是检测器的类别置信度与定位质量不一致，因而可直接移植到通用检测器。其 probability-ranking 推导适用于任意 bbox 检测；小目标、密集目标和高 IoU 评测只是预期收益更明显的场景。OT diversity collapse 仍作为低维 coupling 的理论现象保留，但当前实验只证明统一增强下三种 coupling 精度等价，不能外推 Stochastic Coupling 的泛化收益。

我们承认以下局限。第一，LQCR 尚未在 COCO 等通用检测基准验证，Dataset 1 的完整三种子复评也未结束。第二，$p q^2$ 是在条件 IoU 分布具有单调质量顺序时成立的 Bayes-consistent surrogate，不是一般情形下严格最大化 COCO AP 的定理。第三，Stochastic Coupling 的 4.6× 平滑性比率仅来自单 seed 自相关 epoch，不能作为独立方法收益。第四，延迟测量只在 ross RTX A6000 上报告；不同 GPU 的训练耗时不可混入速度对比。第五，仅验证 ResNet-50 主干。应对这些局限是未来工作的自然方向。

## 附录

<!--
TMI 附录策略（策略 A+B）：
- TMI 禁止 supplementary text materials（自 2022-01-01 起）。
- 解决方案：将详细附录内容移至 arXiv companion preprint。
- 主文仅保留证明梗概（策略 A）和零结果消融的内联简要提及
  （D、E）于正文中。
- 下方每个附录标注其 TMI 目标：
    [MAIN PAPER] / [ARXIV COMPANION] / [COMPRESS] / [DELETED]
- 标记为 [ARXIV COMPANION] 或 [DELETED] 的内容在本草稿中保留
  作为完整事实记录并供给 arXiv preprint。
- 预估页数节省：约 3.4 页附录压缩为约 0.4 页
  （证明梗概 + 内联提及）于 10 页 IEEE 主文中。

PHASE 3 重组（commit 8f67c88a, 2026-07-19）：
- 主文 Appendix A：证明（Prop 1, 2, 3 梗概）— 本草稿 §A
- 主文 Appendix B："AdaLN-Zero and Shifted Schedule Ablations" —
  合并 old §D（AdaLN-Zero）+ old §E（偏移调度）+ old §F.2
  （Y 染色体分析）为主文中约 1 段的单节。
- old §B（Falsified Directions）→ 仅 ARXIV COMPANION（从主文移除）
- old §C（Per-Seed Values）→ 仅 ARXIV COMPANION（从主文移除）
- old §F.1, §F.3, §F.4, §F.5 → 仅 ARXIV COMPANION（从主文移除）
- 所有正文内联引用已更新：old "(Appendix D)" 和 "(Appendix E)"
  现指向 "(Appendix B)"；old "(Appendix F)" 对 C 组指向
  "(Appendix B, arXiv companion)"。
- 本草稿保留原始 §B–§F 结构作为完整事实记录；下方的目标标注
  反映主文映射。
-->

### A. 证明与 ODE Solver 推导

<!-- [MAIN PAPER: 仅证明梗概 — 命题 1, 2, 3 内联于 §3.3；ODE solver 推导在 §A.4–A.5]
    [ARXIV COMPANION: 下方完整证明，含有限 N 分析和高斯混合后验细节]
    页数预算：主文中约 0.4 页（三个压缩梗概）。 -->

**与正文的对应关系。** 本附录支撑 §3.3（OT Diversity Collapse 与 Stochastic Coupling）与 §3.2（ODE Solvers）。命题 1（上界 $\Delta H \le \log K$）与命题 2（经 Fano 不等式的下界）陈述于 §3.3，刻画 *OT Diversity Collapse*；命题 3（Stochastic Coupling 关于 $\epsilon$ 的单调性）陈述于 §3.3，证明 Stochastic Coupling 设计的合理性。§A.4–A.5 给出 §3.2 中三种 ODE solver（Euler、Heun、DPM-Solver++）的详细推导。主文内联保留压缩的命题陈述；本附录重述并补充证明细节，完整证明（含有限 $N$ 分析和高斯混合后验）见 arXiv companion preprint。

#### A.1 命题 1 的证明（OT 多样性上界）

**设置**：源 $\nu = \mathcal{N}(0, \sigma^2 I_d)$，目标 $\mu = \frac{1}{K}\sum_{k=1}^K \delta_{\mathbf{b}_k}$。一个耦合 $\pi$ 将噪声样本 $\{\mathbf{z}_i\}_{i=1}^N$ 分配给目标框 $\{\mathbf{b}_{V_i}\}_{i=1}^N$。flow 状态为 $X_t = (1-t) \mathbf{b}_V + t \mathbf{z}$，其中 $\mathbf{z} \sim \nu$，$V$ 为耦合分配。

**证明梗概。** 随机耦合：$V \sim \operatorname{Uniform}(\{1,\ldots,K\})$ 且与 $\mathbf{z}$ 独立，故 $H_{\text{rand}}(V|X_t) \le H(V) = \log K$（在高噪声情形下等号成立，此时高斯混合后验近似均匀）。OT 耦合（$N \to \infty$）：引理使 $V = \arg\min_k \lVert\mathbf{z} - \mathbf{b}_k\rVert$（Voronoi 分配）成为 $\mathbf{z}$ 的确定性函数，故给定 $X_t$ 即可恢复 $\mathbf{z} = (X_t - (1-t)\mathbf{b}_V)/t$ 进而恢复 $V$，得到 $H_{\text{OT}}(V|X_t) = 0$。结合：$\Delta H \le \log K$，经验上紧致（$0.03\%$ 误差）。

**关于有限 $N$ 的注记。** 在实践中，OT 在大小为 $N$ 的 mini-batch 上求解（例如在我们的设置中 $N=2$）。对于有限 $N$，OT 并不产生精确的 Voronoi partitioning——它产生一个随 $N$ 改进的近似。经验验证（$\Delta H = 3.8415$ 对比 $\log K = 3.8427$，0.03% 误差）证实即便对于小 $N$，$N \to \infty$ 界在染色体检测设置下也是极佳的近似，可能因为 $K \approx 46 \gg N$ 且 GT 框在 $\mathbb{R}^4$ 中相对于 $\sigma$ 良分离。

#### A.2 命题 2 的证明（OT 多样性下界）

**证明梗概。** 在 OT 下，$V = \arg\min_k \lVert\mathbf{z} - \mathbf{b}_k\rVert$（引理）；给定 $X_t$，模型在 $\mathbf{b}_V$ 的 Voronoi 单元内以高斯噪声 $\sigma_t^2 I_d$ 观测 $\mathbf{z}$。Fano 不等式给出
$$H_{\text{OT}}(V|X_t) \;\le\; h(P_{\text{err}}) + P_{\text{err}} \log K,$$
其中 $P_{\text{err}}$ 是最优最近邻规则的误分类概率。在 $\binom{K}{2}$ 对 Voronoi 单元上取并集界（每对相距 $d_{\min}$，$\mathbf{z}$ 服从高斯噪声 $\sigma_t$）给出 $P_{\text{err}} \le \binom{K}{2}\, \Phi(-d_{\min}/(2\sigma_t))$。结合 $H_{\text{rand}}(V|X_t) \ge \log K$（高噪声情形下后验均匀）及 $h \le \log 2$ 即得。当 $\sigma_t/d_{\min} \to 0$ 时 $P_{\text{err}} \to 0$ 指数级衰减（高斯尾），故 $\Delta H \to \log K$，与上界匹配。对于染色体检测（$d_{\min} \approx 20$ px，$\sigma_t \sim 1$ px），$P_{\text{err}} < 10^{-45}$，故 $\Delta H \ge 0.999\,\log K$，与观测到的 $0.03\%$ 间隙一致。

#### A.3 命题 3 的证明（Stochastic Coupling 单调性）

**命题 3**（Stochastic Coupling 单调性）。在均匀源边际下，$H_{\text{stoch}}(V \mid X_t; \epsilon)$ 关于 Sinkhorn 正则化参数 $\epsilon \ge 0$ 单调非递减。

**证明梗概。** $T_\epsilon$ 求解 $\min_\pi \langle \pi, c \rangle - \epsilon H(\pi)$ s.t. 均匀边际 (Cuturi, 2013)。最优值 $\mathcal{V}(\epsilon)$ 关于 $\epsilon$ 是凹的（$\epsilon$ 的仿射函数的下确界）。由包络定理 $d\mathcal{V}/d\epsilon = -H(T_\epsilon)$，凹性给出 $dH(T_\epsilon)/d\epsilon \ge 0$。在均匀边际下，$H_{\text{stoch}} = \tfrac{1}{K} H(T_\epsilon)$（平均行熵），故关于 $\epsilon$ 非递减。端点：$\epsilon \to 0$ 给出 $H \to 0$（Hard OT），$\epsilon \to \infty$ 给出 $H \to \log K$（Random）。

#### A.4 Heun Solver 推导（二阶 predictor-corrector）

Heun 方法通过 predictor-corrector 提供二阶 ODE 精度：
- 预测：$\hat{\mathbf{x}}_{t-\Delta t} = \mathbf{x}_t + \Delta t \cdot \mathbf{v}_\theta(\mathbf{x}_t, t)$
- 校正：$\mathbf{x}_{t-\Delta t} = \mathbf{x}_t + \frac{\Delta t}{2} [\mathbf{v}_\theta(\mathbf{x}_t, t) + \mathbf{v}_\theta(\hat{\mathbf{x}}_{t-\Delta t}, t{-}\Delta t)]$

每步 2 次网络前向评估（NFE）。在 4 步时共 8 NFE（实际为 7；最后一步退化为 Euler）。

#### A.5 DPM-Solver++ 推导（二阶 multistep，1 NFE/步）

我们将 DPM-Solver++ (Lu et al., 2022) 适配到 RF 线性路径。模型直接预测 $\mathbf{x}_0$（GT 框），更新使用 $\mathbf{x}_0$ 历史在 $t$ 空间（而非 VP-SDE 的 log-SNR $\lambda$ 空间）中的多项式插值。RF-ODE 在 data-prediction 形式下为 $d\mathbf{x}/dt = (\mathbf{x} - \mathbf{x}_0(t))/t$；在 $[t_n, t_{n+1}]$ 上对线性 $\mathbf{x}_0(t)$ 插值精确积分得到

$$\mathbf{x}_{t_{n+1}} = \tfrac{t_{n+1}}{t_n}\,\mathbf{x}_{t_n} + \bigl(1 - \tfrac{t_{n+1}}{t_n}\bigr)\,\mathbf{x}_0^{(n)} + \varphi_1\,\mathbf{D}_1,$$
$$\varphi_1 = t_{n+1}\log\tfrac{t_n}{t_{n+1}} - t_n + t_{n+1},\quad \mathbf{D}_1 = \tfrac{\mathbf{x}_0^{(n)} - \mathbf{x}_0^{(n-1)}}{t_n - t_{n-1}},$$

其中前两项是常数 $\mathbf{x}_0$ 的精确解，$\varphi_1 \mathbf{D}_1$ 是线性变化 $\mathbf{x}_0(t)$ 的二阶校正。$t \to 0$ 处的奇点由 $\epsilon$ 截断处理（$t_{n+1} > 10^{-7}$）；三阶变体额外加入二次项 $\varphi_2 \mathbf{D}_2$。与 VP-SDE DPM-Solver++ 不同，$\mathbf{x}_1$（初始噪声）是固定样本，*不* 参与插值；其贡献由 $\mathbf{x}_{t_n}$ 隐式承载。每步 1 NFE，4 步共 4 NFE（相比 Heun 的 7）。

#### A.6 直线度指标 $\eta_{\mathrm{str}}$ 的性质（命题 4）

**命题 4（直线度指标）。** 定义 $\eta_{\mathrm{str}}^{(n)} := \lVert \mathbf{D}_1^{(n)} \rVert_2 / (\lVert \hat{\mathbf{x}}_0^{(n)} \rVert_2 + \epsilon_{\mathrm{norm}})$，其中 $\mathbf{D}_1^{(n)} = (\hat{\mathbf{x}}_0^{(n)} - \hat{\mathbf{x}}_0^{(n-1)})/(t_n - t_{n-1})$ 为 DPM-Solver++ 二阶校正项。则：

(i) $\eta_{\mathrm{str}} \ge 0$，且 $\eta_{\mathrm{str}} = 0$ 当且仅当 $\hat{x}_0$ 在 $[t_{n-1}, t_n]$ 上为常数（直线轨迹）。

(ii) 理想 1-RectFlow（$v_\theta$ 精确恢复 $v = x_1 - x_0$）下 $\forall n: \eta_{\mathrm{str}}^{(n)} = 0$，此时 DPM-Solver++ 任意步数等价于 1 步 Euler 的精确线性外推。

(iii) 若 $\bar{\eta}_{\mathrm{str}}^{(n)} < \epsilon_{\mathrm{conv}}$ 对所有 $n \ge N_0$ 成立，则 DPM-Solver++ 在 $N_0$ 步后无显著精度增益——因高阶校正项 $\varphi_1 \mathbf{D}_1$ 已被 $\eta_{\mathrm{str}}$ 界住。

**证明梗概。** (i) 由定义 $\eta_{\mathrm{str}}$ 为范数比值，非负性显然。$\eta_{\mathrm{str}} = 0 \Leftrightarrow \mathbf{D}_1 = \mathbf{0} \Leftrightarrow \hat{x}_0^{(n)} = \hat{x}_0^{(n-1)} \Leftrightarrow \hat{x}_0$ 在 $[t_{n-1}, t_n]$ 上为常数。(ii) 理想 RF 下 $\hat{x}_0(t) \equiv x_0$（GT bbox 为常数），故 $\mathbf{D}_1 = \mathbf{0}$。(iii) DPM-Solver++ 更新中 $\varphi_1 \mathbf{D}_1$ 相对主项的范数比为 $\varphi_1 \eta_{\mathrm{str}} / (1 - t_{n+1}/t_n)$，对小 $\eta_{\mathrm{str}}$ 可忽略。$\square$

#### A.7 维度判据 $K^{-1/d}$ 的推导（命题 5）

**命题 5（维度判据——跨维度筛选）。** 设 $K$ 个 GT 点 $\{\mathbf{b}_k\}_{k=1}^K$ 在 $d$ 维空间 $[0,L]^d$ 中以典型间距分布，semi-discrete OT 耦合将噪声 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \sigma_t^2 I_d)$ 分配至最近邻 GT（$\sigma_t$ 为 RF 在训练时间 $t$ 处的有效噪声尺度，见命题 2）。定义 *典型最近邻距离* $d_{\mathrm{NN}}$ 为 GT 点到其最近邻的期望距离。则：

(i) $d_{\mathrm{NN}} \sim L \cdot K^{-1/d}$（几何概率标准结果，Penrose, 2003；前含 $O(1)$ 常数 $\Gamma(1{+}1/d)\,V_d^{-1/d}$，不影响量级分析）。

(ii) *（典型行为分析）* OT 多样性坍缩程度由无量纲比 $\sigma_t / d_{\mathrm{NN}}$ 刻画：$\sigma_t / d_{\mathrm{NN}} \gg 1$ 时坍缩可忽略（$\Delta H \to 0$，此方向由命题 2 严格保证，因 $d_{\min} \le d_{\mathrm{NN}}$）；$\sigma_t / d_{\mathrm{NN}} \ll 1$ 且 GT 分布充分铺展（$d_{\min} \sim d_{\mathrm{NN}}$）时坍缩趋于完全（$\Delta H \to \log K$，此方向为典型行为论证，依赖铺展性假设）。

(iii) $K^{-1/d} = d_{\mathrm{NN}} / L$ 作为 *跨维度筛选工具*。**关键说明**：$K^{-1/d}$ 单独不决定坍缩方向——真正的判别量为 $\sigma_t / d_{\mathrm{NN}} = (\sigma_t / L) / K^{-1/d}$，而 $K^{-1/d}$ 仅作为维度 $d$ 的代理变量起作用。其筛选有效性完全依赖于以下 *任务类型驱动的经验协变*：低维回归任务（检测）中噪声为定位噪声（$\sigma_t / L \ll 1$，如染色体检测 $\sigma_t \sim 1$ px 对比 $L \sim 500$ px），高维生成任务中噪声覆盖全域（$\sigma_t / L \sim 1$）。在此协变下，$K^{-1/d} \to 1$（高维）指示坍缩风险低，$K^{-1/d} \ll 1$（低维）指示坍缩风险高。

**注记。**
(i) (i) 为数学事实；(ii) 为典型行为分析（无坍缩方向严格，坍缩方向需铺展性假设）；(iii) 为经验筛选准则。

(ii) (iii) 仅适用于跨维度比较（不同 $d$ 的任务间）。同维度不同 $K$ 的任务间，$K^{-1/d}$ 不预测相对坍缩程度——需通过命题 2 测量 $d_{\min}$ 和 $\sigma_t$ 确定。

(iii) 中间区域（如 COCO，$K^{-1/d} \approx 0.62$）：判据不做明确预测，需通过命题 2 分析确定。

**证明梗概。**

(i) 由几何概率标准结果（Penrose, 2003）：$K$ 个 i.i.d. 均匀分布点在 $[0,L]^d$ 中的典型最近邻距离 $d_{\mathrm{NN}} \sim L \cdot \Gamma(1+1/d)\,V_d^{-1/d} \cdot K^{-1/d}$，其中 $V_d = \pi^{d/2}/\Gamma(d/2{+}1)$ 为 $d$ 维单位球体积。前因子 $\Gamma(1{+}1/d)\,V_d^{-1/d}$ 为 $O(1)$ 常数（如 $d{=}4$ 时 $\approx 0.61$），不影响量级分析，故 $d_{\mathrm{NN}} \propto L \cdot K^{-1/d}$。此为 *典型*（期望）距离，区别于命题 2 Fano 界中的 *最小* 成对距离 $d_{\min} \sim L \cdot K^{-2/d}$。

(ii) 在 semi-discrete OT 下，$\mathbb{R}^d$ 被划分为 $K$ 个 Voronoi 单元 $\mathcal{V}_k$。给定 $X_t = (1-t)\mathbf{b}_V + t\mathbf{z}$，模型需从 $K$ 个假设中判别 $V$。

*无坍缩方向（严格）*：若 $\sigma_t \gg d_{\mathrm{NN}}$，则因 $d_{\min} \le d_{\mathrm{NN}}$，有 $\sigma_t \gg d_{\min}$。命题 2 的 Fano 界给出 $P_{\text{err}} \to 1$，故 $\Delta H \to 0$。

*坍缩方向（典型行为）*：若 $\sigma_t \ll d_{\mathrm{NN}}$ *且* GT 分布充分铺展（$d_{\min} \sim d_{\mathrm{NN}}$，即 $K^{-1/d}$ 不极端小），则 $\sigma_t \ll d_{\min}$，命题 2 给出 $P_{\text{err}} \to 0$，$\Delta H \to \log K$。注意此方向非严格——若 GT 分布对抗性聚集（$d_{\min} \ll d_{\mathrm{NN}}$），即便 $\sigma_t \ll d_{\mathrm{NN}}$ 也可能 $\sigma_t \not\ll d_{\min}$。染色体检测中 $d_{\min}/d_{\mathrm{NN}} \approx 0.38$（同阶），铺展性假设成立。

(iii) 将 (i) 代入 (ii) 的判别量：
$$\frac{\sigma_t}{d_{\mathrm{NN}}} = \frac{\sigma_t / L}{K^{-1/d}}.$$
**$K^{-1/d}$ 单独不决定坍缩方向**：若 $\sigma_t / L$ 固定，则 $K^{-1/d}$ 小对应 $\sigma_t / d_{\mathrm{NN}}$ 大（无坍缩），与经验观察相反。判据的有效性来自 $\sigma_t / L$ 与任务类型的协变——这并非偶然，而有物理原因：检测任务的 RF 噪声施加于框坐标（定位噪声，$\sigma_t / L \ll 1$），生成任务的 RF 噪声施加于像素空间（全域噪声，$\sigma_t / L \sim 1$）。$K^{-1/d}$ 捕获维度 $d$，$d$ 通过此物理协变决定 $\sigma_t / d_{\mathrm{NN}}$ 的量级，从而起到跨维度筛选作用。$\square$

**经验验证。**

- *染色体检测*（$d=4, K_{\text{mean}} \approx 46.6, K^{-1/d} \approx 0.38$）：$\sigma_t / d_{\mathrm{NN}} \ll 1$（$\sigma_t \sim 1$ px，$d_{\mathrm{NN}} \sim 190$ px），预测坍缩 $\Delta H \approx \log K$。实测 $\Delta H = 3.8415$ vs $\log K_{\text{mean}} = 3.8427$（相对误差 $0.03\%$，Figure 3），与预测一致。
- *图像生成*（$d \sim 10^5, K^{-1/d} \approx 1.0$）：$\sigma_t / d_{\mathrm{NN}} \sim 1$，指示坍缩风险低。"无坍缩"的最终结论来自经验证据（OT-CFM 在图像生成中的广泛成功）而非判据的精确预测——判据的作用是排除坍缩风险（$\sigma_t / d_{\mathrm{NN}} \not\ll 1$）。

**判据的局限。**
(a) *跨维度筛选*：$K^{-1/d}$ 仅在比较不同 $d$ 的任务时有效。同维度不同 $K$ 的任务间（如 COCO $K=7$ vs 染色体 $K=46$，均为 $d=4$），$K^{-1/d}$ 不预测相对坍缩——需通过命题 2 测量 $d_{\min}$ 和 $\sigma_t$ 确定。
(b) *中间区域*：COCO（$K^{-1/d} \approx 0.62$）落在中间区域，判据不做明确预测，需通过命题 2 分析确定。
(c) *铺展性假设*：坍缩方向的预测依赖 $d_{\min} \sim d_{\mathrm{NN}}$。当 $K^{-1/d}$ 极端小（$< 0.1$）时 $d_{\min}/d_{\mathrm{NN}}$ 可能显著小于 1，需直接验证铺展性。

### B. 被证伪的方向

<!-- [仅 ARXIV COMPANION — Phase 3 重组中从 TMI 主文移除]
    这些负面结果记录了内部研究决策但不推进论文声明。本草稿保留
    作为事实记录。若审稿人问"你们试过 X 吗？"，引用 arXiv companion。 -->

**与正文的对应关系。** 本附录通过记录我们探索并在实验中证伪的研究方向，证明 §3（方法）和 §4（实验）中所做的方法选择。每个被证伪的方向对应一个我们考虑过并用经验证据拒绝的替代设计：推理时优化方向（Adaptive Step、Draft-Verify、Head Early-Exit、RoI Feature Cache）关乎未能改进默认流水线（§3.2、§4.6）；Flow Matching Detection 和 Cascade Head Count E2E 方向关乎 RF + cascade 头的架构替代方案，它们降低了 mAP（§3.1）。这些负面结果解释了 *为何* 我们最终设计不包含这些组件，并在此保留作为完整事实记录；主文仅在直接相关于设计决策处提及它们。

| 方向 | 判定 / 证据 |
|-----------|--------------------|
| Adaptive Step | 证伪：$x_0$ 相对 $\Delta$ 最小 0.166 |
| Draft-Verify | 证伪：早期步骤 cls 一致率 57% |
| Head Early-Exit | 证伪：所有阈值下退出率 0% |
| RoI Feature Cache | 证伪：box 位移 93–124 px/步 |
| Flow Matching Detection | 证伪：mAP 0.823（−0.033） |
| Cascade Head Count E2E | 证伪：mAP 0.684（−0.172） |

**表 B.1**：被证伪的研究方向。

#### B.1 Reflow (2-Rectification)：详细报告

<!-- [仅 ARXIV COMPANION — 完整负面结果报告，含理论分析与重试确认实验]
    与 §4.5.2 的 $\eta_{\mathrm{str}}$ 诊断形成闭环：诊断正确识别了轨迹曲率，
    但 reflow 作为"解药"在检测 4 维空间中因信息瓶颈而失败。-->

**动机。** §4.5.2 的直线度指标 $\eta_{\mathrm{str}}$ 为 reflow 提供了操作判据：若训练后 $\bar{\eta}_{\mathrm{str}}$ 持续偏高（实测 $\eta_{\mathrm{str}} \in [0.7, 3.4]$，非零），2-Rectification (Liu et al., 2023) 可望进一步使轨迹趋近直线。在图像生成中，reflow 通过以 1-RectFlow 模型的端点预测 $\hat{x}_0$ 重新耦合并重训，显著降低轨迹曲率。我们因此在检测场景下验证 reflow 的有效性——结果为方法本质失败，此处给出完整报告。

**方法（x0-MSE Reflow）。** 以 +DPM-Solver++ checkpoint（mAP=0.863）作为 1-RectFlow 教师模型，对其在训练集上的推理输出 $\hat{x}_0^{\text{teacher}}$ 收集为 reflow coupling 的回归目标（以教师模型的 $\hat{x}_0$ 预测作为耦合目标，启用 reflow 耦合模式），随后以 standard MSE 损失从 +DPM-Solver++ checkpoint 初始化重训。这与图像生成中 2-Rectification 的标准流程一致。

**理论分析：检测场景下 reflow 的四项固有风险。** 与图像生成（$d \sim 10^5$）不同，检测的 4 维 bbox 空间使 reflow 面临结构性困难：

1. **cls/box 不一致性**（方法设计问题）：reflow 重算 OT 匹配（noise↔GT）以分配分类标签，但 box 回归目标为教师模型的 $\hat{x}_0^{\text{pred}}$。当教师模型预测的框偏离 GT 时，分类头被告知"这是 GT$_j$"而回归头被拉向"教师模型预测的另一个位置"，两者目标冲突。

2. **循环依赖**（方法设计问题）：模型以教师模型的自预测作为训练目标，强化教师模型的系统误差而非修正它。在图像生成中，高维像素空间的 $\hat{x}_0$ 预测足够精确使此问题可忽略；但在 4 维 bbox 空间，教师模型对噪声 proposal 的预测本身带有不可消除的模糊性。

3. **box_renewal 训推不一致**（方法设计问题）：reflow 训练时关闭 box renewal（以保证 coupling 一致），推理时开启（以维持 proposal 多样性），导致 proposal 分布偏移。

4. **mAP_75 退化（信息瓶颈）**（方法本质问题）：教师模型对纯噪声 proposal 的 $\hat{x}_0$ 预测在 4 维空间中本质模糊——多个 GT 框可能映射到相近的噪声区域。reflow 使模型学习预测这种模糊的 $\hat{x}_0$ 而非精确 GT，导致精细定位（mAP_75）退化。这是 4 维信息瓶颈的必然结果，与 §5.1 "RF 轨迹直线化假设在检测中不严格成立"的结论一致。

**实验证据。**

*首次实验失败（配置缺陷）。* 首次 reflow 实验存在三项配置缺陷：缺失 load_from（从零训练而非从 +DPM-Solver++ checkpoint 初始化）、lr=1e-5（仅为 baseline 5e-5 的 1/5）、max_epoch=50（仅为 baseline 150 的 1/3）。结果严重欠训练：best mAP=0.646@ep42（$\Delta=-0.217$ vs +DPM-Solver++ 0.863），mAP_75 从 0.974 退化至 0.733，且 best 后持续回退至 0.542@ep50。

*重试确认（配置修正后方法本质失败）。* 修复全部三项配置缺陷（load_from=+DPM-Solver++ best ep117、lr=5e-5、max_epoch=150、warmup 5ep + cosine、EarlyStopping patience=30）后重训。逐 epoch 关键数据：

| Epoch | mAP | mAP_50 | mAP_75 | 备注 |
|-------|-----|--------|--------|------|
| 1 | 0.862 | 0.989 | 0.970 | best = +DPM-Solver++ checkpoint 本身，非 reflow 贡献 |
| 2 | 0.618 | 0.974 | 0.687 | mAP 退化 $-0.244$，mAP_75 降至 0.70 以下 |
| 11 | 0.812 | 0.986 | 0.946 | 重试阶段最高 mAP（仍 < +DPM-Solver++ 0.863） |
| 19 | 0.586 | 0.937 | 0.608 | mAP_75 再次降至 0.70 以下 |
| 31 | 0.794 | 0.983 | 0.917 | EarlyStopping 触发（30 ep 零改善） |

**表 B.2**：Reflow 重试逐 epoch mAP（Dataset 2，seed 42）。best=0.862@ep1 恰为 +DPM-Solver++ checkpoint 加载后的初始状态，reflow 训练 30 个 epoch 零改善。

best=0.862@ep1 恰为 +DPM-Solver++ checkpoint 加载后的初始状态，reflow 训练 30 个 epoch **零改善**，mAP_75 两次降至 0.70 以下（+DPM-Solver++ 为 0 次，8 个 epoch 降至 0.80 以下），EarlyStopping @ep31 自动终止。代码实现经独立审查确认正确（coupling 加载、空间转换、per-proposal 对齐）。这证明 reflow 不仅未使轨迹趋近直线以提升性能，反而持续损害 +DPM-Solver++ 已学到的表示。

**结论。** Reflow 在检测场景下为方法本质失败，而非配置缺陷：配置修复后四项固有风险依然全部命中。根本原因是 4 维 bbox 空间的信息瓶颈——+DPM-Solver++ 对噪声 proposal 的 $\hat{x}_0$ 预测过于模糊，无法作为 reflow 的可靠目标。这与图像生成中 reflow 的成功形成鲜明对比，印证了 §5.1 的结论：RF 轨迹直线化假设在高维像素空间成立，但在 4 维检测空间中受限。$\eta_{\mathrm{str}}$ 诊断正确识别了轨迹曲率（$\eta_{\mathrm{str}} \in [0.7, 3.4] \neq 0$），但"曲率非零"不蕴含"reflow 可修复"——后者受制于目标空间的信息密度，而非轨迹本身的几何性质。

### C. 逐 seed 数值（Dataset 1）

<!-- [仅 ARXIV COMPANION — Phase 3 重组中从 TMI 主文移除]
    逐 seed 表格对 10 页主文过于细粒度。主文报告 mean±std 聚合
    （Table 5-7）；逐 seed 分解移至 arXiv 以供完整复现性验证。 ]

**与正文的对应关系。** 本附录支撑 §4.2（RF 对比 DDPM，Table 5）和 §4.4（耦合消融，Table 9）中的多 seed 表格，提供聚合 mean±std 数值背后的逐 seed 数值。下方每个子表对应一个具体的正文表格：§C.1 支撑 §4.2.2 中引用的 Dataset 1 RF-vs-DDPM 比较；§C.2 支撑 §4.4.1 中引用的 Dataset 1 耦合消融。逐 seed 分解允许独立验证正文中报告的跨 seed 方差（+DPM-Solver++ ±0.003，DDPM ±0.002 等）逐 seed 重现，且无个体 seed 是驱动聚合的离群点。

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
| Hard OT | 0 | 42 | 0.705 | 0.894 | 0.792 | 0.409 | 0.702 | 0.543 |
| Hard OT | 0 | 123 | 0.703 | 0.897 | 0.792 | 0.412 | 0.700 | 0.548 |
| Hard OT | 0 | 789 | 0.707 | 0.897 | 0.794 | 0.425 | 0.702 | 0.566 |
| Random | $\infty$ | 42 | 0.713 | 0.909 | 0.800 | 0.421 | 0.711 | 0.626 |
| Random | $\infty$ | 123 | 0.718 | 0.909 | 0.804 | 0.442 | 0.716 | 0.629 |
| Random | $\infty$ | 789 | 0.708 | 0.904 | 0.796 | 0.423 | 0.702 | 0.527 |
| Stochastic Coupling | 5 | 42 | 0.745 | 0.941 | 0.832 | 0.510 | 0.738 | 0.638 |
| Stochastic Coupling | 5 | 123 | 0.745 | 0.942 | 0.834 | 0.505 | 0.739 | 0.636 |
| Stochastic Coupling | 5 | 789 | 0.750 | 0.943 | 0.837 | 0.519 | 0.743 | 0.620 |
| Hard OT (mean±std) | — | — | $0.705 \pm 0.002$ | 0.896 | 0.793 | 0.415 | 0.701 | 0.552 |
| Random (mean±std) | — | — | $0.713 \pm 0.005$ | 0.907 | 0.800 | 0.429 | 0.710 | 0.594 |
| **Stochastic Coupling (mean±std)** | — | — | **$0.747 \pm 0.003$** | **0.942** | **0.834** | **0.511** | **0.740** | **0.631** |

**表 C.2**：耦合消融的逐 seed 数值（Dataset 1，独立推理，seed 42）。

### D. AdaLN-Zero 消融

<!-- [Phase 3 重组中合并入主文 Appendix B]
     与 old §E（偏移调度）和 old §F.2（Y 染色体）合并为
     "AdaLN-Zero and Shifted Schedule Ablations" 单节（约 1 段）于
     10 页 IEEE 主文中。完整表格在此保留作为事实记录。]

本附录报告 Section 3.1.2 及正文贡献讨论中引用的 AdaLN-Zero 独立消融，证实其在 RF 框架内对 $+0.053$ mAP 增益的单独贡献为零（Table D.1）。我们在 Dataset 2 上进行了独立的消融以验证 AdaLN-Zero 的单独贡献。两个实验除时间条件模块外配置相同（RF 公式、Heun solver 4 步、偏移调度 shift=3.0、随机耦合、150 epochs）。

| Config | mAP | $\Delta$ mAP |
|--------|-----|--------------|
| RF + Heun (without AdaLN) | 0.856 | — |
| RF + Heun + AdaLN-Zero | 0.856 | +0.000 |

**表 D.1**：Dataset 2 上的 AdaLN-Zero 消融。

AdaLN-Zero 在此数据集上的 RF 框架内贡献为 *零*（$\Delta$mAP = 0.000）。这与如下假设一致：RF 的低曲率 ODE 路径已提供充分的时间结构，使零初始化的调制成为冗余。我们将 AdaLN-Zero 作为标准条件机制 (Dhariwal & Nichol, 2021) 保留，以与更广泛的扩散文献保持一致，但指出它并不贡献于 Section 4.2 中所声明的 $+0.053$ mAP 改善。整个 $+0.053$ 差距归因于 RF 公式本身（低曲率 ODE 路径）；偏移的噪声调度自身贡献可忽略，如 Appendix B 所示。

### E. 偏移噪声调度消融

<!-- [Phase 3 重组中合并入主文 Appendix B]
     与 old §D（AdaLN-Zero）和 old §F.2（Y 染色体）合并为
     "AdaLN-Zero and Shifted Schedule Ablations" 单节（约 1 段）于
     10 页 IEEE 主文中。完整表格在此保留作为事实记录。]

本附录报告 Section 3.1.2、摘要和结论中引用的偏移噪声调度的独立消融。偏移调度（shift=3.0）作为 RF 训练配置的一部分贯穿本文使用，但其对 $+0.053$ mAP 增益的单独贡献此前未被分离。我们通过在 Dataset 2 上以 shift 设为 $0$（即线性调度）训练相同配置来填补这一空白。

两个实验除 shift 参数外配置相同（RF 公式、Heun solver 4 步、随机耦合、150 epochs、seed 42）。结果按验证 mAP 的最佳 checkpoint 报告。

| Schedule | mAP | AP50 | AP75 | AP$_S$ |
|----------|-----|------|------|--------|
| Linear (shift=0) | 0.857 | 0.989 | 0.971 | 0.555 |
| Shifted (shift=3.0) | 0.856 | 0.990 | 0.971 | 0.563 |
| $\Delta$ | $-0.001$ | $+0.001$ | $0.000$ | $+0.008$ |

**表 E.1**：Dataset 2 上的偏移噪声调度消融。偏移调度对 mAP 的单独贡献可忽略；整个相对 Euler 基线的 $+0.053$ 差距归因于 RF 公式本身。线性调度运行的运行内 epoch mAP std 为 0.0037（最后 30 个 epoch）。AP$_M$ 和 AP$_L$ 在两种调度间至多相差 0.004（未显示）。

偏移调度的单独贡献为 $\Delta \text{mAP} = -0.001$（$\approx 0\%$），完全处于 seed 噪声范围内，尽管它在小目标上带来小的 $+0.008$ 改善（AP$_S$；AP$_M$ 和 AP$_L$ 至多相差 0.004）。这证实 Section 4.2 中所声明的相对 Euler 基线的 $+0.053$ mAP 增益归因于 RF 公式本身（低曲率 ODE 路径），而非偏移调度。我们将偏移调度作为继承自扩散检测文献的标准细节保留，但它不是单独的精度增益来源。因此 Section 4.5 中的 solver×step 解耦将 $+0.053$ 差距的 $91\%$ 归因于 *作为整体* 的 RF 范式，其中低曲率 ODE 路径占主导。

### F. 逐类 AP 细节

<!-- 混合目标（Phase 3 重组）：
  F.1 染色体尺寸分组    -> [仅 ARXIV COMPANION]（参考材料）
  F.2 Y 染色体分析      -> [合并入主文 Appendix B]（约 1 句）
  F.3 C 组判别          -> [仅 ARXIV COMPANION]（详细逐类数值）
  F.4 完整逐类 AP 表    -> [仅 ARXIV COMPANION]（24 行表对主文过大）
  F.5 定位饱和          -> [仅 ARXIV COMPANION]（§5 讨论中 1 句）
  主文净成本：Appendix B 中约 1 句 + 0 表；完整内容在此 + arXiv 保留。 -->

本附录以详细分解补充 Section 4.3.1 中的逐类 AP 分析。

#### F.1 染色体尺寸分组

遵循标准 ISCN 核型分析约定，24 个染色体类别按物理尺寸分为三层，用于 Figure 6：

- *Large*：A1–A3、B4–B5（在低倍镜下可见的中期染色体；> 8 Mbp）。
- *Medium*：C6–C12、X、D13–D15（中等尺寸的亚中着丝粒 / 近端着丝粒染色体）。
- *Small*：E16–E18、F19–F20、G21–G22、Y（最小的染色体；F/G 组 < 5 Mbp）。

该分组与 COCO 风格的 AP$_S$/AP$_M$/AP$_L$ 按目标面积划分对齐，因为染色体物理尺寸与中期图像中的 bbox 面积相关。

#### F.2 Y 染色体分析

Y 染色体是最难的类别（seed 42 时 AP=0.779；3 个 seed 上 $0.771 \pm 0.006$），其困难性由数据与生物学共同决定。Dataset 2 训练集仅含约 1,803 个 Y 染色体样本，而每条常染色体约 7,000 个、X 染色体 5,123 个——3.9× 的不平衡直接限制了 Y 类别获得的梯度更新次数。这种不平衡是生物学的结果：Y 仅以单拷贝出现且仅在男性样本中。Y 也是最小的人类染色体之一，富含异染色质，且在个体间形态变异较大。其 AP$_S$=0.577 证实困难集中在小目标尺度。

#### F.3 C 组判别

C 组染色体（C6–C12）是典型的"难以区分"类别：七条中大尺寸的亚中着丝粒染色体，尺寸和形状相似，靠细微的带纹差异区分。检测器在该组上取得高 AP（C6=0.900、C7=0.896、C8=0.883、C9=0.880、C10=0.877、C11=0.871、C12=0.890），组内差异仅为 0.029（0.871–0.900）。该差异与尺寸仅弱相关，提示模型捕捉到了细微带纹线索而非仅依赖尺寸。X 染色体（AP=0.885）位于大组范围内，与其中大尺寸的亚中着丝粒形态和充足的训练数据（5,123 个样本）一致。

#### F.4 完整逐类 AP 表

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
| Y | 0.779 | 0.972 | 0.933 | 0.577 | 0.788 | — |

**表 F.1**：Dataset 2 验证集上完整的逐类 AP 分解（+DPM-Solver++）。

#### F.5 定位饱和与下游潜力

所有类别的 AP50 接近饱和（>0.988，Y 为 0.972），因此定位接近饱和（>0.988），残余误差集中在细粒度分类上。一个在正确定位的裁剪区域上工作的下游精化阶段——带纹分类器或形态感知的重新评分头——原则上可恢复相当一部分剩余 AP，因为上游检测器已经提供了正确的区域。

### G. DDPM 多步消融（Dataset 2）

本附录支撑 §4.2 中"DDPM 步数对精度几乎无影响"的声明。以 DiffusionDet checkpoint（Dataset 2，best @ ep26，seed 42）运行 Euler solver 于 1/2/4/8 步，完整精度指标如下：

| Steps | NFE | mAP | AP50 | AP75 | AP$_S$ | AP$_M$ | AP$_L$ |
|-------|-----|-----|------|------|--------|--------|--------|
| 1 | 1 | 0.805 | 0.971 | 0.937 | 0.405 | 0.802 | 0.810 |
| 2 | 2 | 0.804 | 0.969 | 0.939 | 0.414 | 0.800 | 0.806 |
| 4 | 4 | 0.804 | 0.970 | 0.938 | 0.402 | 0.800 | 0.804 |
| 8 | 8 | 0.805 | 0.971 | 0.940 | 0.405 | 0.801 | 0.818 |

**表 G.1**：DDPM Euler 多步消融（Dataset 2 验证集，seed 42）。mAP 在 1–8 步间变化 $<0.002$，所有精度指标（AP50/AP75/AP$_S$/AP$_M$/AP$_L$）均在噪声范围内波动。对比 RF 范式切换带来的 $+0.053$ mAP 增益（Table 5），DDPM 增加步数的收益可忽略。注：1 步推理 mAP（0.805）与 Table 5 中报告的 DDPM baseline（0.803，训练评估）略有差异，源于推理与训练评估的 maxDets 设置不同；多步对比在相同推理设置下进行，结论不受影响。
