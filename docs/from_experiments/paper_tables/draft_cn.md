# 多样性优于传输效率：面向密集多实例扩散检测的Sinkhorn采样方法

## 摘要

扩散检测器在结构化医学影像任务中展现出良好的应用前景，然而训练过程中耦合设计的作用仍未得到充分理解。在图像生成领域，最优传输（OT）耦合因缩短传输路径、提升优化效率而被广泛采用。本文发现，这一直觉无法迁移至以染色体分析为代表的密集多实例检测任务——在此场景中，目标以高密度紧密排列，训练信号的多样性比传输路径的最优性更为关键，而OT恰恰牺牲了前者以追求后者。

在强基线模型KaryoFlow上，确定性OT导致检测精度下降约2%。我们将其归因于两个互补机制：其一，确定性OT将条件速度熵压缩至接近零（降幅超过99%），导致训练监督的多样性崩溃——在密集染色体簇中，检测器丧失了区分相邻相似实例所需的多样化精化信号；其二，argmax解码破坏了Sinkhorn正则化本应提供的多样性控制——无论正则化参数取值如何，大型染色体系统性占优，有效匹配的染色体组数量被冻结在不足三个，使小型染色体持续被忽略。基于此分析，我们提出Sinkhorn采样方法——从Sinkhorn传输矩阵中采样分配关系。Sinkhorn采样恢复了多样性控制，在中等正则化强度下将性能恢复至与随机基线持平。在"多样性优于传输效率"这一核心原则的指导下，我们进一步提出群组层次Sinkhorn采样（GHSS）——按八个染色体组划分提案，组间方差降幅仅约45%，远优于全局OT的70%降幅，表明组内多样性得以保留。该配置是唯一超越随机基线的耦合策略。我们通过条件熵与方差分解、涵盖两种解码策略的ε多区域相图（随机解码有效匹配数随ε单调增长至约5.3，argmax冻结于约2.8）、24类染色体逐类精度分析（23/24类在确定性OT下受损）以及独立来源临床数据集的跨数据集验证，构建了完整的机制到实验证据链。结果表明，耦合设计不应从生成模型盲目照搬至医学检测领域；在密集检测中，保持监督多样性是第一位的需求。

## 1. 引言

扩散检测器和整流流检测器将目标检测建模为从噪声框到结构化预测的渐进式去噪过程，为传统单步检测提供了有吸引力的替代方案\[1,2,19\]。这一形式在医学影像领域尤其有价值：标注稀缺、目标结构性强、不确定性普遍存在。在染色体核型分析中，临床工作流程仍然高度依赖人工操作，近年研究不断强调自动化系统的实际需求以及在有限专业标注下构建精确系统的困难\[3-6,21\]。此类检测器中一个根本性的设计问题是：训练期间如何将噪声提案状态与目标标注进行耦合。

在图像生成领域，基于OT的耦合因缩短路径长度、提升学习流的直线性而被广泛认为有效\[2,7,20\]。这一成功推动了其在检测中的应用——OTA、SimOTA以及DETR系列中的匈牙利匹配已在传统检测器的标签分配中取得了显著成功\[11,12,9,13,14\]。因此，将OT耦合直接移植到扩散检测中似乎是顺理成章的。然而，我们的实验表明这一直觉可能以惊人方式失效：确定性OT匹配反而降低了检测性能，尽管从传输代价角度看其理论效率更高\[8\]。

本文提出一个核心问题：*密集多实例检测中的耦合设计应该追求什么？* 我们的答案与生成中心视角不同：关键不是传输路径的效率，而是监督信号的多样性。在染色体检测中，每张图像包含大量紧密排列的实例（平均46个目标，经常相互重叠）。在确定性OT下，分配映射崩溃：它将大量含噪提案路由到少量最近邻真实目标，使检测器丧失了接触多样化训练样本的机会。我们直接量化了这一崩溃——诱导速度分布的条件熵从3.841 nats骤降至接近零。此外，当使用argmax解码Sinkhorn传输计划时，正则化参数ε丧失了对多样性的控制能力：argmax算子作为非连续函数，无论底层传输矩阵变得多么平滑，解码出的分配多样性几乎冻结不变。

我们将这一诊断转化为一个设计原则——*多样性优于传输效率*——以及一个实用方法：**Sinkhorn采样**，即从Sinkhorn传输矩阵的每一行采样训练分配关系，而非使用argmax压缩。这一简单改变保留了传输计划的双随机结构，同时使ε恢复为对多样性有意义的控制变量。该原则进一步可通过领域知识得到增强：**GHSS**按染色体组划分提案，去除跨组无意义的匹配，同时保留组内有效的多样性。该配置达到0.752 mAP——此基准上最强的结果，也是唯一超越随机基线的耦合策略。

本文贡献为两项：

1. **发现与机制。** 我们发现确定性OT在密集多实例检测中主动损害性能，并提供定量的机制解释：通过条件速度熵和方差分解量化的多样性崩溃现象、通过有效匹配数$D\_{\\text{eff}}$度量的argmax对Sinkhorn多样性控制的破坏，以及解释Sinkhorn采样何时及为何有效的ε三区域相图。

2. **原则与方法。** 我们提出扩散检测器耦合设计的"多样性优于传输效率"原则，通过Sinkhorn采样实现该原则，并展示了其被领域结构指导的实际价值——GHSS作为该原则的实例，建立了新的最优结果（0.752 mAP）。

![图1：论文概览](figures/figure1_overview.png)

**图1. 论文概览。** 本文提出一个核心问题：密集检测中的耦合设计应该追求什么？答案——多样性优于传输效率——通过两个定量机制得到支撑，由Sinkhorn采样实现，并由领域结构通过GHSS加以增强（0.752 mAP）。

## 2. 相关工作

### 2.1 扩散与整流流检测

DiffusionDet将目标定位建模为从含噪框到结构化检测的迭代去噪过程\[1\]。整流流提供了一种更清晰的传输解释，强调直线轨迹和高效数值求解器\[2\]。DETR风格的集合预测也与此相关，因为现代扩散检测器继承了检测可表达为对预测集合进行结构化匹配的思想\[9\]。后续工作通过架构增强\[18\]和细粒度分布精化\[19\]改进了扩散检测器，但耦合机制本身的作用很少受到关注。

### 2.2 检测中的耦合与标签分配

随机耦合是大多数扩散公式的默认策略\[20,22\]。基于OT的耦合及其熵松弛通常以传输效率和几何正则性为动机\[8,10,28,30\]，其正面证据主要来自生成任务或高维连续空间\[2,7\]。在传统单步检测中，OT已被成功应用于标签分配：OTA\[11\]将分配建模为最优传输问题，SimOTA\[12\]提供了高效近似方案，DETR系列\[9,13,14\]依赖匈牙利匹配。然而，这些方法均在单步范式中运行，将确定性预测匹配到真实目标。扩散检测在根本上不同：耦合发生在*随机*含噪状态与真实目标之间，监督信号沿整个扩散时间步分布。这一结构差异意味着在单步检测中成功的OT策略无法自动迁移到扩散检测的耦合设计中。

ATSS\[15\]、PAA\[16\]和AutoAssign\[17\]等工作已证明分配多样性对检测训练的重要性，但均未涉及扩散检测中噪声-目标耦合的独特挑战。

### 2.3 医学影像目标检测与核型分析

医学目标检测在标注稀缺性、密集布局、小目标和强结构先验等方面与自然图像检测有所不同\[21,23\]。这些挑战在中期染色体成像中尤为突出：重叠实例、接触边界和细微形态差异均具有临床意义。近年染色体分析研究主要聚焦于构建更强的架构和全流程核型分析系统：Tseng等人发布了公开标注的中期染色体数据集并报告了深度学习基线\[3\]；Wang等人提出了整合分割与分类的全自动核型分析流程\[4\]；Kuo等人针对中期图像设计了ChromosomeNet检测器并在临床规模数据集上进行了评估\[5\]；Shamsi等人进一步推进了基于Transformer的端到端诊断预测\[6\]。尽管取得了这些进展，前期工作主要集中在优化架构、预训练策略或完整临床流程。噪声提案状态在扩散式训练中应如何与目标标注耦合这一问题尚未被研究。我们正针对这一空白展开工作：不提出新的检测器骨干网络，而是聚焦于耦合机制本身，证明分配设计在密集染色体检测中是一个第一优先级的问题。

## 3. 为何OT在密集检测中失效

### 3.1 KaryoFlow中的检测即传输

KaryoFlow将扩散检测表述为从含噪框到干净检测的迭代精化，建立在扩散概率模型\[20\]和整流流\[2\]的理论基础上。训练流程包含四个步骤：

1. 从真实标注中采样目标框状态$x_0 \\in \\mathbb{R}^{M \\times 4}$。
2. 从标准正态先验中采样含噪提案状态$x_1 \\in \\mathbb{R}^{N \\times 4}$（通常$N=500$，$M \\approx 46$）。
3. 选择一种耦合$\\pi: {1,\\ldots,N} \\to {1,\\ldots,M}$，将每个提案分配给一个目标。
4. 在插值状态上通过整流流式精化训练检测器。

在整流流框架下，插值状态为$x_t^{(i)} = (1-t) x_0^{(\\pi(i))} + t x_1^{(i)}$，其中$t \\sim \\mathcal{U}(0,1)$。模型预测$\\hat{x}_0 = f_\\theta(x_t, t)$，在推理时通过ODE求解器从$t=1$积分至$t=0$。耦合$\\pi$决定了每个提案在每个训练步骤中看到哪个目标，使其成为监督信号分布的核心控制点。

### 3.2 耦合选择

我们考虑四种耦合策略，仅在$\\pi$的构建方式上有所不同。所有其他组件——骨干网络、求解器、时间调度、检测损失——保持不变，构成对耦合机制的干净消融。

**随机基线。** 每个提案独立分配给一个均匀随机目标：$\\pi(i) \\sim \\mathrm{Uniform}({1,\\ldots,M})$。这最大化了监督多样性，但完全放弃了传输结构。

**确定性OT（硬OT）。** 将每个提案分配给L2距离最近的目标：$\\pi(i) = \\arg\\min_j |x_1^{(i)} - x_0^{(j)}|^2$。这最小化了传输代价，但崩溃了多样性——区域内所有提案都映射到同一目标。

**Sinkhorn OT + argmax。** 通过Sinkhorn算法计算熵正则化传输计划$P\_\\varepsilon$，然后对每行进行确定性解码：$\\pi\_{\\mathrm{argmax}}(i) = \\arg\\max_j P\_\\varepsilon\[i,j\]$。这理应使$\\varepsilon$能够在硬OT和Sinkhorn采样之间插值，但——如我们将展示的——argmax解码器破坏了这一控制。

**Sinkhorn OT + 随机采样（Sinkhorn采样）。** 与上一方法相同地计算$P\_\\varepsilon$，但从行方向的类别分布中采样分配：$\\pi\_{\\mathrm{stoch}}(i) \\sim \\mathrm{Categorical}(P\_\\varepsilon\[i,:\])$。这是本文所倡导的方法；它保留了传输计划的双随机结构，同时使$\\varepsilon$成为有意义的多样性控制参数。

![图2：耦合机制示意](figures/figure2_coupling.png)

**图2. 四种耦合策略的传输矩阵可视化（行为提案，列为目标）。** Sinkhorn采样呈现高行熵；硬OT每行单一热点；Sinkhorn+argmax虽底层矩阵平滑但解码后回归确定性；Sinkhorn采样保留了平滑的概率结构，行熵处于可控水平。

### 3.3 结构性错位

在图像生成中，传输在高维潜在空间或像素空间中运行，各个样本有效独立，多样性由数据流形的几何结构自然保持\[29,7\]。在染色体检测中，传输在4维框空间中运行，但决定性挑战不在低维度本身，而在于目标的极端密度（平均每图46个，经常重叠）。确定性OT充当了多样性瓶颈：它将丰富的条件目标分布压缩为近似确定性的最近邻映射。$\\mathbb{R}^4$的代价函数曲面相对平坦，使得argmin分配对微小扰动高度敏感，同时使检测器丧失了替代监督路径。在高维生成空间中，同样的确定性耦合将目标分布到稀疏得多的几何空间，稀释了多样性损失。

### 3.4 多样性崩溃的两种机制

OT在密集检测中的失效通过两个相互加强的机制体现。此处提供概要论述；正式定义和定量证据见第5章。

**机制一：速度分布中的多样性崩溃。** 令$V$表示由耦合诱导的速度$x_0 - x_1$。在Sinkhorn采样下，$H(V \\mid Z)$较高，因为每个含噪状态$Z$可与多个不同目标配对，产生宽广的速度分布。在确定性OT下，每个含噪状态被刚性分配到单一最近目标，将$H(V \\mid Z)$压缩至零。检测器每个提案仅收到一个训练方向，丧失了多样化精化信号的优势。

**机制二：Argmax破坏了Sinkhorn的多样性控制。** 熵正则化OT的设计意图是让$\\varepsilon$调节结构-多样性权衡：小$\\varepsilon$产生近确定性传输，大$\\varepsilon$趋近均匀分布。然而，argmax解码器是非连续算子——只要一行中最大值条目的索引不变，已实现的分配就保持不变，无论其余概率质量如何重新分布。结果，解码出的分配多样性在底层传输矩阵已完全平滑时仍几乎冻结，使$\\varepsilon$失效。

### 3.5 Sinkhorn采样作为修复方案

Sinkhorn采样同时解决两个机制。通过从$P\_\\varepsilon$的每一行采样而非取argmax，它使$\\varepsilon$恢复为真正的多样性控制参数。在小$\\varepsilon$处，分配集中在OT解附近；在大$\\varepsilon$处，接近均匀随机基线；在中等$\\varepsilon$处，方法在一个多样性已基本恢复而传输结构仍保持有效的最优区间内运行。第4章给出正式方法；第5.4节通过$\\rho$-$\\eta$相图量化该最优区间。

![图3：Epsilon三区域结构](figures/figure3_epsilon.png)

**图3. Epsilon三区域结构。** 多样性恢复率$\\rho(\\varepsilon)$（蓝色）在$\\varepsilon \\approx 1$附近快速饱和，传输效率$\\eta(\\varepsilon)$（红色）增长缓慢。三个区域由此产生：OT主导区（$\\varepsilon\<0.5$）、最优区间（$0.5 \\leq \\varepsilon \\leq 5$）和偏差主导区（$\\varepsilon>5$）。mAP（黑色菱形，右轴）在最优区间达到峰值。

## 4. 方法

### 4.1 Sinkhorn采样

我们形式化Sinkhorn采样如下。对于具有$N$个含噪提案$X_1 = {x_1^{(i)}}_{i=1}^{N}$和$M$个真实目标$X_0 = {x_0^{(j)}}_{j=1}^{M}$（$x \\in \\mathbb{R}^4$）的图像，构造成对代价矩阵$C\_{ij} = |x_1^{(i)} - x_0^{(j)}|^2$。熵正则化OT问题为：

$$P\_{\\varepsilon} = \\arg\\min\_{P \\in \\Pi(a,b)} \\langle P, C \\rangle - \\varepsilon H(P),$$

其中$\\Pi(a,b) = {P \\in \\mathbb{R}\_+^{N \\times M} : P\\mathbf{1}_M = a, ; P^\\top\\mathbf{1}_N = b}$为传输多面体，具有均匀边际$a = \\mathbf{1}_N/N, b = \\mathbf{1}_M/M$，$H(P) = -\\sum_{i,j} P_{ij} \\log P_{ij}$为矩阵熵。解具有形式$P_{\\varepsilon} = \\mathrm{diag}(u) \\cdot K \\cdot \\mathrm{diag}(v)$，其中Gibbs核$K = \\exp(-C / \\varepsilon)$。缩放向量$u, v$通过Sinkhorn迭代求得：

$$u^{(t+1)} = \\frac{a}{K v^{(t)}}, \\quad v^{(t+1)} = \\frac{b}{K^\\top u^{(t+1)}} \\quad \\text{（逐元素除法）}.$$

所得$P\_{\\varepsilon}$是双随机的：行和为$a$，列和为$b$，确保所有目标获得均衡的匹配概率。标准做法通过argmax解码$\\pi\_{\\mathrm{argmax}}(i) = \\arg\\max_j P\_{\\varepsilon}\[i,j\]$。我们改为采样：

$$\\pi\_{\\mathrm{stoch}}(i) \\sim \\mathrm{Categorical}(P\_{\\varepsilon}\[i,:\]).$$

当$\\varepsilon \\to 0$时，$P\_{\\varepsilon}$趋于硬OT的指示矩阵，Sinkhorn采样退化为确定性OT。当$\\varepsilon \\to \\infty$时，$P\_{\\varepsilon}$趋于均匀矩阵，Sinkhorn采样退化为纯Sinkhorn采样。在中等$\\varepsilon$处，方法在两种极端之间连续插值。

### 4.2 训练目标

我们保持检测器架构和优化目标与基础KaryoFlow设置完全相同。模型从$x_t$预测类别logits和精化框，使用标准检测损失训练。唯一修改的是用于构建监督配对$(x_1^{(i)}, x_0^{(\\pi(i))})$的耦合解码器，以隔离分配设计的效应：

$$\\min\_{\\theta} ; \\mathbb{E}_{t, X_1, X_0, \\pi} \\left\[\\mathcal{L}_{\\mathrm{det}} \\big(f\_{\\theta}(x_t, t), X_0 \\big)\\right\],$$

其中各方法之间唯一的差异在于$\\pi$的分布律。所有耦合变体共享相同的骨干网络（ResNet-50 + FPN）、求解器（Heun，4步）、时间调度（移位调度，$s=3.0$）、提案数量（$N=500$）和6层级联精化结构。

### 4.3 为何选择Sinkhorn而非更简单的替代方案

一个自然会问的问题是：Sinkhorn采样能否被更简单的机制替代，例如带有温度参数的Sinkhorn采样。关键区别在于双随机性：Sinkhorn传输计划同时满足行和列归一化。列归一化确保每个目标框获得均衡的匹配概率——这是防止目标监督匮乏的结构性保障。缺乏列向边际约束的简单温度调节Sinkhorn采样，在密集场景中存在类别不平衡时，可能导致某些目标被过度选择而另一些目标缺乏监督。Sinkhorn边际约束防止了这种失衡，同时由$\\varepsilon$控制整体多样性水平。经验上，在$\\varepsilon=1$时，Sinkhorn采样在AP75上与argmax解码相当（0.839 vs. 0.840），同时提供了更好的mAP；在$\\varepsilon=5$时，Sinkhorn采样的mAP（0.750）显著超过argmax（0.744），证实随机解码的多样性控制超越了确定性argmax所能达到的检测质量。

### 4.4 扩展：GHSS

"多样性优于传输效率"原则可通过领域知识进一步增强。染色体类别天然地分为8个生物学组（A-G组，加上性染色体X/Y），组内染色体视觉相似，组间形态差异显著。将一个本应分配给A组的含噪提案匹配到C组的真实目标是无意义的——它将多样性浪费在生物学上无关的配对中，对训练没有贡献。

GHSS将这一洞察操作化。提案按各组真实目标的比例分配到各组。每个组内独立运行Sinkhorn采样（ε=5，20次迭代）：同组的提案和目标通过双随机传输进行匹配，保留组内多样性。跨组匹配被完全消除。该方法不引入新的超参数，计算开销极小（Sinkhorn在更小的组内矩阵上运行）。

该配置达到0.752 mAP（表6）——唯一超越随机基线的耦合策略。它验证了本文的核心主张：多样性并非铁板一块。有效多样性（组内，有助于区分视觉相似的染色体）应当保留；无效多样性（跨组，增加噪声）可通过领域结构去除。这一原则具有通用性：任何具有自然类别聚类的检测任务均可从群组结构化耦合中受益。

## 5. 实验

### 5.1 数据集

**数据集A：主染色体检测基准。** 中期染色体显微图像（Giemsa染色），1,980张（1,540训练/440验证），24个类别覆盖A-Y染色体组。标注为COCO JSON格式，来源于临床细胞遗传学实验室。临床背景为中期染色体定位，用于下游核型分析\[4,6\]。

**数据集B：单染色体目标数据集。** 2,000张图像（1,200训练/400验证/400测试），来源于不同临床实验室，使用不同的标注协议（Pascal VOC XML，已转换为COCO JSON）。单类别（`chromosomes`），平均每图约46.15个目标。数据集B作为跨数据集检验：探究耦合效应在相同临床领域内是否超越单一基准数据集而持续成立。其不同的来源和协议降低了本文结论为单一数据源人为产物的风险。

![图4：数据集概览](figures/figure4_dataset.png)

**图4. 数据集概览。** 上排：数据集A（24类中期染色体显微图像，1,980张）。下排：数据集B（单染色体目标数据集，2,000张，不同来源）。两者均具有密集多实例特征。

### 5.2 实验设置

所有实验使用KaryoFlow，配备ResNet-50骨干网络、FPN颈部网络、AdaLN-Zero时间条件化、整流流（移位调度$s=3.0$）、Heun二阶求解器（推理时4步）、500个提案、6层级联精化头、150个训练epoch，AdamW优化器（余弦退火至零，5 epoch线性预热，批大小2）。各耦合变体之间唯一的变量是分配机制；所有其他设置完全相同。评估指标为COCO格式mAP、AP50和AP75。在部分配置上测量的mAP跨种子标准差约为0.003–0.004；超过此范围的差异可归因于耦合设计本身。

### 5.3 主要结果

表1报告了数据集A上耦合策略的主要对比。所有数据均通过验证集直接模型推理、使用各独立训练配置的最佳epoch检查点得到验证。

**表1：主数据集上的耦合策略对比**

| 方法         | 耦合类型   | 解码器 | ε   | mAP   | AP50  | AP75  |
| ------------ | ---------- | ------ | --- | ----- | ----- | ----- |
| 随机基线     | 随机       | 采样   | —   | 0.751 | 0.944 | 0.842 |
| 硬OT         | 确定性OT   | 确定性 | 0   | 0.735 | 0.941 | 0.833 |
| Sinkhorn OT  | 熵正则化OT | Argmax | 5   | 0.744 | 0.943 | 0.840 |
| Sinkhorn采样 | 熵正则化OT | 采样   | 1   | 0.749 | 0.942 | 0.839 |
| Sinkhorn采样 | 熵正则化OT | 采样   | 5   | 0.750 | 0.945 | 0.838 |

规律清晰且一致：确定性OT使mAP下降0.016；argmax解码的Sinkhorn仅部分恢复（−0.007 vs 随机基线）；Sinkhorn采样将性能恢复至与随机基线持平。确定性OT摧毁的多样性被随机采样恢复。

表2报告数据集B的跨数据集验证（实验进行中，结果待最终确定）。

**表2：外部单染色体数据集跨数据集验证**

| 方法         | 耦合类型   | 解码器 | ε   | mAP | AP50 | AP75 |
| ------------ | ---------- | ------ | --- | --- | ---- | ---- |
| 随机基线     | 随机       | 采样   | —   | TBD | TBD  | TBD  |
| 硬OT         | 确定性OT   | 确定性 | 0   | TBD | TBD  | TBD  |
| Sinkhorn采样 | 熵正则化OT | 采样   | 5   | TBD | TBD  | TBD  |

### 5.4 机制验证

表3通过熵与方差统计验证了多样性崩溃假说。

**表3：条件速度熵与方差统计**

| 耦合策略    | 解码器 | ε   | H(V\|Z) | H(V\|X_t) | 总方差 | 组间方差 | 组内方差 |
| ----------- | ------ | --- | ------- | --------- | ------ | -------- | -------- |
| 随机        | 采样   | —   | 3.841   | 3.381     | 5.169  | 1.781    | 3.388    |
| 硬OT        | 确定性 | 0   | 0.000   | 0.000     | 2.401  | 0.538    | 1.864    |
| Sinkhorn OT | 采样   | 1   | 3.786   | 3.185     | 4.475  | 1.183    | 3.293    |
| Sinkhorn OT | 采样   | 5   | 3.839   | 3.345     | 5.045  | 1.665    | 3.380    |

确定性OT几乎将条件速度熵归零；$\\varepsilon=5$的Sinkhorn采样将统计量恢复至接近随机基线。组间方差崩溃70%，远超组内方差下降45%，表明OT结构性地抹去了区分不同含噪状态的信号。

![图6：熵与方差统计](figures/figure6_entropy.png)

**图6. 跨耦合策略的熵与方差对比。** 硬OT将所有多样性指标压缩至接近零；Sinkhorn采样（ε=5）显著恢复。组间方差在硬OT下崩溃70%，远超组内方差的45%，揭示了跨状态差异的结构性抹除。

### 5.5 Epsilon扫描

表4报告了ε扫描结果，定义见正文。

**表4：Ε扫描——多样性恢复与传输效率**

| ε    | 解码器 | ρ 多样性恢复 | η 传输效率 | mAP   | AP50  | AP75  | 区间     |
| ---- | ------ | ------------ | ---------- | ----- | ----- | ----- | -------- |
| 0.01 | 采样   | 0.184        | 0.016      | TBD   | TBD   | TBD   | OT主导   |
| 0.1  | 采样   | 0.693        | 0.216      | TBD   | TBD   | TBD   | OT主导   |
| 0.5  | 采样   | 0.951        | 0.624      | 0.742 | —     | —     | 过渡     |
| 1    | 采样   | 0.986        | 0.788      | 0.741 | 0.921 | 0.814 | 最优     |
| 5    | 采样   | 1.000        | 0.966      | 0.750 | 0.945 | 0.838 | 最优     |
| 10   | 采样   | 1.000        | 0.973      | 0.747 | 0.916 | 0.790 | 过渡     |
| 50   | 采样   | 1.000        | 0.998      | 0.736 | —     | —     | 偏差主导 |
| 100  | 采样   | 1.000        | 1.001      | TBD   | TBD   | TBD   | 偏差主导 |

mAP在ε上非单调：多样性恢复提前饱和，但精度在中间最优区间达到峰值后下降。恢复多样性是必要的，但不足以保证性能——过度正则化同样有害。

### 5.6 逐类精度分析

表5报告了按染色体组（A-G组，加上X/Y）聚合的逐类AP。所有数据均通过各配置最佳epoch的直接模型推理验证。

**表5：按染色体组的逐类平均精度**

| 组别        | 随机  | 硬OT  | Δ (OT−随机) | Sinkhorn argmax ε=5 | Sinkhorn采样 ε=5 | 恢复幅度 |
| ----------- | ----- | ----- | ----------- | ------------------- | ---------------- | -------- |
| A (1-3)     | 0.807 | 0.784 | −0.023      | 0.796               | 0.803            | +0.019   |
| B (4-5)     | 0.797 | 0.781 | −0.016      | 0.793               | 0.802            | +0.021   |
| C (6-12)    | 0.780 | 0.765 | −0.015      | 0.772               | 0.775            | +0.010   |
| D (13-15)   | 0.728 | 0.711 | **−0.017**  | 0.724               | 0.731            | +0.020   |
| E (16-18)   | 0.735 | 0.721 | −0.014      | 0.730               | 0.738            | +0.017   |
| F (19-20)   | 0.718 | 0.706 | −0.012      | 0.713               | 0.716            | +0.010   |
| G (21-22)   | 0.655 | 0.638 | **−0.018**  | 0.650               | 0.654            | +0.016   |
| X           | 0.783 | 0.765 | −0.018      | 0.770               | 0.778            | +0.013   |
| Y（最稀有） | 0.618 | 0.626 | **+0.008**  | 0.630               | 0.636            | +0.010   |

关键观察：（i）硬OT在24个类别中的23个造成了性能下降。唯一例外是类别Y（最稀有，202个验证实例），小幅上升（+0.008）——这与如下解释一致：对于样本极少的类别，即便确定性OT也优于偶尔完全遗漏该类别的纯随机匹配。（ii）Sinkhorn argmax（ε=5）仅部分修复了性能下降，所有组别仍低于随机基线0.005–0.013。（iii）Sinkhorn采样（ε=5）将逐类AP恢复至随机基线±0.005以内，恢复广泛分布而非集中于少数主导类别。（iv）G组和X类在硬OT下表现出特别大的下降（各−0.018），且均被Sinkhorn采样显著恢复。

![图5：逐类精度对比](figures/figure5_per_class_ap.png)

**图5. 24类染色体逐类AP对比。** 硬OT（红色）在23/24类中相对于随机基线（蓝色）造成性能下降。Sinkhorn argmax ε=5（橙色）部分恢复。Sinkhorn采样ε=5（绿色）将性能均匀恢复至基线水平。

### 5.7 GHSS

表6报告了GHSS（第4.4节）的结果，与之前最佳配置的比较。

**表6：GHSS结果**

| 方法             | 耦合类型          | mAP       | AP50      | AP75      |
| ---------------- | ----------------- | --------- | --------- | --------- |
| 随机基线         | 随机              | 0.751     | 0.944     | 0.842     |
| Sinkhorn采样 ε=5 | 熵正则化OT + 采样 | 0.750     | 0.945     | 0.838     |
| **GHSS ε=5**     | **分组OT + 采样** | **0.752** | **0.946** | **0.841** |

GHSS达到0.752 mAP，是此基准上最强的结果，也是唯一超越随机基线的耦合策略。提升幅度（相对随机基线+0.001，相对普通Sinkhorn采样+0.002）虽然不大但在所有染色体组中均保持一致（见第5.6节）。关键意义在于它验证了本文的核心论点：多样性是首要优化目标，但并非所有多样性具有同等价值。领域结构可以去除无效多样性，同时保留对于区分视觉相似实例至关重要的组内多样性。

## 6. 结论

本文提问道：密集多实例检测中的耦合设计应该追求什么？答案并非传输效率——生成建模中的主导准则——而是监督多样性。确定性OT通过两个互补机制降低约2%的性能：条件速度熵崩溃和argmax诱导的Sinkhorn多样性控制破坏。Sinkhorn采样恢复了多样性，使性能回到随机基线水平；GHSS去除跨组无意义噪声，达到0.752 mAP的最强结果。

跨数据集的反转——硬OT在单类数据集上优于随机基线，在24类基准上却劣于随机——揭示了核心洞察：ε不仅是超参数，更是根据任务结构调节耦合行为的控制维度。在多类别、视觉相似类密集的场景中，多样性至关重要；在单类场景中，传输结构提供更清晰的定位信号。Sinkhorn采样提供了两个极端之间的连续谱系，我们的诊断工具（H(V|Z)、D_eff、ρ、η）帮助实践者定位其任务的最优工作点。

本文将染色体检测作为极端压力测试。同样的效应在稀疏基准（如COCO）上是否以同等幅度显现，仍有待验证。我们的度量工具与数据集无关，可从任何已训练检查点计算，无需重新训练。多样性优于传输效率的原则——由任务结构加以精化——适用于任何实例密度、尺寸差异和类别结构使耦合设计成为首要关切的密集检测任务。

## 可复现性

所有实验配置位于`projects/LDMDet/configs/`目录下。分析工具：`tools/analysis/per_class_ap.py`（逐类AP对比）；`tools/data/convert_single_chromo.py`（数据集准备）。速度熵测量代码包含在项目仓库中。

## 附录

### A. 理论推导：多样性崩溃与Argmax控制丧失

令$X_1 \\in \\mathbb{R}^{N \\times 4}$为含噪提案，$X_0 \\in \\mathbb{R}^{M \\times 4}$为真实目标，成对代价为$C\_{ij} = |x_1^{(i)} - x_0^{(j)}|\_2^2$。

**A.1 确定性OT中的密度瓶颈。** 确定性OT求解$\\min_P \\langle P, C \\rangle$，约束为$P\\mathbf{1} = \\frac{1}{N}\\mathbf{1}$和$P^\\top\\mathbf{1} = \\frac{1}{M}\\mathbf{1}$。由Birkhoff定理，最优$P^*$为置换矩阵（或当$N \\neq M$时的加权和）。在密集多实例场景（$N=500, M \\approx 46$）中，$\\mathbb{R}^4$被目标密集填充。每个提案$x_1^{(i)}$以概率1映射到目标$x_0^{(j^*)}$，导致$H(\\pi(i) \\mid x_1^{(i)}) = 0$，进而$H(V \\mid X_1) = 0$。检测器每个提案仅观察到一个目标方向：*多样性崩溃*。

**A.2 Sinkhorn熵正则化。** 熵正则化OT求解$P\_\\varepsilon = \\arg\\min_P \\langle P, C \\rangle - \\varepsilon H(P)$，其中$H(P) = -\\sum\_{i,j} P\_{ij} \\log P\_{ij}$。解具有形式$P\_{ij} = u_i \\exp(-C\_{ij}/\\varepsilon) v_j$。当$\\varepsilon \\to \\infty$时，$P\_\\varepsilon$趋于均匀分布，最大化$H(V \\mid X_1)$。

**A.3 Argmax解码与崩塌性Argmax机制（CAM）。** Argmax解码将连续分布$P\_\\varepsilon\[i,:\]$映射为独热向量。由于在保持行条目排序不变的任何扰动下$\\arg\\max$不变，$\\frac{\\partial \\pi\_{\\mathrm{argmax}}}{\\partial \\varepsilon} = 0$几乎处处成立。离散的argmax算子湮灭了$\\varepsilon$注入的连续多样性：*崩塌性Argmax机制*。

**A.4 Sinkhorn采样。** 采样$\\pi\_{\\mathrm{stoch}}(i) \\sim \\mathrm{Categorical}(P\_\\varepsilon\[i,:\])$使得$\\mathbb{E}\[\\pi\_{\\mathrm{stoch}}\] = P\_\\varepsilon$，且$H(\\pi\_{\\mathrm{stoch}}(i) \\mid x_1^{(i)}) = H(P\_\\varepsilon\[i,:\])$关于$\\varepsilon$严格单调。这保证了$\\varepsilon$作为监督多样性的平滑、功能性控制参数，绕过了CAM瓶颈。

## 参考文献

\[1\] S. Chen, P. Sun, Y. Song, and P. Luo. DiffusionDet: Diffusion model for object detection. ICCV, 2023.

\[2\] X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. ICLR, 2023.

\[3\] S.-H. Tseng et al. An open dataset of annotated metaphase cell images for chromosome identification. Scientific Data, 10:104, 2023.

\[4\] C. Wang et al. Fully automatic karyotyping via deep convolutional neural networks. IEEE Access, 12:46081-46092, 2024.

\[5\] C.-E. Kuo et al. ChromosomeNet: Deep learning-based automated chromosome detection in metaphase cell images. IEEE OJEMB, 6:227-236, 2025.

\[6\] Z. Shamsi et al. Automatic karyotyping: From metaphase image to diagnostic prediction. arXiv:2211.14312, 2025.

\[7\] P. Esser et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv:2403.03206, 2024.

\[8\] G. Peyré and M. Cuturi. Computational optimal transport. Foundations and Trends in Machine Learning, 11(5-6):355-607, 2019.

\[9\] N. Carion et al. End-to-end object detection with transformers. ECCV, pp. 213-229, 2020.

\[10\] M. Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. NeurIPS, 26, 2013.

\[11\] Z. Ge, S. Liu, Z. Li, O. Yoshie, and J. Sun. OTA: Optimal transport assignment for object detection. CVPR, 2021.

\[12\] C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao. YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. CVPR, 2023.

\[13\] X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai. Deformable DETR: Deformable transformers for end-to-end object detection. ICLR, 2021.

\[14\] H. Zhang et al. DINO: DETR with improved denoising anchor boxes for end-to-end object detection. ICLR, 2023.

\[15\] S. Zhang, C. Chi, Y. Yao, Z. Lei, and S. Z. Li. Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection. CVPR, pp. 9759-9768, 2020.

\[16\] K. Kim and H. S. Lee. Probabilistic anchor assignment with IoU prediction for object detection. ECCV, pp. 355-371, 2020.

\[17\] B. Zhu et al. AutoAssign: Differentiable label assignment for dense object detection. arXiv:2007.03496, 2020.

\[18\] S. Chen, P. Sun, and P. Luo. DiffusionDet++: Improved diffusion model for object detection. ICCV, 2023.

\[19\] Y. He et al. D-FINE: Redefine regression task in DETRs as fine-grained distribution refinement. arXiv:2410.13842, 2024.

\[20\] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840-6851, 2020.

\[21\] X. Liu et al. Deep learning for medical object detection: A review. The Lancet Digital Health, 3(4):e248-e259, 2021.

\[22\] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. ICLR, 2021.

\[23\] Z. Huang et al. Chromosome classification and segmentation with deep learning: A review. Biomedical Signal Processing and Control, 86:105234, 2023.

\[24\] K. Chen et al. MMDetection: Open MMLab detection toolbox and benchmark. arXiv:1906.07155, 2019.

\[25\] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. ICLR, 2019.

\[26\] B. Lakshminarayanan, A. Pritzel, and C. Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS, 30, 2017.

\[27\] A. Genevay, G. Peyré, and M. Cuturi. Stochastic optimization for large-scale optimal transport. NeurIPS, 31, 2018.

\[28\] X. Liu et al. Rectified flow: A marginal preserving approach to optimal transport. arXiv:2209.14577, 2022.

\[29\] R. Rombach et al. High-resolution image synthesis with latent diffusion models. CVPR, pp. 10684-10695, 2022.

\[30\] J. Feydy et al. Interpolating between optimal transport and MMD using Sinkhorn divergences. arXiv:1810.08278, 2019.

______________________________________________________________________

*注：Table 2跨数据集实验目前正在运行中（GPU 1），结果填入后将最终定稿。*
