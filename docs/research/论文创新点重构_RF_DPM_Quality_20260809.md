# KaryoFlow 创新点重构：RF、DPM-Solver++ 与定位质量校准排序

> 更新日期：2026-08-09  
> 证据原则：只把具有可追溯配置、checkpoint、日志和公平对照的数据写成方法收益；
> 单 seed 曲线统计或增强策略混杂结果不得作为核心精度贡献。

## 1. 修正后的三项核心创新

### 1.1 Rectified Flow 检测建模：主要精度来源

第一项贡献是把 DiffusionDet 的 DDPM 去噪过程改写为低维框空间中的 Rectified
Flow。设噪声框为 $x_1$、GT 框为 $x_0$，线性条件路径为

$$x_t=(1-t)x_0+t x_1,$$

并学习 data-prediction 形式的 $\hat x_0(x_t,t,I)$。相对 DDPM，RF 的主要优势不是
“用了 flow matching”这一名称，而是目标路径曲率更低，使少步离散积分的全局误差
不再被复杂随机轨迹主导。在速度场足够光滑时，$p$ 阶求解器的全局误差满足

$$\|x(0)-x_N\|\le C\,h^p\max_t\|x^{(p+1)}(t)\|,$$

RF 通过降低高阶时间导数项而降低误差常数。项目的 solver×step 解耦结果表明，
Dataset 2 的主要增益来自 RF 公式本身，而不是 AdaLN-Zero 或 shifted schedule。

可信收益口径：

- Dataset 1：纯 RF 主线约 $0.746$ vs DDPM $0.729$，约 **+0.017 mAP**；
- Dataset 2：RF $0.856$ vs DDPM $0.803$，约 **+0.053 mAP**；
- 这是当前三项中最大的绝对精度来源。

### 1.2 RF 适配的 DPM-Solver++：主要效率来源

第二项贡献是把 DPM-Solver++ 的 data-prediction 多步形式适配到 RF 框轨迹。它利用
历史 $\hat x_0$ 对时间导数作多步近似，在每个 solver step 只需一次新的网络评估；
Heun 的 predictor-corrector 则除首末边界外需要约两次评估。

因此 4-step 设置中：

$$\mathrm{NFE}_{\mathrm{DPM++}}=4,\qquad
\mathrm{NFE}_{\mathrm{Heun}}=7,$$

理论计算量比为 $7/4=1.75$。ross RTX A6000 的统一基准为约 12.9 FPS。必须修正的
表述是：**DPM-Solver++ 的坚实贡献是以更少 NFE 匹配精度，而不是一个稳定的大幅
mAP 增益**。匹配步数、同 checkpoint 的 aggregate mAP 差异约在 $\pm0.001$ 噪声内；
跨 checkpoint 的逐图像检验曾得到 +0.006，但不能与同 checkpoint 因果消融混写。

### 1.3 Localization-Quality Calibrated Ranking（LQCR）：新的独立精度创新

第三项贡献由原 Stochastic Coupling 替换为 **Localization-Quality Calibrated
Ranking（LQCR）**。代码中的实现来源为 CAPR-C2 quality-only 分支，但论文命名不再
使用尚未实现的 canonical-axis/profile 含义。

检测器的分类分数 $p_j=P(C_j=1\mid F_j)$ 主要表达类别正确性，却不保证与框的定位
质量 $U_j=\operatorname{IoU}(b_j,b_j^*)$ 单调。COCO 在多个 IoU 阈值
$\tau\in\{0.50,\ldots,0.95\}$ 上计算 AP；对阈值 $\tau$，预测成为真正例的事件是

$$Z_{j,\tau}=\mathbf 1[C_j=1,\ U_j\ge\tau].$$

由 probability-ranking principle，在固定 $\tau$ 下，最大化期望排序质量应按后验

$$\pi_{j,\tau}=P(Z_{j,\tau}=1\mid F_j)
=P(C_j=1\mid F_j)P(U_j\ge\tau\mid C_j=1,F_j)$$

排序。LQCR 用末级 proposal feature 预测连续定位质量
$q_j\approx E[U_j\mid C_j=1,F_j]$，并采用

$$s_j=p_j q_j^\beta,\qquad \beta=2$$

作为跨阈值后验的低成本代理。当条件 IoU 分布属于由 $q_j$ 排序的单参数、随机单调
族时，$q_j$ 同时保持各 $P(U_j\ge\tau\mid F_j)$ 的排序，因而该融合是 Bayes-consistent
的排序代理。一般分布下它不是“严格最大化 COCO AP”的定理，只是有明确概率排序
依据的可学习近似；这一边界必须在论文中保留。

实现采用严格 `final_only` 因果隔离：solver、box renewal、Top-K 和 GACS 始终读取
原始分类分数，$p q^2$ 只用于最终输出排序，不改变类别、框坐标或扩散轨迹。因此

$$\Delta_{\mathrm{LQCR}}
=AP(\text{final-only})-AP(\text{A4})$$

可直接归因于定位质量排序。相同权重下 solver-coupled 相对 final-only 仅
+0.00008 mAP，说明总收益不是 proposal 轨迹偶然改变造成。

可信收益口径：

| 数据集/设置 | A4 | LQCR | 增量 | 当前证据状态 |
|---|---:|---:|---:|---|
| Dataset 2，seed42，严格诊断 | 0.86301 | 0.87044 | **+0.00743** | 完整逐 IoU、final-only 因果消融 |
| Dataset 2，quality/data seed123 | 0.863 基座 | best 0.871 | 约 **+0.008** | 同一冻结 A4 的条件复现 |
| Dataset 2，quality/data seed789 | 0.863 基座 | best 0.871 | 约 **+0.008** | 同一冻结 A4 的条件复现 |
| Dataset 1，A4 seed42 | 0.746 | 0.751 | **+0.005** | workstation 训练完成；待 ross 统一复评 |
| Dataset 1，A4 seed123 | 0.748 | 当前 best 0.753 | **+0.005** | 独立基座；训练进行中 |

Dataset 2 seed42 的细粒度变化为 AP90 **+0.03041**、AP95 **+0.03251**，而 AP50
基本不变，符合“改进排序与高 IoU 定位可信度、而非发现更多物体”的机制预期。真实
同类 IoU 重排 oracle 为 +0.0369 mAP，说明当前 quality predictor 只兑现约五分之一
的可用排序空间，仍有明确后续研究余量。

## 2. 三项创新的统一理论脉络

三项贡献分别处理生成式检测误差链的不同位置：

$$
\underbrace{x_1\rightarrow x_t\rightarrow x_0}_{\text{RF：降低轨迹建模误差}}
\quad\xrightarrow{\text{DPM++}}\quad
\underbrace{\hat b_j}_{\text{降低少步积分成本}}
\quad\xrightarrow{\text{LQCR}}\quad
\underbrace{(\hat b_j,c_j,s_j)}_{\text{降低排序错配}}.
$$

- RF 处理训练目标与路径几何；
- DPM-Solver++ 处理给定路径的离散积分效率；
- LQCR 处理框已经生成之后，类别置信度与定位质量不一致造成的 AP 损失。

这三者不是同一机制的重复包装：RF 改变学习的动力系统，DPM-Solver++ 改变数值解法，
LQCR 在不改变动力系统和框坐标的条件下改变最终排序。final-only 消融为第三项与前两项
的独立性提供了直接实验证据。

## 3. Stochastic Coupling 的证据审计与降级

### 3.1 可保留的事实

- 低维 $d=4$、高目标数 $K$ 下，hard OT 分配的条件熵下降是可分析的数学现象；
- Sinkhorn 行采样可在 hard OT 与 random pairing 之间插值并恢复条件熵；
- Dataset 2 单 seed 的最后 30 epoch 曲线中，报告过 std 0.006→0.0013、range
  0.023→0.005 的 **4.6× 平滑性观察**。

### 3.2 不能再作为核心创新收益的声明

- 统一标准增强后，Dataset 1 的 Random/StochOT/Hard OT 三种策略 3-seed 精度差异
  不显著；最新记录约为 0.746/0.747/0.748，均在噪声内。
- Dataset 2 的 3-seed val/test 中 StochOT 相对 Random 约为 −0.002，仍不显著。
- 历史“StochOT +0.034、$p<10^{-120}$”来自标准增强与简单增强混杂，已失效。
- “4.6× 稳定性”只来自单 seed、连续 epoch 的自相关样本；30 个 epoch 不能视为
  30 个独立重复实验，也尚无多 seed 置信区间或最终测试泛化收益闭环。

因此 Stochastic Coupling 应从“三个提升点”中移除。论文中只保留为：

1. OT coupling 的理论/负面分析；
2. 标准增强补偿 diversity collapse 的实证发现；
3. 单 seed 训练平滑性观察，明确标为探索性结果；
4. 不声称提升 mAP、泛化或 EarlyStopping 可靠性，除非后续完成多 seed 因果验证。

## 4. 论文修订口径

摘要、引言、方法导览和结论统一使用以下三项：

1. **Rectified Flow detection formulation**：最大精度来源；
2. **RF-adapted DPM-Solver++**：少 NFE 的效率来源；
3. **Localization-Quality Calibrated Ranking**：当前最大的新增独立精度创新。

Stochastic Coupling 移至耦合分析/附录，不再出现在模型定义
`KaryoFlow = RF + StochOT + DPM++` 中。建议主模型定义改为：

$$\text{KaryoFlow-LQCR}=\text{RF detector}+\text{DPM-Solver++}+\text{LQCR}.$$

现阶段 Dataset 1 的 +0.005 需标注为“训练验证结果，ross 统一复评中”；Dataset 2
seed123/789 是同一冻结 A4 基座上的 quality-head 条件复现，不能写成完整模型独立
三种子。Dataset 1 seed123 当前也得到 +0.005 的独立基座早期结果；待 seed123/789
完成并在 ross paired evaluation 后，才能报告完整
三种子均值、标准差和显著性。

## 5. 数据与代码来源

- A4 诊断：`work_dirs/diagnosis/precision_bottleneck_a4_seed42.json`
- LQCR final-only：
  `work_dirs/diagnosis/precision_bottleneck_capr_quality_final_only_epoch2_seed42.json`
- solver-coupled：
  `work_dirs/diagnosis/precision_bottleneck_capr_quality_epoch2_seed42.json`
- Dataset 2 seed123：`work_dirs/capr_quality_final_only_24obj_seed123/`
- Dataset 2 seed789：`work_dirs/capr_quality_final_only_24obj_seed789/`（workstation）
- Dataset 1 seed42：`work_dirs/capr_quality_final_only_chr2024_seed42/`（workstation）
- 实现：`ldmdet/core/head.py`、`ldmdet/core/single_head.py`、
  `ldmdet/criterion/criterion.py`
- 配置：`experiments/configs/ldmdet/directions/capr/`
- StochOT 最新证据审计：`docs/EXPERIMENT_LINEAGE.md` §二。
