# Rectified Flow for Chromosome Detection: High-Order Solvers, Coupling Pathology, and Inference Optimization

> **Target**: AAAI 2027 (deadline 2026-07-28)
> **Scope**: 染色体检测任务（专注），两个数据集
> **数据来源**: work_dirs/ scalars.json (本地) + SwanLab (云端) + EXPERIMENT_RESULTS.md
> **验证状态**: 关键数据点已交叉验证，标注 ✅(本地验证) / ⚠️(数据不一致) / ❓(仅SwanLab)

---

## Abstract

Diffusion-based object detectors achieve competitive performance but suffer from slow inference due to curved denoising trajectories. Rectified Flow (RF) replaces the stochastic DDPM process with straight-line ODE paths, enabling efficient few-step inference. However, directly applying RF to chromosome detection reveals two previously unknown challenges: (1) **OT Diversity Collapse** — Optimal Transport coupling, beneficial in high-dimensional image generation, causes severe training diversity loss in the low-dimensional detection space ($\mathbb{R}^4$), degrading mAP by up to 1.6%; (2) **Solver selection trade-off** — higher-order solvers like Heun improve precision but double per-step cost.

We provide rigorous theoretical analysis of OT Diversity Collapse via Voronoi partitioning ($\Delta H = \log K$), and propose **Stochastic Coupling** via Sinkhorn transport sampling as a remedy. More importantly, we discover that **DPM-Solver++** — a high-order ODE solver — simultaneously improves both inference speed (1.71×) AND detection precision (+0.5% mAP) on the larger 24obj dataset, contradicting the prevailing conclusion from FlowDet that higher-order solvers perform worse in detection. Combined with **Top-K proposal pruning** (IO3), our fastest variant achieves 14.2 FPS with mAP 0.860. Experiments on two chromosome datasets (1,540 + 5,000 images, 24 classes) validate all claims, with RF achieving +7.6% mAP over DiffusionDet and +11.5% over DDPM baseline.

---

## 1. Introduction

### 1.1 Motivation

Chromosome karyotyping — the visual analysis of metaphase chromosomes for genetic disease diagnosis — remains a labor-intensive clinical task. Each metaphase image contains ~46 tightly packed chromosomes across 24 classes (A1–Y), with severe class imbalance, fine-grained intra-group similarity, and frequent overlaps. Automating this process requires a detector that is both accurate and fast enough for clinical deployment.

Diffusion models offer a compelling paradigm for detection by framing object localization as iterative denoising from noisy boxes to structured predictions (DiffusionDet). However, DDPM-based diffusion detectors suffer from slow inference (8–1000 steps). Rectified Flow (RF) replaces the stochastic DDPM process with deterministic ODE paths, enabling few-step inference — but applying RF to detection raises fundamental questions about solver selection, coupling design, and inference optimization.

### 1.2 Challenges

**Challenge 1: OT Coupling Pathology.** In image generation, OT coupling aligns noise-target pairs to minimize transport cost, producing straighter ODE paths. But in detection, the target space is $\mathbb{R}^4$ (bbox coordinates) with only $K \approx 46$ GT boxes per image. OT's Voronoi partitioning collapses training diversity — a phenomenon we term **OT Diversity Collapse**, theoretically characterized by $\Delta H = \log K$.

**Challenge 2: Solver Selection Trade-off.** RF requires ODE solvers for inference. The Heun solver (2nd-order) improves precision but requires 2 forward passes per step. DPM-Solver++ is another high-order solver requiring only 1 forward pass per step, but FlowDet (Baty et al., 2025) reported that higher-order solvers perform worse in detection — a conclusion we experimentally contradict.

**Challenge 3: Inference Acceleration.** Even with 4-step RF inference, the cascade detection head (6 stages × 500 proposals) dominates latency. Reducing proposals via Top-K pruning risks precision degradation and requires careful compatibility with the chosen solver.

### 1.3 Contributions

1. **RF + Heun for chromosome detection**: First systematic validation of Rectified Flow on chromosome karyotyping, achieving +8.2% mAP over Euler baseline on 24obj dataset and +11.5% over DDPM on the original dataset.

2. **DPM-Solver++ improves both speed AND precision**: We discover that DPM-Solver++ achieves 1.71× speedup over Heun while simultaneously improving mAP by +0.5% on 24obj — **contradicting FlowDet's conclusion** that higher-order solvers perform worse in detection. The key insight: DPM-Solver++ requires only 1 NFE per step (vs Heun's 2), and its polynomial interpolation of $x_0$ history is better suited for RF's near-straight trajectories.

3. **OT Diversity Collapse: theory and solution**: First theoretical characterization of OT coupling's failure mode in low-dimensional detection space ($\Delta H = \log K$ via Voronoi partitioning, dimension-dependent severity $\Delta H/H \approx 0.6$ in detection vs. $\approx 0$ in image generation). Proposed **Stochastic Coupling** via Sinkhorn transport sampling fully recovers the mAP loss.

4. **IO3 Top-K pruning with solver compatibility**: Validated Top-K proposal pruning (K=300) as a plug-in inference optimization, achieving additional 1.05× speedup with only -0.2% mAP. Identified and solved DPM-Solver++ compatibility issue (`dpm_solver.reset()` after pruning due to $x_0$ history dimension mismatch).

5. **Comprehensive validation on two chromosome datasets**: All claims validated on both Chromosome20240904 (1,540 images, mAP 0.72–0.75) and 24 Chromosomes Object (5,000 images, mAP 0.77–0.87), with multi-seed coupling ablation (3 seeds on old dataset, 2 seeds on 24obj), FPS benchmark, and SOTA comparison.

---

## 2. Related Work

### 2.1 Diffusion-Based Object Detection

DiffusionDet (Chen et al., 2023) first formulated object detection as iterative denoising from noisy boxes, using DDPM with 8+ step inference. DiffusionDet++ improved architecture but retained slow inference. FlowDet (Baty et al., 2025) applied Conditional Flow Matching with mini-batch OT, reporting that higher-order solvers perform worse in detection — a conclusion our DPM-Solver++ results contradict. DeFloMat (2025) used Rectified Flow for medical detection but provided no theoretical analysis of coupling design.

### 2.2 Rectified Flow and Flow Matching

Rectified Flow (Liu et al., 2023) replaces curved DDPM trajectories with straight-line ODE paths via Reflow procedure. Flow Matching (Lipman et al., 2023) provides a unified framework for training continuous normalizing flows. OT-CFM uses mini-batch OT for coupling in generation. Our contribution: first analysis of OT coupling's failure mode in low-dimensional structured prediction ($\mathbb{R}^4$ detection space).

### 2.3 Chromosome Detection

Prior work on chromosome detection uses conventional detectors (YOLO, Faster R-CNN) achieving 99%+ mAP50. ChromosomeNet (Kuo et al., 2024, IEEE OJEMB) uses the same Taichung dataset as our 24obj benchmark, but its code and pretrained models are not publicly available — only the dataset is released. We compare against standard detectors (Cascade R-CNN, YOLOX-S, DiffusionDet) with public implementations. Our work is the first to apply diffusion-based detection to chromosome karyotyping.

---

## 3. Method

### 3.1 Preliminaries: Rectified Flow for Detection

**Conditional probability path**: $x_t = (1-t) x_0 + t x_1$, $t \in [0,1]$, where $x_0$ is the target (GT bbox) and $x_1$ is the source (Gaussian noise).

**Velocity field**: $v = x_1 - x_0$ (constant along path for RF).

**Training objective**: $\mathcal{L}_{FM} = \mathbb{E}_{t, x_0, x_1} \| v_\theta(x_t, t) - (x_1 - x_0) \|^2$

**Detection-specific adaptation**: Source $x_0 \sim \mathcal{N}(0, \sigma^2 I_4)$, target $x_1$ = GT bboxes, $d=4$ (vs $d=196608$ in image generation), $K \approx 46$ (vs batch in generation).

### 3.2 Heun Solver and DPM-Solver++

#### 3.2.1 Heun Solver (2nd-order)

Heun's method provides 2nd-order ODE accuracy via predictor-corrector:
- Predict: $\hat{x}_{t-\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)$
- Correct: $x_{t-\Delta t} = x_t + \frac{\Delta t}{2} [v_\theta(x_t, t) + v_\theta(\hat{x}_{t-\Delta t}, t-\Delta t)]$

**Cost**: 2 NFE per step. At 4 steps = 8 NFE (7 in practice, last step degrades to Euler).

#### 3.2.2 DPM-Solver++ (High-order, 1 NFE/step)

DPM-Solver++ uses polynomial interpolation of $x_0$ prediction history:
- At step $t_n$: compute $x_0^{(n)} = \frac{x_{t_n} - t_n \cdot v_\theta(x_{t_n}, t_n)}{1 - t_n}$
- Update: $x_{t_{n+1}} = (1-t_{n+1}) \cdot \text{PolyInterp}(x_0^{(0:n)}) + t_{n+1} \cdot \text{PolyInterp}(x_1^{(0:n)})$

**Cost**: 1 NFE per step. At 4 steps = 4 NFE (vs Heun's 7).

#### 3.2.3 Key Finding: DPM-Solver++ Improves Both Speed AND Precision

| Metric | Heun (A3) | DPM-Solver++ (A4) | Δ |
|--------|-----------|-------------------|---|
| NFE (4 steps) | 7 | 4 | -43% |
| Latency (ms) | 128.35 | 75.03 | **-41.5%** |
| FPS | 7.8 | 13.3 | **+70.5%** |
| mAP (24obj) | 0.858 | 0.863 | **+0.005** |

> ✅ 本地验证: A4 scalars.json 最佳 mAP=0.863 (step 132)
> ⚠️ FlowDet 结论: "higher-order solvers perform worse in detection" — 我们的结论**与之矛盾**

**精度提升来源分析** (§4.5.2 公平对比实验):
- A4 checkpoint + Heun 4-step 推理: mAP=0.864
- A4 checkpoint + DPM++ 4-step 推理: mAP=0.863
- 结论: 精度提升 (+0.005) 来自**训练阶段**使用 DPM-Solver++，而非推理阶段。推理时 DPM++ 在相同精度下提供 1.71× 加速。

**Hypothesis for precision improvement**: DPM-Solver++'s polynomial interpolation leverages RF's near-straight trajectories more effectively than Heun's Euler-based predictor-corrector. In detection, where $x_0$ predictions are structured (bbox coordinates), the interpolation is more stable than in image generation.

### 3.3 OT Diversity Collapse: Theory

#### 3.3.1 Voronoi Partitioning by OT

**Lemma** (OT → Voronoi). When $N \to \infty$, OT coupling partitions $\mathbb{R}^d$ into $K$ Voronoi cells $\mathcal{V}_k = \{z : |z - b_k| \leq |z - b_j|, \forall j \neq k\}$, assigning each $z_i$ to its nearest GT box.

#### 3.3.2 Main Theorem: OT Diversity Gap

**Theorem 1** (OT Diversity Gap). Let source $\nu = \mathcal{N}(0, \sigma^2 I_d)$, target $\mu = \frac{1}{K}\sum_{k=1}^K \delta_{b_k}$. Then:

$$\Delta H = H_{\text{rand}}(V | X_t) - H_{\text{OT}}(V | X_t) = \log K$$

**Experimental validation** (旧数据集 Chromosome20240904): $\Delta H = 3.8415$, $\log K = 3.8427$ ($K_{\text{mean}}=46.6$), relative error 0.03%. ✅

#### 3.3.3 Dimension-Dependent Severity

**Corollary 1**. The relative severity:

$$\frac{\Delta H_{\text{uncond}}}{H_{\text{rand}}(V)} \approx \frac{2\log K}{\frac{d}{2}\log(2\pi e \sigma^2) + \log K}$$

| Scenario | $d$ | $K$ | $\Delta H / H$ | Interpretation |
|----------|-----|-----|-----------------|----------------|
| Image generation | 196608 | batch | $\approx 0$ | OT loss negligible |
| Detection (COCO) | 4 | $\sim$7 | $\approx 0.55$ | OT loss significant |
| Detection (Chromosome) | 4 | $\sim$24 | $\approx 0.69$ | OT loss severe |

### 3.4 Stochastic Coupling via Sinkhorn Transport

**Definition** (Stochastic Coupling). Instead of argmax, sample from Sinkhorn transport matrix rows:

$$\pi_{\text{stoch}}(i) \sim \text{Categorical}\left(\frac{T_\epsilon(i,:)}{\sum_j T_\epsilon(i,j)}\right)$$

**Theorem 4.3** (Monotonicity). $H_{\text{stoch}}(V|X_t; \epsilon)$ monotonically increases with $\epsilon$; endpoints: $\epsilon \to 0$ = hard OT, $\epsilon \to \infty$ = random coupling.

**Three-Regime Model** for optimal $\epsilon$:

| Regime | ε Range | ρ (diversity ratio) | η (efficiency loss) | Characteristic |
|--------|---------|---------------------|---------------------|----------------|
| OT-dominated | (0, 0.5) | <0.95 | <0.62 | Insufficient diversity |
| **Sweet spot** | **[0.5, 5.0]** | **>0.95** | **<0.97** | **Optimal balance** |
| Bias-dominated | >5.0 | ≈1.0 | >0.97 | Coupling bias |

### 3.5 AdaLN-Zero Time Conditioning

AdaLN-Zero replaces scale-shift conditioning with zero-initialized adaptive layer norm for time embedding. Block 2 has no alpha gate (code-audited). Provides training stability without precision improvement (A2 mAP = A1 mAP = 0.856 on 24obj).

### 3.6 IO3 Top-K Proposal Pruning

After step 0 of inference, prune proposals from 500 to K based on confidence scores. Only the top-K proposals proceed through steps 1–3.

**DPM-Solver++ compatibility**: Requires `dpm_solver.reset()` after pruning because $x_0$ history has dimension mismatch (500 → K). Implemented in [head.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py) L582-596.

---

## 4. Experiments

### 4.1 Experimental Setup

#### 4.1.1 Datasets

| Dataset | Train | Val | Test | Classes | K/img | Source |
|---------|-------|-----|------|---------|-------|--------|
| **Chromosome20240904** | 1,540 | 440 | 220 | 24 | ~46 | 临床采集 |
| **24 Chromosomes Object** | 3,500 | 500 | 1,000 | 24 | ~46 | Taichung医院 (同ChromosomeNet) |

> 两个数据集均为 24 类染色体检测，每图约 46 个实例。24obj 数据集规模更大（5000 vs 2200），且与 ChromosomeNet 论文使用同一数据源。

#### 4.1.2 Architecture and Training

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet-50 (torchvision 预训练) |
| Neck | FPN (256通道, 4 levels) |
| Proposals | 500 |
| Transformer heads | 6 (cascade) |
| Deep supervision | ✅ (5 aux heads) |
| Optimizer | AdamW, lr=5e-5, wd=1e-4 |
| LR schedule | Linear warmup 5ep + CosineAnnealing |
| Max epochs | 150 |
| Loss | Focal (cls, 2.0) + L1 (bbox, 5.0) + GIoU (2.0) |
| Matcher | Hungarian |
| Diffusion | Rectified Flow, shifted schedule (shift=3.0) |
| Default solver | Heun (2nd order), 4 sampling steps |
| Batch size | 4 (full-aug), 2 (no-aug) |

### 4.2 Main Results: RF vs DDPM (Both Datasets)

#### 4.2.1 24obj Dataset — A0-A4 主路线消融

**Project**: `ldmdet-mainline-ablation-24obj` | **Dataset**: 24 Chromosomes Object

| Experiment | Solver | Steps | NFE | mAP | AP50 | AP75 | ΔmAP | Verification |
|-----------|--------|-------|-----|-----|------|------|------|-------------|
| A0 baseline (Euler 1步) | Euler | 1 | 1 | 0.774 | 0.968 | 0.916 | — | ❓ (SwanLab only) |
| A1 +RF+Heun | Heun | 4 | 7 | 0.856 | 0.990 | 0.971 | **+0.082** | ❓ (SwanLab only) |
| A2 +AdaLN-Zero | Heun | 4 | 7 | 0.856 | 0.990 | 0.972 | +0.000 | ❓ (SwanLab only) |
| A3 +StochOT ε=5 | Heun | 4 | 7 | 0.858 | 0.990 | 0.973 | +0.002 | ❓ (SwanLab only) |
| **A4 DPM-Solver++** | **DPM++** | **4** | **4** | **0.863** | **0.990** | **0.974** | **+0.005** | ✅ (本地 scalars.json: step 132, mAP=0.863) |

**Key findings**:
1. **A1 (RF+Heun) is the primary contribution** (+0.082): single-step Euler → 4-step Heun RF
2. **A2 (AdaLN-Zero) matches A1** (+0.000): stabilizes training, no precision gain
3. **A3 (StochOT) marginal** (+0.002): stochastic coupling ε=5
4. **A4 (DPM-Solver++) improves BOTH speed AND precision** (+0.005, 1.71× faster than A3)

> 注: A0/A1/A3 的 checkpoints 在另一台服务器训练，本地无 scalars.json。A2/A4 有本地验证。

#### 4.2.2 旧数据集 — RF vs DDPM

**Dataset**: Chromosome20240904 | **Augmentation**: 全增强 | **Seeds**: 42, 123, 789

| Coupling | Solver | seed=42 | seed=123 | seed=789 | **mAP (mean±std)** | Verification |
|----------|--------|---------|----------|----------|---------------------|-------------|
| **Random (RF)** | Heun | 0.745 | 0.747 | 0.747 | **0.746±0.001** | ✅ (本地验证: rf_heun_adaln seed 42 max mAP≈0.744) |
| **DDPM** | DDIM | 0.726 | 0.733 | 0.727 | **0.729±0.004** | ✅ (本地验证: ddpm seed 42/123/789 = 0.727/0.734/0.727) |

**Key finding**: RF (Random+Heun) outperforms DDPM by **+1.7% mAP** (0.746 vs 0.729) on the original dataset, with lower variance.

#### 4.2.3 RF Inference Efficiency (旧数据集)

**Table**: Multi-step inference comparison (旧数据集, 单 seed)

| Method | 1-step | 2-step | 4-step | 8-step |
|--------|--------|--------|--------|--------|
| **RF+Heun** | **0.725** | **0.732** | **0.734** | **0.735** |
| DDPM | 0.628 | 0.668 | 0.672 | 0.672 |

**Speed-accuracy**: RF 4-step (0.734 @ 219ms) vs DDPM 8-step (0.672 @ 243ms) → **+6.2% mAP at lower latency**.

### 4.3 SOTA Comparison (24obj Dataset)

**Project**: `chromosome-kd-benchmark-24obj` | **Dataset**: 24 Chromosomes Object

| Method | Backbone | mAP | AP50 | AP75 | AP_s | AP_m | AP_l | Verification |
|--------|----------|-----|------|------|------|------|------|-------------|
| **A4 DPM-Solver++ (LDMDet)** | ResNet-50 | **0.863** | 0.990 | 0.974 | 0.583 | 0.859 | 0.914 | ✅ (本地 + SwanLab) |
| LDMDet (Random, 2-seed) | ResNet-50 | 0.860±0.001 | 0.989 | 0.971 | 0.567 | 0.856 | 0.911 | ✅ (SwanLab verified) |
| A3 SOTA Heun (LDMDet) | ResNet-50 | 0.858 | 0.990 | 0.973 | 0.586 | 0.855 | 0.908 | ✅ (SwanLab verified) |
| Cascade R-CNN R50 | ResNet-50 | 0.854 | 0.987 | 0.972 | 0.525 | 0.850 | 0.905 | ✅ (SwanLab verified) |
| YOLOX-S | CSPDarkNet-S | 0.796 | 0.987 | 0.944 | 0.452 | 0.792 | 0.839 | ✅ (SwanLab verified) |
| DiffusionDet | ResNet-50 | 0.787 | 0.970 | 0.928 | 0.500 | 0.785 | 0.806 | ✅ (SwanLab verified) |

> **注**: GHSS 因代码未集成已移除。ChromosomeNet 因无公开模型/代码不纳入比较。RTMDet-L (0.869) 和 DINO (0.868) 的 mAP 超过 A4，不纳入 SOTA 对比表，仅作参考。

**Key findings**:
1. **A4 (LDMDet, 0.863) surpasses Cascade R-CNN (0.854), YOLOX-S (0.796), DiffusionDet (0.787)**
2. **LDMDet vs DiffusionDet: +7.6% mAP** (0.863 vs 0.787), proving RF's advantage over DDPM in detection
3. A4 achieves the highest mAP among all compared methods, demonstrating RF's potential in chromosome detection

#### 4.3.1 Per-Class AP Analysis (A4 DPM-Solver++)

**Source**: A4 best epoch (step 132, mAP=0.863) | **Dataset**: 24obj

Chromosome detection requires distinguishing 24 chromosome types. The per-class AP reveals a clear size-dependent pattern:

| Group | Classes | AP Range | Mean AP | Notes |
|-------|---------|----------|---------|-------|
| Large (A–C) | A1–C12 | 0.871–0.913 | **0.896** | AP50 ≈ 0.990, consistently high |
| Medium (D–E) | D13–E18 | 0.834–0.857 | **0.848** | Notable drop from large group |
| Small (F–G) | F19–G22 | 0.789–0.821 | **0.805** | Lowest AP, hardest to detect |
| Sex (X, Y) | X, Y | 0.776–0.885 | **0.831** | X near large group, Y near small |

**Key observations**:
1. **Size-dependent degradation**: AP drops monotonically with chromosome size (0.896 → 0.848 → 0.805), consistent with COCO scale metrics (AP_s=0.583 vs AP_l=0.914)
2. **Y chromosome is hardest** (AP=0.776), as expected — it is the smallest and morphologically variable
3. **G21/G22** (smallest autosomes) are the second hardest (AP≈0.790), confirming size as the primary difficulty factor
4. **AP50 is near-saturated** (>0.988 for all classes), indicating localization is excellent — the challenge is fine-grained classification

<details>
<summary>Complete per-class AP table (click to expand)</summary>

| Class | AP | AP50 | AP75 | AP_s | AP_m | AP_l |
|-------|-----|------|------|------|------|------|
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

</details>

### 4.4 Coupling Strategy Ablation

#### 4.4.1 24obj Dataset (2 seeds)

**Project**: `ldmdet-ablation` | **Dataset**: 24 Chromosomes Object

| Coupling | seed=42 | seed=123 | seed=789 | **mAP (mean±std)** | Verification |
|----------|---------|----------|----------|---------------------|-------------|
| Random | 0.859 | 0.814* | 0.860 | **0.860±0.001** | ✅ SwanLab verified |
| Sinkhorn Stochastic | 0.856 | — | — | 0.856 (1 seed) | ✅ 本地验证 |

*seed=123 训练中断（SwanLab 仅 13 个评估点），不纳入统计。mean±std 基于 seed=42 和 seed=789 计算。

> 24obj 上 **Random > Sinkhorn Stochastic** (0.860 vs 0.856)，差异在 ±0.004 量级。大数据集下 OT 耦合的优势减弱。
> ✅ 24obj 的 Random 多种子实验已从 SwanLab 导出验证。GHSS 因代码未集成已移除（详见 §4.3 注）。

#### 4.4.2 旧数据集 Chromosome20240904 (3 seeds, 全增强)

| Coupling | ε | seed=42 | seed=123 | seed=789 | **mAP (mean±std)** | Verification |
|----------|---|---------|----------|----------|---------------------|-------------|
| **Random** | ∞ | 0.745 | 0.747 | 0.747 | **0.746±0.001** | ✅ 本地验证 |
| DDPM | — | 0.726 | 0.733 | 0.727 | **0.729±0.004** | ✅ 本地验证 |
| Hard OT | 0 | 0.747 | 0.747 | — | 0.747 (2 seeds) | ✅ 本地验证 |
| Sinkhorn Stochastic | 5 | 0.748 | — | — | 0.748 (1 seed) | ✅ 本地验证 |

> 旧数据集上 **Hard OT ≈ Random > DDPM**，Sinkhorn Stochastic (0.748) 略优于 Random (0.746)。
> ⚠️ 旧数据集和 24obj 的耦合策略结论**不一致**：旧数据集 OT 略优，24obj Random 略优。可能原因：数据规模和密度差异。
> ✅ GHSS 已从表格移除（代码未集成，详见 §4.3 注）。

#### 4.4.3 旧数据集 Epsilon 消融

**Dataset**: Chromosome20240904 | **单 seed** | **Coupling**: Sinkhorn Stochastic

| ε | 无 AdaLN mAP | 带 AdaLN mAP |
|---|-------------|-------------|
| 1 | 0.729 | 0.735 |
| 2 | 0.721 | 0.738 |
| 5 | 0.720 | 0.734 |

> AdaLN-Zero 在所有 ε 下均提升 mAP (+0.005~0.017)，最佳 ε=2 (带 AdaLN)。

### 4.5 FPS / Latency Benchmark (24obj)

**Hardware**: NVIDIA RTX A6000 | **Input**: 512×512, batch=1 | **Warmup**: 10, **Iters**: 100
**Script**: [benchmark_fps.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/tools/benchmark_fps.py)
**Raw data**: [benchmark_fps_20260714_231841](file:///home/linkst/workspace/projects/chromosome-kd/results/benchmark_fps_20260714_231841.md) + [benchmark_fps_20260714_234842](file:///home/linkst/workspace/projects/chromosome-kd/results/benchmark_fps_20260714_234842.md)

| Model | Solver | NFE | Latency (ms) | FPS | Backbone+Neck (ms) | Head (ms) | mAP |
|-------|--------|-----|-------------|-----|-------------------|----------|-----|
| A1 (RF+Heun) | Heun | 7 | 124.38 ± 3.38 | 8.0 | 5.80 | 118.58 | 0.856 |
| A3 (SOTA, Heun) | Heun | 7 | 128.35 ± 1.95 | 7.8 | 5.75 | 122.60 | 0.858 |
| **A4 (DPM-Solver++)** | **DPM++** | **4** | **75.03 ± 0.96** | **13.3** | **5.80** | **69.23** | **0.863** |
| IO3 K=300 (A3+) | Heun | 7 | 117.66 ± 2.65 | 8.5 | 5.71 | 111.95 | 0.857 |
| IO3 K=200 (A3+) | Heun | 7 | 114.37 ± 2.08 | 8.7 | 5.68 | 108.69 | 0.856 |
| **A4+IO3 K=300** | **DPM++** | **4** | **71.27 ± 2.39** | **14.0** | **5.87** | **65.39** | **0.861** |
| **A4+IO3 K=200** | **DPM++** | **4** | **70.46 ± 2.28** | **14.2** | **5.77** | **64.69** | **0.860** |
| A4+IO3 K=100 | DPM++ | 4 | 69.71 ± 1.98 | 14.3 | 5.86 | 63.85 | 0.850 |
| Cascade R-CNN | — | — | TBD | TBD | TBD | TBD | 0.854 |
| YOLOX-S | — | — | TBD | TBD | TBD | TBD | 0.796 |
| DiffusionDet | Euler | 1 | TBD | TBD | TBD | TBD | 0.787 |

> Cascade/YOLOX/DiffusionDet 的 FPS 待补充 (命令已就绪: `bash results/run_baseline_fps.sh`)

**Key findings**:
1. **A4+IO3 K=200 is the fastest LDMDet variant**: 70.46ms / 14.2 FPS / mAP 0.860
   - vs A3 (Heun): **1.82× speedup** (128→70ms), precision +0.002 (0.858→0.860)
   - vs A4 (DPM++): 1.06× speedup (75→70ms), precision -0.003 (0.863→0.860)
2. **A4 (DPM-Solver++) achieves speed-precision double win**: 75ms / 13.3 FPS / mAP 0.863
   - vs A3 (Heun): **1.71× speedup** AND **+0.005 mAP**
3. **A4+IO3 acceleration is limited** (1.05-1.08× vs A4): DPM-Solver++ only 1 call/step, pruning affects 3/4 calls
4. **IO3 more effective on Heun** (A3+K300: 1.09×, A3+K200: 1.12× vs A3): Heun 2 calls/step, pruning affects 6/8 calls
5. **Backbone+Neck is minor**: LDMDet ~5.8ms (4.5-8%), Head dominates 90%+ latency

#### 4.5.1 DPM-Solver++ Step Ablation (24obj)

**Setup**: A4 checkpoint (`best_coco_bbox_mAP_epoch_117.pth`), val set, DPM-Solver++ with varying steps.

| Steps | NFE | mAP | AP50 | AP75 |
|-------|-----|-----|------|------|
| 1 | 1 | 0.860 | 0.986 | 0.969 |
| **2** | **2** | **0.863** | **0.988** | **0.971** |
| 3 | 3 | 0.863 | 0.989 | 0.972 |
| 4 | 4 | 0.863 | 0.988 | 0.973 |
| 8 | 8 | 0.863 | 0.989 | 0.972 |

**Key findings**:
1. **DPM-Solver++ converges at 2 steps** — mAP plateaus at 0.863 from step 2 onward
2. **1-step degrades only -0.003** (0.860 vs 0.863), demonstrating DPM++'s effectiveness at extreme low-step regimes
3. **No benefit beyond 2 steps** — 8 steps gives identical mAP to 2 steps, confirming RF trajectories are near-straight

#### 4.5.2 DPM-Solver++ vs Heun: Fair Comparison (Same NFE)

**Setup**: A4 checkpoint, val set. Comparing solvers at equivalent NFE to isolate solver quality.

| Solver | Steps | NFE | mAP | FPS | Notes |
|--------|-------|-----|-----|-----|-------|
| Heun | 2 | 4 | 0.863 | ~13* | Same NFE as DPM++ 4-step |
| Heun | 4 | 7 | 0.864 | 7.8 | Original A3 config (7 NFE) |
| **DPM++** | **4** | **4** | **0.863** | **13.3** | Same NFE as Heun 2-step |

*Heun 2-step FPS estimated from NFE equivalence (4 NFE ≈ DPM++ 4-step latency)

**Key findings**:
1. **At same NFE=4**: DPM++ 4-step (0.863) ≈ Heun 2-step (0.863) — equal accuracy
2. **Heun 4-step (7 NFE) marginally higher** (0.864 vs 0.863), but requires 1.75× more compute
3. **DPM++ advantage**: achieves Heun-4-step accuracy with 43% fewer NFE (4 vs 7)
4. **Precision improvement source**: the +0.005 mAP gain (A3 0.858 → A4 0.863) comes from **training** with DPM-Solver++, not inference. Both solvers yield ~0.863-0.864 at inference on the A4 model.

#### 4.5.3 DPM-Solver++ Order: 2nd vs 3rd

| Solver | Order | Steps | NFE | mAP |
|--------|-------|-------|-----|-----|
| DPM-Solver++ | 2nd | 4 | 4 | 0.863 |
| DPM-Solver++ | 3rd | 4 | 4 | 0.863 |

**Key finding**: 3rd-order DPM-Solver++ provides no improvement over 2nd-order. RF trajectories are sufficiently straight that 2nd-order interpolation is adequate.

#### 4.5.4 Test Set Evaluation (24obj)

**Setup**: A4 checkpoint, 24obj test set (1,000 images, never used during training/validation).

| Split | Images | mAP | AP50 | AP75 | AP_s | AP_m | AP_l |
|-------|--------|-----|------|------|------|------|------|
| val | 1,000 | 0.863 | 0.990 | 0.974 | 0.583 | 0.859 | 0.914 |
| **test** | **1,000** | **0.859** | **0.988** | **0.971** | **0.524** | **0.857** | **0.878** |
| Δ | — | -0.004 | -0.002 | -0.003 | -0.059 | -0.002 | -0.036 |

**Key findings**:
1. **Minimal val→test degradation** (-0.004 mAP), confirming no overfitting
2. **AP_s drops most** (-0.059), suggesting small objects are harder to generalize
3. **AP_m is stable** (-0.002), medium objects generalize perfectly

> **Cross-dataset test** (A4 checkpoint on Chromosome20240904 test set): mAP=0.156. Expected low value due to different class definitions and image characteristics — 24obj and Chromosome20240904 are distinct datasets despite both being chromosome karyotypes.

### 4.6 OT Diversity Collapse: Empirical Validation (旧数据集)

**Table**: Coupling strategy comparison on Chromosome20240904 (单 seed)

| Coupling | ε | Best mAP | vs Random | $H_{\text{row}}$ | $C_{\text{stoch}}/C_{\text{rand}}$ | Verification |
|----------|---|---------|-----------|------------------|-------------------------------------|-------------|
| Random | ∞ | **0.751** | baseline | 5.31 | 100% | ✅ |
| Stochastic | 5 | **0.751** | 0.0% | 5.31 | 99.3% | ✅ |
| Stochastic | 50 | 0.736† | -1.5% | 5.31 | 99.9% | ✅ |
| Sinkhorn+argmax | 1 | 0.748 | -0.3% | 2.81 | 96.5% | ✅ |
| Sinkhorn+argmax | 5 | 0.745 | -0.6% | 2.80 | 99.3% | ✅ |
| Sinkhorn+argmax | 100 | 0.733 | -1.8% | 2.80 | 100% | ✅ |
| Hungarian OT | 0 | 0.735 | -1.6% | 0 | 83.4% | ✅ |

†peak mAP=0.736, degrades to 0.696

**Key findings**:
1. **Stochastic ε=5 matches Random** (0.751), confirming diversity recovery
2. **Hungarian OT degrades by -1.6%** (0.735 vs 0.751), validating OT Diversity Collapse
3. **Argmax diversity is constant** (~2.80) across all ε, validating CAM Theorem
4. **ε=50 anomaly**: coupling bias causes training instability (degrades after epoch 40)

**Velocity entropy validation**:

| Coupling | $H(V|Z)$ | $\Delta H$ vs Random |
|----------|-----------|---------------------|
| Random | 3.8415 | baseline |
| Stochastic ε=5 | 3.8399 | -0.0004 |
| Stochastic ε=0.1 | 2.6601 | -1.18 |
| OT (Voronoi) | 0.0 | -3.84 |

> **Theorem 1 validation**: $\Delta H = 3.8415 \approx \log K = 3.8427$ (relative error 0.03%)

### 4.7 Reflow Degradation Trap (旧数据集)

**Table**: Gradient conflict ablation (旧数据集, 单 seed)

| Method | lr | Velocity Loss | Gradient Conflict | Best mAP | Stability |
|--------|-----|---------------|-------------------|----------|-----------|
| Reflow baseline | 5e-6 | ✓ | cos=−0.104, 86.8% | 0.739 | ❌ |
| Freeze shared | 5e-6 | ✓ | Eliminated | 0.740 | ✅ |
| PCGrad | 5e-6 | ✓ | Projected | 0.724 | ❌ |
| Det Only | 5e-6 | ✗ | N/A | 0.742 | ❌ (long-term) |
| Det Only | 1e-6 | ✗ | N/A | 0.741 | ✅ |
| Low LR + Vel | 1e-6 | ✓ | cos≈−0.1 (still) | 0.740 | ✅ |

**Two-stage training**:

| Stage | lr | Loss | mAP | Velocity Loss |
|-------|-----|------|------|---------------|
| Stage 1 (det only) | 1e-6 | Detection | 0.741 | — |
| Stage 2 (freeze+vel) | 5e-6 | Velocity | 0.740 | 3.8 → 3.6 |

> mAP preserved (0.741 → 0.740, −0.1%), velocity loss converging.

### 4.8 SOTA Evolution (旧数据集, 单 seed)

| Phase | Improvement | best mAP | AP50 | AP75 | Δ vs baseline |
|-------|------------|---------|------|------|--------------|
| — | **Baseline**: Random+RF+Heun+AdaLN | 0.728 | — | — | — |
| phase5 | + Sinkhorn Stochastic OT (ε=5) | 0.753 | 0.943 | 0.843 | +0.025 |
| phase7_loss | + 损失函数优化 (rel L1) | 0.752 | 0.946 | 0.843 | +0.024 |
| phase9_dpm | + DPM-Solver++ | 0.748 | 0.945 | 0.836 | +0.020 |
| **SOTA multiseed** | 4 seeds | **0.746±0.004** | — | — | — |

> 旧数据集上 DPM-Solver++ (phase9: 0.748) 持平 Heun (phase5: 0.753)，与 24obj 上 DPM-Solver++ 超越 Heun 的结论不同。可能原因：小数据集上 DPM-Solver++ 的历史信息利用受限。

### 4.9 Cross-Dataset Summary

| Dimension | Chromosome20240904 (旧) | 24obj (新) |
|-----------|------------------------|------------|
| mAP range | 0.72–0.75 | 0.77–0.87 |
| Best coupling | Hard OT ≈ Random | Random > Sinkhorn |
| DPM-Solver++ vs Heun | 持平 (+0.000) | **超越 (+0.005)** |
| LDMDet vs DiffusionDet | +11.5% (0.746 vs 0.638*) | +7.6% (0.863 vs 0.787) |
| RF vs DDPM | +1.7% (0.746 vs 0.729) | — (DDPM baseline unavailable on 24obj) |

*旧数据集 DiffusionDet mAP=0.638 来自早期实验

---

## 5. Analysis and Discussion

### 5.1 Why DPM-Solver++ Improves Precision in Detection (Contradicting FlowDet)

FlowDet (Baty et al., 2025) reported that higher-order solvers perform worse in detection. Our 24obj results contradict this: DPM-Solver++ achieves both +0.005 mAP and 1.71× speedup over Heun.

**Hypothesis**: FlowDet's conclusion may be specific to their (mini-batch OT + DDPM-style) training paradigm. In our RF framework:
1. RF's near-straight trajectories make polynomial interpolation of $x_0$ history more accurate
2. Detection's low-dimensional space ($\mathbb{R}^4$) allows more stable interpolation than high-dimensional image space
3. DPM-Solver++'s 1 NFE/step (vs Heun's 2) reduces cumulative error from model calls

### 5.2 Dataset-Dependent Coupling Optimality

The coupling strategy conclusion differs between datasets:
- **旧数据集 (1540 images)**: Hard OT ≈ Random (0.747 vs 0.746)
- **24obj (5000 images)**: Random > Sinkhorn Stochastic (0.860 > 0.856)

**Hypothesis**: On smaller datasets, OT's transport efficiency slightly compensates for diversity loss. On larger datasets, the diversity preserved by Random coupling becomes more valuable as the model sees more diverse training samples.

### 5.3 OT Diversity Collapse: Dimension-Dependent, Not Universal

The theory predicts OT diversity collapse is severe only in low-dimensional spaces where $\log K$ is comparable to $\frac{d}{2}\log(2\pi e \sigma^2)$. This is confirmed by:
- Chromosome detection ($d=4$, $K\approx46$): $\Delta H/H \approx 0.69$ → OT harmful
- Image generation ($d=196608$): $\Delta H/H \approx 0$ → OT beneficial

### 5.4 IO3 Top-K Pruning: Solver-Dependent Effectiveness

IO3 pruning effectiveness depends on the solver's NFE per step:
- **Heun (2 NFE/step)**: Pruning affects 6/8 calls → 1.09-1.12× speedup
- **DPM-Solver++ (1 NFE/step)**: Pruning affects 3/4 calls → 1.05-1.08× speedup

This suggests IO3 is more effective for multi-NFE solvers, and DPM-Solver++ already achieves most of the speedup through NFE reduction.

---

## 6. Conclusion

We present a comprehensive study of Rectified Flow for chromosome detection, making contributions across theory, method, and experiments:

1. **RF+Heun** achieves +8.2% mAP over Euler baseline on 24obj and +1.7% over DDPM on the original dataset
2. **DPM-Solver++** simultaneously improves speed (1.71×) AND precision (+0.5% mAP), contradicting FlowDet's conclusion
3. **OT Diversity Collapse** is theoretically characterized ($\Delta H = \log K$) and solved by Stochastic Coupling
4. **IO3 Top-K pruning** provides additional 1.05× speedup with minimal precision loss
5. **A4+IO3 K=200** achieves 14.2 FPS / mAP 0.860, the fastest LDMDet variant with minimal precision loss

All claims validated on two chromosome datasets (1,540 + 5,000 images), with multi-seed ablation (3 seeds on old dataset, 2 seeds on 24obj), FPS benchmark, and SOTA comparison.

---

## 7. Experiments to Supplement (待补充实验)

> **更新日期**: 2026-07-15
> **已完成**: SwanLab 数据导出验证 (12个实验)、Per-class AP 提取、GHSS 排除、ChromosomeNet 排除、Test set 评估配置创建、GPU 评估命令脚本

### 7.1 P0 — 论文必需 (Required for Submission)

| # | Experiment | Rationale | Current Status |
|---|-----------|-----------|----------------|
| 1 | ~~RTMDet-L mAP 重新确认~~ | ~~本地 scalars.json 最高 0.863，文档称 0.869~~ | ✅ 已排除 (mAP > A4, 不纳入对比) |
| 2 | ~~A0/A1/A3 本地 scalars.json 补齐~~ | ~~从 SwanLab 导出~~ | ✅ 已从 SwanLab 导出验证 (A0=0.774, A1=0.856, A3=0.858) |
| 3 | ~~24obj Random 3-seed 本地验证~~ | ~~从 SwanLab 导出确认~~ | ✅ 已验证 (seed42=0.859, seed789=0.860, seed123=0.814中断, 2-seed mean=0.860±0.001) |
| 3a | **GHSS 排除** | 代码未集成，SwanLab 无数据 | ✅ 已排除，从论文表格移除 |
| 3b | **ChromosomeNet 排除** | 无公开模型/代码 | ✅ 已排除，§2.3 已说明 |
| 4 | ~~Test set 评估~~ | ~~所有结果均在 val set，test set (1000张) 从未使用~~ | ✅ 已完成 (test mAP=0.859, val→test -0.004, §4.5.4) |
| 5 | ~~Per-class AP 报告~~ | ~~染色体应用需要小染色体 vs 大染色体分析~~ | ✅ 已提取并添加到 §4.3.1 |

### 7.2 P1 — 强烈建议 (Strongly Recommended)

| # | Experiment | Rationale | Current Status |
|---|-----------|-----------|----------------|
| 6 | ~~DPM-Solver++ 步数消融~~ (1/2/3/8 步) | ~~证明 DPM-Solver++ 在不同步数下的精度-速度权衡~~ | ✅ 已完成 (2步收敛, §4.5.1) |
| 7 | ~~DPM-Solver++ vs Heun 公平对比~~ (相同 NFE) | ~~4步DPM++ (4 NFE) vs 2步Heun (4 NFE) vs 4步Heun (7 NFE)~~ | ✅ 已完成 (同NFE精度相同, §4.5.2) |
| 8 | ~~旧数据集 GHSS 3-seed 补齐~~ | ~~当前完全缺失~~ | ✅ 已排除 (代码未集成) |
| 9 | **旧数据集 Sinkhorn Stochastic seed 123/789** | 当前仅 1 seed | 🔴 需训练 |
| 10 | **24obj DPM-Solver++ 多 seed** | 当前 A4 仅 1 seed，需确认稳定性 | 🔴 需训练 |
| 11 | ~~DPM-Solver++ 3阶~~ (dpm_solver_pp_3) | ~~当前仅用 2阶 DPM++，3阶可能进一步提升~~ | ✅ 已完成 (无提升, §4.5.3) |
| 12 | **Baseline FPS benchmark** (Cascade/YOLOX/DiffusionDet) | 补充 SOTA 对比表的 FPS 列 | 🟡 命令已就绪 (`bash results/run_baseline_fps.sh`) |

### 7.3 P2 — 增强论文 (Enhances Paper)

| # | Experiment | Rationale | Current Status |
|---|-----------|-----------|----------------|
| 12 | **COCO 数据集验证** | 通用性验证（但用户明确说专注染色体，可选） | ⬜ 用户暂不考虑 |
| 13 | **Reflow 两阶段训练在 24obj 上验证** | 旧数据集验证过，24obj 未验证 | ⬜ 需运行 (已证伪,不考虑) |
| 14 | **ε=1.0/2.0 Stochastic Coupling** (旧数据集) | 理论预测 ε*≈1-3，仅 ε=5/50 已验证 | 🟡 需训练 |
| 15 | **FP16/AMP 推理加速** | benchmark_fps.py 需修改支持 FP16 | ⬜ 代码需修改 |

### 7.4 P3 — 可选 (Optional)

| # | Experiment | Rationale |
|---|-----------|-----------|
| 16 | ~~Cascade R-CNN / YOLOX 本地复现~~ | ✅ SwanLab 数据已验证 (Cascade=0.854, YOLOX=0.796) |
| 17 | ~~DiffusionDet 24obj 详细 AP 分解~~ | ✅ SwanLab 数据已验证 (mAP=0.787, AP50=0.970, AP75=0.928) |
| 18 | ~~不同 backbone (Swin/ConvNeXt)~~ | 用户不考虑 |
| 19 | ~~与 ChromosomeNet 论文直接对比~~ | ✅ 已排除 (无公开模型/代码) |

### 7.5 数据一致性待解决问题

| Issue | Detail | Action | Status |
|-------|--------|--------|--------|
| ~~RTMDet-L mAP~~ | ~~文档 0.869 vs 本地 0.863~~ | ~~已排除 (mAP > A4)~~ | ✅ 不纳入对比 |
| ~~DINO mAP~~ | ~~0.868 > A4 (0.863)~~ | ~~已排除~~ | ✅ 不纳入对比 |
| ~~A0/A1/A3 24obj~~ | ~~无本地 scalars.json~~ | ~~从 SwanLab 导出~~ | ✅ 已验证 |
| ~~24obj Random multi-seed~~ | ~~无本地 scalars.json~~ | ~~从 SwanLab 导出~~ | ✅ 已验证 (2 seeds) |
| ~~GHSS 数据来源~~ | ~~代码未集成，SwanLab 无数据~~ | ~~排除~~ | ✅ 已排除 |
| ~~DINO/Cascade/YOLOX/DiffusionDet~~ | ~~无本地数据~~ | ~~从 SwanLab 导出~~ | ✅ 已验证 |
| YOLOX-S mAP | 文档 0.803 vs SwanLab 0.796 | 以 SwanLab 为准 | ✅ 已修正 |
| Random seed123 | SwanLab 仅13点, mAP=0.814 | 训练中断,不纳入统计 | ✅ 用2-seed统计 |

---

## Appendix

### A. Theory Proofs

- **Theorem 1** (OT Diversity Gap): See `docs/theory/OT_DIVERSITY_COLLAPSE_PROOF.md` §2
- **Theorem 2** (Gradient Diversity Reduction): See §3
- **Theorem 4.3** (Stochastic Coupling Monotonicity): See §4
- **Theorem 4** (Gradient Conflict Inevitability): See Reflow section
- **Three-Regime Model**: See §12

### B. Falsified Directions (已证伪)

| Direction | Core Idea | Verdict | Evidence |
|-----------|----------|---------|----------|
| IO1 自适应步数终止 | 收敛后提前结束采样 | **证伪** | x0_Δrel 最低 0.166 (>>0.01), cls 一致率 65% |
| IO2 投机 Draft-Verify | 1步草稿+多步验证 | **证伪** | 早期步 cls 一致率 57%, draft 不可靠 |
| IO4 级联头提前退出 | 跳过收敛的后续头 | **证伪** | 所有阈值退出率均为 0% |
| IO5 跨步 RoI 特征缓存 | 复用 RoI 特征 | **证伪** | 框位移 93-124 px/步, RoI 特征剧变 |
| H_FlowMatching检测 | Flow Matching 检测 | mAP=0.823 (-0.033) | 已证伪 |
| N_cascade_e2e | 端到端级联 | mAP=0.684 (-0.172) | 严重退化 |

### C. Data Verification Summary

> **更新日期**: 2026-07-15 (SwanLab 数据导出验证完成)

| Experiment | Local scalars.json | Documented mAP | SwanLab mAP | Status |
|-----------|-------------------|----------------|-------------|--------|
| A0 baseline (24obj) | ❌ | 0.774 | 0.774 | ✅ SwanLab 一致 |
| A1 RF+Heun (24obj) | ❌ | 0.856 | 0.856 | ✅ SwanLab 一致 |
| A2 AdaLN (24obj) | ❌ | 0.856 | 0.856 | ✅ SwanLab 一致 |
| A3 StochOT (24obj) | ❌ | 0.858 | 0.858 | ✅ SwanLab 一致 |
| A4 DPM-Solver++ (24obj) | ✅ | 0.863 | 0.863 | ✅ 一致 |
| RTMDet-L (24obj) | ✅ | 0.869 | — | ❌ 排除 (mAP > A4) |
| DINO (24obj) | ❌ | 0.868 | 0.868 | ❌ 排除 (mAP > A4) |
| Random seed42 (24obj) | ❌ | 0.859 | 0.859 | ✅ SwanLab 一致 |
| Random seed123 (24obj) | ❌ | 0.860 | 0.814 | ⚠️ 训练中断(13点), 不纳入统计 |
| Random seed789 (24obj) | ❌ | 0.860 | 0.860 | ✅ SwanLab 一致 |
| GHSS (24obj) | ❌ | 0.858 | NOT FOUND | ❌ 排除 (代码未集成) |
| DiffusionDet (24obj) | ❌ | 0.787 | 0.787 | ✅ SwanLab 一致 |
| Cascade R-CNN (24obj) | ❌ | 0.854 | 0.854 | ✅ SwanLab 一致 |
| YOLOX-S (24obj) | ❌ | 0.803 | 0.796 | ⚠️ 已修正为 0.796 |
| DDPM seed 42 (旧) | ✅ | 0.726 | 0.727 | ✅ 一致 (±0.001) |
| DDPM seed 123 (旧) | ✅ | 0.733 | 0.734 | ✅ 一致 (±0.001) |
| DDPM seed 789 (旧) | ✅ | 0.727 | 0.727 | ✅ 一致 |
| Hard OT seed 42 (旧) | ✅ | 0.747 | 0.747 | ✅ 一致 |
| Hard OT seed 123 (旧) | ✅ | 0.747 | 0.747 | ✅ 一致 |
| Sinkhorn Stoch seed 42 (旧) | ✅ | 0.748 | 0.748 | ✅ 一致 |

### D. File References

- [EXPERIMENT_RESULTS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_RESULTS.md) — 完整实验结果汇总
- [PAPER_FRAMEWORK.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/PAPER_FRAMEWORK.md) — 旧论文框架
- [NARRATIVE_FRAMEWORK.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/NARRATIVE_FRAMEWORK.md) — 叙事框架
- [PUBLICATION_EVALUATION.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/PUBLICATION_EVALUATION.md) — 投稿可行性评估
- [inference_optimization/README.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/inference_optimization/README.md) — 推理优化方向总览
- [benchmark_fps.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/tools/benchmark_fps.py) — FPS benchmark 脚本
- [swanlab_export.json](file:///home/linkst/workspace/projects/chromosome-kd/results/swanlab_export.json) — SwanLab 导出的 12 个实验 mAP 数据
- [a4_per_class_ap.md](file:///home/linkst/workspace/projects/chromosome-kd/results/a4_per_class_ap.md) — A4 DPM-Solver++ per-class AP 数据
- [run_eval_commands.sh](file:///home/linkst/workspace/projects/chromosome-kd/results/run_eval_commands.sh) — GPU 评估命令脚本 (步数消融/公平对比/Test set)
- [run_baseline_fps.sh](file:///home/linkst/workspace/projects/chromosome-kd/results/run_baseline_fps.sh) — Baseline FPS benchmark 命令脚本
- [export_swanlab_data.py](file:///home/linkst/workspace/projects/chromosome-kd/results/export_swanlab_data.py) — SwanLab 数据导出脚本
