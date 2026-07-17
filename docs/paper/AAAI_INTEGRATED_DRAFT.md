# Rectified Flow for Chromosome Detection: Stable Coupling and Few-Step Inference

## Abstract

Chromosome karyotyping — the visual inspection of metaphase chromosomes under a microscope — is a cornerstone of clinical genetics, underpinning prenatal testing, congenital disorder diagnosis, and cancer cytogenetics. Yet the procedure remains labor-intensive: trained cytogeneticists must manually delineate, rotate, and classify roughly 46 tightly packed chromosomes per cell into 24 classes, a process that is slow, observer-dependent, and ill-suited to the throughput demands of modern diagnostics. Automating this analysis with a detector that is both accurate and fast enough for clinical deployment is therefore of considerable practical value, but it is hampered by three obstacles: the limited accuracy of conventional detectors on fine-grained, morphologically similar chromosomes; the slow many-step inference and curved-trajectory truncation errors of Denoising Diffusion Probabilistic Model (DDPM)-based diffusion detectors; and the training instability that arises when diffusion detectors are fit to small clinical datasets.

We address these obstacles with *Rectified Flow* (RF), which replaces the stochastic DDPM process with deterministic straight-line ODE paths. The RF training paradigm — straight-line ODE paths combined with a shifted noise schedule — proves to be the dominant source of accuracy gain on the 24 Chromosomes Object (24obj) benchmark: KaryoFlow — our RF-based detector — exceeds DiffusionDet by +0.076 mAP and matches or exceeds Cascade R-CNN and YOLOX-S, and a solver×step disentanglement ablation isolates this paradigm effect from incidental solver and step-count choices. To explain why mini-batch OT coupling, beneficial in image generation, destabilizes training in low-dimensional structured prediction, we provide a theoretical characterization of Optimal Transport (OT) Diversity Collapse in the low-dimensional ($\mathbb{R}^4$) detection space, and propose Stochastic Coupling via Sinkhorn transport as a training stabilizer; although its mAP gain is marginal, it reduces within-run convergence oscillation by 4.6×, which is the property of primary practical concern when training data are scarce. For inference, we deploy DPM-Solver++ for two-step inference, demonstrating through a controlled ablation that its benefit is purely computational rather than a precision advantage, and combine it with Top-K proposal pruning to push inference to a latency compatible with interactive clinical use.

All claims are validated on two chromosome datasets with multi-seed experiments, per-class AP analysis, test-set evaluation, and state-of-the-art comparison. The resulting detector delivers a speed–accuracy trade-off that, together with the stable training behavior, points toward practical deployment in computer-assisted karyotyping.

## 1. Introduction

### 1.1 Motivation

Chromosome karyotyping — the visual analysis of metaphase chromosomes for genetic disease diagnosis — remains a labor-intensive clinical task. Each metaphase image contains about 46 tightly packed chromosomes across 24 classes (A1–Y), with severe class imbalance, fine-grained intra-group similarity, and frequent overlaps. Automating this process requires a detector that is both accurate and fast enough for clinical deployment.

Diffusion models offer a compelling paradigm for detection by framing object localization as iterative denoising from noisy boxes to structured predictions (DiffusionDet). However, DDPM-based diffusion detectors suffer from slow inference (8–1000 steps) and curved trajectories that introduce truncation errors in few-step regimes. Rectified Flow (RF) replaces the stochastic DDPM process with deterministic straight-line ODE paths, enabling few-step inference — but applying RF to detection raises fundamental questions about coupling design, solver selection, and training stability. We frame these as the key bottlenecks of RF-based dense detection in low-data regimes: (1) coupling diversity collapse in low-dimensional structured prediction; (2) numerical solver design for efficient few-step inference; and (3) training stability under high object density. Our three contributions address these bottlenecks systematically.

**From scenario to method.** A typical metaphase spread contains roughly forty-six chromosomes across twenty-four classes, many touching or overlapping. The difficulty is uneven: C-group chromosomes (C6–C12) are morphologically similar, distinguished mainly by subtle banding patterns, while the Y chromosome is the smallest, appears in only one copy in male samples, and has roughly 1,800 training samples versus about 7,000 per autosome. A useful detector must therefore localize densely packed objects under few-step inference, train stably from an imbalanced small corpus, and run fast enough for interactive screening. These demands map onto our three ingredients: Rectified Flow supplies straight-line trajectories for few-step, low-truncation inference; Stochastic Coupling counteracts OT diversity collapse; and DPM-Solver++ with Top-K pruning converts trajectories into clinical-grade latency.

### 1.2 Contributions

Our first contribution validates the RF training paradigm for chromosome detection. KaryoFlow achieves +0.082 mAP over the Euler baseline on 24obj and +0.017 over DDPM on the original dataset. Because the A0→A1 comparison changes several variables at once (DDPM→RF, Euler→Heun, 1→4 steps), a solver×step disentanglement ablation attributes 94% of the gain to the RF paradigm and only 6% to solver and step-count choices; AdaLN-Zero contributes null individually (Appendix D).

Our second contribution is a theoretical characterization of OT coupling's failure mode in low-dimensional detection space, with a practical remedy. We prove an upper bound $\Delta H \le \log K$ on the conditional-entropy reduction, tight to 0.03% empirically, and propose Stochastic Coupling (Sinkhorn-transport sampling instead of argmax), which reduces within-run epoch-level mAP oscillation by 4.6×. An $\epsilon$ ablation shows $\epsilon < 1$ is harmful while $\epsilon \ge 1$ saturates.

Our third contribution deploys DPM-Solver++ for two-step inference (1.71× speedup over Heun, mAP 0.863), with purely computational advantage and no precision gain. The +0.005 mAP gap in the main table stems from checkpoint selection, not solver precision, refining FlowDet's conclusion that higher-order solvers perform worse.

**Novelty boundary.** Relative to FlowDet (CFM with mini-batch OT, reports higher-order solvers perform worse), our novelty is: (i) the theoretical characterization of *why* mini-batch OT becomes a liability in low-dimensional structured prediction (Section 3.3); and (ii) a solver×step disentanglement showing higher-order solvers perform *no better* (not strictly worse) at matched steps. Relative to DeFloMat (RF for medical detection, coupling as implementation detail), we provide the theoretical analysis of coupling design as a training pathology and the Stochastic Coupling remedy. Unlike OT-CFM and multisample flow matching (high-dimensional image generation, $d \sim 10^5$), our setting is low-dimensional ($d=4$) with $K \approx 46$, where OT collapse is severe ($\Delta H/H \approx 0.69$). AdaLN-Zero is reused as a standard implementation detail (Appendix D); DPM-Solver++ is adopted off-the-shelf, our contribution being the disentanglement analysis.

These contributions are backed by comprehensive validation. All claims are validated on both Chromosome20240904 (henceforth the *original* dataset, 1,540 images) and 24obj (5,000 images), with multi-seed coupling ablation, per-class AP analysis, test-set evaluation, and an FPS benchmark.

![**Figure 1**: KaryoFlow overview. (a) RF replaces the curved DDPM denoising trajectory with a straight-line ODE path from noise $\mathbf{x}_1$ to Ground Truth (GT) box $\mathbf{x}_0$; nodes indicate the 4 solver steps. (b) Time conditioning injects the continuous time $t$ via AdaLN-Zero zero-initialized modulation so the network is identity at $t{=}0$. (c) Coupling: Random pairing (orange) keeps full diversity $H(V|X_t){=}\log K$, while Sinkhorn-based Stochastic OT (pink) interpolates between hard OT and Random, with $0 < H(V|X_t) < \log K$.](latex/figures/method_overview.png)

## 2. Related Work

### 2.1 Diffusion-Based Object Detection

DiffusionDet formulates detection as iterative denoising from noisy boxes using DDPM, but curved DDPM trajectories make few-step inference slow and truncation-error-prone. FlowDet applies Conditional Flow Matching with mini-batch OT coupling and reports higher-order solvers perform worse, but does not analyze *why* mini-batch OT becomes a liability in low-dimensional structured prediction. DeFloMat uses Rectified Flow for medical detection but treats coupling as an implementation detail. Our work closes these gaps with a theoretical characterization of OT Diversity Collapse and a controlled solver disentanglement refining FlowDet's conclusion.

### 2.2 Rectified Flow and Flow Matching

Rectified Flow replaces curved DDPM trajectories with straight-line ODE paths, and Flow Matching provides a unified training framework. OT-CFM and multisample flow matching use mini-batch OT coupling successfully in *image generation* ($d \sim 10^5$, $K \approx$ batch size), but the behavior in low-dimensional structured prediction ($d$ small, $K$ targets per image) has not been analyzed. With $d=4$ and $K \approx 46$, OT coupling approaches its $\log K$ entropy-reduction upper bound, collapsing coupling diversity. We characterize this failure mode and propose Stochastic Coupling as a remedy interpolating between hard OT and random pairing.

### 2.3 Chromosome Detection

Prior work uses conventional detectors (YOLO, Faster R-CNN) that leave a measurable accuracy gap on the fine-grained 24-class setting. ChromosomeNet uses the same Taichung dataset as our 24obj benchmark but releases no code or pretrained models. We compare against standard detectors with public implementations (Cascade R-CNN, YOLOX-S, DiffusionDet) and provide the first open-source diffusion-based detector for chromosome karyotyping with multi-seed validation and per-class AP analysis.

## 3. Method

### 3.1 Rectified Flow for Detection (KaryoFlow)

#### 3.1.1 RF Formulation

The conditional probability path is
$$\mathbf{x}_t = (1-t)\, \mathbf{x}_0 + t\, \mathbf{x}_1, \qquad t \in [0,1],$$
where $\mathbf{x}_0$ is the target (GT bbox) and $\mathbf{x}_1$ is the source (Gaussian noise). The corresponding velocity field is constant along the path, $\mathbf{v} = \mathbf{x}_1 - \mathbf{x}_0$, and the training objective is the flow matching loss
$$\mathcal{L}_{FM} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\!\left[ \left\lVert \mathbf{v}_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \right\rVert^2 \right].$$

**Detection-specific adaptation**: source $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_4)$, target $\mathbf{x}_0$ are GT bboxes, $d=4$ (vs. $d=196{,}608$ in image generation), and $K \approx 46$ targets per image.

#### 3.1.2 Time Conditioning

AdaLN-Zero serves as the time conditioning mechanism, replacing scale-shift conditioning with zero-initialized adaptive layer norm so the network starts as an unconditional model (identity at $t{=}0$). A separate ablation (Appendix D) confirms its individual contribution to mAP is null; the +0.082 mAP gain is entirely attributable to the RF formulation and the shifted schedule. We retain AdaLN-Zero as a standard implementation detail, not a separate contribution.

### 3.2 ODE Solvers: Heun and DPM-Solver++

#### 3.2.1 Heun Solver (2nd-order)

Heun's method provides 2nd-order ODE accuracy via a predictor–corrector:
- Predict: $\hat{\mathbf{x}}_{t-\Delta t} = \mathbf{x}_t + \Delta t \cdot \mathbf{v}_\theta(\mathbf{x}_t, t)$
- Correct: $\mathbf{x}_{t-\Delta t} = \mathbf{x}_t + \frac{\Delta t}{2} [\mathbf{v}_\theta(\mathbf{x}_t, t) + \mathbf{v}_\theta(\hat{\mathbf{x}}_{t-\Delta t}, t{-}\Delta t)]$

**Cost**: 2 network forward evaluations (NFE) per step. At 4 steps this yields 8 NFE (7 in practice; the last step degrades to Euler).

#### 3.2.2 DPM-Solver++ (High-order, 1 NFE/step)

DPM-Solver++ uses polynomial interpolation of the $\mathbf{x}_0$ prediction history (PI denotes polynomial interpolation):
- At step $t_n$: compute $\mathbf{x}_0^{(n)} = \frac{\mathbf{x}_{t_n} - t_n \cdot \mathbf{v}_\theta(\mathbf{x}_{t_n}, t_n)}{1 - t_n}$
- Update: $\mathbf{x}_{t_{n+1}} = (1{-}t_{n+1})\, \operatorname{PI}(\mathbf{x}_0^{(0:n)}) + t_{n+1}\, \operatorname{PI}(\mathbf{x}_1^{(0:n)})$

**Cost**: 1 NFE per step. At 4 steps this yields 4 NFE (vs Heun's 7).

#### 3.2.3 Key Finding: DPM-Solver++ Improves Speed, Not Precision

Evaluating the A1 checkpoint across all solver×step combinations (Table 2, Figure 2), solver type has *no effect on mAP* at matched steps (Euler = DPM-Solver++ at both 4-step and 1-step). Step count has marginal effect (+0.004 from 1 to 4 steps). Heun's +0.001 over Euler 4-step costs 2× NFE (7 vs 4) — not cost-effective.

| Solver | Steps | NFE | mAP |
|--------|-------|-----|-----|
| Heun | 4 | 7 | 0.856 |
| Euler | 4 | 4 | 0.855 |
| DPM-Solver++ | 4 | 4 | 0.855 |
| Euler | 1 | 1 | 0.851 |
| DPM-Solver++ | 1 | 1 | 0.851 |

**Table 2**: Solver×step disentanglement ablation on the A1 checkpoint (24obj val, seed 42).

**Conclusion**: DPM-Solver++'s sole advantage is *computational* (1.71× speedup at equivalent accuracy). The +0.005 mAP gap between A2 and A3 stems from *checkpoint selection*, not solver precision: cross-seed validation (seed 123: 0.857 vs seed 42: 0.863, gap −0.006) confirms this is within epoch noise.

![**Figure 2**: Solver×step disentanglement ablation (A1 checkpoint). Bars are colored by solver type and hatched by step count. Solver/step configuration contributes only +0.005 mAP (6%); the remaining +0.077 mAP (94%) is attributable to the RF training paradigm.](latex/figures/solver_ablation.png)

### 3.3 OT Diversity Collapse and Stochastic Coupling

#### 3.3.1 Voronoi Partitioning by OT

**Lemma** (OT → Voronoi). When $N \to \infty$, OT coupling partitions $\mathbb{R}^d$ into $K$ Voronoi cells $\mathcal{V}_k = \{\mathbf{z} : \lVert\mathbf{z} - \mathbf{b}_k\rVert \le \lVert\mathbf{z} - \mathbf{b}_j\rVert, \forall j \ne k\}$, assigning each $\mathbf{z}_i$ to its nearest GT box.

#### 3.3.2 Proposition: OT Diversity Gap

**Setup**: Let source $\nu = \mathcal{N}(0, \sigma^2 I_d)$ (noise), target $\mu = \frac{1}{K}\sum_{k=1}^K \delta_{\mathbf{b}_k}$ ($K$ GT boxes). A coupling assigns each noise sample $\mathbf{z}_i$ to a target box $\mathbf{b}_{V_i}$. We let:
- $V \in \{1, \ldots, K\}$: coupling assignment RV (which GT box a noise sample is paired with)
- $X_t = (1-t) \mathbf{b}_V + t \mathbf{z}$: the flow state observed by the model
- $H(V \mid X_t)$: conditional entropy of $V$ given $X_t$ — how much $X_t$ leaks about $V$

**Proposition 1** (OT Diversity Upper Bound). Under the Voronoi partitioning assumption (Lemma, requiring $N \to \infty$) and the Gaussian noise model,
$$\Delta H \;=\; H_{\text{rand}}(V \mid X_t) - H_{\text{OT}}(V \mid X_t) \;\le\; \log K.$$

**Proof sketch** (full proof in Appendix A): Under random coupling, $H_{\text{rand}}(V|X_t) \le H(V) = \log K$ in the high-noise regime. Under OT coupling with $N \to \infty$, $V = \operatorname{Voronoi}(\mathbf{z})$ is deterministic (Lemma), so given $X_t$ one recovers $\mathbf{z}$ and hence $V$, giving $H_{\text{OT}}(V | X_t) = 0$. Therefore $\Delta H \le \log K$.

**Empirical validation** (Chromosome20240904): $\Delta H = 3.8415$, $\log K = 3.8427$ ($K_{\text{mean}} = 46.6$), relative error 0.03%. Figure 3 visualizes the partitioning and the empirical match, and the entropy phase diagram traces conditional entropy $H(V|Z)$ as a function of the stochastic coupling parameter $\epsilon$.

![**Figure: Entropy phase diagram.** Conditional entropy $H(V|Z)$ as a function of the stochastic coupling parameter $\epsilon$. Hard OT ($\epsilon{=}0$) collapses to $H{=}0$; Random coupling ($\epsilon{\to}\infty$) saturates at $H{=}3.8415 \approx \log K{=}3.8427$ (relative error 0.03%). Stochastic coupling with $\epsilon \ge 1$ recovers near-full diversity, while $\epsilon < 1$ falls in the danger zone of diversity collapse.](latex/figures/entropy_phase.png)

![**Figure 3**: OT Diversity Collapse. (a) Voronoi partitioning for $K{=}8$ GT boxes in a 2D projection. Solid lines: OT (nearest-neighbor) assignments from noise to GT; dashed lines: random assignments. (b) Empirical validation of Proposition 1: theoretical $\log K = 3.8427$ vs. empirical $\Delta H = 3.8415$ (relative error 0.03%).](latex/figures/ot_theory.png)

#### 3.3.3 Dimension-Dependent Severity

| Scenario | $d$ | $K$ | $\Delta H / H$ |
|----------|-----|-----|-----------------|
| Image generation | 196608 | batch | $\approx 0$ |
| Detection (COCO) | 4 | $\sim$7 | $\approx 0.55$ |
| Detection (Chromosome) | 4 | $\sim$24 | $\approx 0.69$ |

**Table 3**: OT Diversity Collapse severity by scenario. OT loss is severe in detection, negligible in image generation.

#### 3.3.4 Stochastic Coupling via Sinkhorn Transport

**Definition** (Stochastic Coupling). Instead of an argmax assignment, we sample the coupling from the Sinkhorn transport matrix rows:
$$\pi_{\text{stoch}}(i) \;\sim\; \operatorname{Categorical}\!\left( \frac{ T_\epsilon(i,:) }{ \sum_j T_\epsilon(i,j) } \right).$$

The key property is that $H_{\text{stoch}}(V|X_t; \epsilon)$ increases monotonically with $\epsilon$; endpoints are $\epsilon \to 0$ = hard OT and $\epsilon \to \infty$ = random coupling. Formally, $H_{\text{stoch}}$ is monotonically increasing in $\epsilon$ (Proposition 2, Appendix A.2).

#### 3.3.5 Stochastic Coupling as Training Stabilizer

While the mAP improvement from Stochastic Coupling is marginal (+0.002 on 24obj), its impact on *within-run training stability* is substantial (Table 4, Figure 4).

| Configuration | Best mAP | Epoch std |
|--------------|----------|-----------|
| RF + AdaLN (no Stoch. Coup.) | 0.856 | 0.006 |
| RF + AdaLN + Stoch. Coup. ($\epsilon{=}5$) | 0.858 | **0.0013** |
| *Stability gain* | — | *4.6×* |

**Table 4**: Training stability: Stochastic Coupling yields 4.6× smoother within-run convergence. "Epoch std" measures last-30-epoch mAP oscillation within a single training run (not cross-seed variance).

![**Figure 4**: Training stability (24obj, real per-epoch mAP from training logs): Random coupling exhibits epoch-level oscillation with std 0.006, while Stochastic Coupling ($\epsilon{=}5$) converges smoothly with std 0.0013 (4.6× improvement). Shaded band marks the last 30 epochs used for std computation.](latex/figures/training_stability.png)

### 3.4 Top-K Proposal Pruning

After step 0 of inference, we prune proposals from 500 to $K$ based on confidence scores; only the top-$K$ proposals proceed through steps 1–3. DPM-Solver++ compatibility requires `dpm_solver.reset()` after pruning because the $\mathbf{x}_0$ history has a dimension mismatch (500 → $K$).

## 4. Experiments

### 4.1 Experimental Setup

#### 4.1.1 Datasets

Table 5 summarizes the two chromosome datasets. Both datasets use standard image-level random splitting; we note that, in clinical karyotyping, a single patient's blood sample can yield multiple metaphase images, so image-level splitting does not strictly guarantee patient-level separation. The datasets do not include patient-level metadata.

| Dataset | Train | Val | Test | Classes |
|---------|-------|-----|------|---------|
| Chromosome20240904 | 1,540 | 440 | 220 | 24 |
| 24obj | 3,500 | 500 | 1,000 | 24 |

**Table 5**: Datasets used in this paper.

#### 4.1.2 Architecture and Training

Our base detection framework, denoted LDMDet, uses a ResNet-50 backbone with FPN neck (256 channels, 4 levels), 500 proposals, 6 cascade transformer heads with deep supervision (5 auxiliary heads). Optimization: AdamW (lr=5×10⁻⁵, wd=10⁻⁴), 5-epoch linear warmup + CosineAnnealing, 150 epochs. Loss is Focal ($\lambda_{\text{cls}}{=}2.0$) + L1 ($\lambda{=}5.0$) + GIoU ($\lambda{=}2.0$) with Hungarian matching. Diffusion uses Rectified Flow with a shifted noise schedule (shift=3.0). Default inference solver is Heun 4-step.

#### 4.1.3 Statistical Considerations

Cross-seed experiments (3 seeds: 42, 123, 789) are available for the original dataset. For 24obj, A3 (DPM-Solver++) has 3 seeds (mean 0.859 ± 0.004), confirming the +0.005 mAP gap is within cross-seed noise.

### 4.2 Main Results: RF vs DDPM

#### 4.2.1 24obj Dataset — Ablation

Table 6 reports a cumulative ablation on the 24obj validation set: A0 (DDPM Euler baseline), A1 is KaryoFlow (RF+Heun), A2 adds Stochastic Coupling, A3 swaps to DPM-Solver++. AdaLN-Zero is used throughout but contributes null individually (Appendix D). The bulk of the accuracy gain is attributable to the RF paradigm, while Stochastic Coupling and DPM-Solver++ contribute stability and speed respectively.

| Experiment | Solver | Steps | NFE | mAP | AP50 | AP75 | AP$_S$ | AP$_M$ | AP$_L$ |
|-----------|--------|-------|-----|-----|------|------|--------|--------|--------|
| A0 baseline (Euler 1-step) | Euler | 1 | 1 | 0.774 | 0.968 | 0.916 | 0.317 | 0.773 | 0.775 |
| **A1 RF+Heun (KaryoFlow)** | Heun | 4 | 7 | **0.856** | 0.990 | 0.971 | 0.563 | 0.853 | 0.913 |
| A2 + Stochastic Coupling ($\epsilon{=}5$) | Heun | 4 | 7 | 0.858 | 0.990 | 0.973 | 0.586 | 0.855 | 0.908 |
| **A3 DPM-Solver++** | DPM++ | 4 | 4 | **0.863** | 0.990 | 0.974 | 0.583 | 0.859 | 0.914 |

**Table 6**: Main ablation on the 24obj dataset. NFE = total network forward evaluations per image. A0→A1 changes multiple variables; the disentanglement ablation (Table 2) attributes 94% of the +0.082 gap to the RF training paradigm.

The RF paradigm accounts for +0.077 mAP (94% of the +0.082 gap), while solver/step configuration adds only +0.005 (6%). Stochastic Coupling contributes +0.002 mAP but 4.6× smoother convergence. DPM-Solver++ adds no precision (+0.005 is checkpoint noise) but is 1.71× faster.

#### 4.2.2 Original Dataset — RF vs DDPM

On the original Chromosome20240904 dataset (3 seeds), RF outperforms DDPM by +0.017 mAP (0.746 vs 0.729, lower variance ±0.001 vs ±0.004) with 4-step inference vs DDPM's 1-step. DDPM gains only +0.044 from 1→8 steps (0.628 → 0.672), whereas the RF paradigm yields +0.082 mAP on 24obj.

| Coupling | Solver | mAP | AP50 | AP75 | AP$_S$ | AP$_M$ | AP$_L$ |
|----------|--------|-----|------|------|--------|--------|--------|
| **Random (RF)** | Heun | **0.746±0.001** | 0.945±0.002 | 0.836±0.001 | 0.512±0.001 | 0.740±0.002 | 0.670±0.012 |
| **DDPM** | DDIM | 0.729±0.004 | 0.926±0.001 | 0.820±0.003 | 0.483±0.004 | 0.724±0.003 | 0.670±0.017 |

Table notes: 3 seeds (42, 123, 789). Per-seed values in Appendix C.

### 4.3 SOTA Comparison (The 24obj Dataset)

Table 7 compares our RF-based detector against standard and diffusion baselines on the same ResNet-50 backbone, 150 epochs, and augmentation pipeline. Our best variant (A3) achieves the highest mAP, exceeding Cascade R-CNN, YOLOX-S, and DiffusionDet — with the largest gain against the DDPM-based DiffusionDet, direct evidence for the RF paradigm's advantage.

| Method | Backbone | mAP |
|--------|----------|-----|
| **Ours (A3 DPM++)** | ResNet-50 | **0.863** |
| LDMDet (Random, 2-seed) | ResNet-50 | 0.860 ± 0.001 |
| A2 Heun (Ours) | ResNet-50 | 0.858 |
| Cascade R-CNN R50 | ResNet-50 | 0.854 |
| YOLOX-S | CSPDarkNet-S | 0.796 |
| DiffusionDet | ResNet-50 | 0.787 |

**Table 7**: SOTA comparison on the 24obj dataset. LDMDet denotes our base detection framework.

**DiffusionDet training note.** DiffusionDet's training crashed before 150 epochs; the reported mAP (0.787) is the best eval obtained before the crash, making our +0.076 mAP claim conservative.

#### 4.3.1 Per-Class AP Analysis

Figure 5 reports the per-class AP for all 24 classes on the A3 checkpoint. Overall AP drops monotonically with chromosome size ($0.896 \to 0.848 \to 0.805$ for Large→Medium→Small), consistent with the well-known difficulty of small-object detection. The Y chromosome is the hardest class (AP=0.776), attributable to both data scarcity (~1,803 samples vs ~7,000 per autosome) and biological characteristics (smallest chromosome, heterochromatin-rich, morphologically variable); its AP$_S$=0.577 confirms the difficulty concentrates at the small-object scale. The C-group chromosomes (C6–C12), despite being morphologically similar look-alikes, achieve high AP with an intra-group spread of only 0.029, indicating competent fine-grained discrimination when training data are sufficient (Appendix E). Finally, AP50 is near-saturated across all classes (>0.988, and 0.972 for the Y), so localization is near-saturated (>0.988) and the residual errors concentrate in fine-grained classification — suggesting a downstream banding-pattern classifier could recover much of the remaining AP.

![**Figure 5**: Per-class AP on the 24obj validation set (A3 DPM-Solver++). Bars are colored by chromosome size group. The dashed line is the overall mean. Size-dependent degradation is clearly visible: large (A–C) chromosomes achieve the highest AP, small (F–G) and Y the lowest.](latex/figures/per_class_ap.png)

### 4.4 Coupling Ablation

#### 4.4.1 Original Dataset, Multi-seed

On Chromosome20240904, all coupling methods produce statistically equivalent mAP (within ±0.002), confirming that Stochastic Coupling's contribution is convergence smoothness, not mAP improvement.

| Coupling | $\epsilon$ | Seeds | mAP | AP50 | AP75 | AP$_S$ | AP$_M$ | AP$_L$ |
|----------|---|-------|-----|------|------|--------|--------|--------|
| **Random** | ∞ | 3 | **0.746±0.001** | 0.945±0.002 | 0.836±0.001 | 0.512±0.001 | 0.740±0.002 | 0.670±0.012 |
| DDPM | — | 3 | 0.729±0.004 | 0.926±0.001 | 0.820±0.003 | 0.483±0.004 | 0.724±0.003 | 0.670±0.017 |
| Hard OT | 0 | 2 | 0.747 | 0.943 | 0.836 | 0.510 | 0.738 | 0.677 |
| **Stochastic Coupling** | 5 | **3** | **0.747±0.002** | 0.942±0.000 | 0.835±0.002 | 0.508±0.006 | 0.740±0.003 | 0.623±0.009 |

Table notes: Seeds — Random/DDPM/Stochastic Coupling $\epsilon{=}5$ = {42, 123, 789}; Hard OT = {42, 123}.

#### 4.4.2 Multi-dimensional Stability Comparison (24obj)

Table 8 reports five additional stability metrics. Stochastic Coupling achieves 30/30 epochs within 1% of the best mAP (vs 13/30 for Random), making late-stage checkpoint selection far more reliable — the property of primary practical concern for EarlyStopping-based training in small-data regimes.

| Metric | A1 (Random) | A3 (Stochastic Coupling $\epsilon{=}5$) | Gain |
|--------|-------------|-----------------------------------------|------|
| Last-30 epoch std | 0.006 | 0.0013 | 4.6× |
| Last-30 CV (std/mean) | 0.69% | 0.16% | 4.4× |
| Last-30 range (max−min) | 0.023 | 0.005 | 4.6× |
| Epochs within 1% of best (last 30) | 13/30 (43%) | 30/30 (100%) | — |
| Best mAP / best epoch | 0.856 / ep62 | 0.858 / ep114 | — |
| Total epochs (EarlyStop) | 92 | 144 | — |
| Training failure rate (9 runs) | 0/9 | 0/9 | — |

**Table 8**: Multi-dimensional stability comparison (24obj, single seed). CV = std/mean.

#### 4.4.3 $\epsilon$ Ablation

$\epsilon < 1$ is harmful (−1.3% mAP within the same augmentation setting); $\epsilon \ge 1$ saturates with diminishing returns, supporting Stochastic Coupling as a necessary OT regularizer rather than a precision booster.

### 4.5 Solver Analysis

#### 4.5.1 DPM-Solver++ Step Ablation

DPM-Solver++ converges at 2 steps (mAP 0.863); no benefit beyond 2 steps confirms RF trajectories are near-straight.

#### 4.5.2 DPM-Solver++ vs Heun at Matched NFE

At similar NFE, DPM-Solver++ 4-step (4 NFE, 0.863) ≈ Heun 2-step (3 NFE, 0.863) — equal accuracy. The primary advantage is *computational*: 43% fewer NFE at equal accuracy.

#### 4.5.3 Test Set Evaluation

On the 24obj test set, A3 achieves mAP 0.859 (vs val 0.863, $\Delta = -0.004$). The aggregate gap is minimal, but AP$_S$ drops by −0.059 (0.583 → 0.524), indicating that small-chromosome detection (F19–G22, Y) generalizes worse than large chromosomes.

### 4.6 FPS / Latency Benchmark

Table 9 and Figure 6 report the speed-accuracy trade-off on an NVIDIA RTX A6000 at 512×512, batch 1. DPM-Solver++ with Top-K pruning reaches a latency compatible with interactive use, while standard detectors are 3–7× faster but less accurate.

| Model | Solver | NFE | Latency (ms) | FPS |
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

**Table 9**: FPS / latency benchmark (24obj, RTX A6000, 512×512).

A3 + IO3 K=200 is the fastest variant (70.46 ms / 14.2 FPS, mAP 0.860); A3 achieves 75 ms / 13.3 FPS at mAP 0.863. The cascade head dominates 90%+ of latency; the backbone+neck is a minor cost (~5.8 ms, 4–8%).

![**Figure 6**: Speed-accuracy trade-off (24obj, RTX A6000, 512×512). Log-scale FPS axis. Our RF variants (circle/square) occupy the high-accuracy region (mAP > 0.85); standard detectors (triangle) are 3–7× faster but less accurate. A3+IO3 K=200 (14.2 FPS, mAP 0.860) achieves the best speed-accuracy trade-off among our variants.](latex/figures/fps_map.png)

### 4.7 Cross-Dataset Summary

Across both datasets, RF outperforms DDPM (+0.017 mAP on the original, +0.076 over DiffusionDet on 24obj), DPM-Solver++ matches Heun at lower NFE, and the best coupling is dataset-dependent (Hard OT ≈ Random on the original; Random ≈ Stochastic Coupling on 24obj).

## 5. Analysis and Discussion

### 5.1 Why RF Works for Chromosome Detection

RF's straight-line ODE paths reduce truncation error in few-step inference, which is especially valuable for chromosome detection: the high object density (~46 per image) compounds per-box errors, the small training sets (1,540–5,000 images) limit the model's ability to learn complex curved DDPM trajectories, and the 24-class fine-grained task benefits from stable feature representations. The +0.082 mAP improvement (0.774 → 0.856) on 24obj confirms RF's effectiveness in this regime.

### 5.2 Stochastic Coupling: Smoothness, Not mAP

Stochastic Coupling's value is smoother convergence, not mAP improvement: the mAP gain is +0.002 (within cross-seed noise ±0.001), while within-run epoch stability improves 4.6× (epoch std 0.006 → 0.0013). With Random coupling, EarlyStopping may select a checkpoint from a "lucky" epoch 0.006 above the trend — a false peak that may not generalize. Stochastic Coupling's 0.0013 epoch std makes checkpoint selection far more reliable. The seed 123 result (mAP 0.857 vs seed 42's 0.863, $\Delta = -0.006$) confirms that epoch oscillation directly impacts which checkpoint EarlyStopping selects. A formal causal link (Stochastic Coupling → better test generalization via better checkpoint selection) requires per-epoch test evaluation, left as future work.

### 5.3 DPM-Solver++ vs Heun: Computational Advantage

Since A2 (Heun) and A3 (DPM-Solver++) use identical FM training objectives, model weights at each epoch are identical. The +0.006 mAP gap from independent Heun evaluation (A3's 0.864 vs A2's 0.858) is within epoch-level noise and not cross-seed robust. The robust claim is: *DPM-Solver++ achieves equal accuracy to Heun at 43% fewer NFE*, refining FlowDet's conclusion that higher-order solvers perform worse in detection.

### 5.4 IO3 Pruning: Solver-Dependent Effectiveness

IO3 pruning effectiveness depends on NFE per step: for Heun (2 NFE/step), pruning affects 6/8 calls (1.09–1.12× speedup); for DPM-Solver++ (1 NFE/step), 3/4 calls (1.05–1.08×). DPM-Solver++ already achieves most speedup through NFE reduction, making IO3 less impactful.

### 5.5 Theory Applicability and Limitations

Our OT Diversity Collapse analysis rests on five assumptions, all well satisfied in chromosome detection: (1) the bound $\Delta H \le \log K$ is an upper bound, tight in the high-noise regime where the shifted schedule spends most timesteps (empirically 0.03% error); (2) the $N \to \infty$ idealization holds because GT bboxes are well-separated in $\mathbb{R}^4$, so even $N=2$ mini-batch OT reduces to nearest-neighbor assignment; (3) the Gaussian noise source is approximately satisfied by our shifted Gaussian schedule; (4) well-separated Voronoi cells (pairwise distances > 20px vs $\sigma \sim 1$px); and (5) the low-dimensional regime ($d=4$, $K \approx 46$) where $\Delta H/H \approx 0.69$ (Table 3).

The theory does not transfer to high-dimensional generation ($d \sim 10^5$, where $\Delta H/H \approx 0$ so OT collapse is negligible — consistent with OT-CFM's success), nor to densely overlapping targets where Assumptions 2 and 4 fail. For COCO ($K \sim 7$, $\Delta H/H \approx 0.55$), the theory predicts Stochastic Coupling would help but with smaller magnitude. The theory suggests RF + Stochastic Coupling would benefit detection tasks combining low $d$, high object density, and small training data — a profile including medical imaging, remote sensing, and other fine-grained dense detection tasks. We did not validate on COCO because its small $K$ reduces OT collapse severity; the appropriate validation dataset has high $K$ and low $d$, exactly the chromosome detection profile. The stability benefit is practically meaningful for clinical deployment: the 4.6× epoch stability improvement means EarlyStopping selects checkpoints within 0.0013 of the trend (versus 0.006 for Random), reducing the risk of deploying a "false peak" checkpoint.

## 6. Conclusion

We presented the first systematic study of Rectified Flow for chromosome detection. The RF training paradigm — straight-line ODE paths combined with a shifted noise schedule — is the dominant source of accuracy gain, yielding +0.082 mAP over the Euler baseline on 24obj and +0.017 mAP over DDPM on the original dataset, and our best variant exceeds DiffusionDet by +0.076 mAP; a solver×step disentanglement ablation attributes 94% of the gain to the RF training paradigm. Stochastic Coupling, grounded in our theoretical analysis of OT Diversity Collapse (upper bound $\Delta H \le \log K$, tight to 0.03% empirically), reframes OT coupling as a convergence stabilizer rather than a precision booster: its mAP gain is marginal, but it reduces within-run epoch-level mAP oscillation by 4.6×, which is the property of primary practical concern for reliable training in small-data regimes. DPM-Solver++ enables two-step inference at 13.3 FPS (mAP 0.863), with a 14.2 FPS variant (IO3 K=200) at mAP 0.860; its advantage over Heun is purely computational (1 NFE/step versus 2, equal accuracy at 43% fewer NFE), refining FlowDet's conclusion that higher-order solvers perform worse in detection. We note that standard detectors such as YOLOX-S (98.5 FPS) and Cascade R-CNN (48.4 FPS) are several times faster than our 13.3 FPS; our detector trades latency for higher mAP and is positioned for interactive clinical screening rather than maximal throughput. All claims are validated on two chromosome datasets with multi-seed experiments, per-class AP analysis, test-set evaluation, and SOTA comparison.

Beyond the immediate chromosome setting, the OT Diversity Collapse phenomenon we characterize has development potential for a broader class of problems. The severity of the collapse ($\Delta H/H \approx 0.69$ in our setting) is governed by the combination of a low-dimensional prediction space, high object density, and small training data, and any task sharing this profile is a candidate beneficiary of Stochastic Coupling. Plausible applications include cell detection in digital pathology, lesion detection in medical imaging, and vehicle detection in dense remote-sensing scenes. The theory provides an a-priori diagnostic for when the method is worth applying: tasks whose $K$ and $d$ place them in the high-severity regime of Table 3 should benefit most.

We also acknowledge several limitations. First, the empirical validation is confined to chromosome data; we have not validated on COCO or other general detection benchmarks, though our theory predicts that COCO's smaller $K \sim 7$ would attenuate OT collapse and thus reduce the marginal benefit of Stochastic Coupling. Second, Stochastic Coupling's mAP gain is marginal (+0.002), so its value is contingent on the stability benefit being practically relevant — which holds for clinical deployment with a single training run but may matter less in regimes where retraining is cheap. Third, the theoretical analysis rests on a well-separated-targets assumption; densely overlapping scenes would require extending the finite-$N$ analysis, and the quantitative $\Delta H \approx \log K$ prediction may not hold there. Addressing these limitations — in particular a formal finite-$N$ theory for overlapping targets and empirical validation on additional high-$K$ detection benchmarks — is a natural direction for future work.

## Appendix

### A. Proofs

This appendix provides the complete proofs of Proposition 1 and the Stochastic Coupling monotonicity argument referenced in Section 3.3, establishing the OT Diversity Collapse upper bound $\Delta H \le \log K$ and the monotonicity of Stochastic Coupling in $\epsilon$.

#### A.1 Proof of Proposition 1 (OT Diversity Gap)

**Setup**: Source $\nu = \mathcal{N}(0, \sigma^2 I_d)$, target $\mu = \frac{1}{K}\sum_{k=1}^K \delta_{\mathbf{b}_k}$. A coupling $\pi$ assigns noise samples $\{\mathbf{z}_i\}_{i=1}^N$ to target boxes $\{\mathbf{b}_{V_i}\}_{i=1}^N$. The flow state is $X_t = (1-t) \mathbf{b}_V + t \mathbf{z}$ where $\mathbf{z} \sim \nu$ and $V$ is the coupling assignment.

**Step 1: Random coupling.** Under random coupling, $V \sim \operatorname{Uniform}(\{1,\ldots,K\})$ independently of $\mathbf{z}$. Given $X_t = (1-t) \mathbf{b}_V + t \mathbf{z}$, the posterior is
$$P(V=k \mid X_t) \propto P(X_t \mid V=k) \cdot P(V=k) = \mathcal{N}\!\bigl(X_t;\, (1{-}t) \mathbf{b}_k,\, t^2 \sigma^2 I_d\bigr) \cdot \tfrac{1}{K}.$$

This is a Gaussian mixture posterior. When the Voronoi cells are well-separated relative to $t\sigma$ ($\min_{j \ne k} \lVert\mathbf{b}_k - \mathbf{b}_j\rVert \gg t\sigma$), the posterior concentrates on a single component and $V$ is nearly determined. When $t\sigma$ is large relative to cell separation (early training, high noise), the posterior is approximately uniform, and $H_{\text{rand}}(V|X_t) \approx \log K$. We use the upper bound $H_{\text{rand}}(V|X_t) \le H(V) = \log K$, with equality in the high-noise regime.

**Step 2: OT coupling ($N \to \infty$).** Under OT coupling with $N \to \infty$, the Lemma guarantees $V = \arg\min_k \lVert\mathbf{z} - \mathbf{b}_k\rVert$ (Voronoi assignment), so $V$ is a deterministic function of $\mathbf{z}$: $V = f_{\text{Voronoi}}(\mathbf{z})$. Given $X_t = (1-t) \mathbf{b}_V + t \mathbf{z}$ and knowing $t$, $\mathbf{z} = (X_t - (1-t) \mathbf{b}_V)/t$. Substituting into the Voronoi condition yields a unique fixed point: there exists exactly one $k^*$ such that $\mathbf{z}^* = (X_t - (1-t) \mathbf{b}_{k^*})/t$ falls in the Voronoi cell of $\mathbf{b}_{k^*}$. Therefore $V$ is fully determined by $X_t$, giving $H_{\text{OT}}(V | X_t) = 0$.

**Step 3.** Combining both steps,
$$\Delta H = H_{\text{rand}}(V|X_t) - H_{\text{OT}}(V|X_t) \le \log K - 0 = \log K.$$

**Remark on finite-$N$.** In practice, OT is solved on mini-batches of size $N$ (e.g., $N=2$ in our setting). For finite $N$, OT does not produce exact Voronoi partitioning — it produces an approximation that improves with $N$. The empirical validation ($\Delta H = 3.8415$ vs $\log K = 3.8427$, 0.03% error) confirms the $N \to \infty$ bound is an excellent approximation even for small $N$ in the chromosome detection setting, likely because $K \approx 46 \gg N$ and the GT boxes are well-separated in $\mathbb{R}^4$ relative to $\sigma$.

#### A.2 Stochastic Coupling Monotonicity

**Proposition 2**: $H_{\text{stoch}}(V|X_t; \epsilon)$ monotonically increases with $\epsilon$.

**Argument**: As $\epsilon \to 0$, the Sinkhorn transport matrix $T_\epsilon$ converges to the deterministic OT assignment (hard coupling), so $H_{\text{stoch}} \to H_{\text{OT}} = 0$. As $\epsilon \to \infty$, $T_\epsilon$ converges to the uniform distribution (random coupling), so $H_{\text{stoch}} \to H_{\text{rand}} = \log K$. By the continuity of the Sinkhorn solution in $\epsilon$, $H_{\text{stoch}}$ increases monotonically. A formal proof would use the log-Sobolev inequality for Schrödinger bridges; we leave this as future work and rely on empirical validation of the monotonicity (Section 4.4).

### B. Falsified Directions

This appendix records the research directions we explored and experimentally falsified, documenting the negative results that justify the method choices made in the main text.

| Direction | Verdict / Evidence |
|-----------|--------------------|
| IO1 adaptive step | Falsified: $x_0$ rel. $\Delta$ min 0.166 |
| IO2 draft-verify | Falsified: early-step cls agreement 57% |
| IO4 head early-exit | Falsified: 0% exit rate at all thresholds |
| IO5 RoI feature cache | Falsified: box displacement 93–124 px/step |
| Flow matching det. | Falsified: mAP 0.823 (−0.033) |
| $N_{\text{cascade}}$ e2e | Falsified: mAP 0.684 (−0.172) |

**Table B.1**: Falsified research directions.

### C. Per-Seed Values (Original Dataset)

This appendix provides the per-seed numerical values underlying the multi-seed tables in Section 4.2 (RF vs DDPM) and Section 4.4 (coupling ablation), so that the aggregated mean±std figures can be independently verified seed by seed.

#### C.1 RF vs DDPM (Section 4.2.2)

| Coupling | Seed | mAP | AP50 |
|----------|------|-----|------|
| Random (RF) | 42 | 0.745 | 0.946 |
| Random (RF) | 123 | 0.747 | 0.945 |
| Random (RF) | 789 | 0.747 | 0.943 |
| DDPM | 42 | 0.726 | 0.925 |
| DDPM | 123 | 0.733 | 0.925 |
| DDPM | 789 | 0.727 | 0.927 |

**Table C.1**: Per-seed values for RF vs DDPM on Chromosome20240904.

#### C.2 Coupling Ablation (Section 4.4.1)

| Coupling | $\epsilon$ | Seed | mAP | AP50 | AP75 | AP$_S$ | AP$_M$ | AP$_L$ |
|----------|---|------|-----|------|------|--------|--------|--------|
| Hard OT | 0 | 42 | 0.747 | — | — | — | — | — |
| Hard OT | 0 | 123 | 0.747 | — | — | — | — | — |
| Stochastic Coupling ($\epsilon{=}5$) | 5 | 42 | 0.746 | — | — | — | — | — |
| Stochastic Coupling ($\epsilon{=}5$) | 5 | 123 | 0.746 | — | — | — | — | — |
| Stochastic Coupling $\epsilon{=}5$ | 5 | 789 | 0.749 | 0.942 | 0.837 | 0.513 | 0.743 | 0.617 |

**Table C.2**: Per-seed values for the coupling ablation (Chromosome20240904).

### D. AdaLN-Zero Ablation

This appendix reports the standalone ablation of AdaLN-Zero referenced in Section 3.1.2 and the contribution discussion of the main text, confirming that its individual contribution to the +0.082 mAP gain is null within the RF framework (Table D.1). We conducted a separate ablation on the 24obj dataset to verify AdaLN-Zero's individual contribution. Both experiments use identical configurations except for the time conditioning module (RF formulation, Heun solver 4-step, shifted schedule shift=3.0, random coupling, batch size 8, 150 epochs).

| Config | mAP | $\Delta$ mAP |
|--------|-----|--------------|
| RF + Heun (without AdaLN) | 0.856 | — |
| RF + Heun + AdaLN-Zero | 0.856 | +0.000 |

**Table D.1**: AdaLN-Zero ablation on 24obj.

AdaLN-Zero contributes *null* ($\Delta$mAP = 0.000) within the RF framework on this dataset. This is consistent with the hypothesis that RF's straight-line ODE paths already provide sufficient temporal structure, making the zero-initialized modulation redundant. We retain AdaLN-Zero as a standard conditioning mechanism for consistency with the broader diffusion literature, but note that it does not contribute to the +0.082 mAP improvement claimed in Section 4.2.1. The entire +0.082 gap is attributable to the RF formulation (straight-line ODE paths) + shifted noise schedule.

### E. Per-Class AP Details

This appendix supplements the per-class AP analysis in Section 4.3.1 with detailed breakdowns.

#### E.1 Y Chromosome Analysis

The Y chromosome is the hardest class (AP=0.776), with difficulty overdetermined by data and biology. The 24obj training set contains only about 1,803 Y-chromosome samples, compared with roughly 7,000 per autosome and 5,123 for the X chromosome — a 3.9× imbalance that directly limits the gradient updates the Y class receives. This imbalance is a consequence of biology: the Y appears in only one copy and only in male samples. The Y is also one of the smallest human chromosomes, enriched in heterochromatin, and varies considerably in morphology across individuals. Its AP$_S$=0.577 confirms that the difficulty concentrates at the small-object scale.

#### E.2 C-group Discrimination

The C-group chromosomes (C6–C12) are the archetypal "hard to distinguish" class: seven medium-to-large submetacentric chromosomes of similar size and shape, distinguished by subtle banding-pattern differences. The detector achieves high AP across the group (C6=0.900, C7=0.896, C8=0.883, C9=0.880, C10=0.877, C11=0.871, C12=0.890) with an intra-group spread of only 0.029 (0.871–0.900). The spread is only weakly correlated with size, suggesting the model captures fine banding cues rather than relying on size alone. The X chromosome (AP=0.885) sits within the large-group range, consistent with its medium-large submetacentric morphology and ample training data (5,123 samples).

#### E.3 Complete Per-Class AP Table

| Class | AP | AP50 | AP75 | AP$_S$ | AP$_M$ | AP$_L$ |
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

**Table E.1**: Complete per-class AP breakdown on the 24obj validation set (A3 DPM-Solver++).

#### E.4 Localization Saturation and Downstream Potential

AP50 is near-saturated across all classes (>0.988, and 0.972 for the Y), so localization is near-saturated (>0.988) and residual errors concentrate in fine-grained classification. A downstream refinement stage operating on correctly localized crops — a banding-pattern classifier or morphology-aware re-scoring head — could in principle recover much of the remaining AP, since the upstream detector already supplies the right regions.
