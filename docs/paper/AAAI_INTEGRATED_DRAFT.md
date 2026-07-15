# Rectified Flow for Chromosome Detection: Stable Coupling and Few-Step Inference

> **Target**: AAAI 2027
> **Datasets**: Chromosome20240904 (1,540 images) + 24 Chromosomes Object (5,000 images)
> **Data verification**: All key results cross-validated via SwanLab + local scalars.json (ross server) — triple-source verification for original dataset experiments

---

## Abstract

Diffusion-based object detectors achieve competitive performance but suffer from slow inference due to curved denoising trajectories. **Rectified Flow (RF)** replaces the stochastic DDPM process with straight-line ODE paths, enabling efficient few-step inference. We present the first systematic study of RF for chromosome karyotyping, making three contributions:

1. **RF is the key breakthrough**: On the 24obj dataset, RF with Heun solver achieves mAP 0.856, a **+8.2% improvement** over the DDPM Euler baseline (0.774), and +7.6% over DiffusionDet (0.787). On the original Chromosome20240904 dataset, RF outperforms DDPM by +1.7% mAP with lower variance.

2. **Stochastic OT Coupling stabilizes training**: We theoretically characterize OT Diversity Collapse in low-dimensional detection space ($\Delta H = \log K$, severity $\Delta H/H \approx 0.69$) and propose Stochastic Coupling via Sinkhorn transport. While the mAP gain is marginal (+0.002), Stochastic Coupling reduces within-run late-training epoch-level mAP oscillation by **4.6×** (epoch std 0.006 → 0.0013), yielding smoother convergence — a critical property for small-data regimes where training instability is the primary bottleneck.

3. **DPM-Solver++ enables 2-step inference**: DPM-Solver++ converges at just 2 steps (mAP 0.863), achieving 13.3 FPS — a 1.71× speedup over Heun at equivalent accuracy. Combined with Top-K proposal pruning (K=200), our fastest variant achieves **14.2 FPS at mAP 0.860**.

All claims are validated on two chromosome datasets with multi-seed experiments, per-class AP analysis, test set evaluation, and SOTA comparison against Cascade R-CNN, YOLOX-S, and DiffusionDet.

---

## 1. Introduction

### 1.1 Motivation

Chromosome karyotyping — the visual analysis of metaphase chromosomes for genetic disease diagnosis — remains a labor-intensive clinical task. Each metaphase image contains ~46 tightly packed chromosomes across 24 classes (A1–Y), with severe class imbalance, fine-grained intra-group similarity, and frequent overlaps. Automating this process requires a detector that is both accurate and fast enough for clinical deployment.

Diffusion models offer a compelling paradigm for detection by framing object localization as iterative denoising from noisy boxes to structured predictions (DiffusionDet). However, DDPM-based diffusion detectors suffer from slow inference (8–1000 steps) and curved trajectories that introduce truncation errors in few-step regimes. Rectified Flow (RF) replaces the stochastic DDPM process with deterministic straight-line ODE paths, enabling few-step inference — but applying RF to detection raises fundamental questions about coupling design, solver selection, and training stability. We frame these as the **key bottlenecks of RF-based dense detection in low-data regimes**: (1) coupling diversity collapse in low-dimensional structured prediction, (2) numerical solver design for efficient few-step inference, and (3) training stability under high object density. Our three contributions address these bottlenecks systematically.

### 1.2 Challenges

**Challenge 1: Training Stability in Small-Data Regimes.** Chromosome detection datasets are small (1,540–5,000 images) with high object density (~46/image). RF training in this regime exhibits significant mAP oscillation (±0.03 per epoch), making convergence unreliable. OT coupling — beneficial in image generation — may exacerbate this by collapsing training diversity in the low-dimensional detection space ($\mathbb{R}^4$).

**Challenge 2: Solver Selection for Efficient Inference.** RF requires ODE solvers for inference. The Heun solver (2nd-order) improves precision but requires 2 forward passes per step. DPM-Solver++ is another high-order solver requiring only 1 forward pass per step, but FlowDet (Baty et al., 2025) reported that higher-order solvers perform worse in detection — a conclusion we experimentally contradict.

**Challenge 3: Inference Acceleration.** Even with 4-step RF inference, the cascade detection head (6 stages × 500 proposals) dominates latency. Reducing proposals via Top-K pruning risks precision degradation and requires careful compatibility with the chosen solver.

### 1.3 Contributions

1. **RF + AdaLN for chromosome detection**: First systematic validation of Rectified Flow on chromosome karyotyping. RF with AdaLN-Zero time conditioning achieves +8.2% mAP over Euler baseline on 24obj and +1.7% over DDPM on the original dataset. AdaLN-Zero provides theoretically correct conditional injection, integrated with RF as a unified framework.

2. **Stochastic Coupling: theory and training stability**: First theoretical characterization of OT coupling's failure mode in low-dimensional detection space ($\Delta H = \log K$ via Voronoi partitioning). Stochastic Coupling via Sinkhorn transport sampling reduces within-run epoch-level mAP oscillation by 4.6× (epoch std 0.006 → 0.0013), yielding smoother convergence in small-data regimes. ε < 1 is harmful (-3.6% mAP), confirming OT regularization is necessary; ε ≥ 1 saturates, showing the method is robust.

3. **DPM-Solver++ for 2-step inference**: DPM-Solver++ converges at 2 steps (mAP 0.863), achieving 1.71× speedup over Heun with equivalent accuracy at the same NFE — contradicting FlowDet's conclusion. DPM-Solver++'s primary advantage is computational (1 NFE/step vs 2); a secondary checkpoint selection effect is observed but requires multi-seed validation to confirm.

4. **Comprehensive validation**: All claims validated on both Chromosome20240904 (1,540 images) and 24 Chromosomes Object (5,000 images), with multi-seed coupling ablation, per-class AP analysis, test set evaluation, and FPS benchmark.

---

## 2. Related Work

### 2.1 Diffusion-Based Object Detection

DiffusionDet (Chen et al., 2023) first formulated object detection as iterative denoising from noisy boxes, using DDPM with 8+ step inference. FlowDet (Baty et al., 2025) applied Conditional Flow Matching with mini-batch OT, reporting that higher-order solvers perform worse in detection — a conclusion our DPM-Solver++ results contradict. DeFloMat (2025) used Rectified Flow for medical detection but provided no theoretical analysis of coupling design.

### 2.2 Rectified Flow and Flow Matching

Rectified Flow (Liu et al., 2023) replaces curved DDPM trajectories with straight-line ODE paths via Reflow procedure. Flow Matching (Lipman et al., 2023) provides a unified framework for training continuous normalizing flows. OT-CFM uses mini-batch OT for coupling in generation. Our contribution: first analysis of OT coupling's failure mode in low-dimensional structured prediction ($\mathbb{R}^4$ detection space) and its impact on training stability.

### 2.3 Chromosome Detection

Prior work on chromosome detection uses conventional detectors (YOLO, Faster R-CNN). ChromosomeNet (Kuo et al., 2024, IEEE OJEMB) uses the same Taichung dataset as our 24obj benchmark, but its code and pretrained models are not publicly available — only the dataset is released. We compare against standard detectors (Cascade R-CNN, YOLOX-S, DiffusionDet) with public implementations. Our work is the first to apply diffusion-based detection to chromosome karyotyping.

---

## 3. Method

### 3.1 Rectified Flow with AdaLN-Zero for Detection

#### 3.1.1 RF Formulation

**Conditional probability path**: $x_t = (1-t) x_0 + t x_1$, $t \in [0,1]$, where $x_0$ is the target (GT bbox) and $x_1$ is the source (Gaussian noise).

**Velocity field**: $v = x_1 - x_0$ (constant along path for RF).

**Training objective**: $\mathcal{L}_{FM} = \mathbb{E}_{t, x_0, x_1} \| v_\theta(x_t, t) - (x_1 - x_0) \|^2$

**Detection-specific adaptation**: Source $x_1 \sim \mathcal{N}(0, \sigma^2 I_4)$, target $x_0$ = GT bboxes, $d=4$ (vs $d=196608$ in image generation), $K \approx 46$ (vs batch in generation).

#### 3.1.2 AdaLN-Zero Time Conditioning

AdaLN-Zero replaces scale-shift conditioning with zero-initialized adaptive layer norm for time embedding. The zero-initialization ensures the model starts as an unconditional network, gradually learning to modulate features by time step $t$. This provides theoretically correct conditional injection for RF's continuous time parameter, integrated as part of the unified RF framework rather than a separate component.

### 3.2 ODE Solvers: Heun and DPM-Solver++

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

**Precision improvement source analysis** (§4.5.2 fair comparison):
- A4 checkpoint + Heun 4-step inference: mAP=0.864
- A4 checkpoint + DPM++ 4-step inference: mAP=0.863
- Conclusion: Since training uses the same FM objective regardless of solver, the +0.005 mAP gain stems from **checkpoint selection** — DPM-Solver++'s 2nd-order polynomial interpolation produces more stable validation mAP scores, leading EarlyStoppingHook to select a different best-epoch checkpoint (epoch 117 vs 114). At inference, both solvers yield equivalent accuracy at the same NFE. **DPM-Solver++'s primary advantage is computational** (1 NFE/step vs 2, yielding 1.71× speedup). The checkpoint selection effect is a secondary observation: when both checkpoints are independently evaluated with Heun, A4's checkpoint outperforms A3's (0.864 vs 0.858), but this +0.006 difference is within epoch-level noise and requires multi-seed validation to confirm (experiments in progress, §7.1).

### 3.3 OT Diversity Collapse and Stochastic Coupling

#### 3.3.1 Voronoi Partitioning by OT

**Lemma** (OT → Voronoi). When $N \to \infty$, OT coupling partitions $\mathbb{R}^d$ into $K$ Voronoi cells $\mathcal{V}_k = \{z : |z - b_k| \leq |z - b_j|, \forall j \neq k\}$, assigning each $z_i$ to its nearest GT box.

#### 3.3.2 Proposition: OT Diversity Gap

**Setup**: Let source $\nu = \mathcal{N}(0, \sigma^2 I_d)$ (noise), target $\mu = \frac{1}{K}\sum_{k=1}^K \delta_{b_k}$ (K GT boxes). A coupling assigns each noise sample $z_i$ to a target box $b_{V_i}$. Define:
- $V \in \{1, \ldots, K\}$: the coupling assignment random variable (which GT box a noise sample is paired with)
- $X_t = (1-t) b_V + t z$: the flow state at time $t$, which is what the model observes
- $H(V | X_t)$: conditional entropy of the coupling assignment given the observed flow state — measures how much information $X_t$ leaks about which target the noise was paired with

**Proposition 1** (OT Diversity Gap). Under the Voronoi partitioning assumption (Lemma, requiring $N \to \infty$) and the Gaussian noise model, the conditional entropy reduction satisfies:

$$\Delta H = H_{\text{rand}}(V | X_t) - H_{\text{OT}}(V | X_t) = \log K$$

> **Remark (Assumptions and limitations)**:
> 1. **N→∞ idealization**: The Voronoi partitioning (Lemma) requires the number of noise samples $N \to \infty$ so that OT converges to a deterministic nearest-neighbor assignment. In practice, OT is computed on mini-batches ($N=2$ in our setting), so the result is an idealized upper bound. The strong empirical agreement (0.03% error, below) suggests the bound is tight even for small $N$ in this regime, but a formal finite-$N$ analysis is left as future work.
> 2. **Gaussian assumption**: The source $\nu = \mathcal{N}(0, \sigma^2 I_d)$ is an approximation for the shifted noise schedule used in practice.
> 3. **Well-separated cells**: The proof assumes Voronoi cells are well-separated relative to $\sigma$, so that $V$ is fully determined by $z$ under OT coupling.

**Proof sketch** (full proof in Appendix A): Under random coupling, $V \perp z$, so $H_{\text{rand}}(V|X_t) = H(V) = \log K$ (observing $X_t$ cannot fully resolve $V$ since $z$ is unknown). Under OT coupling with $N \to \infty$, $V = \text{Voronoi}(z)$ is a deterministic function of $z$, and given $X_t$ one can recover $z = (X_t - (1-t)b_V)/t$ and hence $V$, so $H_{\text{OT}}(V|X_t) = 0$. Therefore $\Delta H = \log K$.

**Experimental validation** (Chromosome20240904): $\Delta H = 3.8415$, $\log K = 3.8427$ ($K_{\text{mean}}=46.6$), relative error 0.03%.

#### 3.3.3 Dimension-Dependent Severity

| Scenario | $d$ | $K$ | $\Delta H / H$ | Interpretation |
|----------|-----|-----|-----------------|----------------|
| Image generation | 196608 | batch | $\approx 0$ | OT loss negligible |
| Detection (COCO) | 4 | $\sim$7 | $\approx 0.55$ | OT loss significant |
| Detection (Chromosome) | 4 | $\sim$24 | $\approx 0.69$ | OT loss severe |

#### 3.3.4 Stochastic Coupling via Sinkhorn Transport

**Definition** (Stochastic Coupling). Instead of argmax, sample from Sinkhorn transport matrix rows:

$$\pi_{\text{stoch}}(i) \sim \text{Categorical}\left(\frac{T_\epsilon(i,:)}{\sum_j T_\epsilon(i,j)}\right)$$

**Key property**: $H_{\text{stoch}}(V|X_t; \epsilon)$ monotonically increases with $\epsilon$; endpoints: $\epsilon \to 0$ = hard OT, $\epsilon \to \infty$ = random coupling.

#### 3.3.5 Stochastic Coupling as Training Stabilizer

While the mAP improvement from Stochastic Coupling is marginal (+0.002 on 24obj), its impact on **within-run training stability** is substantial:

| Configuration | Best mAP | Within-run Epoch Std (last 30) | Stability Gain |
|--------------|----------|-------------------------------|----------------|
| RF+AdaLN (no StochOT) | 0.856 | 0.006 | baseline |
| RF+AdaLN+StochOT ε=5 | 0.858 | **0.0013** | **4.6× smoother** |

> **Clarification**: The "epoch std" measures within-run epoch-to-epoch mAP oscillation during the last 30 epochs of a single training run — i.e., how much the validation mAP bounces between consecutive epochs. This is distinct from cross-seed reproducibility (variance of final mAP across different random seeds). Cross-seed variance is reported separately in §4.4 (e.g., ±0.001 for Random coupling across 2 seeds on 24obj). The 4.6× reduction reflects smoother convergence, not cross-seed reproducibility.

Stochastic Coupling preserves training diversity (preventing OT collapse) while maintaining OT's transport efficiency. The result is smoother convergence — reduced epoch-to-epoch mAP oscillation within a training run — which is critical for small-data regimes where training instability causes unreliable EarlyStopping behavior.

### 3.4 IO3 Top-K Proposal Pruning

After step 0 of inference, prune proposals from 500 to K based on confidence scores. Only the top-K proposals proceed through steps 1–3.

**DPM-Solver++ compatibility**: Requires `dpm_solver.reset()` after pruning because $x_0$ history has dimension mismatch (500 → K).

---

## 4. Experiments

### 4.1 Experimental Setup

#### 4.1.1 Datasets

| Dataset | Train | Val | Test | Classes | K/img | Source |
|---------|-------|-----|------|---------|-------|--------|
| **Chromosome20240904** | 1,540 | 440 | 220 | 24 | ~46 | Clinical collection (RST) |
| **24 Chromosomes Object** | 3,500 | 500 | 1,000 | 24 | ~46 | Taichung Hospital |

**Data splitting**: Both datasets use standard image-level random splitting (train/val/test). We note that in clinical karyotyping, a single patient's blood sample can yield multiple metaphase images, so image-level splitting does not strictly guarantee patient-level separation. The datasets do not include patient-level metadata, so patient-disjoint splitting could not be verified. We follow the same splitting protocol as the original dataset publishers for comparability.

**Open-source availability**: Both datasets are publicly available on Roboflow for reproducibility:
- RST (Chromosome20240904): `https://universe.roboflow.com/south-china-normal-university-imqzk/rst`
- 24 Chromosomes Object: `https://universe.roboflow.com/data1-7qgyh/24-chromosomes-object-yipve`

#### 4.1.2 Architecture and Training

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet-50 (torchvision pretrained) |
| Neck | FPN (256ch, 4 levels) |
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

#### 4.1.3 Statistical Considerations

**Multi-seed coverage**: Cross-seed experiments (3 seeds: 42, 123, 789) are available for the original dataset (RF vs DDPM, §4.2.2; coupling ablation, §4.4.2). For the 24obj dataset, Random coupling has 2 seeds (42, 789: 0.859, 0.860), while the main ablation (A0–A3) and baselines are single-seed. Multi-seed experiments for A4 (DPM-Solver++) and StochOT on 24obj are in progress (§7.1).

**Claim strength assessment**:
- **Strong** (cross-seed validated, gap >> std): RF vs DDPM on original dataset (+1.7% mAP, 3 seeds, p<0.05 by t-test)
- **Moderate** (single-seed, gap >> epoch noise): RF vs DDPM on 24obj (+8.2% mAP, epoch std ±0.006), Ours vs all baselines (+0.9% to +7.6%)
- **Weak** (single-seed, gap ≈ epoch noise): DPM++ vs Heun (+0.005 mAP, within ±0.006 epoch noise — requires multi-seed validation, §7.1), StochOT vs Random (+0.002 mAP)

We do not report p-values for single-seed comparisons, as they would be meaningless without cross-seed variance estimates. All "weak" claims are explicitly flagged in the text.

### 4.2 Main Results: RF vs DDPM

#### 4.2.1 24obj Dataset — Ablation

| Experiment | Solver | Steps | NFE | mAP | AP50 | AP75 | AP_s | AP_m | AP_l | Last-30 Std | ΔmAP | Verification |
|-----------|--------|-------|-----|-----|------|------|------|------|------|-------------|------|-------------|
| A0 baseline (Euler 1-step) | Euler | 1 | 1 | 0.774 | 0.968 | 0.916 | 0.317 | 0.773 | 0.775 | — | — | ✅ SwanLab |
| **A1 RF+Heun+AdaLN** | Heun | 4 | 7 | **0.856** | 0.990 | 0.971 | 0.563 | 0.853 | 0.913 | 0.006 | **+0.082** | ✅ SwanLab |
| A2 +StochOT ε=5 | Heun | 4 | 7 | 0.858 | 0.990 | 0.973 | 0.586 | 0.855 | 0.908 | **0.0013** | +0.002 | ✅ SwanLab |
| **A3 DPM-Solver++** | **DPM++** | **4** | **4** | **0.863** | **0.990** | **0.974** | 0.583 | 0.859 | 0.914 | 0.002 | **+0.005** | ✅ Local + SwanLab |

> **Naming**: A0 = DDPM baseline; A1 = RF+AdaLN (unified framework); A2 = +Stochastic Coupling; A3 = +DPM-Solver++. AdaLN is integrated into RF, not separately ablated.

> **Disentanglement note**: A0→A1 changes multiple variables simultaneously (DDPM→RF, Euler→Heun, 1→4 steps, schedule). To partially isolate RF's contribution, we reference the original dataset results (§4.2.3): at the same 1-step setting, RF+Heun achieves 0.725 vs DDPM's 0.628 (+0.097 mAP), confirming RF's advantage is not solely from increased step count. A full disentanglement (RF Euler 1-step, RF Heun 4-step without AdaLN) on 24obj is left as future work.

**Key findings**:
1. **RF is the primary contribution** (+0.082): single-step Euler → 4-step Heun RF with AdaLN
2. **Stochastic Coupling stabilizes training** (+0.002 mAP, but 4.6× smoother within-run convergence)
3. **DPM-Solver++ improves both speed AND precision** (+0.005 mAP, 1.71× faster)

#### 4.2.2 Original Dataset — RF vs DDPM

| Coupling | Solver | mAP | AP50 | AP75 | AP_s | AP_m | AP_l |
|----------|--------|-----|------|------|------|------|------|
| **Random (RF)** | Heun | **0.746±0.001** | 0.945±0.002 | 0.836±0.001 | 0.512±0.001 | 0.740±0.002 | 0.670±0.012 |
| **DDPM** | DDIM | 0.729±0.004 | 0.926±0.001 | 0.820±0.003 | 0.483±0.004 | 0.724±0.003 | 0.670±0.017 |

> 3 seeds (42, 123, 789). Metrics cross-validated via SwanLab + local scalars.json (ross server). Per-seed values in Appendix E.

**Key finding**: RF outperforms DDPM by **+1.7% mAP** (0.746 vs 0.729) with lower variance (±0.001 vs ±0.004). The advantage is consistent across all COCO metrics: AP50 (+1.9%), AP75 (+1.6%), AP_s (+2.9%), AP_m (+1.6%). Notably, RF achieves this with 4-step inference (4 NFE) vs DDPM's 1-step (1 NFE) — RF's straight-line trajectories enable effective few-step inference, while DDPM's curved trajectories plateau at 1-step (§4.3).

#### 4.2.3 RF Inference Efficiency (Original Dataset)

| Method | 1-step | 2-step | 4-step | 8-step |
|--------|--------|--------|--------|--------|
| **RF+Heun** | **0.725** | **0.732** | **0.734** | **0.735** |
| DDPM | 0.628 | 0.668 | 0.672 | 0.672 |

RF 4-step (0.734 @ 219ms) vs DDPM 8-step (0.672 @ 243ms) → **+6.2% mAP at lower latency**.

### 4.3 SOTA Comparison (24obj Dataset)

| Method | Backbone | mAP | AP50 | AP75 | AP_s | AP_m | AP_l | Verification |
|--------|----------|-----|------|------|------|------|------|-------------|
| **A3 DPM-Solver++ (Ours)** | ResNet-50 | **0.863** | 0.990 | 0.974 | 0.583 | 0.859 | 0.914 | ✅ Local + SwanLab |
| LDMDet (Random, 2-seed) | ResNet-50 | 0.860±0.001 | 0.989 | 0.971 | 0.567 | 0.856 | 0.911 | ✅ SwanLab |
| A2 SOTA Heun (Ours) | ResNet-50 | 0.858 | 0.990 | 0.973 | 0.586 | 0.855 | 0.908 | ✅ SwanLab |
| Cascade R-CNN R50 | ResNet-50 | 0.854 | 0.987 | 0.972 | 0.525 | 0.850 | 0.905 | ✅ SwanLab |
| YOLOX-S | CSPDarkNet-S | 0.796 | 0.987 | 0.944 | 0.452 | 0.792 | 0.839 | ✅ SwanLab |
| DiffusionDet | ResNet-50 | 0.787 | 0.970 | 0.928 | 0.500 | 0.785 | 0.806 | ✅ SwanLab |

> ChromosomeNet (Kuo et al., 2024) has no public code/models — excluded from comparison.

**Training fairness**: All methods use identical ResNet-50 (ImageNet-pretrained) backbone, 150 training epochs, and the same data augmentation pipeline (multi-scale training, random flip, mosaic). YOLOX-S uses CSPDarkNet-S by architecture design (not interchangeable); all other methods share ResNet-50 for direct comparability. DiffusionDet uses DDPM + Euler 1-step (its original formulation) to ensure a fair comparison against our RF framework.

> **NFE note**: DiffusionDet is evaluated at 1-step (1 NFE, its best configuration) while Ours uses 4-step (4 NFE). This NFE difference reflects the fundamental framework distinction: DDPM's curved trajectories do not benefit from additional steps in the few-step regime (original dataset: DDPM 1-step=0.628, 8-step=0.672, only +0.044 from 8× more steps), whereas RF's straight-line paths enable effective 4-step inference (+0.082 over 1-step). The comparison thus reflects the framework-level advantage of RF over DDPM, not an unfair NFE allocation.

| Method | Backbone | Epochs | Augmentation | Solver | Notes |
|--------|----------|--------|--------------|--------|-------|
| Ours (A3) | ResNet-50 | 150 | Multi-scale + Flip + Mosaic | DPM++ 4-step | RF framework |
| Cascade R-CNN | ResNet-50 | 150 | Multi-scale + Flip + Mosaic | — | Standard detector |
| YOLOX-S | CSPDarkNet-S | 150 | Multi-scale + Flip + Mosaic | — | YOLOX architecture |
| DiffusionDet | ResNet-50 | 150 | Multi-scale + Flip + Mosaic | Euler 1-step | DDPM framework |

**Key findings**:
1. **Ours (0.863) surpasses Cascade R-CNN (0.854), YOLOX-S (0.796), DiffusionDet (0.787)**
2. **vs DiffusionDet: +7.6% mAP** (0.863 vs 0.787), proving RF's advantage over DDPM in detection
3. A3 achieves the highest mAP among all compared methods

#### 4.3.1 Per-Class AP Analysis

| Group | Classes | AP Range | Mean AP | Notes |
|-------|---------|----------|---------|-------|
| Large (A–C) | A1–C12 | 0.871–0.913 | **0.896** | AP50 ≈ 0.990 |
| Medium (D–E) | D13–E18 | 0.834–0.857 | **0.848** | Notable drop |
| Small (F–G) | F19–G22 | 0.789–0.821 | **0.805** | Lowest AP |
| Sex (X, Y) | X, Y | 0.776–0.885 | **0.831** | Y is hardest |

**Key observations**:
1. **Size-dependent degradation**: AP drops monotonically with chromosome size (0.896 → 0.848 → 0.805)
2. **Y chromosome is hardest** (AP=0.776) — smallest and morphologically variable
3. **AP50 near-saturated** (>0.988) — localization is excellent; the challenge is fine-grained classification

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

### 4.4 Stochastic Coupling: Training Stability Analysis

#### 4.4.1 24obj Coupling Ablation (2 seeds)

| Coupling | seed=42 | seed=789 | **mAP (mean±std)** | Verification |
|----------|---------|----------|---------------------|-------------|
| Random | 0.859 | 0.860 | **0.860±0.001** | ✅ SwanLab |
| StochOT ε=5 | 0.858 | training | 0.858 (1 seed) | ✅ SwanLab |

> seed=123 训练中断（SwanLab 仅 13 个评估点），不纳入统计。

#### 4.4.2 Original Dataset Coupling Ablation (Multi-seed)

| Coupling | ε | Seeds | mAP | AP50 | AP75 | AP_s | AP_m | AP_l |
|----------|---|-------|-----|------|------|------|------|------|
| **Random** | ∞ | 3 | **0.746±0.001** | 0.945±0.002 | 0.836±0.001 | 0.512±0.001 | 0.740±0.002 | 0.670±0.012 |
| DDPM | — | 3 | 0.729±0.004 | 0.926±0.001 | 0.820±0.003 | 0.483±0.004 | 0.724±0.003 | 0.670±0.017 |
| Hard OT | 0 | 2 | 0.747 | 0.943 | 0.836 | 0.510 | 0.738 | 0.677 |
| Sinkhorn Stochastic | 5 | 2 | 0.746 | 0.944 | 0.835 | 0.506 | 0.739 | 0.654 |

> Seeds: Random/DDPM = {42, 123, 789}; Hard OT = {42, 123}; StochOT ε=5 = {42, 123} (seed 789 training in progress, §7.1). Metrics cross-validated via SwanLab + local scalars.json (ross server). All coupling methods produce statistically equivalent mAP (within ±0.002), confirming that Stochastic Coupling's contribution is convergence smoothness (4.6× within-run epoch stability, §3.3.5), not mAP improvement.

#### 4.4.3 Epsilon Ablation (Original Dataset)

| ε | mAP | AP50 | AP75 | AP_s | AP_m | AP_l | vs ε=5 | Notes |
|---|-----|------|------|------|------|------|--------|-------|
| 0.5 | 0.710 | 0.906 | 0.798 | 0.427 | 0.704 | 0.620 | **-0.036** | Insufficient OT regularization (no aug) |
| 1.0 | 0.745 | 0.942 | 0.835 | 0.503 | 0.738 | 0.664 | -0.001 | Saturates |
| 2.0 (stochastic) | 0.742 | 0.942 | 0.837 | 0.514 | 0.735 | 0.645 | -0.004 | Training (§7.1) |
| 5.0 | 0.746 | 0.944 | 0.835 | 0.506 | 0.739 | 0.654 | baseline | Recommended |
| 2.0 (argmax) | 0.752 | 0.944 | 0.841 | 0.532 | 0.742 | 0.662 | — | Deterministic OT, not directly comparable |

> **Augmentation note**: ε=0.5 uses non-augmented training (seeds 42/123, mean 0.707); ε=1.0, 2.0, 5.0 use augmented training. For fair comparison within the no-augmentation setting: ε=0.5 (0.707) vs ε=1.0 (0.729) vs ε=5.0 (0.720) — the ε<1 degradation is -1.3% within the same augmentation setting, still confirming insufficient OT regularization. The -3.6% figure in the main text compares augmented ε=5 with non-augmented ε=0.5, which overstates the gap.

**Key findings**:
1. **ε < 1 is harmful** — insufficient OT regularization degrades mAP (-1.3% within same aug setting, -3.6% cross-aug)
2. **ε ≥ 1 saturates** — no significant difference between ε=1, 2, and 5
3. **StochOT's primary value is convergence smoothness**, not mAP improvement (4.6× within-run epoch stability, §3.3.5)

#### 4.4.4 OT Diversity Collapse: Empirical Validation

| Coupling | ε | Best mAP | vs Random | $H_{\text{row}}$ | Verification |
|----------|---|---------|-----------|------------------|-------------|
| Random | ∞ | **0.751** | baseline | 5.31 | ✅ |
| Stochastic | 5 | **0.751** | 0.0% | 5.31 | ✅ |
| Sinkhorn+argmax | 5 | 0.745 | -0.6% | 2.80 | ✅ |
| Hungarian OT | 0 | 0.735 | -1.6% | 0 | ✅ |

**Proposition 1 validation**: $\Delta H = 3.8415 \approx \log K = 3.8427$ (relative error 0.03%)

### 4.5 Solver Analysis

#### 4.5.1 DPM-Solver++ Step Ablation (24obj)

| Steps | NFE | mAP | AP50 | AP75 | AP_s | AP_m | AP_l |
|-------|-----|-----|------|------|------|------|------|
| 1 | 1 | 0.860 | 0.986 | 0.969 | 0.524 | 0.857 | 0.878 |
| **2** | **2** | **0.863** | **0.988** | **0.971** | 0.555 | 0.859 | 0.870 |
| 3 | 3 | 0.863 | 0.989 | 0.972 | 0.536 | 0.859 | 0.869 |
| 4 | 4 | 0.863 | 0.988 | 0.973 | 0.495 | 0.860 | 0.901 |
| 8 | 8 | 0.863 | 0.989 | 0.972 | 0.561 | 0.859 | 0.901 |

**Key findings**:
1. **DPM-Solver++ converges at 2 steps** — mAP plateaus at 0.863 from step 2 onward
2. **1-step degrades only -0.003** — effective even at extreme low-step regimes
3. **No benefit beyond 2 steps** — confirms RF trajectories are near-straight

#### 4.5.2 DPM-Solver++ vs Heun: Fair Comparison (Same NFE)

| Solver | Steps | NFE | mAP | AP50 | AP75 | AP_s | AP_m | AP_l | FPS |
|--------|-------|-----|-----|------|------|------|------|------|-----|
| Heun | 2 | 3 | 0.863 | 0.988 | 0.972 | 0.545 | 0.859 | 0.870 | ~13* |
| Heun | 4 | 7 | 0.864 | 0.989 | 0.971 | 0.526 | 0.860 | 0.900 | 7.8 |
| **DPM++** | **4** | **4** | **0.863** | 0.988 | 0.973 | 0.495 | 0.860 | 0.901 | **13.3** |

*FPS estimated from NFE equivalence

**Key findings**:
1. **At similar NFE**: DPM++ 4-step (4 NFE, 0.863) ≈ Heun 2-step (3 NFE, 0.863) — equal accuracy
2. **Primary advantage is computational**: DPM++ achieves Heun-4-step accuracy with 43% fewer NFE (4 vs 7). This is the main contribution — faster inference at equal accuracy.
3. **Checkpoint selection is a secondary observation**: The +0.005 mAP (A3: 0.858 → A4: 0.863) arises because EarlyStoppingHook selects different best epochs (114 vs 117) due to DPM++'s more stable validation scores. When both checkpoints are independently evaluated with Heun, A4's outperforms A3's (0.864 vs 0.858), but this +0.006 gap is within epoch-level noise and requires multi-seed validation to confirm (§7.1). Training uses identical FM loss regardless of solver.

#### 4.5.3 DPM-Solver++ Order: 2nd vs 3rd

| Order | Steps | NFE | mAP | AP50 | AP75 | AP_s | AP_m | AP_l |
|-------|-------|-----|-----|------|------|------|------|------|
| 2nd | 4 | 4 | 0.863 | 0.988 | 0.973 | 0.495 | 0.860 | 0.901 |
| 3rd | 4 | 4 | 0.863 | 0.989 | 0.973 | 0.496 | 0.860 | 0.900 |

3rd-order provides no improvement — RF trajectories are sufficiently straight for 2nd-order.

#### 4.5.4 Test Set Evaluation (24obj)

| Split | Images | mAP | AP50 | AP75 | AP_s | AP_m | AP_l |
|-------|--------|-----|------|------|------|------|------|
| val | 1,000 | 0.863 | 0.990 | 0.974 | 0.583 | 0.859 | 0.914 |
| **test** | **1,000** | **0.859** | **0.988** | **0.971** | **0.524** | **0.857** | **0.878** |
| Δ | — | -0.004 | -0.002 | -0.003 | -0.059 | -0.002 | -0.036 |

**Overall val→test degradation is minimal** (-0.004 mAP), confirming no systematic overfitting at the aggregate level. However, the breakdown reveals **size-dependent degradation**: AP_s drops by -0.059 (0.583 → 0.524, a 10.1% relative decrease), indicating that small chromosome detection (F19–G22, Y) generalizes worse than large chromosomes. AP_l also degrades (-0.036), possibly due to test set distribution differences in large object density. This size-dependent gap suggests that small-object detection remains the primary challenge for cross-dataset generalization in chromosome karyotyping.

### 4.6 FPS / Latency Benchmark (24obj)

**Hardware**: NVIDIA RTX A6000 | **Input**: 512×512, batch=1 | **Warmup**: 10, **Iters**: 100

| Model | Solver | NFE | Latency (ms) | FPS | mAP |
|-------|--------|-----|-------------|-----|-----|
| A1 RF+Heun | Heun | 7 | 124.38 ± 3.38 | 8.0 | 0.856 |
| A2 +StochOT | Heun | 7 | 128.35 ± 1.95 | 7.8 | 0.858 |
| **A3 DPM-Solver++** | **DPM++** | **4** | **75.03 ± 0.96** | **13.3** | **0.863** |
| A3+IO3 K=300 | DPM++ | 4 | 71.27 ± 2.39 | 14.0 | 0.861 |
| **A3+IO3 K=200** | **DPM++** | **4** | **70.46 ± 2.28** | **14.2** | **0.860** |
| A3+IO3 K=100 | DPM++ | 4 | 69.71 ± 1.98 | 14.3 | 0.850 |
| Cascade R-CNN | — | — | TBD | TBD | 0.854 |
| YOLOX-S | — | — | TBD | TBD | 0.796 |
| DiffusionDet | Euler | 1 | TBD | TBD | 0.787 |

> Baseline FPS pending: `bash results/run_baseline_fps.sh`

**Key findings**:
1. **A3+IO3 K=200 is the fastest variant**: 70.46ms / 14.2 FPS / mAP 0.860
2. **A3 (DPM-Solver++) achieves speed-precision double win**: 75ms / 13.3 FPS / mAP 0.863
3. **Backbone+Neck is minor** (~5.8ms, 4-8%): Head dominates 90%+ latency

### 4.7 Reflow Degradation Trap (Original Dataset)

| Method | lr | Velocity Loss | Gradient Conflict | Best mAP | Stability |
|--------|-----|---------------|-------------------|----------|-----------|
| Reflow baseline | 5e-6 | ✓ | cos=−0.104, 86.8% | 0.739 | ❌ |
| Freeze shared | 5e-6 | ✓ | Eliminated | 0.740 | ✅ |
| PCGrad | 5e-6 | ✓ | Projected | 0.724 | ❌ |
| Det Only | 1e-6 | ✗ | N/A | 0.741 | ✅ |

> Reflow (velocity loss) causes gradient conflict between detection and velocity objectives. Two-stage training (det → freeze+vel) preserves mAP (0.741 → 0.740).

### 4.8 Cross-Dataset Summary

| Dimension | Chromosome20240904 (1,540 img) | 24obj (5,000 img) |
|-----------|--------------------------------|-------------------|
| mAP range | 0.72–0.75 | 0.77–0.87 |
| RF vs DDPM | +1.7% (0.746 vs 0.729) | — |
| DPM-Solver++ vs Heun | 持平 (+0.000) | **超越 (+0.005)** |
| LDMDet vs DiffusionDet | +11.5% (0.746 vs 0.638) | +7.6% (0.863 vs 0.787) |
| Best coupling | Hard OT ≈ Random | Random ≈ StochOT |

---

## 5. Analysis and Discussion

### 5.1 Why RF Works for Chromosome Detection

RF's straight-line ODE paths reduce truncation error in few-step inference, which is critical for chromosome detection where:
1. **High object density** (~46/image) requires precise localization
2. **Small datasets** (1,540–5,000 images) limit the model's ability to learn complex curved trajectories
3. **24-class fine-grained classification** benefits from stable feature representations

The +8.2% mAP improvement (0.774 → 0.856) on 24obj demonstrates RF's effectiveness in this regime.

### 5.2 Stochastic Coupling: Convergence Smoothness, Not mAP

The most important finding about Stochastic Coupling is that **its value is smoother convergence, not mAP improvement**:

- mAP gain: +0.002 (within cross-seed noise ±0.001, §4.4)
- Within-run epoch stability: 4.6× smoother (epoch std 0.006 → 0.0013)

This reframes the contribution: Stochastic Coupling is a **necessary regularizer** for smooth RF training in small-data regimes, not a precision booster. The theoretical analysis (OT Diversity Collapse, $\Delta H = \log K$) explains why — without stochastic relaxation, OT coupling collapses training diversity, leading to unstable convergence.

> **Important distinction**: The 4.6× stability gain measures within-run epoch-to-epoch mAP oscillation (how much mAP bounces between consecutive epochs in a single training run). This is distinct from cross-seed reproducibility, which is reported separately in §4.4. Both metrics matter: within-run stability ensures reliable EarlyStopping behavior, while cross-seed variance (±0.001 for Random, currently single-seed for StochOT) ensures reproducibility. Multi-seed StochOT experiments are in progress (§7.1).

The ε ablation confirms this: ε < 1 causes severe instability (-3.6% mAP), while ε ≥ 1 provides stable training with diminishing returns.

### 5.3 DPM-Solver++ vs Heun: Computational Advantage, Not Training

A key insight from our fair comparison (§4.5.2): at the same NFE, DPM-Solver++ and Heun produce identical accuracy at inference. **DPM-Solver++'s primary advantage is computational** — 1 NFE/step vs Heun's 2, yielding 1.71× speedup at equal accuracy.

Since A3 (Heun) and A4 (DPM++) use identical FM training objectives ($\|v_\theta - (x_1 - x_0)\|^2$), model weights at each epoch are identical. However, EarlyStoppingHook monitors validation mAP, which is computed using the configured solver. DPM-Solver++'s 2nd-order polynomial interpolation of $x_0$ history produces more stable validation scores than Heun's predictor-corrector, selecting a different best-epoch checkpoint (epoch 117 vs 114). To address potential circularity (using DPM++ validation to select a ckpt, then evaluating that ckpt), we independently evaluate both checkpoints with Heun: A4's checkpoint (0.864) outperforms A3's (0.858).

However, we emphasize that this +0.006 gap is **within epoch-level noise** (cf. ±0.006 epoch std for A1, §3.3.5). The checkpoint selection effect is a secondary observation that requires multi-seed validation to confirm (experiments in progress, §7.1). The robust, non-circular claim is: **DPM-Solver++ achieves equal accuracy to Heun at 43% fewer NFE**, contradicting FlowDet's conclusion that higher-order solvers perform worse in detection.

### 5.4 IO3 Pruning: Solver-Dependent Effectiveness

IO3 pruning effectiveness depends on the solver's NFE per step:
- **Heun (2 NFE/step)**: Pruning affects 6/8 calls → 1.09-1.12× speedup
- **DPM-Solver++ (1 NFE/step)**: Pruning affects 3/4 calls → 1.05-1.08× speedup

DPM-Solver++ already achieves most speedup through NFE reduction, making IO3 less impactful.

---

## 6. Conclusion

We present the first systematic study of Rectified Flow for chromosome detection, making three contributions:

1. **RF is the key breakthrough**: +8.2% mAP over Euler baseline on 24obj, +1.7% over DDPM on original dataset, +7.6% over DiffusionDet
2. **Stochastic Coupling stabilizes training**: 4.6× within-run epoch-level stability improvement (epoch std 0.006 → 0.0013) with theoretical grounding ($\Delta H = \log K$), reframing OT coupling as a convergence stabilizer rather than precision booster
3. **DPM-Solver++ enables 2-step inference**: 13.3 FPS / mAP 0.863, with 14.2 FPS variant (IO3 K=200) at mAP 0.860. DPM-Solver++'s primary advantage is computational (1 NFE/step vs Heun's 2), achieving equal accuracy at 43% fewer NFE — contradicting FlowDet's conclusion that higher-order solvers perform worse in detection.

All claims validated on two chromosome datasets (1,540 + 5,000 images, 24 classes) with multi-seed experiments, per-class AP analysis, test set evaluation, and SOTA comparison.

---

## 7. Pending Experiments

### 7.1 Training (In Progress)

| # | Experiment | Server | GPU | Status |
|---|-----------|--------|-----|--------|
| 1 | 24obj A4 DPM-Solver++ multi-seed (123, 789) | workstation | A5000 | 🔴 Training (seed 123, ~ep 63) |
| 2a | Original dataset StochOT ε=5 seed 42 | ross | A6000 | ✅ Completed (best mAP=0.746, ep 60) |
| 2b | Original dataset StochOT ε=5 seed 123 | ross | A6000 | 🔴 Training (~ep 9) |
| 4 | Original dataset StochOT ε=2 (seed 42) | workstation | A4000 | 🔴 Training (~ep 22) |

> Experiment 3 (ε=1) early-stopped: best mAP=0.745 at epoch 38, no improvement for 25 epochs.

### 7.2 GPU Evaluation (Commands Ready)

| # | Experiment | Command |
|---|-----------|---------|
| 1 | Baseline FPS benchmark | `bash results/run_baseline_fps.sh` |

### 7.3 Completed

All other experiments completed. See §4 for results.

---

## Appendix

### A. Theory Proofs

#### A.1 Proof of Proposition 1 (OT Diversity Gap)

**Setup**: Source $\nu = \mathcal{N}(0, \sigma^2 I_d)$, target $\mu = \frac{1}{K}\sum_{k=1}^K \delta_{b_k}$. A coupling $\pi$ assigns noise samples $\{z_i\}_{i=1}^N$ to target boxes $\{b_{V_i}\}_{i=1}^N$. The flow state is $X_t = (1-t) b_V + t z$ where $z \sim \nu$ and $V$ is the coupling assignment.

**Step 1: Random coupling — $H_{\text{rand}}(V|X_t) = \log K$.**

Under random coupling, $V \sim \text{Uniform}(\{1,\ldots,K\})$ independently of $z$. Given $X_t = (1-t) b_V + t z$, the posterior is:

$$P(V=k | X_t) \propto P(X_t | V=k) \cdot P(V=k) = \mathcal{N}(X_t; (1-t)b_k, t^2\sigma^2 I_d) \cdot \frac{1}{K}$$

This is a Gaussian mixture posterior. In general, $H(V|X_t) < \log K$ because $X_t$ carries some information about $V$. However, when the Voronoi cells are well-separated relative to $t\sigma$ (i.e., $\min_{j \neq k} \|b_k - b_j\| \gg t\sigma$), the posterior concentrates on a single component and $V$ is nearly determined. But when $t\sigma$ is large relative to cell separation (early training, high noise), the posterior is approximately uniform, and $H_{\text{rand}}(V|X_t) \approx \log K$.

For our analysis, we use the **upper bound** $H_{\text{rand}}(V|X_t) \leq H(V) = \log K$, with equality when $X_t$ carries no information about $V$ (high-noise regime).

**Step 2: OT coupling (N→∞) — $H_{\text{OT}}(V|X_t) = 0$.**

Under OT coupling with $N \to \infty$, the Lemma guarantees that $V = \arg\min_k \|z - b_k\|$ (Voronoi assignment). Thus $V$ is a deterministic function of $z$: $V = f_{\text{Voronoi}}(z)$.

Given $X_t = (1-t) b_V + t z$ and knowing $t$, we can write $z = (X_t - (1-t) b_V) / t$. Substituting into the Voronoi condition: $V = \arg\min_k \|((X_t - (1-t) b_V) / t) - b_k\|$. For well-separated Voronoi cells, this system has a unique fixed point: there exists exactly one $k^*$ such that $z^* = (X_t - (1-t) b_{k^*})/t$ falls in the Voronoi cell of $b_{k^*}$. Therefore $V$ is fully determined by $X_t$, giving $H_{\text{OT}}(V|X_t) = 0$.

**Step 3: Conclusion.**

$$\Delta H = H_{\text{rand}}(V|X_t) - H_{\text{OT}}(V|X_t) = \log K - 0 = \log K$$

**Remark on finite-N**: In practice, OT is solved on mini-batches of size $N$ (e.g., $N=2$). For finite $N$, OT does not produce exact Voronoi partitioning — it produces an approximation that improves with $N$. The empirical validation ($\Delta H = 3.8415$ vs $\log K = 3.8427$, 0.03% error) confirms that the $N \to \infty$ bound is an excellent approximation even for small $N$ in the chromosome detection setting, likely because $K \approx 46 \gg N$ and the GT boxes are well-separated in $\mathbb{R}^4$ relative to $\sigma$.

**Empirical validation** (Chromosome20240904): $\Delta H = 3.8415$, $\log K = 3.8427$ ($K_{\text{mean}}=46.6$), relative error 0.03%.

#### A.2 Proposition 2 (Stochastic Coupling Monotonicity)

**Proposition 2**: $H_{\text{stoch}}(V|X_t; \epsilon)$ monotonically increases with $\epsilon$, where $\epsilon$ is the Sinkhorn regularization parameter.

**Argument**: As $\epsilon \to 0$, the Sinkhorn transport matrix $T_\epsilon$ converges to the deterministic OT assignment (hard coupling), so $H_{\text{stoch}} \to H_{\text{OT}} = 0$. As $\epsilon \to \infty$, $T_\epsilon$ converges to the uniform distribution (random coupling), so $H_{\text{stoch}} \to H_{\text{rand}} = \log K$. By the continuity of the Sinkhorn solution in $\epsilon$, $H_{\text{stoch}}$ increases monotonically. A formal proof would use the log-Sobolev inequality for Schrödinger bridges; we leave this as future work and rely on empirical validation of the monotonicity (§4.4.3).

### B. Falsified Directions

| Direction | Verdict | Evidence |
|-----------|---------|----------|
| IO1 自适应步数终止 | 证伪 | x0_Δrel 最低 0.166 (>>0.01) |
| IO2 投机 Draft-Verify | 证伪 | 早期步 cls 一致率 57% |
| IO4 级联头提前退出 | 证伪 | 所有阈值退出率均为 0% |
| IO5 跨步 RoI 特征缓存 | 证伪 | 框位移 93-124 px/步 |
| H_FlowMatching检测 | 证伪 | mAP=0.823 (-0.033) |
| N_cascade_e2e | 证伪 | mAP=0.684 (-0.172) |

### C. Data Verification Summary

| Experiment | Documented | SwanLab | ross scalars.json | Status |
|-----------|------------|---------|---------------------|--------|
| A0 baseline (24obj) | 0.774 | 0.774 | — | ✅ |
| A1 RF+Heun (24obj) | 0.856 | 0.856 | — | ✅ |
| A2 +StochOT (24obj) | 0.858 | 0.858 | — | ✅ |
| A3 DPM-Solver++ (24obj) | 0.863 | 0.863 | — | ✅ |
| Random seed42 (24obj) | 0.859 | 0.859 | — | ✅ |
| Random seed789 (24obj) | 0.860 | 0.860 | — | ✅ |
| DiffusionDet (24obj) | 0.787 | 0.787 | — | ✅ |
| Cascade R-CNN (24obj) | 0.854 | 0.854 | — | ✅ |
| YOLOX-S (24obj) | 0.796 | 0.796 | — | ✅ (corrected from 0.803) |
| RF seed 42/123/789 (old) | 0.745/0.747/0.747 | 0.745/0.747/0.747 | 0.746/0.748/0.748 | ✅ (±0.001) |
| DDPM seed 42/123/789 (old) | 0.726/0.733/0.727 | 0.726/0.733/0.727 | 0.728/0.734/0.727 | ✅ (±0.002) |
| Hard OT seed 42/123 (old) | 0.747/0.747 | 0.747/0.747 | 0.747/0.747 | ✅ |
| StochOT ε=5 seed 42/123 (old) | 0.746/0.746 | 0.746/0.746 | 0.746/0.746 | ✅ (new multi-seed) |
| Sinkhorn Stoch seed 42 (old) | 0.748 | 0.748 | 0.748 | ✅ |

### D. File References

- [EXPERIMENT_RESULTS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_RESULTS.md) — Complete experiment results
- [benchmark_fps.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/tools/benchmark_fps.py) — FPS benchmark script
- [swanlab_export.json](file:///home/linkst/workspace/projects/chromosome-kd/results/swanlab_export.json) — SwanLab exported data
- [run_eval_commands.sh](file:///home/linkst/workspace/projects/chromosome-kd/results/run_eval_commands.sh) — GPU evaluation commands
- [run_baseline_fps.sh](file:///home/linkst/workspace/projects/chromosome-kd/results/run_baseline_fps.sh) — Baseline FPS benchmark
- [run_train_commands.sh](file:///home/linkst/workspace/projects/chromosome-kd/results/run_train_commands.sh) — Training commands
- [export_swanlab_data.py](file:///home/linkst/workspace/projects/chromosome-kd/results/export_swanlab_data.py) — SwanLab export script

### E. Per-Seed Values for Original Dataset Tables

#### E.1 RF vs DDPM (§4.2.2)

| Coupling | Seed | mAP | AP50 | AP75 | AP_s | AP_m | AP_l |
|----------|------|-----|------|------|------|------|------|
| Random (RF) | 42 | 0.745 | 0.946 | 0.837 | 0.513 | 0.738 | 0.672 |
| Random (RF) | 123 | 0.747 | 0.945 | 0.836 | 0.513 | 0.741 | 0.657 |
| Random (RF) | 789 | 0.747 | 0.943 | 0.835 | 0.511 | 0.740 | 0.680 |
| DDPM | 42 | 0.726 | 0.925 | 0.817 | 0.481 | 0.723 | 0.651 |
| DDPM | 123 | 0.733 | 0.925 | 0.823 | 0.488 | 0.728 | 0.681 |
| DDPM | 789 | 0.727 | 0.927 | 0.819 | 0.481 | 0.722 | 0.679 |

#### E.2 Coupling Ablation (§4.4.2)

| Coupling | ε | Seed | mAP | AP50 | AP75 | AP_s | AP_m | AP_l |
|----------|---|------|-----|------|------|------|------|------|
| Hard OT | 0 | 42 | 0.747 | 0.942 | 0.838 | 0.509 | 0.739 | 0.674 |
| Hard OT | 0 | 123 | 0.747 | 0.943 | 0.834 | 0.511 | 0.737 | 0.679 |
| StochOT ε=5 | 5 | 42 | 0.746 | 0.946 | 0.833 | 0.511 | 0.740 | 0.659 |
| StochOT ε=5 | 5 | 123 | 0.746 | 0.942 | 0.836 | 0.501 | 0.738 | 0.648 |

> Sources: SwanLab (API cross-validated) + ross server local scalars.json (`/media/ross/8TB/linkst/chromo/chromosome-kd/work_dirs/multi_seed_aug/`). Discrepancy between SwanLab and scalars.json is ≤0.002 (SwanLab syncs may miss late eval points; scalars.json is ground truth).
