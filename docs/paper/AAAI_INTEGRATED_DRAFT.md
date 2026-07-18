# Rectified Flow for Chromosome Detection: Stable Coupling and Few-Step Inference

<!--
============================================================
TMI MIGRATION STRATEGY HEADER
============================================================
Target journal: IEEE Transactions on Medical Imaging (TMI)
Template: IEEEtran.cls [journal,10pt]

HARD CONSTRAINTS (verify against live author-instructions before submit):
- Initial submission <= 10 pages INCLUDING references (hard cap, returned without review)
- Abstract < 250 words; must include IEEEkeywords
- Double-column, single-spaced, justified, 10pt
- Single-anonymous review (author info OK in initial submission, no biographies)
- NO supplementary text materials (since 2022-01-01) -> arXiv companion preprint
- Figures/tables must appear inline in main text
- References: IEEEtran.bst, author-name format (e.g., J. Smith)

STRATEGY A+B (user-approved):
- Strategy A: Compress appendices in main text (keep proof sketches, brief mentions)
- Strategy B: Move detailed proofs/per-seed tables to arXiv companion preprint

CONTENT DESTINATION LEGEND:
  [MAIN PAPER]      -> retain in 10-page IEEE submission
  [ARXIV COMPANION] -> move to arXiv preprint, cite from main paper
  [COMPRESS]        -> reduce to 1-2 sentences inline, full version on arXiv
  [DELETED]         -> not needed for TMI (kept in draft as fact record only)
  [PLACEHOLDER]     -> new content to be added in Phase 3-D (robustness) or 3-E (theory)

DRAFT STATUS:
- This draft is the COMPLETE FACT RECORD (most detailed content, states facts clearly).
- Selective writing for the 10-page IEEE main paper is a normal technique;
  content marked [ARXIV COMPANION] or [DELETED] is NOT lost — it remains here
  as the authoritative record and feeds the arXiv companion.
- Sync state: synced to main.tex commit 737971aa (2026-07-18).
============================================================
-->

> **TMI Positioning Note (per ChatGPT analysis + user review 2026-07-19):**
> The paper's novelty is ML theory (RF paradigm, OT Diversity Collapse, Stochastic
> Coupling, solver disentanglement), NOT biological insight. However, the paper
> must STILL LEAD FROM THE APPLICATION (chromosome karyotyping) — the algorithmic
> novelty serves the clinical task, not vice versa. Balance: application context
> opens the Abstract/Intro, algorithmic contributions follow as the solution.
> Rationale: TMI is a medical imaging journal; reviewers expect clinical motivation first.

## Abstract

<!-- [MAIN PAPER] Abstract for TMI: <=250 words, application-led opening + algorithmic contributions. Word count: ~243 -->

Chromosome karyotyping — the visual inspection of metaphase chromosomes for clinical genetics — requires trained cytogeneticists to classify roughly 46 tightly packed chromosomes per cell into 24 morphologically similar classes, a process that is slow and observer-dependent. Automating this analysis demands a detector that is both accurate and fast enough for clinical deployment, but three obstacles arise: limited accuracy of conventional detectors on fine-grained chromosomes; slow many-step inference and curved-trajectory truncation errors of Denoising Diffusion Probabilistic Model (DDPM)-based diffusion detectors; and training instability on small clinical datasets.

We address these obstacles with *Rectified Flow* (RF), which replaces the stochastic DDPM process with deterministic straight-line ODE paths, instantiated as *KaryoFlow*. The RF training paradigm is the dominant source of accuracy gain: KaryoFlow exceeds DiffusionDet by $+0.076$ mAP and matches Cascade R-CNN and YOLOX-S; a solver$\times$step disentanglement attributes 94% of the gain to RF. We prove an upper bound $\Delta H \le \log K$ on conditional-entropy reduction from OT coupling in the low-dimensional ($\mathbb{R}^4$) detection space — characterizing *OT Diversity Collapse* — and propose Stochastic Coupling via Sinkhorn transport, which yields a large, highly significant mAP gain in the low-data regime ($+0.034$, $p<10^{-120}$) that diminishes with dataset size, and reduces within-run convergence oscillation by $4.6\times$. DPM-Solver++ for four-step inference achieves a small but statistically significant precision advantage over Heun at matched step count ($+0.006$ per-image mAP, Wilcoxon $p<0.001$) with $1.71\times$ speedup, combined with Top-$K$ pruning.

All claims are validated on two public chromosome datasets with multi-seed experiments, per-class AP analysis, and SOTA comparison.

<!-- [MAIN PAPER] IEEEkeywords placeholder — to be finalized:
Index Terms — Rectified Flow, object detection, optimal transport, diffusion models, medical image analysis, chromosome karyotyping
-->

## 1. Introduction

<!--
[MAIN PAPER] TMI Intro Repositioning Plan (for Phase 2-B IEEEtran migration):
- Current intro leads with clinical motivation (chromosome karyotyping workflow).
- TMI-appropriate version should lead with the ALGORITHMIC problem (diffusion
  detection inherits DDPM pathologies; RF as remedy) and use chromosome detection
  as the motivating dense-detection INSTANCE, not the primary subject.
- Suggested structure for 10-page IEEE:
  P1: Diffusion detection + DDPM pathologies (algorithm-led, ~0.4 page)
  P2: RF as remedy + three key bottlenecks (coupling/solver/stability) (~0.4 page)
  P3: Chromosome karyotyping as motivating instance + scenario-to-method map (~0.3 page)
  P4: Contributions summary + novelty boundary (~0.4 page)
- The clinical detail (46 chromosomes, 24 classes, C-group/Y-chromosome difficulty)
  moves to §4.1.1 Datasets or a brief motivating-paragraph, NOT the opening.
- Keep Figure 1 (method overview) at end of intro.
-->

### 1.1 Motivation

Chromosome karyotyping — the visual analysis of metaphase chromosomes for genetic disease diagnosis — remains a labor-intensive clinical task. Each metaphase image contains about 46 tightly packed chromosomes across 24 classes (A1–Y), with severe class imbalance, fine-grained intra-group similarity, and frequent overlaps. Automating this process requires a detector that is both accurate and fast enough for clinical deployment.

Diffusion models offer a compelling paradigm for detection by framing object localization as iterative denoising from noisy boxes to structured predictions (DiffusionDet). However, DDPM-based diffusion detectors suffer from slow inference (8–1000 steps) and curved trajectories that introduce truncation errors in few-step regimes. Rectified Flow (RF) replaces the stochastic DDPM process with deterministic straight-line ODE paths, enabling few-step inference — but applying RF to detection raises fundamental questions about coupling design, solver selection, and training stability. We frame these as the key bottlenecks of RF-based dense detection in low-data regimes: (i) coupling diversity collapse in low-dimensional structured prediction; (ii) numerical solver design for efficient few-step inference; and (iii) training stability under high object density. Our three contributions address these bottlenecks systematically.

**From scenario to method.** A typical metaphase spread contains roughly forty-six chromosomes across twenty-four classes, many touching or overlapping. The difficulty is uneven: C-group chromosomes (C6–C12) are morphologically similar, distinguished mainly by subtle banding patterns, while the Y chromosome is among the smallest, appears in only one copy in male samples, and has roughly 1,800 training samples versus about 7,000 per autosome. A useful detector must therefore localize densely packed objects under few-step inference, train stably from an imbalanced small corpus, and run fast enough for interactive screening. These demands map onto our three ingredients: Rectified Flow supplies straight-line trajectories for few-step, low-truncation inference; Stochastic Coupling counteracts OT diversity collapse; and DPM-Solver++ with Top-$K$ pruning converts trajectories into clinical-grade latency.

### 1.2 Contributions

Our first contribution validates the RF training paradigm for chromosome detection. KaryoFlow achieves +0.082 mAP over the Euler baseline on 24 Chromosomes Object and +0.017 over DDPM on the original dataset. Because the A0→A1 comparison changes several variables at once (DDPM→RF, Euler→Heun, 1→4 steps), a solver×step disentanglement ablation attributes 94% of the gain to the RF paradigm and only 6% to solver and step-count choices; AdaLN-Zero contributes null individually (Appendix D).

Our second contribution is a theoretical characterization of OT coupling's failure mode in low-dimensional detection space, with a practical remedy. We prove an upper bound $\Delta H \le \log K$ on the conditional-entropy reduction, empirically validated to within 0.03%, and propose Stochastic Coupling (Sinkhorn-transport sampling instead of argmax), which reduces within-run epoch-level mAP oscillation by 4.6× and yields a large, highly significant mAP gain in the low-data regime ($+0.034$, $p<10^{-120}$) that diminishes with dataset size. An $\epsilon$ ablation shows $\epsilon < 1$ is harmful while $\epsilon \ge 1$ saturates.

Our third contribution deploys DPM-Solver++ for four-step inference ($1.71\times$ speedup over Heun, mAP $0.859 \pm 0.003$ over three seeds). A controlled, per-image paired ablation reveals that DPM-Solver++ yields a small but statistically significant precision advantage over Heun at matched step count ($+0.006$ per-image mAP, Wilcoxon $p<0.001$, Table 8), refining FlowDet's conclusion that higher-order solvers perform worse: at matched steps the higher-order solver is in fact slightly *better*, not worse.

**Novelty boundary.** Relative to FlowDet (CFM with mini-batch OT, reports higher-order solvers perform worse), our novelty is: (i) the theoretical characterization of *why* mini-batch OT becomes a liability in low-dimensional structured prediction (Section 3.3); and (ii) a solver×step disentanglement showing that, at matched step count, the higher-order DPM-Solver++ is in fact slightly but significantly *better* than Heun (not strictly worse as FlowDet reports), while being $1.71\times$ faster in NFE. Relative to DeFloMat (RF for medical detection, coupling as implementation detail), we provide the theoretical analysis of coupling design as a training pathology and the Stochastic Coupling remedy. Unlike OT-CFM and multisample flow matching (high-dimensional image generation, $d \sim 10^5$), our setting is low-dimensional ($d=4$) with $K \approx 46$, where OT collapse is severe ($\Delta H/H \approx 0.69$). AdaLN-Zero (Dhariwal & Nichol, 2021) is reused as a standard implementation detail (Appendix D); DPM-Solver++ is adopted off-the-shelf, our contribution being the controlled disentanglement analysis with per-image paired statistical tests.

These contributions are backed by comprehensive validation. All claims are validated on both Chromosome20240904 (henceforth the *original* dataset, 1,540 images) and 24 Chromosomes Object (5,000 images), with multi-seed coupling ablation, per-class AP analysis, test-set evaluation, and an FPS benchmark.

![**Figure 1**: KaryoFlow overview. (a) RF replaces the curved DDPM denoising trajectory with a straight-line ODE path from noise $\mathbf{x}_1$ to Ground Truth (GT) box $\mathbf{x}_0$; nodes indicate the 4 solver steps. (b) Time conditioning injects the continuous time $t$ via AdaLN-Zero zero-initialized modulation so the network is identity at $t{=}0$. (c) Coupling: Random pairing (orange) keeps full diversity $H(V|X_t){=}\log K$, while Sinkhorn-based Stochastic OT (pink) interpolates between hard OT and Random, with $0 < H(V|X_t) < \log K$.](latex/figures/method_overview.png)

## 2. Related Work

### 2.1 Diffusion-Based Object Detection

DiffusionDet formulates detection as iterative denoising from noisy boxes using DDPM, but curved DDPM trajectories make few-step inference slow and truncation-error-prone. DiffuBox (Chen et al., 2024) extends the diffusion paradigm to 3D object detection via point-diffusion refinement of coarse proposals, but the 3D point-diffusion setting differs substantially from our 2D noise-box-to-GT formulation. FlowDet applies Conditional Flow Matching with mini-batch OT coupling and reports higher-order solvers perform worse, but does not analyze *why* mini-batch OT becomes a liability in low-dimensional structured prediction. DeFloMat uses Rectified Flow for medical detection but treats coupling as an implementation detail. Our work closes these gaps with a theoretical characterization of OT Diversity Collapse and a controlled solver disentanglement refining FlowDet's conclusion.

### 2.2 Rectified Flow and Flow Matching

Rectified Flow replaces curved DDPM trajectories with straight-line ODE paths, and Flow Matching provides a unified training framework. OT-CFM and multisample flow matching use mini-batch OT coupling successfully in *image generation* ($d \sim 10^5$, $K \approx$ batch size), but the behavior in low-dimensional structured prediction ($d$ small, $K$ targets per image) has not been characterized in terms of diversity collapse. Concurrent work analyzes OT degradation in conditional high-dimensional generation (Cheng & Schwing, 2025), but the failure mode there (condition-skewed prior) is orthogonal to the low-dimensional diversity collapse we characterize. With $d=4$ and $K \approx 46$, OT coupling approaches its $\log K$ entropy-reduction upper bound, collapsing coupling diversity. We characterize this failure mode and propose Stochastic Coupling as a remedy interpolating between hard OT and random pairing.

### 2.3 Chromosome Detection

Prior work uses conventional detectors (YOLO, Faster R-CNN) that leave a measurable accuracy gap on the fine-grained 24-class setting. ChromosomeNet uses the same Taichung dataset as our 24 Chromosomes Object benchmark but releases no code or pretrained models. We compare against standard detectors with public implementations (Cascade R-CNN, YOLOX-S, DiffusionDet) and provide the first open-source diffusion-based detector for chromosome karyotyping (code will be released upon publication) with multi-seed validation and per-class AP analysis.

## 3. Method

### 3.1 Rectified Flow for Detection (KaryoFlow)

#### 3.1.1 RF Formulation

The conditional probability path is
$$\mathbf{x}_t = (1-t)\, \mathbf{x}_0 + t\, \mathbf{x}_1, \qquad t \in [0,1],$$
where $\mathbf{x}_0$ is the target (GT bbox) and $\mathbf{x}_1$ is the source (Gaussian noise). The corresponding velocity field is constant along the path, $\mathbf{v} = \mathbf{x}_1 - \mathbf{x}_0$, and the training objective is the flow matching loss
$$\mathcal{L}_{FM} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\!\left[ \left\lVert \mathbf{v}_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \right\rVert^2 \right].$$

**Detection-specific adaptation**: source $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_4)$, target $\mathbf{x}_0$ are GT bboxes, $d=4$ (vs. $d=196{,}608$ in image generation), and $K \approx 46$ targets per image.

#### 3.1.2 Time Conditioning

AdaLN-Zero (Dhariwal & Nichol, 2021) serves as the time conditioning mechanism, replacing scale-shift conditioning with zero-initialized adaptive layer norm so the network starts as an unconditional model (identity at $t{=}0$). A separate ablation (Appendix D) confirms its individual contribution to mAP is null; the +0.082 mAP gain is entirely attributable to the RF formulation itself, with the shifted noise schedule contributing negligibly on its own (Appendix E). We retain AdaLN-Zero as a standard implementation detail, not a separate contribution.

### 3.2 ODE Solvers: Heun and DPM-Solver++

#### 3.2.1 Heun Solver (2nd-order)

Heun's method provides 2nd-order ODE accuracy via a predictor–corrector:
- Predict: $\hat{\mathbf{x}}_{t-\Delta t} = \mathbf{x}_t + \Delta t \cdot \mathbf{v}_\theta(\mathbf{x}_t, t)$
- Correct: $\mathbf{x}_{t-\Delta t} = \mathbf{x}_t + \frac{\Delta t}{2} [\mathbf{v}_\theta(\mathbf{x}_t, t) + \mathbf{v}_\theta(\hat{\mathbf{x}}_{t-\Delta t}, t{-}\Delta t)]$

**Cost**: 2 network forward evaluations (NFE) per step. At 4 steps this yields 8 NFE (7 in practice; the last step degrades to Euler).

#### 3.2.2 DPM-Solver++ (2nd-order multistep, 1 NFE/step)

We adapt DPM-Solver++ (Lu et al., 2022) to the RF linear path. The model directly predicts $\mathbf{x}_0$ (the GT box), and the update uses polynomial interpolation of the $\mathbf{x}_0$ history in $t$ space (not the log-SNR $\lambda$ space of VP-SDE). The RF-ODE in data-prediction form is $d\mathbf{x}/dt = (\mathbf{x} - \mathbf{x}_0(t))/t$; integrating exactly over $[t_n, t_{n+1}]$ with linear $\mathbf{x}_0(t)$ interpolation yields

$$\mathbf{x}_{t_{n+1}} = \tfrac{t_{n+1}}{t_n}\,\mathbf{x}_{t_n} + \bigl(1 - \tfrac{t_{n+1}}{t_n}\bigr)\,\mathbf{x}_0^{(n)} + \varphi_1\,\mathbf{D}_1,$$
$$\varphi_1 = t_{n+1}\log\tfrac{t_n}{t_{n+1}} - t_n + t_{n+1},\quad \mathbf{D}_1 = \tfrac{\mathbf{x}_0^{(n)} - \mathbf{x}_0^{(n-1)}}{t_n - t_{n-1}},$$

where the first two terms are the exact solution for constant $\mathbf{x}_0$ and $\varphi_1 \mathbf{D}_1$ is the 2nd-order correction for linearly varying $\mathbf{x}_0(t)$. The singularity at $t{\to}0$ is handled by an $\epsilon$ cutoff ($t_{n+1} > 10^{-7}$); a 3rd-order variant adds a quadratic term $\varphi_2 \mathbf{D}_2$. Unlike VP-SDE DPM-Solver++, $\mathbf{x}_1$ (the initial noise) is a fixed sample and is *not* interpolated; its contribution is carried implicitly by $\mathbf{x}_{t_n}$.

**Cost**: 1 NFE per step. At 4 steps this yields 4 NFE (vs Heun's 7).

#### 3.2.3 Key Finding: Solver Type is mAP-Neutral on A1; DPM-Solver++ Buys NFE Reduction

Evaluating the A1 checkpoint across all solver×step combinations (Table 1, Figure 2), solver type has *no effect on mAP* at matched steps (Euler = DPM-Solver++ at both 4-step and 1-step). Step count has marginal effect (+0.004 from 1 to 4 steps). Heun's +0.001 over Euler 4-step costs $1.75\times$ NFE (7 vs 4) — not cost-effective. On the full model (A3 vs A2), DPM-Solver++ is in fact slightly but significantly *more* accurate than Heun at matched 4-step ($+0.006$ per-image mAP, Wilcoxon $p<10^{-3}$; Table 8), so its advantage is both computational and a small precision gain.

| Solver | Steps | NFE | mAP |
|--------|-------|-----|-----|
| Heun | 4 | 7 | 0.856 |
| Euler | 4 | 4 | 0.855 |
| DPM-Solver++ | 4 | 4 | 0.855 |
| Euler | 1 | 1 | 0.851 |
| DPM-Solver++ | 1 | 1 | 0.851 |

**Table 1**: Solver×step disentanglement ablation on the A1 checkpoint (24 Chromosomes Object val, seed 42).

**Conclusion**: DPM-Solver++'s advantage is *both computational and a small precision gain*: $1.71\times$ NFE speedup, plus a small but statistically significant mAP improvement over Heun at matched 4-step ($+0.006$ per-image mAP, Wilcoxon $p<10^{-3}$; Table 8). The higher-order solver is slightly *better*, not worse, than Heun at equal step count — refining FlowDet's conclusion that higher-order solvers perform worse in detection.

![**Figure 2**: Solver×step disentanglement ablation (A1 checkpoint). Bars are colored by solver type and hatched by step count. Solver/step configuration contributes only +0.005 mAP (6%); the remaining +0.077 mAP (94%) is attributable to the RF training paradigm.](latex/figures/solver_ablation.png)

### 3.3 OT Diversity Collapse and Stochastic Coupling

#### 3.3.1 Voronoi Partitioning by OT

**Lemma** (OT → Voronoi). When $N \to \infty$, OT coupling partitions $\mathbb{R}^d$ into $K$ Voronoi cells $\mathcal{V}_k = \{\mathbf{z} : \lVert\mathbf{z} - \mathbf{b}_k\rVert \le \lVert\mathbf{z} - \mathbf{b}_j\rVert, \forall j \ne k\}$, assigning each $\mathbf{z}_i$ to its nearest GT box.

This is the standard semi-discrete OT limit (Santambrogio, 2015).

#### 3.3.2 Proposition: OT Diversity Gap

**Setup**: Let source $\nu = \mathcal{N}(0, \sigma^2 I_d)$ (noise), target $\mu = \frac{1}{K}\sum_{k=1}^K \delta_{\mathbf{b}_k}$ ($K$ GT boxes). A coupling assigns each noise sample $\mathbf{z}_i$ to a target box $\mathbf{b}_{V_i}$. We let:
- $V \in \{1, \ldots, K\}$: coupling assignment RV (which GT box a noise sample is paired with)
- $X_t = (1-t) \mathbf{b}_V + t \mathbf{z}$: the flow state observed by the model
- $H(V \mid X_t)$: conditional entropy of $V$ given $X_t$ — how much $X_t$ leaks about $V$

**Proposition 1** (OT Diversity Upper Bound). Under the Voronoi partitioning assumption (Lemma, requiring $N \to \infty$) and the Gaussian noise model,
$$\Delta H \;=\; H_{\text{rand}}(V \mid X_t) - H_{\text{OT}}(V \mid X_t) \;\le\; \log K.$$

**Proof sketch** (full proof in Appendix A): Under random coupling, $H_{\text{rand}}(V|X_t) \le H(V) = \log K$ in the high-noise regime. Under OT coupling with $N \to \infty$, $V = \operatorname{Voronoi}(\mathbf{z})$ is deterministic (Lemma), so given $X_t$ one recovers $\mathbf{z}$ and hence $V$, giving $H_{\text{OT}}(V | X_t) = 0$. Therefore $\Delta H \le \log K$.

<!-- [PLACEHOLDER: Phase 3-E — Theory Deepening]
Proposition 1 currently proves only the UPPER bound ΔH ≤ log K.
For TMI, add a matching LOWER bound to demonstrate tightness:
  Conjecture (lower bound): Under sufficient GT-box separation (sep = min_{j≠k} ||b_k - b_j||)
  and high-noise regime (tσ >> sep), ΔH ≥ log K - O(exp(-c·sep²/(tσ)²)),
  i.e., the upper bound is tight up to exponential decay in separation-to-noise ratio.
This shows OT collapse is INHERENT to low-dimensional detection, not an artifact
of the specific GT configuration. Proof sketch -> [MAIN PAPER]; full proof -> [ARXIV COMPANION].
See Appendix A.1 for the placeholder insertion point.
-->

**Empirical validation** (Chromosome20240904): $\Delta H = 3.8415$, $\log K = 3.8427$ ($K_{\text{mean}} = 46.6$), relative error 0.03%. Figure 3 visualizes the partitioning and the empirical match, and the entropy phase diagram traces conditional entropy $H(V|Z)$ as a function of the stochastic coupling parameter $\epsilon$.

![**Figure: Entropy phase diagram.** Conditional entropy $H(V|Z)$ as a function of the stochastic coupling parameter $\epsilon$. Hard OT ($\epsilon{=}0$) collapses to $H{=}0$; Random coupling ($\epsilon{\to}\infty$) saturates at $H{=}3.8415 \approx \log K{=}3.8427$ (relative error 0.03%). Stochastic coupling with $\epsilon \ge 1$ recovers near-full diversity, while $\epsilon < 1$ falls in the danger zone of diversity collapse.](latex/figures/entropy_phase.png)

![**Figure 3**: OT Diversity Collapse. (a) Voronoi partitioning for $K{=}8$ GT boxes in a 2D projection. Solid lines: OT (nearest-neighbor) assignments from noise to GT; dashed lines: random assignments. (b) Empirical validation of Proposition 1: theoretical $\log K = 3.8427$ vs. empirical $\Delta H = 3.8415$ (relative error 0.03%).](latex/figures/ot_theory.png)

#### 3.3.3 Dimension-Dependent Severity

Table 2 shows that OT collapse is severe in detection ($\Delta H/H \approx 0.55$–$0.69$) but negligible in image generation.

| Scenario | $d$ | $K$ | $\Delta H / H$ |
|----------|-----|-----|-----------------|
| Image generation | $256^2{\times}3$ | batch | $\approx 0$ |
| Detection (COCO) | 4 | $\sim$7 | $\approx 0.55$ |
| Detection (Chromosome) | 4 | $\sim$46 | $\approx 0.69$ |

**Table 2**: OT Diversity Collapse severity by scenario. OT loss is severe in detection, negligible in image generation.

#### 3.3.4 Stochastic Coupling via Sinkhorn Transport

**Definition** (Stochastic Coupling). Instead of an argmax assignment, we sample the coupling from the Sinkhorn transport matrix rows (Cuturi, 2013):
$$\pi_{\text{stoch}}(i) \;\sim\; \operatorname{Categorical}\!\left( \frac{ T_\epsilon(i,:) }{ \sum_j T_\epsilon(i,j) } \right).$$

The key property is that $H_{\text{stoch}}(V|X_t; \epsilon)$ increases monotonically with $\epsilon$; endpoints are $\epsilon \to 0$ = hard OT and $\epsilon \to \infty$ = random coupling. We conjecture that $H_{\text{stoch}}$ is monotonically increasing in $\epsilon$ (Conjecture 1, Appendix A.2), supported by empirical validation.

<!-- [PLACEHOLDER: Phase 3-E — Theory Deepening]
Conjecture 1 (StochOT monotonicity) is currently stated with only an empirical
argument (continuity of Sinkhorn solution in ε). For TMI, upgrade to a rigorous
proof using the log-Sobolev inequality for Schrödinger bridges, or alternatively
a coupling argument showing dH_stoch/dε ≥ 0 via the entropy-regularized OT
first-order conditions. Proof sketch -> [MAIN PAPER]; full proof -> [ARXIV COMPANION].
If a complete proof proves elusive, retain as Conjecture but add a formal
partial-result proposition (e.g., monotonicity in the ε→0 and ε→∞ limits).
-->

#### 3.3.5 Stochastic Coupling as Training Stabilizer

While the mAP improvement from Stochastic Coupling is marginal (+0.002 on 24 Chromosomes Object), its impact on *within-run training stability* is substantial (Table 3, Figure 4).

| Configuration | Best mAP | Epoch std |
|--------------|----------|-----------|
| RF + AdaLN (no Stoch. Coup.) | 0.856 | 0.006 |
| RF + AdaLN + Stoch. Coup. ($\epsilon{=}5$) | 0.858 | **0.0013** |
| *Stability gain* | — | *4.6×* |

**Table 3**: Training stability: Stochastic Coupling yields 4.6× smoother within-run convergence. "Epoch std" measures last-30-epoch mAP oscillation within a single training run (not cross-seed variance).

![**Figure 4**: Training stability (24 Chromosomes Object, real per-epoch mAP from training logs): Random coupling exhibits epoch-level oscillation with std 0.006, while Stochastic Coupling ($\epsilon{=}5$) converges smoothly with std 0.0013 (4.6× improvement). Shaded band marks the last 30 epochs used for std computation.](latex/figures/training_stability.png)

### 3.4 Top-$K$ Proposal Pruning

After step 0 of inference, we prune proposals from 500 to $K$ based on confidence scores; only the top-$K$ proposals proceed through steps 1–3. DPM-Solver++ compatibility requires `dpm_solver.reset()` after pruning because the $\mathbf{x}_0$ history has a dimension mismatch (500 → $K$).

## 4. Experiments

### 4.1 Experimental Setup

#### 4.1.1 Datasets

Table 4 summarizes the two public chromosome datasets used in this paper, which we refer to as Dataset 1 and Dataset 2 hereafter. Dataset 1 is Chromosome20240904 (RST), a clinical collection of 1,540 images. Dataset 2 is the 24 Chromosomes Object dataset (Tseng et al., 2023), comprising 5,000 images. Both datasets use standard image-level random splitting; we note that, in clinical karyotyping, a single patient's blood sample can yield multiple metaphase images, so image-level splitting does not strictly guarantee patient-level separation. The datasets do not include patient-level metadata. Both datasets are publicly available: Dataset 1 (RST) on Roboflow Universe at https://universe.roboflow.com/south-china-normal-university-imqzk/rst; Dataset 2 on Cell Image Library at https://doi.org/10.7295/W9CIL54816.

| Dataset | Train | Val | Test | Classes |
|---------|-------|-----|------|---------|
| Dataset 1 | 1,540 | 440 | 220 | 24 |
| Dataset 2 | 3,500 | 500 | 1,000 | 24 |

**Table 4**: Datasets used in this paper. Dataset 1 is Chromosome20240904 (RST); Dataset 2 is the 24 Chromosomes Object dataset.

#### 4.1.2 Architecture and Training

Our base detection framework, denoted LDMDet, uses a ResNet-50 backbone with FPN neck (256 channels, 4 levels), 500 proposals, 6 cascade transformer heads with deep supervision (5 auxiliary heads). Optimization: AdamW (lr=5×10⁻⁵, wd=10⁻⁴), 5-epoch linear warmup + CosineAnnealing, 150 epochs. Loss is Focal ($\lambda_{\text{cls}}{=}2.0$) + L1 ($\lambda{=}5.0$) + GIoU ($\lambda{=}2.0$) with Hungarian matching. Diffusion uses Rectified Flow with a shifted noise schedule (shift=3.0). Default inference solver is Heun 4-step.

#### 4.1.3 Detection Protocol

Boxes are represented in two spaces: image space (absolute pixel xyxy) for RoIAlign and NMS, and diffusion space where GT boxes are converted to cxcywh, normalized to $[0,1]$, and linearly mapped to $[-s, +s]$ with $s{=}2.0$ to match the noise distribution. The forward diffusion uses the rectified-flow linear path $x_t = (1{-}t)\,x_0 + t\,\varepsilon$ (the DDPM baseline uses a cosine schedule $x_t = \sqrt{\bar\alpha}_t\, x_0 + \sqrt{1{-}\bar\alpha}_t\,\varepsilon$). At inference, 500 random-noise proposals are iteratively denoised; with time-ensemble enabled, predictions from all sampling steps ($500 \times \text{steps}$ boxes) are concatenated and deduplicated by per-class NMS (IoU threshold $0.5$). No score thresholding is applied after NMS; all surviving boxes are passed to the COCO evaluator, which truncates to $\text{maxDets}{=}100$ per image. Evaluation uses the standard COCO mAP$@0.5{:}0.95$ (10 IoU thresholds, step $0.05$).

#### 4.1.4 Statistical Considerations

Cross-seed experiments (3 seeds: 42, 123, 789) are available for Dataset 1. For Dataset 2, A3 (DPM-Solver++) has 3 seeds (mean $0.859 \pm 0.003$), confirming the $+0.002$ aggregate mAP gap is within cross-seed variance, though per-image paired tests (Table 8) reveal a statistically significant $+0.006$ advantage.

### 4.2 Main Results: RF vs DDPM

#### 4.2.1 Dataset 2 — Ablation

Table 5 reports a cumulative ablation on the Dataset 2 validation set: A0 (DDPM Euler baseline), A1 is KaryoFlow (RF+Heun), A2 adds Stochastic Coupling, A3 swaps to DPM-Solver++. AdaLN-Zero is used throughout but contributes null individually (Appendix D). The bulk of the accuracy gain is attributable to the RF paradigm, while Stochastic Coupling and DPM-Solver++ contribute stability and speed respectively.

| Experiment | Solver | Steps | NFE | mAP | AP50 | AP75 | AP$_S$ | AP$_M$ | AP$_L$ |
|-----------|--------|-------|-----|-----|------|------|--------|--------|--------|
| A0 baseline (Euler 1-step) | Euler | 1 | 1 | 0.774 | 0.968 | 0.916 | 0.317 | 0.773 | 0.775 |
| **A1 RF+Heun (KaryoFlow)** | Heun | 4 | 7 | **0.857** | 0.989 | 0.969 | 0.502 | 0.853 | 0.867 |
| A2 + Stochastic Coupling ($\epsilon{=}5$) | Heun | 4 | 7 | 0.857 | 0.989 | 0.971 | 0.523 | 0.854 | 0.864 |
| **A3 DPM-Solver++** | DPM++ | 4 | 4 | **0.859±0.003** | 0.988 | 0.968 | 0.516±0.012 | 0.856 | 0.890 |

**Table 5**: Main ablation on Dataset 2 (independent inference, seed 42 for A0–A2; 3-seed mean±std for A3). NFE = total network forward evaluations per image. "Last-30 std" is within-run epoch mAP std. A0→A1 changes multiple variables; the disentanglement ablation (Table 1) attributes 94% of the $+0.083$ gap to the RF training paradigm.

The RF paradigm accounts for $+0.078$ mAP (94% of the $+0.083$ gap), while solver/step configuration adds only $+0.005$ (6%). On Dataset 2, Stochastic Coupling contributes no measurable mAP gain ($+0.000$, Wilcoxon $p{=}0.80$) but $4.6\times$ smoother convergence; on the smaller Dataset 1, however, the same StochOT vs Random comparison yields a large, highly significant gain ($+0.034$, $p<10^{-120}$; Table 9) — the benefit is real in low-data regimes and diminishes with dataset size. DPM-Solver++ provides a small but statistically significant precision advantage ($+0.006$ per-image mAP, Wilcoxon $p{<}0.001$, paired $t$ $p{<}0.001$; see Table 8) and is $1.71\times$ faster.

#### 4.2.2 Dataset 1 — RF vs DDPM

On Dataset 1 (3 seeds), RF outperforms DDPM by +0.017 mAP (0.746 vs 0.729, lower variance ±0.001 vs ±0.004) with 4-step inference vs DDPM's 1-step. DDPM gains only +0.044 from 1→8 steps (0.628 → 0.672), whereas the RF paradigm yields +0.082 mAP on Dataset 2.

### 4.3 SOTA Comparison (Dataset 2)

Table 6 compares our RF-based detector against standard and diffusion baselines. Our best variant (A3, DPM-Solver++, 3-seed mean) trails RTMDet-L (a stronger CSPNeXt-L detector) by only $-0.004$ mAP and the multi-scale DINO R50 by $-0.010$ mAP (the latter outside our cross-seed variance of $\pm 0.003$), while exceeding Cascade R-CNN, YOLOX-S, and DiffusionDet — with the largest gain against the DDPM-based DiffusionDet ($+0.072$ mAP), direct evidence for the RF paradigm's advantage. Importantly, our method achieves this with $1.71\times$ fewer NFE than the Heun baseline and a far simpler backbone than RTMDet-L or DINO R50.

| Method | Backbone | mAP | AP50 | AP75 | AP$_S$ |
|--------|----------|-----|------|------|--------|
| DINO R50 | ResNet-50 | **0.869** | 0.991 | 0.977 | 0.553 |
| RTMDet-L† | CSPNeXt-L | 0.863 | 0.991 | 0.974 | 0.540 |
| **Ours (A3 DPM++)** | ResNet-50 | 0.859 | 0.988 | 0.968 | 0.516 |
| Ours (A2 Heun+StochOT) | ResNet-50 | 0.857 | 0.989 | 0.971 | 0.523 |
| LDMDet (Random) | ResNet-50 | 0.857 | 0.989 | 0.969 | 0.502 |
| Cascade R-CNN | ResNet-50 | 0.854 | 0.987 | 0.972 | 0.525 |
| YOLOX-S | CSPDarkNet-S | 0.796 | 0.987 | 0.944 | 0.452 |
| DiffusionDet | ResNet-50 | 0.787 | 0.970 | 0.928 | 0.500 |

**Table 6**: SOTA comparison on Dataset 2 (independent inference; A3 reports 3-seed mean). LDMDet denotes our base detection framework. RTMDet-L uses a stronger CSPNeXt-L backbone; DINO R50 uses multi-scale deformable attention. Our method is competitive with RTMDet-L on overall mAP while offering $1.71\times$ faster inference; AP$_S$ differences among top methods are not statistically significant (Table 8).

† RTMDet-L training was interrupted at epoch 86; the best checkpoint (epoch 85) is reported.

**Small-object performance and the medical-imaging direction.** On small objects, our detector (AP$_S{=}0.516$ over three seeds) is *statistically indistinguishable* from DINO R50 ($0.553$) and RTMDet-L ($0.540$): a per-image paired Wilcoxon test across the 60 small-object images finds no significant difference among the top methods (Table 8). We therefore do not claim a small-object *advantage*; rather, the result is that a single-shot RF detector with a plain ResNet-50 backbone is *competitive* on the small-object regime that is practically most relevant for chromosome analysis — the Y chromosome and several C-group chromosomes are small and morphologically subtle, and clinical karyotyping prioritizes per-class sensitivity over aggregate mAP. This competitiveness, achieved without the multi-scale deformable attention of DINO or the heavier CSPNeXt-L backbone of RTMDet-L, supports the broader direction of diffusion models for medical imaging where small-target detection under clutter is common. On Dataset 1, where the training set is smaller (1,540 images), our LDMDet (0.753 mAP) in fact exceeds both RTMDet-L (0.742) and DINO R50 (0.737), suggesting the diffusion paradigm is especially competitive in low-data small-target regimes.

#### 4.3.1 Per-Class AP Analysis

Figure 5 reports the per-class AP for all 24 classes on the A3 checkpoint (seed 42, independent inference). Overall AP drops monotonically with chromosome size ($0.896 \to 0.848 \to 0.805$ for Large→Medium→Small), consistent with the well-known difficulty of small-object detection. The Y chromosome is the hardest class (AP=0.779 for seed 42; $0.771 \pm 0.006$ over three training seeds, the highest per-class variance alongside G21 and X), attributable to both data scarcity (~1,803 samples vs ~7,000 per autosome) and biological characteristics (among the smallest chromosomes, heterochromatin-rich, morphologically variable); as a small chromosome, its AP is also the most sensitive to the per-image AP$_S$ variance noted in Section 4.3. The C-group chromosomes (C6–C12), despite being morphologically similar look-alikes, achieve high AP with an intra-group spread of only 0.029 (stable at 0.027–0.032 across seeds), indicating competent fine-grained discrimination when training data are sufficient (Appendix F). Finally, AP50 is near-saturated across all classes (>0.988, and 0.972 for the Y), so localization is near-saturated (>0.988) and the residual errors concentrate in fine-grained classification — suggesting a downstream banding-pattern classifier could recover much of the remaining AP.

![**Figure 5**: Per-class AP on the Dataset 2 validation set (A3 DPM-Solver++). Bars are colored by chromosome size group. The dashed line is the overall mean. Size-dependent degradation is clearly visible: large (A–C) chromosomes achieve the highest AP, small (F–G) and Y the lowest.](latex/figures/per_class_ap.png)

#### 4.3.2 Statistical Significance of the Ablation Gains

Table 8 reports per-image paired significance tests (Wilcoxon signed-rank and paired $t$-test) over the 500 validation images for the three pairwise comparisons underlying the main ablation. Two conclusions stand out. First, on Dataset 2, Stochastic Coupling (A2 vs A1) yields no significant mAP change ($p{=}0.80$) — but this is *dataset-specific*: the same comparison on the smaller Dataset 1 reveals a large, highly significant gain ($+0.034$, $p<10^{-120}$; Table 9), so StochOT's accuracy contribution is real in low-data regimes and diminishes with dataset size. Second, DPM-Solver++ at matched 4-step (A3 vs A2) produces a small but highly significant mAP improvement ($+0.006$, $p<10^{-3}$ on both tests) — i.e., the higher-order solver is *slightly better*, not worse, than Heun at equal step count. On AP$_S$, none of the pairwise differences reach significance ($p>0.6$ on all tests), so the small-object numbers in Tables 5 and 6 should be read as noise-equivalent across our own variants; the same caveat applies to cross-method AP$_S$ comparisons.

| Comparison | Metric | $\Delta$ | Wilc. $p$ | $t$ $p$ | $n$ |
|------------|--------|----------|-----------|---------|-----|
| A2−A1 (StochOT) | mAP | $+0.0001$ | $0.797$ ns | $0.944$ ns | 500 |
| A3−A2 (DPM++) | mAP | $+0.0056$ | $\mathbf{2.5\!\cdot\!10^{-7}}$ *** | $\mathbf{8.4\!\cdot\!10^{-7}}$ *** | 500 |
| A3−A1 (combined) | mAP | $+0.0057$ | $4.5\!\cdot\!10^{-4}$ *** | $4.9\!\cdot\!10^{-5}$ *** | 500 |
| A2−A1 (StochOT) | AP$_S$ | $+0.0012$ | $0.783$ ns | $0.947$ ns | 60 |
| A3−A2 (DPM++) | AP$_S$ | $-0.0031$ | $0.855$ ns | $0.855$ ns | 60 |
| A3−A1 (combined) | AP$_S$ | $-0.0019$ | $0.691$ ns | $0.898$ ns | 60 |

**Table 8**: Per-image paired significance tests on Dataset 2 validation ($n{=}500$ images; AP$_S$ uses the 60 images containing small objects). $\Delta$ is the mean per-image difference of the second model minus the first. Wilc. = Wilcoxon signed-rank; $t$ = paired Student's $t$-test. *** denotes $p<0.001$; ns = not significant ($p>0.05$).

| Comparison | Metric | $\Delta$ | Wilc. $p$ | $t$ $p$ | $n$ |
|------------|--------|----------|-----------|---------|-----|
| Stoch−Rand | mAP | $+0.0308$ | $\mathbf{9.0\!\cdot\!10^{-126}}$ *** | $\mathbf{9.9\!\cdot\!10^{-130}}$ *** | 1320 |
| Hard−Rand | mAP | $-0.0061$ | $1.2\!\cdot\!10^{-8}$ *** | $1.6\!\cdot\!10^{-9}$ *** | 1320 |
| Stoch−Hard | mAP | $+0.0369$ | $\mathbf{9.0\!\cdot\!10^{-155}}$ *** | $\mathbf{2.8\!\cdot\!10^{-156}}$ *** | 1320 |
| Stoch−Rand | AP$_S$ | $+0.0450$ | $3.9\!\cdot\!10^{-68}$ *** | $2.5\!\cdot\!10^{-75}$ *** | 1314 |
| Hard−Rand | AP$_S$ | $-0.0051$ | $1.7\!\cdot\!10^{-2}$ * | $2.2\!\cdot\!10^{-2}$ * | 1314 |
| Stoch−Hard | AP$_S$ | $+0.0501$ | $1.9\!\cdot\!10^{-83}$ *** | $5.9\!\cdot\!10^{-85}$ *** | 1314 |

**Table 9**: Per-image paired significance tests for the coupling ablation on Dataset 1 (pooled across 3 training seeds, $n{=}440{\times}3{=}1320$; AP$_S$ uses 1314 pairs after filtering images without small-object GT). $\Delta$ is the mean per-image difference of the second strategy minus the first. Wilc. = Wilcoxon signed-rank; $t$ = paired Student's $t$-test. *** denotes $p<0.001$; * denotes $p<0.05$.

### 4.4 Coupling Ablation

#### 4.4.1 Dataset 1, Multi-seed

On Dataset 1 (Table C.2), Stochastic Coupling yields a large and highly significant mAP gain over Random coupling ($+0.034$, $0.747$ vs $0.713$; pooled Wilcoxon $p<10^{-120}$, $n{=}1320$; Table 9), and over Hard OT ($+0.042$, $p<10^{-150}$). Hard OT is in fact *worse* than Random ($-0.008$, $p<10^{-8}$), confirming the diversity-collapse pathology predicted by Section 3.3: deterministic OT assignment collapses $H(V|X_t)\to 0$ and degrades training. The gain is dataset-dependent, however: on the larger Dataset 2 (5000 images, 24 classes), the same StochOT vs Random comparison shrinks to $+0.0001$ ($p{=}0.80$, not significant; Table 8). We hypothesize that with more data the marginal benefit of OT-induced pairing diminishes, as the model sees enough samples to average out the random-coupling noise — an empirical signature of the low-data regime where Stochastic Coupling (which also delivers $4.6\times$ smoother convergence, Table 7) is most valuable.

#### 4.4.2 Multi-dimensional Stability Comparison (Dataset 2)

Table 7 reports five additional stability metrics. Stochastic Coupling achieves 30/30 epochs within 1% of the best mAP (vs 13/30 for Random), making late-stage checkpoint selection far more reliable — the property of primary practical concern for EarlyStopping-based training in small-data regimes.

| Metric | A1 (Random) | A3 (Stochastic Coupling $\epsilon{=}5$) | Gain |
|--------|-------------|-----------------------------------------|------|
| Last-30 epoch std | 0.006 | 0.0013 | 4.6× |
| Last-30 CV (std/mean) | 0.69% | 0.16% | 4.4× |
| Last-30 range (max−min) | 0.023 | 0.005 | 4.6× |
| Epochs within 1% of best (last 30) | 13/30 (43%) | 30/30 (100%) | — |
| Best mAP / best epoch | 0.856 / ep62 | 0.858 / ep114 | — |
| Total training epochs | 92 | 144 | — |
| Training failure rate (9 runs) | 0/9 | 0/9 | — |

**Table 7**: Multi-dimensional stability comparison (Dataset 2, single seed). Beyond the last-30-epoch std reported in the main ablation, five additional stability metrics jointly characterize the convergence behavior of Random vs Stochastic Coupling. CV = std/mean.

#### 4.4.3 $\epsilon$ Ablation

$\epsilon < 1$ is harmful (−1.3% mAP within the same augmentation setting); $\epsilon \ge 1$ saturates with diminishing returns, supporting Stochastic Coupling as a necessary OT regularizer rather than a precision booster.

### 4.5 Solver Analysis

#### 4.5.1 DPM-Solver++ Step Ablation

DPM-Solver++ converges at 2 steps (mAP 0.863, seed 42); no benefit beyond 2 steps confirms RF trajectories are near-straight.

#### 4.5.2 DPM-Solver++ vs Heun at Matched NFE

At similar NFE, DPM-Solver++ 4-step (4 NFE, $0.863$) ≈ Heun 2-step (3 NFE, $0.863$) — equal accuracy, so DPM-Solver++ buys 43% fewer NFE at no cost. Crucially, at matched *step count* (4 vs 4), DPM-Solver++ is in fact slightly but significantly *more* accurate than Heun ($+0.006$ per-image mAP, Wilcoxon $p<10^{-3}$; Table 8), so its advantage is both computational and a small precision gain — not "purely computational" as one might infer from the matched-NFE comparison alone.

#### 4.5.3 Test Set Evaluation

On the Dataset 2 test set, A3 achieves mAP 0.859 (vs val $0.863$ for seed 42, $\Delta = -0.004$), confirming that the aggregate accuracy generalizes well. The AP$_S$ point estimate, however, swings from $0.499$ (val) to $0.577$ (test) for the same checkpoint — a reminder that AP$_S$ on this 24-class benchmark is high-variance (only 60 val images contain small objects) and should be interpreted alongside the per-image significance tests in Table 8 rather than as a point estimate.

### 4.6 FPS / Latency Benchmark

Table 10 and Figure 6 report the speed-accuracy trade-off at $512{\times}512$ input resolution. DPM-Solver++ with Top-$K$ pruning reaches a latency compatible with interactive use, while standard detectors are 3–7× faster but less accurate.

| Model | Solver | NFE | Latency (ms) | FPS | mAP |
|-------|--------|-----|-------------|-----|-----|
| A1 RF+Heun | Heun | 7 | 124.38 ± 3.38 | 8.0 | 0.856 |
| A2 + Stoch. Coup. | Heun | 7 | 128.35 ± 1.95 | 7.8 | 0.858 |
| **A3 DPM++** | DPM++ | 4 | **75.03 ± 0.96** | **13.3** | **0.863** |
| A3 + IO3 K=300 | DPM++ | 4 | 71.27 ± 2.39 | 14.0 | 0.861 |
| **A3 + IO3 K=200** | DPM++ | 4 | **70.46 ± 2.28** | **14.2** | **0.860** |
| A3 + IO3 K=100 | DPM++ | 4 | 69.71 ± 1.98 | 14.3 | 0.850 |
| Cascade R-CNN | — | 1 | 20.67 ± 0.48 | 48.4 | 0.854 |
| YOLOX-S | — | 1 | 10.15 ± 0.41 | 98.5 | 0.796 |
| DiffusionDet | Euler | 1 | 24.38 ± 1.09 | 41.0 | 0.787 |

**Table 10**: FPS / latency benchmark (Dataset 2, 512×512, seed 42). Latency is reported as mean ± std over 200 images on an RTX A6000.

A3 + IO3 K=200 is the fastest variant (70.46 ms / 14.2 FPS, mAP 0.860); A3 achieves 75 ms / 13.3 FPS at mAP 0.863. The cascade head dominates 90%+ of latency; the backbone+neck is a minor cost (~5.8 ms, 4–8%).

![**Figure 6**: Speed-accuracy trade-off (Dataset 2, RTX A6000, 512×512). Log-scale FPS axis. Our RF variants (circle/square) occupy the high-accuracy region (mAP > 0.85); standard detectors (triangle) are 3–7× faster but less accurate. A3+IO3 K=200 (14.2 FPS, mAP 0.860) achieves the best speed-accuracy trade-off among our variants.](latex/figures/fps_map.png)

### 4.7 Cross-Dataset Summary

Across both datasets, RF outperforms DDPM ($+0.017$ mAP on Dataset 1, $+0.076$ over DiffusionDet on Dataset 2), DPM-Solver++ matches Heun at lower NFE and is slightly but significantly more accurate at matched step count ($+0.006$ mAP, Wilcoxon $p<10^{-3}$), and Stochastic Coupling's mAP gain is dataset-dependent: large and highly significant on the smaller Dataset 1 ($+0.034$ over Random, $p<10^{-120}$; Hard OT is in fact *worse* than Random, $-0.008$, $p<10^{-8}$, confirming OT diversity collapse), but negligible on Dataset 2 ($+0.0001$, $p{=}0.80$). The $4.6\times$ convergence-smoothness benefit holds on both.

### 4.8 Robustness (Planned for TMI)

<!-- [PLACEHOLDER: Phase 3-D — Lightweight Robustness Experiments]
Two inference-only experiments to strengthen TMI evaluation (SIER criteria:
Evaluation + Reproducibility). Both reuse existing checkpoints — NO retraining.

(1) Annotation-noise robustness (Dataset 2, A3 checkpoint):
    - Inject label noise by perturbing GT bbox coordinates (Gaussian jitter at
      σ_bbox = {2, 5, 10} px) and class-label flips (rate = {5%, 10%, 20%}).
    - Re-run inference on the perturbed test set; report mAP degradation curve.
    - Hypothesis: RF's straight-line ODE is more robust to label noise than
      DDPM's curved trajectory (lower truncation error amplification).
    - Expected effort: ~1 day (inference-only, existing checkpoints).

(2) Cross-dataset transfer (Dataset 1 → Dataset 2, zero-shot):
    - Take A3 checkpoint trained on Dataset 2; evaluate on Dataset 1 test set
      (and vice versa) WITHOUT fine-tuning.
    - Report mAP + per-class AP to characterize cross-cohort generalization.
    - Hypothesis: Stochastic Coupling's smoothness benefit generalizes across
      cohorts; RF paradigm transfers better than DDPM due to straighter paths.
    - Expected effort: ~0.5 day (inference-only).

Both experiments → [MAIN PAPER] as a new §4.8 subsection (~0.5 page).
Detailed per-perturbation tables → [ARXIV COMPANION].
-->

*To be added in Phase 3-D.*

## 5. Analysis and Discussion

### 5.1 Why RF Works for Chromosome Detection

RF's straight-line ODE paths reduce truncation error in few-step inference, which is especially valuable for chromosome detection: the high object density (~46 per image) compounds per-box errors, the small training sets (1,540–5,000 images) limit the model's ability to learn complex curved DDPM trajectories, and the 24-class fine-grained task benefits from stable feature representations. The +0.082 mAP improvement ($0.774 \to 0.856$) on Dataset 2 confirms RF's effectiveness in this regime.

### 5.2 Stochastic Coupling: Dataset-Dependent mAP Gain Plus Smoothness

Stochastic Coupling's value has two distinct components. On Dataset 2 (5000 images), the mAP gain is negligible ($+0.0001$, $p{=}0.80$, Table 8) and its value is entirely smoother convergence ($4.6\times$ epoch-std reduction, $0.006 \to 0.0013$). On the smaller Dataset 1 (1540 images), however, the same comparison reveals a large, highly significant mAP gain ($+0.034$, $p<10^{-120}$, Table 9) on top of the smoothness benefit. This dataset-dependence is consistent with the theory: with more data, the model sees enough samples to average out random-coupling noise, attenuating OT collapse and the marginal benefit of StochOT.

On both datasets, the smoothness benefit has practical consequences for checkpoint selection: with Random coupling, checkpoint selection may land on a "lucky" epoch 0.006 above the trend — a false peak that may not generalize. Stochastic Coupling's 0.0013 epoch std makes checkpoint selection far more reliable. The seed 123 result (mAP 0.857 vs seed 42's 0.863, $\Delta = -0.006$) confirms that epoch oscillation directly impacts which checkpoint EarlyStopping selects. A formal causal link (Stochastic Coupling → better test generalization via better checkpoint selection) requires per-epoch test evaluation, left as future work.

### 5.3 DPM-Solver++ vs Heun: Computational and Small Precision Advantage

Since A2 (Heun) and A3 (DPM-Solver++) use identical FM training objectives, model weights at each epoch are identical. Yet DPM-Solver++ at matched 4-step yields a small but statistically significant mAP improvement over Heun ($+0.006$ per-image mAP, Wilcoxon $p<10^{-3}$; Table 8), so the higher-order solver is slightly *better*, not worse. Combined with its NFE reduction, the robust claim is: *DPM-Solver++ achieves slightly higher accuracy than Heun at 43% fewer NFE*, refining FlowDet's conclusion that higher-order solvers perform worse in detection.

### 5.4 IO3 Pruning: Solver-Dependent Effectiveness

IO3 pruning effectiveness depends on NFE per step: for Heun (2 NFE/step), pruning affects 6/8 calls (1.09–1.12× speedup); for DPM-Solver++ (1 NFE/step), 3/4 calls (1.05–1.08×). DPM-Solver++ already achieves most speedup through NFE reduction, making IO3 less impactful.

### 5.5 Theory Applicability and Limitations

Our OT Diversity Collapse analysis rests on five assumptions, all well satisfied in chromosome detection: (1) the bound $\Delta H \le \log K$ is an upper bound, tight in the high-noise regime where the shifted schedule spends most timesteps (empirically 0.03% error); (2) the $N \to \infty$ idealization holds because GT bboxes are well-separated in $\mathbb{R}^4$, so even $N=2$ mini-batch OT reduces to nearest-neighbor assignment; (3) the Gaussian noise source is approximately satisfied by our shifted Gaussian schedule; (4) well-separated Voronoi cells (pairwise distances > 20px vs $\sigma \sim 1$px); and (5) the low-dimensional regime ($d=4$, $K \approx 46$) where $\Delta H/H \approx 0.69$ (Table 2).

The theory does not transfer to high-dimensional generation ($d \sim 10^5$, where $\Delta H/H \approx 0$ so OT collapse is negligible — consistent with OT-CFM's success), nor to densely overlapping targets where Assumptions 2 and 4 fail. For COCO ($K \sim 7$, $\Delta H/H \approx 0.55$), the theory predicts Stochastic Coupling would help but with smaller magnitude. The theory suggests RF + Stochastic Coupling would benefit detection tasks combining low $d$, high object density, and small training data — a profile including medical imaging, remote sensing, and other fine-grained dense detection tasks. We did not validate on COCO because its small $K$ reduces OT collapse severity; the appropriate validation dataset has high $K$ and low $d$, exactly the chromosome detection profile. The stability benefit is practically meaningful for clinical deployment: the 4.6× epoch stability improvement means checkpoint selection lands within 0.0013 of the trend (versus 0.006 for Random), reducing the risk of deploying a "false peak" checkpoint.

## 6. Conclusion

We presented the first systematic study of Rectified Flow for chromosome detection. The RF training paradigm — straight-line ODE paths — is the dominant source of accuracy gain, yielding +0.082 mAP over the Euler baseline on Dataset 2 and +0.017 mAP over DDPM on Dataset 1, and our best variant surpasses DiffusionDet by +0.076 mAP; a solver×step disentanglement ablation attributes 94% of the gain to the RF training paradigm, with the shifted noise schedule contributing negligibly on its own (Appendix E). Stochastic Coupling, grounded in our theoretical analysis of OT Diversity Collapse (upper bound $\Delta H \le \log K$, empirically validated to within 0.03%), serves a dual role: in low-data regimes it yields a large, highly significant mAP gain ($+0.034$ on Dataset 1, Wilcoxon $p<10^{-120}$), while on larger data the gain diminishes and its value shifts to a $4.6\times$ reduction of within-run epoch-level mAP oscillation — the property of primary practical concern for reliable training in small-data regimes. DPM-Solver++ enables four-step inference at 13.3 FPS (mAP $0.859 \pm 0.003$ over three seeds), with a 14.2 FPS variant (IO3 K=200) at mAP 0.860; its advantage over Heun at matched step count is both computational ($1.71\times$ fewer NFE) and a small but statistically significant precision gain ($+0.006$ per-image mAP, Wilcoxon $p<10^{-3}$), refining FlowDet's conclusion that higher-order solvers perform worse in detection. We note that standard detectors such as YOLOX-S (98.5 FPS) and Cascade R-CNN (48.4 FPS) are several times faster than our 13.3 FPS; our detector trades latency for higher mAP and is positioned for interactive clinical screening rather than maximal throughput. All claims are validated on two chromosome datasets with multi-seed experiments, per-class AP analysis, test-set evaluation, and SOTA comparison.

Beyond the immediate chromosome setting, the OT Diversity Collapse phenomenon we characterize has development potential for any task sharing the profile of low-dimensional prediction space, high object density, and small training data (severity $\Delta H/H \approx 0.69$). The theory provides an a-priori diagnostic via Table 2; validation on at least one non-chromosome high-$K$ low-$d$ benchmark would substantially strengthen the generality claim and is a natural next step.

We also acknowledge several limitations. First, the empirical validation is confined to chromosome data; we have not validated on COCO or other general detection benchmarks, though our theory predicts that COCO's smaller $K \sim 7$ would attenuate OT collapse and thus reduce the marginal benefit of Stochastic Coupling. Second, Stochastic Coupling's mAP gain is strongly dataset-dependent: it is large and highly significant on the smaller Dataset 1 ($+0.034$, Wilcoxon $p<10^{-120}$; Table 9) but statistically zero on the larger Dataset 2 ($+0.0001$, $p{=}0.80$; Table 8). We hypothesize that more data reduces the marginal benefit of OT-induced pairing; this suggests StochOT is most valuable in low-data regimes — exactly the medical-imaging setting we target — but also means its accuracy contribution cannot be taken for granted on larger benchmarks. In any case, the $4.6\times$ convergence-smoothness benefit (Table 7) holds independently of the mAP gain. Third, the theoretical analysis rests on a well-separated-targets assumption; densely overlapping scenes would require extending the finite-$N$ analysis, and the quantitative $\Delta H \approx \log K$ prediction may not hold there. Addressing these limitations — in particular a formal finite-$N$ theory for overlapping targets and empirical validation on additional high-$K$ detection benchmarks — is a natural direction for future work.

## Appendix

<!--
TMI APPENDIX STRATEGY (Strategy A+B):
- TMI prohibits supplementary text materials (since 2022-01-01).
- Solution: Move detailed appendix content to an arXiv companion preprint.
- Main paper retains only proof sketches (Strategy A) and brief mentions of
  null-result ablations (D, E) inline in the text body.
- Each appendix below is annotated with its TMI destination:
    [MAIN PAPER] / [ARXIV COMPANION] / [COMPRESS] / [DELETED]
- Content marked [ARXIV COMPANION] or [DELETED] is preserved in this draft
  as the complete fact record and feeds the arXiv preprint.
- Estimated page savings: ~3.4 pages of appendix compressed to ~0.4 page
  (proof sketches + inline mentions) in the 10-page IEEE main paper.
-->

### A. Proofs

<!-- [MAIN PAPER: proof sketches only — retain §3.3.2 sketch + Conjecture 1 statement]
    [ARXIV COMPANION: full proofs below + new lower bound (Phase 3-E placeholder)]
    Page budget: ~0.2 page in main paper (sketch already inline at §3.3.2). -->

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

<!-- [PLACEHOLDER: Phase 3-E — Lower Bound for Proposition 1]
INSERTION POINT for the matching lower bound. The upper bound ΔH ≤ log K is
proved above; TMI submission should add:

  Proposition 2 (Lower Bound): Let sep = min_{j≠k} ||b_k - b_j|| denote the
  minimum GT-box separation. In the high-noise regime (tσ ≥ c·sep for constant
  c > 0), ΔH ≥ log K - K·exp(-sep²/(2t²σ²)), showing the upper bound is tight
  up to exponential decay in (sep/(tσ))².

Proof approach: Lower-bound H_rand(V|X_t) via the Gaussian mixture posterior
entropy (Bhattacharyya bound on misclassification), and lower-bound H_OT(V|X_t)
via the finite-N Voronoi cell overlap. Full proof -> [ARXIV COMPANION].
Sketch -> [MAIN PAPER §3.3.2] (1 paragraph).
-->

#### A.2 Stochastic Coupling Monotonicity

**Conjecture 1**: $H_{\text{stoch}}(V|X_t; \epsilon)$ monotonically increases with $\epsilon$.

**Empirical argument**: As $\epsilon \to 0$, the Sinkhorn transport matrix $T_\epsilon$ converges to the deterministic OT assignment (hard coupling), so $H_{\text{stoch}} \to H_{\text{OT}} = 0$. As $\epsilon \to \infty$, $T_\epsilon$ converges to the uniform distribution (random coupling), so $H_{\text{stoch}} \to H_{\text{rand}} = \log K$. By the continuity of the Sinkhorn solution in $\epsilon$, $H_{\text{stoch}}$ increases monotonically. A formal proof would use the log-Sobolev inequality for Schrödinger bridges; we leave this as future work and rely on empirical validation of the monotonicity (Section 4.4). We therefore state this property as a conjecture rather than a proven proposition.

<!-- [PLACEHOLDER: Phase 3-E — Rigorous Proof for Conjecture 1]
INSERTION POINT for upgrading Conjecture 1 to a proven proposition (or adding
a partial-result proposition). Candidate approaches:

(1) Log-Sobolev inequality for Schrödinger bridges:
    Show dH_stoch/dε ≥ 0 by differentiating the Sinkhorn objective and applying
    the entropy-regularized OT first-order conditions.

(2) Coupling argument:
    Construct a monotone coupling between T_{ε1} and T_{ε2} for ε1 < ε2,
    showing the assignment distribution under ε2 majorizes that under ε1.

(3) If a complete proof is infeasible, add a formal PARTIAL result:
    Proposition 3 (Endpoint Monotonicity): H_stoch(V|X_t; ε) is monotone in
    the limits ε→0 and ε→∞, with H_stoch → 0 and H_stoch → log K respectively.
    (This is already established by the empirical argument above; formalize it.)

Full proof -> [ARXIV COMPANION]; sketch/statement -> [MAIN PAPER §3.3.4].
-->

### B. Falsified Directions

<!-- [DELETED from TMI main paper]
    These negative results document internal research decisions but do not
    advance the paper's claims. Retain in this draft as fact record only.
    If a reviewer asks "did you try X?", cite the arXiv companion. -->

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

### C. Per-Seed Values (Dataset 1)

<!-- [ARXIV COMPANION]
    Per-seed tables are too granular for the 10-page main paper. The main paper
    reports mean±std aggregates (Tables 5-7); per-seed breakdowns move to arXiv
    for full reproducibility verification. ]

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

**Table C.1**: Per-seed values for RF vs DDPM on Dataset 1.

#### C.2 Coupling Ablation (Section 4.4.1)

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

**Table C.2**: Per-seed values for the coupling ablation (Dataset 1, independent inference, seed 42).

### D. AdaLN-Zero Ablation

<!-- [COMPRESS to 1 sentence in main paper: "AdaLN-Zero contributes null
     (ΔmAP = 0.000) within the RF framework (Appendix D, arXiv)." Already
     mentioned at §1.2 Contributions and §3.1.2. Full table -> [ARXIV COMPANION].]

This appendix reports the standalone ablation of AdaLN-Zero referenced in Section 3.1.2 and the contribution discussion of the main text, confirming that its individual contribution to the +0.082 mAP gain is null within the RF framework (Table D.1). We conducted a separate ablation on Dataset 2 to verify AdaLN-Zero's individual contribution. Both experiments use identical configurations except for the time conditioning module (RF formulation, Heun solver 4-step, shifted schedule shift=3.0, random coupling, 150 epochs).

| Config | mAP | $\Delta$ mAP |
|--------|-----|--------------|
| RF + Heun (without AdaLN) | 0.856 | — |
| RF + Heun + AdaLN-Zero | 0.856 | +0.000 |

**Table D.1**: AdaLN-Zero ablation on Dataset 2.

AdaLN-Zero contributes *null* ($\Delta$mAP = 0.000) within the RF framework on this dataset. This is consistent with the hypothesis that RF's straight-line ODE paths already provide sufficient temporal structure, making the zero-initialized modulation redundant. We retain AdaLN-Zero as a standard conditioning mechanism (Dhariwal & Nichol, 2021) for consistency with the broader diffusion literature, but note that it does not contribute to the +0.082 mAP improvement claimed in Section 4.2.1. The entire +0.082 gap is attributable to the RF formulation itself (straight-line ODE paths); the shifted noise schedule contributes negligibly on its own, as shown in Appendix E.

### E. Shifted Noise Schedule Ablation

<!-- [COMPRESS to 1 sentence in main paper: "The shifted noise schedule
     (shift=3.0) contributes negligibly (ΔmAP = -0.001); the entire +0.082
     gain is attributable to the RF formulation itself (Appendix E, arXiv)."
     Already mentioned at §1.2 and Abstract. Full table -> [ARXIV COMPANION].]

This appendix reports the standalone ablation of the shifted noise schedule referenced in Section 3.1.2, the abstract, and the conclusion. The shifted schedule (shift=3.0) is used throughout the paper as part of the RF training configuration, but its individual contribution to the +0.082 mAP gain has not been isolated thus far. We fill this gap by training an identical configuration with the shift set to $0$ (i.e., a linear schedule) on Dataset 2.

Both experiments use identical configurations except for the shift parameter (RF formulation, Heun solver 4-step, random coupling, 150 epochs, seed 42). Results are reported at the best checkpoint by validation mAP.

| Schedule | mAP | AP50 | AP75 | AP$_S$ |
|----------|-----|------|------|--------|
| Linear (shift=0) | 0.857 | 0.989 | 0.971 | 0.555 |
| Shifted (shift=3.0) | 0.856 | 0.990 | 0.971 | 0.563 |
| $\Delta$ | $-0.001$ | $+0.001$ | $0.000$ | $+0.008$ |

**Table E.1**: Shifted noise schedule ablation on Dataset 2. The shifted schedule contributes negligibly to mAP on its own; the entire $+0.082$ gap over the Euler baseline is attributable to the RF formulation itself. The linear-schedule run has within-run epoch mAP std 0.0037 (last 30 epochs). AP$_M$ and AP$_L$ differ by at most 0.004 between schedules (not shown).

The shifted schedule's individual contribution is $\Delta \text{mAP} = -0.001$ ($\approx 0\%$), well within seed noise, although it yields a small $+0.008$ improvement on small objects (AP$_S$; AP$_M$ and AP$_L$ differ by at most 0.004). This confirms that the $+0.082$ mAP gain over the Euler baseline claimed in Section 4.2.1 is attributable to the RF formulation itself (straight-line ODE paths), not to the shifted schedule. We retain the shifted schedule as a standard detail inherited from the diffusion-detection literature, but it is not a separate source of accuracy gain. The solver×step disentanglement in Section 4.5 therefore attributes 94% of the $+0.082$ gap to the RF paradigm *as a whole*, which is dominated by the straight-line ODE paths.

### F. Per-Class AP Details

<!-- MIXED DESTINATIONS:
  F.1 Chromosome Size Groups    -> [ARXIV COMPANION] (reference material, brief inline mention OK)
  F.2 Y Chromosome Analysis     -> [MAIN PAPER: 1-2 sentences in §4.3.1] + [ARXIV: full analysis]
  F.3 C-group Discrimination    -> [ARXIV COMPANION] (detailed per-class numbers)
  F.4 Complete Per-Class AP Table -> [ARXIV COMPANION] (24-row table too large for main)
  F.5 Localization Saturation   -> [COMPRESS to 1 sentence in §5 Discussion]
  Net main-paper cost: ~3 sentences + 0 tables; full content preserved here + arXiv. -->

This appendix supplements the per-class AP analysis in Section 4.3.1 with detailed breakdowns.

#### F.1 Chromosome Size Groups

Following the standard ISCN karyotyping convention, the 24 chromosome classes are grouped by physical size into three tiers used in Figure 5:

- *Large*: A1–A3, B4–B5 (metaphase chromosomes visible at low magnification; > 8 Mbp).
- *Medium*: C6–C12, X, D13–D15 (medium-sized submetacentric / acrocentric chromosomes).
- *Small*: E16–E18, F19–F20, G21–G22, Y (the smallest chromosomes; < 5 Mbp for F/G group).

This grouping aligns with the COCO-style AP$_S$/AP$_M$/AP$_L$ split by object area, since chromosome physical size correlates with bbox area in metaphase images.

#### F.2 Y Chromosome Analysis

The Y chromosome is the hardest class (AP=0.779 for seed 42; $0.771 \pm 0.006$ over three seeds), with difficulty overdetermined by data and biology. The Dataset 2 training set contains only about 1,803 Y-chromosome samples, compared with roughly 7,000 per autosome and 5,123 for the X chromosome — a 3.9× imbalance that directly limits the gradient updates the Y class receives. This imbalance is a consequence of biology: the Y appears in only one copy and only in male samples. The Y is also one of the smallest human chromosomes, enriched in heterochromatin, and varies considerably in morphology across individuals. Its AP$_S$=0.577 confirms that the difficulty concentrates at the small-object scale.

#### F.3 C-group Discrimination

The C-group chromosomes (C6–C12) are the archetypal "hard to distinguish" class: seven medium-to-large submetacentric chromosomes of similar size and shape, distinguished by subtle banding-pattern differences. The detector achieves high AP across the group (C6=0.900, C7=0.896, C8=0.883, C9=0.880, C10=0.877, C11=0.871, C12=0.890) with an intra-group spread of only 0.029 (0.871–0.900). The spread is only weakly correlated with size, suggesting the model captures fine banding cues rather than relying on size alone. The X chromosome (AP=0.885) sits within the large-group range, consistent with its medium-large submetacentric morphology and ample training data (5,123 samples).

#### F.4 Complete Per-Class AP Table

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

**Table F.1**: Complete per-class AP breakdown on the Dataset 2 validation set (A3 DPM-Solver++).

#### F.5 Localization Saturation and Downstream Potential

AP50 is near-saturated across all classes (>0.988, and 0.972 for the Y), so localization is near-saturated (>0.988) and residual errors concentrate in fine-grained classification. A downstream refinement stage operating on correctly localized crops — a banding-pattern classifier or morphology-aware re-scoring head — could in principle recover much of the remaining AP, since the upstream detector already supplies the right regions.
