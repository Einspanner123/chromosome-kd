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
- Sync state: synced to main.tex commit 8f67c88a (2026-07-19, Phase 3 complete).
  Phase 3 changes incorporated:
  - Proposition 2 (OT Diversity Lower Bound via Fano's inequality) added
  - Conjecture 1 promoted to Proposition 3 (Monotonicity via envelope theorem)
  - Cross-dataset §4.8 corrected: Dataset 1 = Chromosome20240904 (220 test imgs,
    10262 instances), not AutoKary (118 test imgs). Numbers: mAP=0.157,
    AP50=0.513, AP75=0.039; 14/24 classes AP50>0.5; C-group fails.
  - Conclusion updated to reference two-sided bound
  - Appendix B+C merged into single "AdaLN-Zero and Shifted Schedule Ablations"
    section (B's null-result table moved to arXiv companion)
  - Compressed to ≤10 pages (TMI hard constraint met)
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

<!-- [MAIN PAPER] Abstract for TMI: <=250 words, application-led opening + algorithmic contributions.
    Word count: 247 (verified 2026-07-19).
    Reviewer point 1 (2026-07-19): reduce metric stacking, add clinical value, medical expression.
    Narrative direction (user feedback 2026-07-19): lead with "diffusion-based detector introduced
    into chromosome imaging"; theory serves the task, not vice versa. -->

Chromosome karyotyping --- the microscopic inspection of metaphase chromosomes for genetic diagnosis --- remains a slow, labor-intensive, and observer-dependent cornerstone of clinical cytogenetics. Each metaphase cell contains roughly forty-six densely packed chromosomes across twenty-four morphologically similar classes; automating this analysis requires a detector accurate and fast enough for routine clinical deployment. Conventional anchor-based detectors plateau on fine-grained intra-group discrimination, while diffusion-based detectors inherit the slow many-step inference and trajectory truncation errors of their image-generation ancestors, and small clinical datasets further destabilize training.

We introduce *KaryoFlow*, a diffusion-based detector for chromosome karyotyping built on *Rectified Flow* (RF), which replaces the curved stochastic trajectories of DDPM with deterministic straight-line ODE paths. Three contributions target the specific difficulties of chromosome imaging. The RF paradigm itself yields the dominant accuracy gain, exceeding DiffusionDet by $+0.060$ mAP and matching Cascade R-CNN, with a controlled ablation attributing $94\%$ of the gain to RF rather than to solver or step-count choices. We further characterize an *OT Diversity Collapse* that arises in the low-dimensional ($\mathbb{R}^4$) detection space and propose *Stochastic Coupling* via Sinkhorn transport, which restores coupling diversity and stabilizes training in the low-data regime. Finally, DPM-Solver++ with Top-$K$ proposal pruning delivers four-step inference at clinical-grade latency, with a small but statistically significant precision advantage over the Heun baseline.

All claims are validated on two public chromosome datasets with multi-seed experiments, per-class AP analysis, and SOTA comparison, supporting diffusion-based detection as a practical paradigm for fine-grained medical imaging.

<!-- [MAIN PAPER] IEEEkeywords placeholder — to be finalized:
Index Terms --- Rectified Flow, object detection, optimal transport, diffusion models, medical image analysis, chromosome karyotyping
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

Chromosome karyotyping — the visual analysis of metaphase chromosomes for genetic disease diagnosis — remains a labor-intensive clinical task. Each metaphase image contains about 46 tightly packed chromosomes across 24 classes (A1–Y), with severe class imbalance, fine-grained intra-group similarity, and frequent overlaps. Automating this process requires a detector that is both accurate and fast enough for clinical deployment, yet the three families of methods developed over the past decade each leave a measurable gap on at least one of these requirements. Conventional anchor-based detectors (Cascade R-CNN, YOLOX) reach respectable throughput but plateau on the fine-grained 24-class discrimination that distinguishes, for example, the acrocentric C-group chromosomes from one another. Transformer detectors (DINO) lift accuracy but require multi-scale deformable attention and considerably heavier backbones, which is unattractive when clinical data are scarce. Diffusion-based detectors (DiffusionDet) reformulate detection as iterative denoising from noisy boxes and offer a conceptually elegant fit for the structured-prediction nature of the task, but they inherit the slow many-step inference and curved-trajectory truncation errors of their DDPM ancestors and are further destabilized by the small training sets typical of clinical imaging.

Rectified Flow (RF) offers a principled remedy: by replacing the curved stochastic DDPM trajectory with a deterministic straight-line ODE path from noise to ground truth, RF enables few-step inference with low truncation error, which is especially valuable when the per-image object density is high (~46 boxes per image) and errors compound across proposals. Applying RF to detection, however, raises three concrete questions that must be answered before the paradigm can be deployed in a clinical pipeline: how should noise samples be coupled to ground-truth boxes when the prediction space is low-dimensional and the number of targets per image is large? Which ODE solver delivers the best accuracy–latency trade-off in the few-step regime? And how can training be stabilized on the imbalanced, small-scale corpora that characterize medical imaging? These three questions — coupling design, solver selection, and training stability — are the bottlenecks of RF-based dense detection in low-data regimes, and our three contributions address them systematically.

**From scenario to method.** A typical metaphase spread contains roughly forty-six chromosomes across twenty-four classes, many touching or overlapping. The difficulty is uneven: C-group chromosomes (C6–C12) are morphologically similar, distinguished mainly by subtle banding patterns, while the Y chromosome is among the smallest, appears in only one copy in male samples, and has roughly 1,803 training samples versus about 7,000 per autosome. A useful detector must therefore localize densely packed objects under few-step inference, train stably from an imbalanced small corpus, and run fast enough for interactive screening. These demands map onto our three ingredients: Rectified Flow supplies straight-line trajectories for few-step, low-truncation inference; Stochastic Coupling counteracts OT diversity collapse in the low-dimensional detection space; and DPM-Solver++ with Top-$K$ pruning converts trajectories into clinical-grade latency.

### 1.2 Contributions

Building on the scenario-to-method mapping above, we make three contributions, each tied to a specific difficulty of chromosome detection and validated on two public datasets (Chromosome20240904, 1,540 images; and 24 Chromosomes Object, 5,000 images) with multi-seed experiments, per-class AP analysis, test-set evaluation, and an FPS benchmark.

The first difficulty is the accuracy ceiling of existing detectors on fine-grained 24-class chromosome imagery, where anchor-based designs struggle with dense packing and intra-group morphological similarity. Our *KaryoFlow* detector addresses this by adopting the RF training paradigm, achieving $+0.082$ mAP over the Euler baseline on 24 Chromosomes Object and $+0.017$ over DDPM on the original dataset. In comparison with SOTA detectors (experimental data in Section 4.3), KaryoFlow as a diffusion-based detector approaches the transformer-based SOTA DINO R50 (47M parameters): the aggregate mAP gap is only about $0.63\%$ ($0.863$ vs $0.8685$), competitive with Cascade R-CNN, and significantly superior to the DDPM-based DiffusionDet ($+0.060$ mAP). Although DINO R50 remains statistically significantly better under per-image paired Wilcoxon test, RF as a diffusion-based method has substantially narrowed the gap with transformer-based SOTA. Detailed statistical tests are in the supplementary material (not yet incorporated in the main text). Because the A0→A1 comparison changes several variables at once (DDPM→RF, Euler→Heun, 1→4 steps), we further conduct a solver×step disentanglement ablation that attributes $94\%$ of the gain to the RF paradigm and only $6\%$ to solver and step-count choices; AdaLN-Zero contributes null individually (Appendix B). The practical implication is that the accuracy gain comes from the paradigm itself rather than from solver tuning, which matters for clinical deployment where reproducibility across sites and seeds is essential.

The second difficulty is a training pathology induced by optimal-transport (OT) coupling in the low-dimensional detection space. When the prediction dimension is $d=4$ and each image contains $K \approx 46$ ground-truth boxes, deterministic OT assignment collapses coupling diversity toward zero — a failure mode we characterize theoretically as *OT Diversity Collapse* (upper bound $\Delta H \le \log K$, matched lower bound via Fano's inequality, empirically tight to within $0.03\%$) — and this collapse degrades training, especially when data are scarce. We propose *Stochastic Coupling*, which samples assignments from the Sinkhorn transport matrix rather than taking an argmax, restoring coupling diversity and stabilizing training. The remedy is most valuable exactly where clinical data are scarcest: on the smaller Dataset 1 it yields a large, highly significant mAP gain ($+0.034$, $p<10^{-120}$) that diminishes with dataset size, and on both datasets it reduces within-run epoch-level mAP oscillation by $4.6\times$, making checkpoint selection reliable for EarlyStopping-based training.

The third difficulty is inference latency: the curved, many-step trajectories of DDPM-based detectors are incompatible with interactive clinical screening, while naive few-step DDPM suffers from truncation error. We deploy DPM-Solver++ for four-step RF inference, achieving $1.71\times$ speedup over Heun while preserving accuracy ($0.859 \pm 0.003$ mAP over three seeds), and combine it with Top-$K$ proposal pruning to reach 13.3–14.2 FPS. A controlled, per-image paired ablation reveals that DPM-Solver++ in fact yields a small but statistically significant precision advantage over Heun at matched step count ($+0.006$ per-image mAP, Wilcoxon $p<0.001$, Table 8), refining FlowDet's conclusion that higher-order solvers perform worse in detection: at matched steps the higher-order solver is slightly *better*, not worse, while also being faster in NFE.

**Novelty boundary.** Relative to FlowDet (CFM with mini-batch OT, which reports that higher-order solvers perform worse), our novelty lies in (i) the formal analysis of *why* mini-batch OT becomes a liability in low-dimensional structured prediction (Section 3.3), and (ii) a solver×step disentanglement showing that, at matched step count, the higher-order DPM-Solver++ is in fact slightly but significantly *better* than Heun while being $1.71\times$ faster in NFE. Relative to DeFloMat (RF for medical detection, treating coupling as an implementation detail), we provide the formal analysis of coupling design as a training pathology and the Stochastic Coupling remedy. Unlike OT-CFM and multisample flow matching (high-dimensional image generation, $d \sim 10^5$), our setting is low-dimensional ($d=4$) with $K \approx 46$, where OT collapse is severe ($\Delta H/H \approx 0.69$). AdaLN-Zero (Dhariwal & Nichol, 2021) is reused as a standard implementation detail (Appendix B); DPM-Solver++ is adopted off-the-shelf, our contribution being the controlled disentanglement analysis with per-image paired statistical tests.

![**Figure 1**: KaryoFlow overview. (a) RF replaces the curved DDPM denoising trajectory with a straight-line ODE path from noise $\mathbf{x}_1$ to Ground Truth (GT) box $\mathbf{x}_0$; nodes indicate the 4 solver steps. (b) Time conditioning injects the continuous time $t$ via AdaLN-Zero zero-initialized modulation so the network is identity at $t{=}0$. (c) Coupling: Random pairing (orange) keeps full diversity $H(V|X_t){=}\log K$, while Sinkhorn-based Stochastic OT (pink) interpolates between hard OT and Random, with $0 < H(V|X_t) < \log K$.](latex/figures/method_overview.png)

## 2. Related Work

### 2.1 Diffusion-Based Object Detection

Diffusion-based detectors reformulate detection as iterative denoising from noisy boxes to GT boxes. Existing works leave three key gaps:

**DiffusionDet** relies on DDPM, whose curved stochastic trajectories make few-step inference slow and truncation-error-prone. **DiffuBox** (Chen et al., 2024) extends diffusion to 3D detection, but its point-diffusion refinement paradigm is fundamentally different from our 2D noise-box-to-GT formulation and cannot be directly borrowed. **FlowDet** is closest to our work—it uses Conditional Flow Matching with mini-batch OT coupling, but has two key shortcomings: it reports "higher-order solvers perform worse" without analyzing why, and overlooks the harm of OT coupling in low-dimensional detection spaces. **DeFloMat** applies Rectified Flow to medical detection but treats coupling as an implementation detail, ignoring its critical role in low-data regimes.

Our work fills these three gaps: (1) we formally analyze the OT diversity collapse pathology in low-dimensional spaces, answering the "why" question FlowDet left unaddressed; (2) we show via a controlled solver disentanglement that higher-order solvers are in fact slightly *better* at matched step count, correcting FlowDet's conclusion; (3) we elevate coupling design from an implementation detail to an analyzable training pathology and propose Stochastic Coupling as a remedy.

### 2.2 Rectified Flow and Flow Matching

Rectified Flow (Liu et al., 2023) replaces curved DDPM trajectories with straight-line ODE paths, and Flow Matching (Lipman et al., 2023) provides a unified training framework. Existing OT coupling methods (OT-CFM, multisample flow matching) work well in *image generation*—a high-dimensional ($d \sim 10^5$) setting where $K$ equals batch size and OT loss is negligible.

But detection is fundamentally different: the dimensionality is extremely low ($d=4$) and each image contains many targets ($K \approx 46$). FlowDet and DeFloMat also work in this low-dimensional detection setting with RF/flow matching, but the former only reports empirical results of OT coupling without analyzing its failure mechanism, and the latter treats coupling as an implementation detail—neither recognized the OT diversity collapse in low-dimensional spaces. Our key insight is that precisely in this low-dimensional, high-$K$ regime, OT coupling approaches its $\log K$ entropy-reduction upper bound and coupling diversity collapses to zero. This failure mode is neither touched by OT-CFM and other high-dimensional works (where OT loss is negligible) nor equivalent to the concurrent analysis of Cheng & Schwing (2025) on conditional high-dimensional generation degradation (their failure mode is condition-skewed prior, orthogonal to our low-dimensional collapse). We not only analyze this pathology but also propose Stochastic Coupling as an interpolating remedy between hard OT and random pairing—the first explicit solution for OT collapse in low-dimensional structured prediction.

### 2.3 Chromosome Detection

Automated chromosome detection has been studied for decades, originally through classical image-processing pipelines (thresholding, morphology, watershed segmentation) that require careful per-cohort parameter tuning and break down under staining variation, overlapping chromosomes, and banding-pattern noise. Modern learning-based approaches adopt general-purpose detectors: YOLO and Faster R-CNN variants reach high throughput but plateau on the fine-grained 24-class discrimination that clinical karyotyping demands, because anchor priors and dense prediction heads are not tailored to the morphologically similar intra-group chromosomes (C6–C12, F-group) that dominate residual errors. Transformer detectors such as DINO improve accuracy but rely on multi-scale deformable attention and heavier backbones, which is unattractive when clinical training data are scarce (1,540–5,000 images per cohort). ChromosomeNet uses the same Taichung dataset as our 24 Chromosomes Object benchmark but releases no code or pretrained models, preventing direct comparison; the Q-band-based pipeline of Sharma et al. and several recent CNN classifiers focus on the downstream classification step assuming boxes are already given, sidestepping the detection problem we address.

Diffusion-based detection has not, to our knowledge, been applied to chromosome karyotyping. We close this gap by providing the first open-source diffusion-based detector for chromosome karyotyping (code will be released upon publication), compare against standard detectors with public implementations (Cascade R-CNN, YOLOX-S, DiffusionDet, DINO, RTMDet-L), and validate with multi-seed experiments and per-class AP analysis that exposes where the remaining errors concentrate (small chromosomes and the morphologically similar C-group).

## 3. Method

This section presents the three core components of KaryoFlow, each targeting a specific difficulty of chromosome detection. The Rectified Flow paradigm in §3.1 replaces the curved stochastic trajectory of DDPM with a deterministic straight-line ODE path, addressing the truncation error accumulation problem under few-step inference (Contribution 1). §3.2 adapts three ODE solvers — Euler, Heun, and DPM-Solver++ — to the RF linear path, providing a computation-accuracy trade-off at matched step count (the inference acceleration part of Contribution 3). §3.3 characterizes the diversity collapse pathology induced by OT coupling in the low-dimensional detection space, and proposes Stochastic Coupling via Sinkhorn transport as a remedy (Contribution 2). The Top-$K$ proposal pruning in §3.4 further reduces inference latency. Theoretical details and proofs are moved to Appendix A; the corresponding experimental validations are in §4.2–§4.5.

### 3.1 Rectified Flow for Detection (KaryoFlow)

Chromosome detection requires regressing $K \approx 46$ 4-dimensional bounding boxes from an image. Diffusion-based detectors (DiffusionDet) formulate this task as iterative denoising from noise boxes to GT boxes, but the curved stochastic trajectory inherited from DDPM produces significant truncation error under few-step inference, and this error compounds across dense proposals. Rectified Flow (RF) replaces the curved trajectory with a deterministic straight-line ODE path from noise $\mathbf{x}_1$ to GT $\mathbf{x}_0$, making the velocity field constant along the path and thereby keeping truncation error low in the few-step regime.

#### 3.1.1 RF Formulation

The conditional probability path is
$$\mathbf{x}_t = (1-t)\, \mathbf{x}_0 + t\, \mathbf{x}_1, \qquad t \in [0,1],$$
where $\mathbf{x}_0$ is the target (GT bbox) and $\mathbf{x}_1$ is the source (Gaussian noise). The corresponding velocity field is constant along the path, $\mathbf{v} = \mathbf{x}_1 - \mathbf{x}_0$, and the training objective is the flow matching loss
$$\mathcal{L}_{FM} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\!\left[ \left\lVert \mathbf{v}_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \right\rVert^2 \right].$$

**Detection-specific adaptation**: source $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_4)$, target $\mathbf{x}_0$ are GT bboxes, $d=4$ (vs. $d=196{,}608$ in image generation), and $K \approx 46$ targets per image.

#### 3.1.2 Time Conditioning

AdaLN-Zero (Dhariwal & Nichol, 2021) serves as the time conditioning mechanism, replacing scale_shift conditioning with zero-initialized adaptive layer norm so the network starts as an unconditional model (identity at $t{=}0$). A separate ablation (Appendix B) confirms its individual contribution to mAP is null; the +0.082 mAP gain is entirely attributable to the RF formulation itself, with the shifted noise schedule contributing negligibly on its own (Appendix B). We retain AdaLN-Zero as a standard implementation detail, not a separate contribution.

### 3.2 ODE Solvers

The RF paradigm yields a straight-line ODE path, but the choice of solver for this ODE determines the accuracy-latency trade-off in few-step inference: low-order solvers (Euler) incur large truncation error, while high-order solvers (Heun) trade more NFE for accuracy. We consider three solvers: Euler (1st-order, 1 NFE/step) as the baseline, Heun (2nd-order predictor-corrector, 2 NFE/step) for high-order accuracy, and DPM-Solver++ (2nd-order multistep, 1 NFE/step) which matches Heun's accuracy while halving the NFE. All three are adapted to the RF linear path; detailed derivations are in Appendix A.4–A.5.

**Euler (1st-order).** Steps directly along the velocity field: $\mathbf{x}_{t-\Delta t} = \mathbf{x}_t + \Delta t \cdot \mathbf{v}_\theta(\mathbf{x}_t, t)$. 1 NFE per step, 4 NFE for 4 steps. Used as the DDPM baseline (A0) solver.

**Heun (2nd-order predictor-corrector).** Provides 2nd-order accuracy via predictor-corrector, 2 NFE per step, 7 NFE for 4 steps (the last step degrades to Euler). Detailed formulas are in Appendix A.4.

**DPM-Solver++ (2nd-order multistep).** We adapt DPM-Solver++ (Lu et al., 2022) to the data-prediction form of the RF linear path, using polynomial interpolation of the $\mathbf{x}_0$ history in $t$ space to achieve 1 NFE/step, 4 NFE for 4 steps ($1.71\times$ speedup over Heun's 7 NFE). The singularity at $t \to 0$ is handled by an $\epsilon$ cutoff. Detailed derivation is in Appendix A.5.

### 3.3 OT Diversity Collapse and Stochastic Coupling

RF training requires coupling noise samples $\mathbf{z}_i$ to GT boxes $\mathbf{b}_k$ to define the flow matching objective. In the low-dimensional detection space ($d=4$, vs $d \sim 10^5$ in image generation) with $K \approx 46$ targets per image, deterministic optimal transport (OT) coupling partitions the noise space into Voronoi cells, making the coupling assignment a deterministic function of the noise — coupling diversity collapses to zero, harming training, an effect especially pronounced when data are scarce. We first provide a formal analysis of this *OT Diversity Collapse* (Propositions 1–2), then propose *Stochastic Coupling* via Sinkhorn transport (Proposition 3) that interpolates between hard OT and random coupling, restoring diversity.

**Formal analysis of OT Diversity Collapse.** Let source $\nu = \mathcal{N}(0, \sigma^2 I_d)$, target $\mu = \frac{1}{K}\sum_k \delta_{\mathbf{b}_k}$, $V$ be the coupling assignment random variable, and $X_t = (1-t)\mathbf{b}_V + t\mathbf{z}$ be the flow state observed by the model. As $N \to \infty$, OT coupling partitions $\mathbb{R}^d$ into $K$ Voronoi cells $\mathcal{V}_k = \{\mathbf{z} : \lVert\mathbf{z} - \mathbf{b}_k\rVert \le \lVert\mathbf{z} - \mathbf{b}_j\rVert, \forall j\}$ (semi-discrete OT limit, Santambrogio, 2015), making $V$ a deterministic function of $\mathbf{z}$. Under this assumption, the conditional entropy loss of OT coupling relative to random coupling $\Delta H = H_{\text{rand}}(V|X_t) - H_{\text{OT}}(V|X_t)$ satisfies the two-sided bound:

$$\log K \cdot (1 - P_{\text{err}}) - h(P_{\text{err}}) \;\le\; \Delta H \;\le\; \log K,$$

where the upper bound is Proposition 1 (under OT $V$ can be recovered from $X_t$, so $H_{\text{OT}} = 0$) and the lower bound is Proposition 2 (Fano's inequality + union bound, $P_{\text{err}} \le \binom{K}{2}\Phi(-d_{\min}/(2\sigma_t))$). When $\sigma_t/d_{\min} \to 0$ the two bounds match, $\Delta H \to \log K$. For chromosome detection ($d_{\min} \approx 20$ px, $\sigma_t \sim 1$ px), $P_{\text{err}} < 10^{-45}$, so $\Delta H \ge 0.999\,\log K$, indicating OT collapse is unavoidable in this setting. Empirical validation (Chromosome20240904): $\Delta H = 3.8415$ vs $\log K = 3.8427$ ($K_{\text{mean}} = 46.6$), relative error 0.03% (Figure 3). Full proofs are in Appendix A.1–A.2.

![**Figure 3**: OT Diversity Collapse. (a) OT coupling on a real chromosome detection image (Dataset 2 validation set, cropped from a metaphase spread containing ~46 chromosomes). Black rectangles are 8 GT boxes (with class labels); circles in the noise space above the image are noise samples $\mathbf{z}_i$; solid lines are OT (nearest-neighbor) assignments, dashed lines are random assignments (only 4 shown to avoid clutter). OT deterministically maps each noise sample to the nearest GT, making the coupling assignment a deterministic function of the noise — diversity collapses to zero; while random assignment preserves full diversity. (b) Empirical validation of Proposition 1: theoretical $\log K = 3.8427$ vs. empirical $\Delta H = 3.8415$ (relative error 0.03%).](latex/figures/ot_theory.png)

![**Figure: Entropy phase diagram.** Conditional entropy $H(V|Z)$ as a function of the stochastic coupling parameter $\epsilon$. Hard OT ($\epsilon{=}0$) collapses to $H{=}0$; Random coupling ($\epsilon{\to}\infty$) saturates at $H{=}3.8415 \approx \log K{=}3.8427$ (relative error 0.03%). Stochastic coupling with $\epsilon \ge 1$ recovers near-full diversity, while $\epsilon < 1$ falls in the danger zone of diversity collapse.](latex/figures/entropy_phase.png)

**Dimension dependence.** OT collapse is severe in detection ($\Delta H/H \approx 0.55$–$0.69$) but negligible in image generation (Table 2), which explains why OT-CFM is applicable in high-dimensional image generation but becomes a training bottleneck in low-dimensional detection.

| Scenario | $d$ | $K$ | $\Delta H / H$ |
|----------|-----|-----|-----------------|
| Image generation | $256^2{\times}3$ | batch | $\approx 0$ |
| Detection (COCO) | 4 | $\sim$7 | $\approx 0.55$ |
| Detection (Chromosome) | 4 | $\sim$46 | $\approx 0.69$ |

**Table 2**: OT Diversity Collapse severity by scenario. OT-induced diversity loss is severe in detection, negligible in image generation.

**Stochastic Coupling.** Unlike argmax assignment, we sample the coupling from the rows of the Sinkhorn transport matrix $T_\epsilon$ (Cuturi, 2013):
$$\pi_{\text{stoch}}(i) \sim \operatorname{Categorical}\!\left( \frac{T_\epsilon(i,:)}{\sum_j T_\epsilon(i,j)} \right).$$
The key property is that $H_{\text{stoch}}(V|X_t; \epsilon)$ increases monotonically with $\epsilon$ (Proposition 3, Appendix A.3, proved via the envelope theorem), with endpoints $\epsilon \to 0$ (hard OT, $H \to 0$) and $\epsilon \to \infty$ (random coupling, $H \to \log K$). Experimental validation of the training stability and mAP impact of Stochastic Coupling is in §4.4.

### 3.4 Top-$K$ Proposal Pruning

At inference time, all 500 proposals pass through the 4-step cascade head, and the cascade head accounts for 90%+ of the latency (§4.6). After step 0, we prune proposals from 500 to $K$ based on confidence scores; only the top-$K$ proposals proceed through steps 1–3, reducing the computation of the subsequent 3 steps by a factor of $500/K$.

DPM-Solver++ compatibility requires calling `dpm_solver.reset()` after pruning, because the $\mathbf{x}_0$ history has a dimension mismatch (500 → $K$).

## 4. Experiments

### 4.1 Experimental Setup

#### 4.1.1 Datasets

Table 4 summarizes the two public chromosome datasets used in this paper, which we refer to as Dataset 1 and Dataset 2 hereafter. Dataset 1 is Chromosome20240904 (RST) \cite{south2024chromosome}, a clinical collection of 1,540 images, publicly released via Roboflow Universe. Dataset 2 is the 24 Chromosomes Object dataset \cite{tseng2023dataset,lu2022cil54816}, comprising 5,000 images, publicly released via the Cell Image Library. Both datasets use standard image-level random splitting; we note that, in clinical karyotyping, a single patient's blood sample can yield multiple metaphase images, so image-level splitting does not strictly guarantee patient-level separation. The datasets do not include patient-level metadata. Access URLs for the datasets are provided in the aforementioned reference entries.

| Dataset | Train | Val | Test | Classes |
|---------|-------|-----|------|---------|
| Dataset 1 | 1,540 | 440 | 220 | 24 |
| Dataset 2 | 3,500 | 500 | 1,000 | 24 |

**Table 4**: Datasets used in this paper. Dataset 1 is Chromosome20240904 (RST); Dataset 2 is the 24 Chromosomes Object dataset.

#### 4.1.2 Architecture and Training

KaryoFlow uses a ResNet-50 backbone with FPN neck (256 channels, 4 levels), 500 proposals, 6 cascade transformer heads with deep supervision (5 auxiliary heads). Optimization: AdamW (lr=5×10⁻⁵, wd=10⁻⁴), 5-epoch linear warmup + CosineAnnealing, 150 epochs. Loss is Focal ($\lambda_{\text{cls}}{=}2.0$) + L1 ($\lambda{=}5.0$) + GIoU ($\lambda{=}2.0$) with Hungarian matching. Diffusion uses Rectified Flow with a shifted noise schedule (shift=3.0). Default inference solver is DPM-Solver++ 4-step (the A1/A2 ablation configurations use Heun 4-step to isolate the solver contribution).

#### 4.1.3 Detection Protocol

Boxes are represented in two spaces: image space (absolute pixel xyxy) for RoIAlign and NMS, and diffusion space where GT boxes are converted to cxcywh, normalized to $[0,1]$, and linearly mapped to $[-s, +s]$ with $s{=}2.0$ to match the noise distribution. The forward diffusion uses the rectified-flow linear path $x_t = (1{-}t)\,x_0 + t\,\varepsilon$ (the DDPM baseline uses a cosine schedule $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1{-}\bar\alpha_t}\,\varepsilon$). At inference, 500 random-noise proposals are iteratively denoised; with time-ensemble enabled, predictions from all sampling steps ($500 \times \text{steps}$ boxes) are concatenated and deduplicated by per-class NMS (IoU threshold $0.5$). No score thresholding is applied after NMS; all surviving boxes are passed to the COCO evaluator, which truncates to $\text{maxDets}{=}100$ per image. Evaluation uses the standard COCO mAP$@0.5{:}0.95$ (10 IoU thresholds, step $0.05$).

#### 4.1.4 Statistical Considerations

All experiments in this paper are validated with at least 3 random seeds (42, 123, 789) independently trained. Cross-seed standard deviations (e.g., $0.859 \pm 0.003$ mAP for A3) are used to bound the statistical significance of key comparisons; when the aggregate mAP gap falls within cross-seed variance, we further report per-image paired tests (Table 8, Table 9) to reveal statistically significant differences.

### 4.2 Main Results: RF vs DDPM

Table 5 reports a cumulative ablation: A0 (DDPM Euler baseline), A1 is KaryoFlow (RF+Heun), A2 adds Stochastic Coupling, A3 swaps to DPM-Solver++. AdaLN-Zero is used throughout but contributes null individually (Appendix B). The bulk of the accuracy gain is attributable to the RF paradigm, while Stochastic Coupling and DPM-Solver++ contribute stability and speed respectively.

| Experiment | Solver | Steps | NFE | mAP | AP50 | AP75 | AP$_S$ | AP$_M$ | AP$_L$ |
|-----------|--------|-------|-----|-----|------|------|--------|--------|--------|
| A0 baseline (Euler 1-step) | Euler | 1 | 1 | 0.774 | 0.968 | 0.916 | 0.317 | 0.773 | 0.775 |
| **A1 RF+Heun (KaryoFlow)** | Heun | 4 | 7 | **0.856** | 0.989 | 0.969 | 0.502 | 0.853 | 0.867 |
| A2 + Stochastic Coupling ($\epsilon{=}5$) | Heun | 4 | 7 | 0.858 | 0.989 | 0.971 | 0.523 | 0.854 | 0.864 |
| **A3 DPM-Solver++** | DPM++ | 4 | 4 | **0.859** | 0.988 | 0.968 | 0.516 | 0.856 | 0.890 |

**Table 5**: Main ablation (independent inference; A0–A2 use seed 42, A3 reports the 3-seed best value). *Question:* how much of the A0→A1 accuracy gain is attributable to the RF paradigm versus the joint change of solver and step count? *Conclusion:* the RF training paradigm accounts for $+0.077$ mAP (94% of the $+0.082$ gap), while solver and step-count configuration contribute only $+0.005$ (6%); Stochastic Coupling and DPM-Solver++ further contribute stability and inference speed respectively. Cross-seed std is reported in the text (A3 mAP $0.859 \pm 0.003$, AP$_S$ $0.516 \pm 0.012$ over three seeds). NFE = total network forward evaluations per image.

The RF paradigm accounts for $+0.077$ mAP (94% of the $+0.082$ gap), while solver/step configuration adds only $+0.005$ (6%). On Dataset 2, Stochastic Coupling contributes no measurable mAP gain ($+0.0001$, Wilcoxon $p{=}0.80$) but $4.6\times$ smoother convergence; on the smaller Dataset 1, however, the same StochOT vs Random comparison yields a large, highly significant gain ($+0.034$, $p<10^{-120}$; Table 9) — the benefit is real in low-data regimes and diminishes with dataset size. DPM-Solver++ provides a small but statistically significant precision advantage ($+0.006$ per-image mAP, Wilcoxon $p{<}0.001$, paired $t$ $p{<}0.001$; see Table 8) and is $1.71\times$ faster.

The cross-dataset RF vs DDPM comparison further confirms the paradigm's advantage: RF with 4-step inference vs DDPM's 1-step inference outperforms DDPM on both datasets — +0.082 mAP on the larger Dataset 2, +0.017 mAP on the smaller Dataset 1 (0.746 vs 0.729, lower variance ±0.001 vs ±0.004). DDPM gains only +0.044 from 1→8 steps (0.628 → 0.672), far less than the RF paradigm's gain.

### 4.3 SOTA Comparison

Table 6 compares our RF-based detector against standard and diffusion baselines. Our best variant (A3, DPM-Solver++, 3-seed mean) trails RTMDet-L (a stronger CSPNeXt-L detector) by $0.004$ mAP and the multi-scale DINO R50 by $0.009$ mAP (both outside our cross-seed variance of $\pm 0.003$), while exceeding Cascade R-CNN, YOLOX-S, and DiffusionDet — with the largest gain against the DDPM-based DiffusionDet ($+0.060$ mAP at seed 42 best, $+0.056$ at the 3-seed mean), direct evidence for the RF paradigm's advantage. Importantly, our method achieves this with $1.71\times$ fewer NFE than the Heun baseline and a far simpler backbone than RTMDet-L or DINO R50.

| Method | Backbone | mAP | AP50 | AP75 | AP$_S$ |
|--------|----------|-----|------|------|--------|
| DINO R50 | ResNet-50 | **0.868** | 0.992 | 0.979 | 0.553 |
| RTMDet-L | CSPNeXt-L | 0.863 | 0.992 | 0.976 | 0.540 |
| **Ours (A3 DPM++)** | ResNet-50 | 0.859 | 0.988 | 0.968 | 0.516 |
| Ours (A2 Heun+StochOT) | ResNet-50 | 0.858 | 0.989 | 0.971 | 0.523 |
| Ours (Random) | ResNet-50 | 0.856 | 0.989 | 0.969 | 0.502 |
| Cascade R-CNN | ResNet-50 | 0.854 | 0.987 | 0.972 | 0.525 |
| YOLOX-S | CSPDarkNet-S | 0.796 | 0.987 | 0.944 | 0.452 |
| DiffusionDet | ResNet-50 | 0.803 | 0.970 | 0.928 | 0.500 |

**Table 6**: SOTA comparison (independent inference; A3 reports 3-seed mean). RTMDet-L uses a stronger CSPNeXt-L backbone; DINO R50 uses multi-scale deformable attention. Our method is competitive with RTMDet-L on overall mAP while offering $1.71\times$ faster inference; AP$_S$ differences among top methods are not statistically significant (Table 8). The $+0.060$ mAP gain over DiffusionDet reported in the text uses the seed 42 best checkpoint (mAP 0.863); the 3-seed mean (0.859) yields $+0.056$. RTMDet-L and DINO R50 values are taken from complete training logs (best checkpoint at epoch 85 and final crashed-state checkpoint respectively).

**Small-object performance and the medical-imaging direction.** On small objects, our detector (AP$_S{=}0.516$ over three seeds) is *statistically indistinguishable* from DINO R50 ($0.553$) and RTMDet-L ($0.540$): a per-image paired Wilcoxon test across the 60 small-object images finds no significant difference among the top methods (Table 8). We therefore do not claim a small-object *advantage*; rather, the result is that a single-shot RF detector with a plain ResNet-50 backbone is *competitive* on the small-object regime that is practically most relevant for chromosome analysis — the Y chromosome and several C-group chromosomes are small and morphologically subtle, and clinical karyotyping prioritizes per-class sensitivity over aggregate mAP. This competitiveness, achieved without the multi-scale deformable attention of DINO or the heavier CSPNeXt-L backbone of RTMDet-L, supports the broader direction of diffusion models for medical imaging where small-target detection under clutter is common. On Dataset 1, where the training set is smaller (1,540 images), KaryoFlow (0.753 mAP) in fact exceeds both RTMDet-L (0.742) and DINO R50 (0.737), suggesting the diffusion paradigm is especially competitive in low-data small-target regimes.

#### 4.3.1 Per-Class AP Analysis

Figure 5 reports the per-class AP for all 24 classes on the A3 checkpoint (seed 42, independent inference). Three patterns are clinically relevant.

First, overall AP drops monotonically with chromosome size ($0.896 \to 0.848 \to 0.805$ for Large→Medium→Small), consistent with the well-known difficulty of small-object detection but also with the clinical reality that the smallest chromosomes (F-group, G-group, Y) carry the highest diagnostic stakes — sex-chromosome aneuploidies and trisomy 21 are among the most frequent karyotyping referrals, so the detection accuracy on these small classes matters disproportionately for clinical utility.

Second, the Y chromosome is the hardest class (AP=0.779 for seed 42; $0.771 \pm 0.006$ over three training seeds, the highest per-class variance alongside G21 and X). Three factors compound: (i) *data scarcity* — Y appears in only one copy in male samples, yielding roughly 1,803 training instances versus about 7,000 per autosome; (ii) *morphology* — Y is among the smallest chromosomes, heterochromatin-rich, and morphologically variable across individuals, so its visual appearance is inherently less stable than the autosomes; and (iii) *class imbalance* — the male-to-female sampling ratio in clinical cohorts further reduces the Y prior. As a small chromosome, its AP is also the most sensitive to the per-image AP$_S$ variance noted in Section 4.3. From a clinical standpoint, Y detection is critical for sex determination and sex-chromosome aneuploidy screening, so even this hardest-class AP (0.779) is clinically actionable when paired with a downstream classifier.

Third, the C-group chromosomes (C6–C12) merit detailed examination because they are the morphologically most similar cluster in the karyotype — all medium-sized metacentric/submetacentric chromosomes distinguished mainly by subtle banding-pattern differences and a gradual size gradient (C6 largest, C12 smallest). Despite this similarity, the detector achieves high AP across the group with an intra-group spread of only 0.029 (stable at 0.027–0.032 across seeds), indicating that the RF features capture the subtle size and banding cues that separate C6 from C12. The spread is not random: it tracks the size gradient, with the larger C6–C8 slightly higher than the smaller C10–C12, mirroring the global size–AP relationship but attenuated — suggesting the detector learns group-internal discrimination beyond pure size. This is practically important because C-group trisomies (e.g., trisomy 8, trisomy 9) are clinically significant and require reliable per-class detection.

Finally, AP50 is near-saturated across all classes (>0.988, and 0.972 for the Y), so localization is near-saturated and the residual errors concentrate in fine-grained classification — suggesting a downstream banding-pattern classifier operating on the detected boxes could recover much of the remaining AP, which is the standard two-stage clinical workflow (detect, then classify).

![**Figure 5**: Per-class AP on the Dataset 2 validation set (A3 DPM-Solver++). Bars are colored by chromosome size group. The dashed line is the overall mean. Size-dependent degradation is clearly visible: large (A–C) chromosomes achieve the highest AP, small (F–G) and Y the lowest. The C-group (C6–C12) maintains high AP despite morphological similarity, and the Y chromosome is the hardest class due to compounded data scarcity and biological variability.](latex/figures/per_class_ap.png)

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

**Table 8**: Per-image paired significance tests on Dataset 2 validation ($n{=}500$ images; AP$_S$ uses the 60 images containing small objects). $\Delta$ is the mean per-image difference of the second model minus the first. Wilc. = Wilcoxon signed-rank; $t$ = paired Student's $t$-test. *** denotes $p<0.001$; ns = not significant ($p>0.05$). *Question:* are the A2−A1 (Stochastic Coupling) and A3−A2 (DPM-Solver++) gains statistically significant at the per-image level on Dataset 2, and does the small-object regime (AP$_S$) admit the same conclusions? *Conclusion:* Stochastic Coupling yields no significant mAP change on the larger Dataset 2 ($p{=}0.80$; its accuracy benefit is confined to the low-data Dataset 1, see Table 9), whereas DPM-Solver++ at matched 4-step produces a small but highly significant mAP improvement ($+0.006$, $p<10^{-6}$) — the higher-order solver is slightly *better*, not worse, than Heun at equal step count; none of the pairwise AP$_S$ differences reach significance, so small-object numbers across our own variants are noise-equivalent.

| Comparison | Metric | $\Delta$ | Wilc. $p$ | $t$ $p$ | $n$ |
|------------|--------|----------|-----------|---------|-----|
| Stoch−Rand | mAP | $+0.0308$ | $\mathbf{9.0\!\cdot\!10^{-126}}$ *** | $\mathbf{9.9\!\cdot\!10^{-130}}$ *** | 1320 |
| Hard−Rand | mAP | $-0.0061$ | $1.2\!\cdot\!10^{-8}$ *** | $1.6\!\cdot\!10^{-9}$ *** | 1320 |
| Stoch−Hard | mAP | $+0.0369$ | $\mathbf{9.0\!\cdot\!10^{-155}}$ *** | $\mathbf{2.8\!\cdot\!10^{-156}}$ *** | 1320 |
| Stoch−Rand | AP$_S$ | $+0.0450$ | $3.9\!\cdot\!10^{-68}$ *** | $2.5\!\cdot\!10^{-75}$ *** | 1314 |
| Hard−Rand | AP$_S$ | $-0.0051$ | $1.7\!\cdot\!10^{-2}$ * | $2.2\!\cdot\!10^{-2}$ * | 1314 |
| Stoch−Hard | AP$_S$ | $+0.0501$ | $1.9\!\cdot\!10^{-83}$ *** | $5.9\!\cdot\!10^{-85}$ *** | 1314 |

**Table 9**: Per-image paired significance tests for the coupling ablation on Dataset 1 (pooled across 3 training seeds, $n{=}440{\times}3{=}1320$; AP$_S$ uses 1314 pairs after filtering images without small-object GT). $\Delta$ is the mean per-image difference of the second strategy minus the first. Wilc. = Wilcoxon signed-rank; $t$ = paired Student's $t$-test. *** denotes $p<0.001$; * denotes $p<0.05$.

#### 4.3.3 Qualitative Comparison

Figure 7 visualizes the detection results of each model on 9 representative cases, covering the full difficulty spectrum from large chromosomes (A-group) to small chromosomes (F/G-group) and the Y chromosome. KaryoFlow's localization accuracy on large/medium chromosomes is comparable to DINO R50 and RTMDet-L; on small chromosomes and the Y chromosome, all methods degrade, but KaryoFlow's miss rate is lower than DiffusionDet's, consistent with the per-class AP analysis in §4.3.1.

![**Figure 7**: Qualitative detection comparison (Dataset 2 validation set, 9 representative cases). Each column is a 3×3 detection grid for one model; from left to right: Ground Truth, KaryoFlow (A3 DPM++), DiffusionDet, RTMDet-L, DINO R50. Cases cover Y chromosome (1, 3), F/G-group small chromosomes (2, 7), D-group (4), X chromosome (5), A-group large chromosomes (6, 8), and E16 (9). Box colors are per-model, in-box labels are predicted classes.](latex/figures/qual_mosaic.png)

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

**Table 7**: Multi-dimensional stability comparison (Dataset 2, single seed). Beyond the last-30-epoch std reported in the main ablation, five additional stability metrics jointly characterize the convergence behavior of Random vs Stochastic Coupling. CV = std/mean. *Question:* does Stochastic Coupling improve training reliability only in the narrow sense of epoch-std, or does the stability advantage extend to multiple practically meaningful metrics such as checkpoint-selection robustness, convergence range, and training-failure rate? *Conclusion:* Stochastic Coupling wins on every measured axis: $4.6\times$ lower epoch std, $4.4\times$ lower coefficient of variation, $4.6\times$ narrower mAP range, and crucially $30/30$ epochs within $1\%$ of the best mAP versus $13/30$ for Random — making late-stage checkpoint selection far more reliable for EarlyStopping-based training in small-data regimes, with zero training failures in $9$ runs for both configurations.

Figure 4 visualizes the per-epoch mAP curve, directly illustrating the 4.6× smoothness gain: Random coupling exhibits epoch-level oscillation, while Stochastic Coupling converges smoothly.

![**Figure 4**: Training stability (24 Chromosomes Object, real per-epoch mAP from training logs): Random coupling exhibits epoch-level oscillation with std 0.006, while Stochastic Coupling ($\epsilon{=}5$) converges smoothly with std 0.0013 (4.6× improvement). Shaded band marks the last 30 epochs used for std computation.](latex/figures/training_stability.png)

#### 4.4.3 $\epsilon$ Ablation

$\epsilon < 1$ is harmful (−1.3% mAP within the same augmentation setting); $\epsilon \ge 1$ saturates with diminishing returns, supporting Stochastic Coupling as a necessary OT regularizer rather than a precision booster.

### 4.5 Solver Analysis

#### 4.5.1 Solver×Step Disentanglement Ablation

To disentangle the contributions of the RF training paradigm from solver/step-count choices, we evaluate all solver×step combinations on the A1 checkpoint (Table 1, Figure 2). At matched step count, solver type has *no effect on mAP* (Euler = DPM-Solver++ at both 4-step and 1-step). Step count has marginal effect (+0.004 from 1 to 4 steps). Heun's +0.001 over Euler 4-step costs $1.75\times$ NFE (7 vs 4) — not cost-effective. The joint solver/step configuration therefore accounts for only $+0.005$ mAP (6%) of the $+0.082$ A0→A1 gap, leaving the remaining 94% attributable to the RF training paradigm. On the full model (A3 vs A2), DPM-Solver++ is in fact slightly but significantly *more* accurate than Heun at matched 4-step ($+0.006$ per-image mAP, Wilcoxon $p<10^{-3}$; Table 8), so its advantage is both computational and a small precision gain — refining FlowDet's conclusion that higher-order solvers perform worse in detection.

| Solver | Steps | NFE | mAP |
|--------|-------|-----|-----|
| Heun | 4 | 7 | 0.856 |
| Euler | 4 | 4 | 0.855 |
| DPM-Solver++ | 4 | 4 | 0.855 |
| Euler | 1 | 1 | 0.851 |
| DPM-Solver++ | 1 | 1 | 0.851 |

**Table 1**: Solver×step disentanglement ablation on the A1 checkpoint (24 Chromosomes Object val, seed 42).

![**Figure 2**: Solver×step disentanglement ablation (A1 checkpoint). Bars are colored by solver type and hatched by step count. Solver/step configuration contributes only +0.005 mAP (6%); the remaining +0.077 mAP (94%) is attributable to the RF training paradigm.](latex/figures/solver_ablation.png)

#### 4.5.2 DPM-Solver++ Step Ablation

DPM-Solver++ converges at 2 steps (mAP 0.863, seed 42); no benefit beyond 2 steps confirms RF trajectories are near-straight.

#### 4.5.3 DPM-Solver++ vs Heun at Matched NFE

At similar NFE, DPM-Solver++ 4-step (4 NFE, $0.863$) ≈ Heun 2-step (3 NFE, $0.863$) — equal accuracy, so DPM-Solver++ buys 43% fewer NFE at no cost. Crucially, at matched *step count* (4 vs 4), DPM-Solver++ is in fact slightly but significantly *more* accurate than Heun ($+0.006$ per-image mAP, Wilcoxon $p<10^{-3}$; Table 8), so its advantage is both computational and a small precision gain — not "purely computational" as one might infer from the matched-NFE comparison alone.

#### 4.5.4 Test Set Evaluation

On the Dataset 2 test set, A3 achieves mAP 0.859 (vs val $0.863$ for seed 42, $\Delta = -0.004$), confirming that the aggregate accuracy generalizes well. The AP$_S$ point estimate, however, swings from $0.499$ (val) to $0.577$ (test) for the same checkpoint — a reminder that AP$_S$ on this 24-class benchmark is high-variance (only 60 val images contain small objects) and should be interpreted alongside the per-image significance tests in Table 8 rather than as a point estimate.

### 4.6 FPS / Latency Benchmark

Table 10 and Figure 6 report the speed-accuracy trade-off at $512{\times}512$ input resolution. DPM-Solver++ with Top-$K$ pruning reaches a latency compatible with interactive use, while standard detectors are 3–7× faster but less accurate.

| Model | Solver | NFE | Latency (ms) | FPS | mAP |
|-------|--------|-----|-------------|-----|-----|
| A1 RF+Heun | Heun | 7 | 124.38 | 8.0 | 0.856 |
| A2 + Stoch. Coup. | Heun | 7 | 128.35 | 7.8 | 0.858 |
| **A3 DPM++** | DPM++ | 4 | **75.03** | **13.3** | **0.863** |
| A3 + Top-$K$ (K=300) | DPM++ | 4 | 71.27 | 14.0 | 0.861 |
| **A3 + Top-$K$ (K=200)** | DPM++ | 4 | **70.46** | **14.2** | **0.860** |
| A3 + Top-$K$ (K=100) | DPM++ | 4 | 69.71 | 14.3 | 0.850 |
| Cascade R-CNN | — | 1 | 20.67 | 48.4 | 0.854 |
| YOLOX-S | — | 1 | 10.15 | 98.5 | 0.796 |
| DiffusionDet | Euler | 1 | 24.38 | 41.0 | 0.803 |

**Table 10**: FPS / latency benchmark (Dataset 2, 512×512, seed 42). Latency is the mean over 200 images on an RTX A6000; the per-image std is below 3.4 ms for all variants and is omitted for clarity (the best value is reported). *Question:* does the diffusion-based detector reach a latency compatible with interactive clinical screening, and at what accuracy cost relative to one-shot detectors? *Conclusion:* A3 with DPM-Solver++ and Top-$K$ pruning reaches 13.3–14.2 FPS at mAP 0.860–0.863, an order-of-magnitude improvement over DDPM-based DiffusionDet (41 FPS but mAP 0.803); standard one-shot detectors are 3–7× faster but trail our method by 0.005–0.067 mAP, positioning the RF detector in the interactive-screening latency band rather than the maximal-throughput band.

A3 + Top-$K$ (K=200) is the fastest variant (70.46 ms / 14.2 FPS, mAP 0.860); A3 achieves 75 ms / 13.3 FPS at mAP 0.863. The cascade head dominates 90%+ of latency; the backbone+neck is a minor cost (~5.8 ms, 4–8%).

![**Figure 6**: Speed-accuracy trade-off (Dataset 2, RTX A6000, 512×512). Log-scale FPS axis. Our RF variants (circle/square) occupy the high-accuracy region (mAP > 0.85); standard detectors (triangle) are 3–7× faster but less accurate. A3+Top-$K$ (K=200) (14.2 FPS, mAP 0.860) achieves the best speed-accuracy trade-off among our variants.](latex/figures/fps_map.png)

### 4.7 Cross-Dataset Summary

Across both datasets, RF outperforms DDPM ($+0.017$ mAP on Dataset 1, $+0.060$ over DiffusionDet on Dataset 2), DPM-Solver++ matches Heun at lower NFE and is slightly but significantly more accurate at matched step count ($+0.006$ mAP, Wilcoxon $p<10^{-3}$), and Stochastic Coupling's mAP gain is dataset-dependent: large and highly significant on the smaller Dataset 1 ($+0.034$ over Random, $p<10^{-120}$; Hard OT is in fact *worse* than Random, $-0.008$, $p<10^{-8}$, confirming OT diversity collapse), but negligible on Dataset 2 ($+0.0001$, $p{=}0.80$). The $4.6\times$ convergence-smoothness benefit holds on both.

### 4.8 Robustness

We report one inference-only robustness probe that strengthens the evaluation
(SIER criteria: Evaluation breadth + Reproducibility). It reuses the A3
checkpoint (DPM-Solver++ 4-step + Top-$K$ pruning) trained on Dataset 2 —
*no model is retrained*.

**Annotation-noise robustness.** We perturb the Dataset 2 *test* ground truth
(GT) by (i) adding Gaussian jitter to each GT bbox center
(σ_bbox ∈ {2, 5, 10} px, width/height preserved, center clipped to image
bounds) and (ii) randomly flipping the class label with probability
p ∈ {5%, 10%, 20%} to a uniformly sampled alternative among the remaining 23
classes. The 3×3 grid of perturbed GTs (plus a clean baseline) is generated
once with seed 42 and re-evaluated with the same checkpoint; image pixels are
untouched. Table 11 reports the resulting mAP degradation.

**Table 11**: Annotation-noise robustness (Dataset 2 test, A3 checkpoint,
seed 42, 1000 images / 45,980 GT instances). Rows: GT bbox jitter σ_bbox
(px). Columns: GT class-flip rate p. Cells: mAP@[0.50:0.95]. Clean baseline
(top-left): 0.859.

| σ_bbox \ p | 0%        | 5%   | 10%  | 20%  |
|------------|-----------|------|------|------|
| 0 px       | **0.859** | ---  | ---  | ---  |
| 2 px       | ---       | 0.666 | 0.599 | 0.477 |
| 5 px       | ---       | 0.428 | 0.386 | 0.308 |
| 10 px      | ---       | 0.179 | 0.162 | 0.129 |

Two patterns emerge. First, *bbox jitter dominates high-IoU precision*: at
σ=5 px, AP50 drops only modestly (0.988→0.847, −0.141) whereas AP75 collapses
(0.971→0.375, −0.596), since a 5-px center shift on ~100-px boxes is enough
to break IoU≥0.75 but not IoU≥0.50. Second, *class flips degrade precision
and recall approximately multiplicatively*: doubling the flip rate roughly
halves the remaining mAP at fixed σ. The two noise sources interact
sub-additively at low noise (the joint −0.193 at σ=2, p=5% is less than the
sum of either marginal would be) but compound severely at high noise. Even
at the most adversarial setting, the model retains a 3× margin over the
random-class baseline (1/24 ≈ 0.042), indicating that the learned RF
features do not collapse under annotation corruption.

## 5. Analysis and Discussion

### 5.1 Why RF Works for Chromosome Detection

RF's straight-line ODE paths reduce truncation error in few-step inference, which is especially valuable for chromosome detection: the high object density (~46 per image) compounds per-box errors, the small training sets (1,540–5,000 images) limit the model's ability to learn complex curved DDPM trajectories, and the 24-class fine-grained task benefits from stable feature representations. The +0.082 mAP improvement ($0.774 \to 0.856$) on Dataset 2 confirms RF's effectiveness in this regime.

### 5.2 Stochastic Coupling: Dataset-Dependent mAP Gain Plus Smoothness

Stochastic Coupling's value has two distinct components. On Dataset 2 (5000 images), the mAP gain is negligible ($+0.0001$, $p{=}0.80$, Table 8) and its value is entirely smoother convergence ($4.6\times$ epoch-std reduction, $0.006 \to 0.0013$). On the smaller Dataset 1 (1540 images), however, the same comparison reveals a large, highly significant mAP gain ($+0.034$, $p<10^{-120}$, Table 9) on top of the smoothness benefit. This dataset-dependence is consistent with the theory: with more data, the model sees enough samples to average out random-coupling noise, attenuating OT collapse and the marginal benefit of StochOT.

On both datasets, the smoothness benefit has practical consequences for checkpoint selection: with Random coupling, checkpoint selection may land on a "lucky" epoch 0.006 above the trend — a false peak that may not generalize. Stochastic Coupling's 0.0013 epoch std makes checkpoint selection far more reliable. The seed 123 result (mAP 0.857 vs seed 42's 0.863, $\Delta = -0.006$) confirms that epoch oscillation directly impacts which checkpoint EarlyStopping selects. A formal causal link (Stochastic Coupling → better test generalization via better checkpoint selection) requires per-epoch test evaluation, left as future work.

### 5.3 DPM-Solver++ vs Heun: Computational and Small Precision Advantage

Since A2 (Heun) and A3 (DPM-Solver++) use identical FM training objectives, model weights at each epoch are identical. Yet DPM-Solver++ at matched 4-step yields a small but statistically significant mAP improvement over Heun ($+0.006$ per-image mAP, Wilcoxon $p<10^{-3}$; Table 8), so the higher-order solver is slightly *better*, not worse. Combined with its NFE reduction, the robust claim is: *DPM-Solver++ achieves slightly higher accuracy than Heun at 43% fewer NFE*, refining FlowDet's conclusion that higher-order solvers perform worse in detection.

### 5.4 Top-$K$ Pruning: Solver-Dependent Effectiveness

Top-$K$ pruning effectiveness depends on NFE per step: for Heun (2 NFE/step), pruning affects 6/8 calls (1.09–1.12× speedup); for DPM-Solver++ (1 NFE/step), 3/4 calls (1.05–1.08×). DPM-Solver++ already achieves most speedup through NFE reduction, making Top-$K$ less impactful.

### 5.5 Theory Applicability and Limitations

The two-sided bound (Propositions 1–2) is tight in chromosome detection ($0.03\%$ gap): well-separated Voronoi cells (pairwise distances $\approx 20$ px vs $\sigma \sim 1$ px) make $P_{\text{err}} < 10^{-45}$ and $N=2$ mini-batch OT reduce to nearest-neighbor assignment. The low-dimensional regime ($d=4$, $K \approx 46$) yields $\Delta H/H \approx 0.69$ (Table 2), and the Gaussian noise source is approximately satisfied by our shifted Gaussian schedule.

The theory does not transfer to high-dimensional generation ($d \sim 10^5$, where $\Delta H/H \approx 0$ so OT collapse is negligible — consistent with OT-CFM's success), nor to densely overlapping targets where Assumptions 2 and 4 fail. For COCO ($K \sim 7$, $\Delta H/H \approx 0.55$), the theory predicts Stochastic Coupling would help but with smaller magnitude. The theory suggests RF + Stochastic Coupling would benefit detection tasks combining low $d$, high object density, and small training data — a profile including medical imaging, remote sensing, and other fine-grained dense detection tasks. We did not validate on COCO because its small $K$ reduces OT collapse severity; the appropriate validation dataset has high $K$ and low $d$, exactly the chromosome detection profile. The stability benefit is practically meaningful for clinical deployment: the 4.6× epoch stability improvement means checkpoint selection lands within 0.0013 of the trend (versus 0.006 for Random), reducing the risk of deploying a "false peak" checkpoint.

## 6. Conclusion

We introduced *KaryoFlow*, a diffusion-based detector for chromosome karyotyping that brings Rectified Flow into clinical cytogenetics. The RF training paradigm --- straight-line ODE paths replacing curved DDPM trajectories --- is the dominant source of accuracy gain, yielding $+0.082$ mAP over the Euler baseline on Dataset 2 and $+0.017$ mAP over DDPM on Dataset 1, and our best variant surpasses the DDPM-based DiffusionDet by $+0.060$ mAP while matching Cascade R-CNN; a solver$\times$step disentanglement attributes $94\%$ of this gain to the RF paradigm itself. Stochastic Coupling, grounded in our characterization of OT Diversity Collapse in the low-dimensional ($\mathbb{R}^4$) detection space, restores coupling diversity and stabilizes training: in low-data regimes it yields a large, highly significant mAP gain ($+0.034$, $p<10^{-120}$), while on larger data the gain shifts to a $4.6\times$ reduction of within-run epoch-level oscillation that makes checkpoint selection reliable. DPM-Solver++ with Top-$K$ pruning delivers four-step inference at 13.3--14.2 FPS with a small but statistically significant precision advantage over Heun, placing the detector in the interactive-screening latency band. Standard one-shot detectors remain 3--7$\times$ faster, so our method trades latency for accuracy and is positioned for interactive clinical screening rather than maximal throughput.

Beyond chromosome karyotyping, the OT Diversity Collapse phenomenon we characterize is not specific to chromosomes --- it arises whenever the prediction space is low-dimensional, the target density per image is high, and the training corpus is small. This profile recurs across medical imaging: cell detection in histopathology (many nuclei per tile, $d=4$ bounding boxes, small annotated cohorts), lesion detection in mammography and retinal imaging (small targets, limited positive cases), and microbiological colony counting. In each of these settings, deterministic OT coupling collapses toward $\log K$ and Stochastic Coupling offers the same dual benefit --- accuracy in low-data regimes, stability in general --- that we observed on chromosomes. The theory provides an a-priori diagnostic via Table 2: any task with $d \ll 100$ and $K \gg 10$ is a candidate, and the severity $\Delta H/H$ predicts whether Stochastic Coupling will help. Validation on at least one non-chromosome high-$K$ low-$d$ benchmark --- cell detection being the most natural next step --- would substantially strengthen the generality claim.

We acknowledge three limitations. First, the empirical validation is confined to chromosome data; validating on COCO or cell-detection benchmarks would test the generality of the OT collapse prediction. Second, Stochastic Coupling's mAP gain is dataset-dependent (large on Dataset 1, negligible on Dataset 2), so its accuracy contribution cannot be taken for granted on larger benchmarks --- though the $4.6\times$ stability benefit holds independently. Third, the theoretical analysis assumes well-separated targets; densely overlapping scenes would require extending the finite-$N$ analysis. Addressing these limitations is a natural direction for future work.

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

PHASE 3 REORGANIZATION (commit 8f67c88a, 2026-07-19):
- Main paper Appendix A: Proofs (Prop 1, 2, 3 sketches) — this draft §A
- Main paper Appendix B: "AdaLN-Zero and Shifted Schedule Ablations" —
  merges old §D (AdaLN-Zero) + old §E (Shifted Schedule) + old §F.2
  (Y Chromosome Analysis) into a single ~1-paragraph section in main paper.
- Old §B (Falsified Directions) → ARXIV COMPANION ONLY (removed from main)
- Old §C (Per-Seed Values) → ARXIV COMPANION ONLY (removed from main)
- Old §F.1, §F.3, §F.4, §F.5 → ARXIV COMPANION ONLY (removed from main)
- All inline main-text references updated: old "(Appendix D)" and "(Appendix E)"
  now point to "(Appendix B)"; old "(Appendix F)" for C-group points to
  "(Appendix B, arXiv companion)".
- This draft preserves the original §B–§F structure as the complete fact
  record; the destination annotations below reflect the main-paper mapping.
-->

### A. Proofs and ODE Solver Derivations

<!-- [MAIN PAPER: proof sketches only — Propositions 1, 2, 3 inline at §3.3]
    [ARXIV COMPANION: full proofs below, including finite-N analysis and
    Gaussian-mixture posterior details]
    Page budget: ~0.4 page in main paper (three compressed sketches). -->

**Correspondence to main text.** This appendix supports §3.3 (OT Diversity Collapse and Stochastic Coupling) and §3.2 (ODE Solvers). Proposition 1 (upper bound $\Delta H \le \log K$) and Proposition 2 (lower bound via Fano's inequality) are stated at §3.3 and characterize the *OT Diversity Collapse*; Proposition 3 (monotonicity of Stochastic Coupling in $\epsilon$) is stated at §3.3 and justifies the Stochastic Coupling design. §A.4–A.5 give the detailed derivations of the three ODE solvers (Euler, Heun, DPM-Solver++) used in §3.2. The main paper retains compressed proposition statements inline; this appendix restates them with additional detail, and full proofs (including finite-$N$ analysis and the Gaussian-mixture posterior) appear in the arXiv companion preprint.

#### A.1 Proof of Proposition 1 (OT Diversity Upper Bound)

**Setup**: Source $\nu = \mathcal{N}(0, \sigma^2 I_d)$, target $\mu = \frac{1}{K}\sum_{k=1}^K \delta_{\mathbf{b}_k}$. A coupling $\pi$ assigns noise samples $\{\mathbf{z}_i\}_{i=1}^N$ to target boxes $\{\mathbf{b}_{V_i}\}_{i=1}^N$. The flow state is $X_t = (1-t) \mathbf{b}_V + t \mathbf{z}$ where $\mathbf{z} \sim \nu$ and $V$ is the coupling assignment.

**Proof sketch.** Random coupling: $V \sim \operatorname{Uniform}(\{1,\ldots,K\})$ independently of $\mathbf{z}$, so $H_{\text{rand}}(V|X_t) \le H(V) = \log K$ (equality in the high-noise regime where the Gaussian-mixture posterior is approximately uniform). OT coupling ($N \to \infty$): the Lemma makes $V = \arg\min_k \lVert\mathbf{z} - \mathbf{b}_k\rVert$ (Voronoi assignment) a deterministic function of $\mathbf{z}$, so given $X_t$ one recovers $\mathbf{z} = (X_t - (1-t)\mathbf{b}_V)/t$ and hence $V$, giving $H_{\text{OT}}(V|X_t) = 0$. Combining: $\Delta H \le \log K$, empirically tight ($0.03\%$ error).

**Remark on finite-$N$.** In practice OT is solved on mini-batches of size $N$ (e.g., $N=2$ in our setting). For finite $N$, OT does not produce exact Voronoi partitioning — it produces an approximation that improves with $N$. The empirical validation ($\Delta H = 3.8415$ vs $\log K = 3.8427$, 0.03% error) confirms the $N \to \infty$ bound is an excellent approximation even for small $N$ in the chromosome detection setting, likely because $K \approx 46 \gg N$ and the GT boxes are well-separated in $\mathbb{R}^4$ relative to $\sigma$.

#### A.2 Proof of Proposition 2 (OT Diversity Lower Bound)

**Proof sketch.** Under OT, $V = \arg\min_k \lVert\mathbf{z} - \mathbf{b}_k\rVert$ (Lemma); given $X_t$, the model observes $\mathbf{z}$ with Gaussian noise $\sigma_t^2 I_d$ in the Voronoi cell of $\mathbf{b}_V$. Fano's inequality on the $K$-way nearest-neighbor classifier gives
$$H_{\text{OT}}(V|X_t) \;\le\; h(P_{\text{err}}) + P_{\text{err}} \log K,$$
where $P_{\text{err}}$ is the misclassification probability of the optimal nearest-neighbor rule. A union bound over the $\binom{K}{2}$ pairs of Voronoi cells (each pair separated by $d_{\min}$, with $\mathbf{z}$ subject to Gaussian noise $\sigma_t$) gives $P_{\text{err}} \le \binom{K}{2}\, \Phi(-d_{\min}/(2\sigma_t))$. Combining with $H_{\text{rand}}(V|X_t) \ge \log K$ (high-noise regime where the posterior is uniform) and $h \le \log 2$ yields the bound. As $\sigma_t/d_{\min} \to 0$, $P_{\text{err}} \to 0$ exponentially (Gaussian tail), so $\Delta H \to \log K$, matching the upper bound. For chromosome detection ($d_{\min} \approx 20$ px, $\sigma_t \sim 1$ px), $P_{\text{err}} < 10^{-45}$, so $\Delta H \ge 0.999\,\log K$, consistent with the observed $0.03\%$ gap.

#### A.3 Proof of Proposition 3 (Stochastic Coupling Monotonicity)

**Proposition 3** (Stochastic Coupling Monotonicity). Under uniform source marginal, $H_{\text{stoch}}(V \mid X_t; \epsilon)$ is monotonically non-decreasing in the Sinkhorn regularization $\epsilon \ge 0$.

**Proof sketch.** $T_\epsilon$ solves $\min_\pi \langle \pi, c \rangle - \epsilon H(\pi)$ s.t. uniform marginals (Cuturi, 2013). The optimal value $\mathcal{V}(\epsilon)$ is concave in $\epsilon$ (infimum of affine functions in $\epsilon$). By the envelope theorem $d\mathcal{V}/d\epsilon = -H(T_\epsilon)$, and concavity gives $dH(T_\epsilon)/d\epsilon \ge 0$. Under uniform marginal, $H_{\text{stoch}} = \tfrac{1}{K} H(T_\epsilon)$ (average row entropy), hence non-decreasing in $\epsilon$. Endpoints: $\epsilon \to 0$ gives $H \to 0$ (Hard OT), $\epsilon \to \infty$ gives $H \to \log K$ (Random).

#### A.4 Heun Solver Derivation (2nd-order predictor-corrector)

Heun's method provides 2nd-order ODE accuracy via a predictor–corrector:
- Predict: $\hat{\mathbf{x}}_{t-\Delta t} = \mathbf{x}_t + \Delta t \cdot \mathbf{v}_\theta(\mathbf{x}_t, t)$
- Correct: $\mathbf{x}_{t-\Delta t} = \mathbf{x}_t + \frac{\Delta t}{2} [\mathbf{v}_\theta(\mathbf{x}_t, t) + \mathbf{v}_\theta(\hat{\mathbf{x}}_{t-\Delta t}, t{-}\Delta t)]$

2 network forward evaluations (NFE) per step. At 4 steps this yields 8 NFE (7 in practice; the last step degrades to Euler).

#### A.5 DPM-Solver++ Derivation (2nd-order multistep, 1 NFE/step)

We adapt DPM-Solver++ (Lu et al., 2022) to the RF linear path. The model directly predicts $\mathbf{x}_0$ (the GT box), and the update uses polynomial interpolation of the $\mathbf{x}_0$ history in $t$ space (not the log-SNR $\lambda$ space of VP-SDE). The RF-ODE in data-prediction form is $d\mathbf{x}/dt = (\mathbf{x} - \mathbf{x}_0(t))/t$; integrating exactly over $[t_n, t_{n+1}]$ with linear $\mathbf{x}_0(t)$ interpolation yields

$$\mathbf{x}_{t_{n+1}} = \tfrac{t_{n+1}}{t_n}\,\mathbf{x}_{t_n} + \bigl(1 - \tfrac{t_{n+1}}{t_n}\bigr)\,\mathbf{x}_0^{(n)} + \varphi_1\,\mathbf{D}_1,$$
$$\varphi_1 = t_{n+1}\log\tfrac{t_n}{t_{n+1}} - t_n + t_{n+1},\quad \mathbf{D}_1 = \tfrac{\mathbf{x}_0^{(n)} - \mathbf{x}_0^{(n-1)}}{t_n - t_{n-1}},$$

where the first two terms are the exact solution for constant $\mathbf{x}_0$ and $\varphi_1 \mathbf{D}_1$ is the 2nd-order correction for linearly varying $\mathbf{x}_0(t)$. The singularity at $t{\to}0$ is handled by an $\epsilon$ cutoff ($t_{n+1} > 10^{-7}$); a 3rd-order variant adds a quadratic term $\varphi_2 \mathbf{D}_2$. Unlike VP-SDE DPM-Solver++, $\mathbf{x}_1$ (the initial noise) is a fixed sample and is *not* interpolated; its contribution is carried implicitly by $\mathbf{x}_{t_n}$. 1 NFE per step, 4 NFE for 4 steps (vs Heun's 7).

### B. Falsified Directions

<!-- [ARXIV COMPANION ONLY — removed from TMI main paper in Phase 3 reorganization]
    These negative results document internal research decisions but do not
    advance the paper's claims. Retain in this draft as fact record only.
    If a reviewer asks "did you try X?", cite the arXiv companion. -->

**Correspondence to main text.** This appendix justifies the method choices made in §3 (Method) and §4 (Experiments) by documenting the research directions we explored and experimentally falsified. Each falsified direction corresponds to an alternative design that we considered and rejected with empirical evidence: IO1–IO5 concern inference-time optimizations that failed to improve over the default pipeline (§3.2, §4.6); the flow-matching-detection and $N_{\text{cascade}}$ e2e directions concern architectural alternatives to RF + cascade heads that degraded mAP (§3.1). The negative results explain *why* our final design does not include these components, and are retained here as the complete fact record; the main paper mentions them only where directly relevant to a design decision.

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

<!-- [ARXIV COMPANION ONLY — removed from TMI main paper in Phase 3 reorganization]
    Per-seed tables are too granular for the 10-page main paper. The main paper
    reports mean±std aggregates (Tables 5-7); per-seed breakdowns move to arXiv
    for full reproducibility verification. ]

**Correspondence to main text.** This appendix supports the multi-seed tables in §4.2 (RF vs DDPM, Table 5) and §4.4 (Coupling Ablation, Table 9) by providing the per-seed numerical values underlying the aggregated mean±std figures. Each sub-table below corresponds to a specific main-text table: §C.1 underlies the Dataset 1 RF-vs-DDPM comparison cited in §4.2.2; §C.2 underlies the Dataset 1 coupling ablation cited in §4.4.1. The per-seed breakdowns allow independent verification that the cross-seed variance reported in the main text ($\pm 0.003$ for A3, $\pm 0.002$ for DDPM, etc.) is reproduced seed by seed, and that no individual seed is an outlier driving the aggregate.

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

<!-- [MERGED INTO MAIN PAPER APPENDIX B in Phase 3 reorganization]
     Combined with old §E (Shifted Schedule) and old §F.2 (Y Chromosome) into
     a single "AdaLN-Zero and Shifted Schedule Ablations" section (~1 paragraph)
     in the 10-page IEEE main paper. Full table preserved here as fact record.]

This appendix reports the standalone ablation of AdaLN-Zero referenced in Section 3.1.2 and the contribution discussion of the main text, confirming that its individual contribution to the +0.082 mAP gain is null within the RF framework (Table D.1). We conducted a separate ablation on Dataset 2 to verify AdaLN-Zero's individual contribution. Both experiments use identical configurations except for the time conditioning module (RF formulation, Heun solver 4-step, shifted schedule shift=3.0, random coupling, 150 epochs).

| Config | mAP | $\Delta$ mAP |
|--------|-----|--------------|
| RF + Heun (without AdaLN) | 0.856 | — |
| RF + Heun + AdaLN-Zero | 0.856 | +0.000 |

**Table D.1**: AdaLN-Zero ablation on Dataset 2.

AdaLN-Zero contributes *null* ($\Delta$mAP = 0.000) within the RF framework on this dataset. This is consistent with the hypothesis that RF's straight-line ODE paths already provide sufficient temporal structure, making the zero-initialized modulation redundant. We retain AdaLN-Zero as a standard conditioning mechanism (Dhariwal & Nichol, 2021) for consistency with the broader diffusion literature, but note that it does not contribute to the +0.082 mAP improvement claimed in Section 4.2.1. The entire +0.082 gap is attributable to the RF formulation itself (straight-line ODE paths); the shifted noise schedule contributes negligibly on its own, as shown in Appendix B.

### E. Shifted Noise Schedule Ablation

<!-- [MERGED INTO MAIN PAPER APPENDIX B in Phase 3 reorganization]
     Combined with old §D (AdaLN-Zero) and old §F.2 (Y Chromosome) into
     a single "AdaLN-Zero and Shifted Schedule Ablations" section (~1 paragraph)
     in the 10-page IEEE main paper. Full table preserved here as fact record.]

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

<!-- MIXED DESTINATIONS (Phase 3 reorganization):
  F.1 Chromosome Size Groups    -> [ARXIV COMPANION ONLY] (reference material)
  F.2 Y Chromosome Analysis     -> [MERGED INTO MAIN PAPER APPENDIX B] (~1 sentence)
  F.3 C-group Discrimination    -> [ARXIV COMPANION ONLY] (detailed per-class numbers)
  F.4 Complete Per-Class AP Table -> [ARXIV COMPANION ONLY] (24-row table too large for main)
  F.5 Localization Saturation   -> [ARXIV COMPANION ONLY] (1 sentence in §5 Discussion)
  Net main-paper cost: ~1 sentence in Appendix B + 0 tables; full content preserved here + arXiv. -->

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
| Y | 0.779 | 0.972 | 0.933 | 0.577 | 0.788 | — |

**Table F.1**: Complete per-class AP breakdown on the Dataset 2 validation set (A3 DPM-Solver++).

#### F.5 Localization Saturation and Downstream Potential

AP50 is near-saturated across all classes (>0.988, and 0.972 for the Y), so localization is near-saturated (>0.988) and residual errors concentrate in fine-grained classification. A downstream refinement stage operating on correctly localized crops — a banding-pattern classifier or morphology-aware re-scoring head — could in principle recover much of the remaining AP, since the upstream detector already supplies the right regions.
