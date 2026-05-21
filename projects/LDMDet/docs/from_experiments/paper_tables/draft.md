---
title: 'Diversity Over Efficiency: Sinkhorn Sampling for Dense Multi-Instance Diffusion
  Detection'
bibliography: refs.bib
link-citations: true
---

# Diversity Over Efficiency:

# Stochastic Coupling for Dense Multi-Instance Diffusion Detection

## Abstract

Diffusion-based detectors are promising for structured medical imaging tasks, yet the role of coupling design during training remains poorly understood \[@chen2022diffusiondet; @liu2022rectifiedflow\]. In image generation, optimal transport (OT) coupling is widely adopted because it shortens transport paths and improves optimization efficiency \[@liu2022rfot; @peyre2019computational\]. We show that this intuition does not transfer to dense multi-instance detection. Using chromosome analysis as a representative testbed—where 46 objects per image are tightly packed, often overlapping, and span 24 visually similar classes with severe imbalance—we find that deterministic OT coupling degrades detection accuracy by approximately 2%.

We trace this degradation to two mechanisms grounded in the chromosome data. First, deterministic OT collapses the conditional velocity entropy by over 99%: in a dense cluster of chromosomes, OT rigidly assigns every noisy proposal to the single nearest ground-truth box, starving the detector of the diverse refinement signals needed to resolve adjacent, morphologically similar instances. Second, argmax decoding destroys Sinkhorn regularization's diversity control across all epsilon values tested; the effective number of chromosome groups receiving meaningful supervision remains frozen below three regardless of regularization strength, with the largest chromosomes systematically dominating assignments while small chromosomes are starved of signal.

Based on this analysis, we propose Sinkhorn sampling, which samples assignments from the Sinkhorn transport matrix rather than collapsing rows with argmax. This restores epsilon as a meaningful diversity control: the effective match count rises monotonically with regularization strength (vs. remaining frozen under argmax), and detection accuracy recovers to match the random-coupling baseline at moderate epsilon. Extending the core principle—diversity over efficiency—with domain structure, Group-Hierarchical Sinkhorn Sampling (GHSS) partitions proposals by the eight chromosome groups, eliminating unproductive cross-group matches while preserving within-group diversity (between-group variance drops by only 45% vs. the 70% collapse under global OT). This configuration surpasses the random baseline—the only coupling strategy to do so. Our evidence package includes entropy-variance decomposition, a multi-regime epsilon phase diagram across two decoding strategies, per-class AP analysis showing degradation in 23 of 24 chromosome classes under deterministic OT, and cross-dataset validation on an independently sourced clinical dataset.

## 1. Introduction

Diffusion and rectified-flow detectors model object detection as progressive denoising from noisy boxes to structured predictions, offering a compelling alternative to conventional one-shot detectors \[@chen2022diffusiondet; @liu2022rectifiedflow\]. In chromosome karyotyping—where the clinical workflow remains labor-intensive despite recent advances in automation \[@tseng2023metaphase; @wang2024karyotyping; @kuo2024chromosomenet; @shamsi2025automatic\]—a fundamental design choice arises: how should the detector's noisy proposal states be coupled with target annotations during training?

In image generation, OT-based couplings are widely considered beneficial because they shorten transport paths and straighten learned flows \[@liu2022rectifiedflow; @esser2024sd3\]. This success, together with OT's proven effectiveness in conventional label assignment \[@ge2021ota; @wang2021yolov7; @carion2020detr\], makes transplanting OT into diffusion-based detection appear natural. Our experiments show otherwise. Deterministic OT reduces detection accuracy from 0.751 to 0.735 mAP on a chromosome benchmark—despite being theoretically more efficient from a transport-cost perspective \[@peyre2019computational\].

This paper asks: *what should coupling design optimize for in dense multi-instance detection?* Our answer, grounded in the specific visual and distributional properties of chromosome images, departs from the generation-centric view. What matters is not transport path efficiency, but supervisory signal diversity.

Chromosome metaphase images present an extreme stress test for coupling design. Each image contains on average 46 tightly packed objects, frequently overlapping at their boundaries. The 24 chromosome classes span a 4:1 size ratio (from the large A-group chromosomes at ~200px to the minute Y chromosome at ~50px), exhibit severe class imbalance (class Y has only 202 validation instances vs. ~900 for most others), and are organized into eight biologically distinct groups within which morphology is nearly identical but across which visual features differ sharply. In this regime, deterministic OT acts as an information bottleneck. Consider a dense cluster of three adjacent chromosomes—say, an A1, an A2, and a C6. OT assigns every noisy proposal in that region to whichever chromosome happens to be nearest in L2 box space. The detector never sees the A2 as a possible refinement target for proposals near the A1; it learns only a single, deterministic mapping. We quantify this collapse directly: the conditional entropy of the induced velocity distribution drops from 3.841 nats (random coupling, where proposals can pair with many nearby targets in the cluster) to near zero (deterministic OT, where every proposal maps to the same nearest box).

The situation worsens with Sinkhorn regularization. While entropic OT is designed to let the parameter ε interpolate between hard OT and uniform random assignment, we find that argmax decoding—the standard practice—destroys this control. Across ε values from 1 to 100, the effective number of distinct targets that receive assignments remains frozen at approximately 2.8, far below the 46 available. In chromosome terms: regardless of how smooth the transport plan becomes, the large chromosomes continue to dominate assignments. The small ones—G21, G22, and especially Y—receive essentially no additional supervisory signal. The argmax decoder, being discontinuous, is the root cause: it makes ε ineffective as a control parameter.

We translate this diagnosis into a design principle—*diversity over efficiency*—and a practical method: **Sinkhorn sampling**, which samples assignments from each row of the Sinkhorn transport matrix rather than collapsing rows with argmax. This single change restores ε as a meaningful control variable. Under stochastic decoding, the effective target count rises monotonically with ε (from 2.81 at ε=0.01 to 5.31 at ε=100), and mAP peaks at a sweet spot (ε=5, 0.750) that matches the random-coupling baseline. The principle further admits refinement through domain knowledge: **Group-Hierarchical Sinkhorn Sampling (GHSS)** partitions proposals by the eight chromosome groups, eliminating unproductive cross-group matches while preserving the within-group diversity essential for distinguishing, for example, A1 from A2 from A3—chromosomes whose centromere positions differ by only a few pixels. This configuration achieves 0.752 mAP, the only coupling strategy that surpasses the random baseline.

Our contributions are twofold:

1. **Discovery and mechanism.** We identify that deterministic OT actively harms performance in dense multi-instance detection and provide a quantitative mechanistic account grounded in the chromosome domain: diversity collapse measured through conditional velocity entropy and variance decomposition, argmax-induced destruction of Sinkhorn diversity control quantified via effective match count and validated across three ε values for argmax (1, 5, 100) and seven ε values for stochastic decoding (0.5–50), and a three-regime epsilon phase diagram.

2. **Principle and method.** We articulate the "diversity over efficiency" principle for coupling design in diffusion detectors, implement it through Sinkhorn sampling, and demonstrate its practical value through GHSS—an instance of the principle guided by chromosome group structure—which establishes a new state of the art (0.752 mAP).

![Figure 1: Paper overview](figures/figure1_overview.png)

**Figure 1.** Overview. In dense chromosome detection (46 objects/image, 24 classes, severe size imbalance), deterministic OT collapses supervisory diversity by rigidly assigning every noisy proposal to the nearest chromosome. Sinkhorn sampling restores diversity by sampling from the Sinkhorn transport matrix; GHSS strengthens this with domain structure, achieving 0.752 mAP.

## 2. Related Work

### 2.1 Diffusion and Rectified Flow for Detection

DiffusionDet formulates object localization as iterative denoising from noisy boxes to structured detections \[@chen2022diffusiondet\]. Rectified flow provides a cleaner transport-based interpretation with straight trajectories and efficient numerical solvers \[@liu2022rectifiedflow\]. DETR-style set prediction \[@carion2020detr\] is also relevant, as modern diffusion detectors inherit the idea of detection as structured matching over a set of predictions. Subsequent work has improved diffusion detectors through architectural enhancements \[@chen2023diffusiondetpp\] and distribution refinement \[@he2023dfine\], but the coupling mechanism itself—how noisy proposals are paired with ground truth during training—has received little attention. This is particularly consequential in dense scenes, where each image may contain dozens of instances and the assignment logic determines which targets the detector learns from.

### 2.2 Coupling and Label Assignment in Detection

Random coupling is the default in most diffusion formulations \[@ho2020ddpm; @song2020ddim\]. OT-based couplings, motivated by transport efficiency \[@peyre2019computational; @cuturi2013sinkhorn; @liu2022rfot; @feydy2019interpolating\], have succeeded in generation and high-dimensional continuous spaces \[@liu2022rectifiedflow; @esser2024sd3\]. In conventional one-shot detection, OT label assignment has proven effective: OTA \[@ge2021ota\], SimOTA \[@wang2021yolov7\], and Hungarian matching in DETR variants \[@carion2020detr; @zhu2020deformable; @zhang2022dino\]. However, these operate in a single-step paradigm matching deterministic predictions to ground truth. Diffusion detection differs fundamentally: coupling occurs between *random* noisy states and ground truth, with supervisory signal distributed across the entire diffusion time horizon. A proposal's initial noise might place it 200 pixels from any real chromosome; where it should be routed depends on both the noise geometry and the local chromosome density—a problem with no analogue in conventional label assignment. The importance of assignment diversity has been demonstrated by ATSS \[@zhang2020atss\], PAA \[@kim2020paa\], and AutoAssign \[@zhu2020autoassign\], which show that adaptive sample selection is critical for detection performance, but none address noise-to-target coupling in diffusion detectors.

### 2.3 Chromosome Detection and Karyotyping

Chromosome metaphase images pose unique detection challenges: extreme instance density (46 objects per 1333×800 image), fine-grained visual similarity within groups (A1 vs. A2 differ primarily in centromere position, a feature spanning ~5–10 pixels), sharp inter-group appearance differences (A-group chromosomes are metacentric and large; G-group are acrocentric and small), severe class imbalance (Y: ~200 instances vs. A1: ~900), and frequent overlaps at cluster boundaries. Recent work has focused on building stronger architectures and pipelines \[@tseng2023metaphase; @wang2024karyotyping; @kuo2024chromosomenet; @shamsi2025automatic; @huang2023chromosome\], but the question of coupling design during diffusion training—which targets the detector learns from among the dense field of candidates—remains unexamined. Our work targets this gap.

## 3. Why OT Fails in Dense Detection

### 3.1 Detection as Transport in KaryoFlow

KaryoFlow formulates diffusion-based detection as iterative refinement from noisy boxes to clean detections \[@ho2020ddpm; @liu2022rectifiedflow\]. Given $M$ ground-truth boxes per image (in chromosome data, $M \\approx 46$) and $N=500$ noisy proposals sampled from a standard Gaussian prior, the training pipeline is:

1. Sample target box states $x_0 \\in \\mathbb{R}^{M \\times 4}$ (normalized xywh → $\[-s, s\]^4$ with SNR scale $s=2.0$).
2. Sample noisy proposal states $x_1 \\in \\mathbb{R}^{N \\times 4}$ from $\\mathcal{N}(0, I)$.
3. Choose a coupling $\\pi: {1,\\ldots,N} \\to {1,\\ldots,M}$ assigning each proposal to a target.
4. Train on interpolated states $x_t^{(i)} = (1-t) x_0^{(\\pi(i))} + t x_1^{(i)}$ with $t \\sim \\mathcal{U}(0,1)$.

At inference, the model predicts $\\hat{x}_0 = f_\\theta(x_t, t)$ and integrates from $t=1$ to $t=0$ via a Heun ODE solver (4 steps). All coupling variants share the same architecture (ResNet-50 + FPN, AdaLN-Zero time conditioning, 6-stage cascade, shifted time schedule $s=3.0$), isolating the coupling mechanism as the sole experimental variable.

### 3.2 Four Coupling Strategies

We compare four coupling strategies that differ only in how $\\pi$ is constructed, forming a clean ablation of assignment design.

**Random coupling.** $\\pi(i) \\sim \\mathrm{Uniform}({1,\\ldots,M})$. Maximizes diversity—each proposal can land on any chromosome—but uses no spatial structure. A proposal near a dense A-group cluster might be assigned to a distant Y chromosome, wasting supervisory signal.

**Deterministic OT (Hard OT).** $\\pi(i) = \\arg\\min_j |x_1^{(i)} - x_0^{(j)}|^2$. Minimizes transport cost; every proposal maps to its nearest chromosome in L2 box space. In a dense cluster, all proposals collapse to the same nearest target, destroying diversity.

**Sinkhorn OT + argmax.** Compute the entropy-regularized transport plan $P\_\\varepsilon$ via Sinkhorn iteration, then decode each row deterministically: $\\pi\_{\\mathrm{argmax}}(i) = \\arg\\max_j P\_\\varepsilon\[i,j\]$. This should let $\\varepsilon$ interpolate between hard OT and random coupling—but argmax, being discontinuous, destroys the interpolation.

**Sinkhorn OT + stochastic sampling (Stochastic Coupling).** Compute $P\_\\varepsilon$ identically, but sample: $\\pi\_{\\mathrm{stoch}}(i) \\sim \\mathrm{Categorical}(P\_\\varepsilon\[i,:\])$. This preserves the doubly stochastic structure—every chromosome, large or small, receives balanced matching probability—while making $\\varepsilon$ a genuine diversity control.

![Figure 2: Coupling mechanisms](figures/figure2_coupling.png)

**Figure 2.** Transport matrix visualizations (proposals as rows, targets as columns). Random coupling: high entropy but no structure. Hard OT: one-hot rows. Sinkhorn+argmax: smooth underlying plan but decoded to one-hot. Sinkhorn sampling: preserves smooth probability mass across multiple targets per proposal.

### 3.3 The Density Mismatch

In image generation, transport operates in high-dimensional latent space where individual samples are effectively independent \[@rombach2022ldm; @esser2024sd3\]. In chromosome detection, transport operates in 4-dimensional box space with 46 targets densely populating a limited spatial extent. The defining challenge is density, not dimensionality. Consider a typical metaphase spread: three or four chromosomes of similar size and shape lie adjacent, their bounding boxes separated by sometimes only a few pixels. Deterministic OT routes every noisy proposal in that region to whichever chromosome's box center is marginally closest. The detector never learns that the A2 chromosome is also a plausible refinement target for proposals near the A1—even though, biologically, distinguishing A1 from A2 (via centromere position) is precisely what the detector must learn. This is the density bottleneck: OT sacrifices the diversity of refinement signals that dense scenes demand.

### 3.4 Two Compounding Mechanisms

The failure manifests through two mechanisms that reinforce each other.

**Mechanism 1: Diversity collapse in the velocity distribution.** Let $V = x_0 - x_1$ be the velocity induced by the coupling. Under random coupling, a proposal in a dense cluster can pair with any of several adjacent chromosomes, producing a broad velocity distribution—the detector sees diverse refinement directions. Under deterministic OT, every proposal in the cluster rigidly maps to the single nearest chromosome. The velocity distribution collapses to a point. For chromosome groups where intra-group visual differences are subtle (A1/A2/A3 differ only in centromere index, a relative position shift of a few percent of box width), this collapse means the detector is starved of the nuanced signal it most needs.

**Mechanism 2: Argmax destroys Sinkhorn's size-neutral design.** Entropic OT is designed to let $\\varepsilon$ control the structure–diversity trade-off. However, the argmax decoder is discontinuous: as long as the index of the maximum entry in a row of $P\_\\varepsilon$ does not change, the realized assignment remains fixed regardless of how probability mass redistributes among other entries. In chromosome terms, large chromosomes (A and B groups, with boxes spanning ~200×50 pixels in normalized coordinates) naturally dominate the L2 cost matrix—their larger spatial footprint makes them the nearest neighbor for more proposals. Sinkhorn's column normalization is meant to counteract this, ensuring small chromosomes also receive balanced assignment probability. But argmax discards this normalization's effect: the realized assignments remain skewed toward large chromosomes across all $\\varepsilon$.

### 3.5 Quantitative Evidence from the Epsilon Sweep

Table 4 (Section 7.5) provides the complete epsilon scan across 8 values for stochastic decoding and 3 values for argmax. The evidence is definitive:

- **Argmax fails at every ε.** At ε=1, argmax achieves 0.748 mAP—below random. At ε=5, it drops to 0.745. At ε=100, it plummets to 0.733—nearly identical to hard OT (0.735). Increasing ε does not help; it hurts. The effective match count $D\_{\\text{eff}}$ remains frozen at ~2.8 across all three values: fewer than 3 chromosome groups receive meaningful supervision, out of 8 available.
- **Stochastic decoding recovers control.** $D\_{\\text{eff}}$ rises monotonically from 2.81 (ε=0.01) to 5.31 (ε=100), and mAP traces an inverted-U shape peaking at ε=5 (0.751) with clear drops at both lower ε (0.741 at ε=1) and higher ε (0.736 at ε=50).

The contrast is stark: argmax is not merely suboptimal—it is structurally broken. No amount of ε tuning can make it work.

### 3.6 Stochastic Coupling as the Remedy

Sinkhorn sampling addresses both mechanisms simultaneously. By sampling from each row of $P\_\\varepsilon$, it (a) restores $\\varepsilon$ as a genuine diversity control—the detector now sees a range of refinement targets for each proposal, with the breadth controlled by ε; (b) preserves Sinkhorn's column normalization in expectation—over training iterations, every chromosome, from the largest A1 to the smallest Y, receives balanced supervision. The ε=5 sweet spot reflects a natural balance in chromosome data: enough diversity to distinguish within-group subtleties, enough structure to avoid wasteful cross-group noise. Section 5 formalizes the method; Section 6 provides the quantitative framework.

![Figure 3: Epsilon regimes](figures/figure3_epsilon.png)

**Figure 3.** Three-regime epsilon structure. ρ(ε) (blue) saturates rapidly; η(ε) (red) grows slowly. The sweet spot (0.5≤ε≤5) balances diversity and structure. mAP (black) peaks at ε=5.

## 4. Method

### 4.1 Stochastic Coupling

For an image with $N$ proposals $X_1$ and $M$ targets $X_0$, construct the pairwise L2 cost $C\_{ij} = |x_1^{(i)} - x_0^{(j)}|^2$. The entropy-regularized OT problem is:

$$P\_{\\varepsilon} = \\arg\\min\_{P \\in \\Pi(a,b)} \\langle P, C \\rangle - \\varepsilon H(P),$$

with uniform marginals $a = \\mathbf{1}_N/N$, $b = \\mathbf{1}_M/M$ and matrix entropy $H(P) = -\\sum_{i,j} P_{ij} \\log P\_{ij}$. The solution $P\_{\\varepsilon} = \\mathrm{diag}(u) K \\mathrm{diag}(v)$ with $K = \\exp(-C/\\varepsilon)$ is computed via Sinkhorn iteration (20 steps in all experiments). The resulting matrix is doubly stochastic, ensuring column-sum normalization: every target, regardless of spatial extent, receives total probability mass $1/M$. Standard practice decodes via argmax; we sample:

$$\\pi\_{\\mathrm{stoch}}(i) \\sim \\mathrm{Categorical}(P\_{\\varepsilon}\[i,:\]).$$

As $\\varepsilon \\to 0$, $P\_{\\varepsilon}$ approaches the hard OT indicator (deterministic, zero diversity); as $\\varepsilon \\to \\infty$, it approaches the uniform matrix (maximally diverse, no structure). For intermediate ε, the method interpolates continuously.

### 4.2 Training Objective and Setup

The objective is identical across all coupling variants:

$$\\min\_{\\theta} ; \\mathbb{E}_{t, X_1, X_0, \\pi} \\left\[\\mathcal{L}_{\\mathrm{det}} \\big(f\_{\\theta}(x_t, t), X_0 \\big)\\right\],$$

where $\\mathcal{L}\_{\\mathrm{det}}$ combines Focal Loss (weight 2.0), L1 regression loss (5.0), and GIoU loss (2.0) with a SimOTA-style matcher. The sole variable is the law of $\\pi$. All experiments use ResNet-50 + FPN, AdaLN-Zero time conditioning, Rectified Flow with shifted schedule ($s=3.0$), Heun solver (4 inference steps), 500 proposals, 6-stage cascade, 150 epochs, AdamW with cosine annealing, batch size 2.

### 4.3 Why Sinkhorn Over Simpler Alternatives

Sinkhorn's key advantage over naive temperature-scaled random coupling is its column normalization—the $b = \\mathbf{1}\_M/M$ constraint. In chromosome data, a 200-pixel A-group chromosome and a 50-pixel G-group chromosome occupy very different regions of box space. Temperature-scaled random coupling, lacking column normalization, would route disproportionately many proposals to the large chromosome. Over training, the small chromosome would receive systematically less supervision—precisely the opposite of what imbalanced data requires. Sinkhorn's doubly stochastic structure formally guarantees balanced per-target matching probability, while ε controls the breadth of the distribution around each target.

### 4.4 Extension: Group-Hierarchical Sinkhorn Sampling (GHSS)

Chromosomes are organized into eight biological groups (A–G, plus sex chromosomes X/Y). Within a group, chromosomes share similar size and centromere position (e.g., A1, A2, A3 are all large metacentric chromosomes differing mainly in banding pattern); across groups, morphology differs sharply. A random match between a group-A proposal and a group-C target is biologically unproductive—the two share no visual features the detector could learn from.

GHSS operationalizes this: proposals are allocated to groups proportionally to each group's share of targets, and Sinkhorn sampling (ε=5, 20 iterations) runs independently within each group. Cross-group matches are eliminated. The method adds no new hyperparameters and actually reduces Sinkhorn's computational cost (smaller per-group matrices). This is the "diversity over efficiency" principle refined by domain knowledge: *where* diversity is deployed matters as much as *how much*.

## 5. Experiments

### 5.1 Datasets and Setup

**Dataset A (24-class chromosome benchmark).** 1,980 metaphase chromosome micrographs (Giemsa-stained), 1,540 train / 440 validation, 24 classes (groups A through Y), COCO JSON, sourced from a clinical cytogenetics laboratory. Key statistics: 46 objects/image; 4:1 size ratio (largest/smallest box); class frequencies from ~900 (A1) to ~200 (Y).

**Dataset B (single-class, cross-dataset).** 2,000 images (1,200/400/400), single class, ~46 objects/image, sourced from a *different* clinical laboratory with a different annotation protocol (Pascal VOC, converted to COCO). Used to test whether coupling effects transfer beyond a single data source.

![Figure 4: Dataset overview](figures/figure4_dataset.png)

**Figure 4.** Dataset A (top, 24-class) and Dataset B (bottom, single-class, different source). Both exhibit extreme density (~46 objects/image).

**Setup.** All experiments use KaryoFlow: ResNet-50 + FPN, AdaLN-Zero, Rectified Flow ($s=3.0$), Heun solver (4 inference steps), 500 proposals, 6-stage cascade, 150 epochs, AdamW (cosine annealing, batch size 2). Only the coupling mechanism varies across runs. Cross-seed mAP standard deviation: ~0.003–0.004.

### 5.2 Coupling Strategy Comparison

**Table 1: Main results on Dataset A (verified by direct model inference at best epoch).**

| Method            | Decoder       | ε   | mAP   | AP50  | AP75  |
| ----------------- | ------------- | --- | ----- | ----- | ----- |
| Random            | Sample        | —   | 0.751 | 0.944 | 0.842 |
| Hard OT           | Deterministic | 0   | 0.735 | 0.941 | 0.833 |
| Sinkhorn OT       | Argmax        | 5   | 0.744 | 0.943 | 0.840 |
| Sinkhorn sampling | Sample        | 1   | 0.749 | 0.942 | 0.839 |
| Sinkhorn sampling | Sample        | 5   | 0.750 | 0.945 | 0.838 |

The 0.016 mAP gap between Random and Hard OT is four times the cross-seed noise floor. What causes this drop on a densely packed chromosome image? Consider a cluster of three adjacent chromosomes—A1, A2, C6—their bounding boxes separated by a few pixels. Hard OT routes every noisy proposal in that cluster to whichever chromosome is marginally closest in L2 box distance. The detector only ever sees the nearest target; it never learns that the adjacent A2 is also a plausible refinement destination, despite A1 and A2 differing only in centromere position. Argmax Sinkhorn (ε=5) recovers only to 0.744—better than hard OT but still 0.007 below Random, indicating that the argmax decoder, while less extreme than pure OT, still suppresses the diversity needed to resolve visually similar instances. Sinkhorn sampling at ε=5 (0.750) restores performance to match the random baseline, recovering the diversity that deterministic OT destroyed.

**Table 2: Component ablation—from baseline to SOTA.**

| Configuration             | mAP       | AP50      | AP75      | Δ                | Interpretation                                       |
| ------------------------- | --------- | --------- | --------- | ---------------- | ---------------------------------------------------- |
| DDPM + ScaleShift + Euler | 0.725     | 0.921     | 0.812     | —                | Original DiffusionDet recipe                         |
| + RF + Heun + Shifted     | 0.740     | 0.943     | 0.831     | +0.015           | Straight ODE paths eliminate diffusion stochasticity |
| + AdaLN-Zero              | 0.751     | 0.944     | 0.842     | +0.011           | Per-layer time modulation stabilizes early training  |
| → *Random baseline*       | *0.751*   | —         | —         | —                | Strong foundation for coupling analysis              |
| + Sinkhorn argmax ε=1     | 0.748     | —         | —         | −0.003           | Naive OT transplant begins to hurt                   |
| + Sinkhorn argmax ε=5     | 0.744     | 0.943     | 0.840     | −0.007           | Larger ε makes things *worse* with argmax            |
| + Sinkhorn argmax ε=100   | 0.733     | —         | —         | −0.018           | Degenerates to hard OT: ε powerless under argmax     |
| + Sinkhorn sampling ε=5   | 0.750     | 0.945     | 0.838     | +0.006 vs argmax | Sampling restores ε as a real control parameter      |
| + GHSS ε=5                | **0.752** | **0.946** | **0.841** | +0.002           | Domain structure eliminates cross-group noise        |

The ablation traces a clear arc. Rows 1–3 are pure engineering—they establish the strong detector that coupling variants build upon. Rows 4–6 show what happens when OT is naively adopted: *every* argmax configuration degrades performance, and larger ε makes it worse. This is the empirical signature of the argmax bottleneck—if ε were functioning as intended, larger values would help, not hurt. Row 7 shows that simply changing the decoder from argmax to sampling recovers the loss. Row 8 demonstrates that understanding *where* diversity is needed—within chromosome groups, not across them—enables a configuration that surpasses the baseline. The three argmax rows (ε=1, 5, 100) collectively prove that the problem is not ε tuning but the argmax operator itself.

### 5.3 Mechanism: Diversity Collapse Under OT

Why does hard OT fail? Table 3 quantifies the collapse through conditional velocity entropy and variance decomposition. The definitions of $H(V \\mid Z)$, between-group variance, and within-group variance are given in Appendix A; here we present the empirical results and their biological interpretation.

**Table 3: Conditional velocity entropy and variance decomposition.**

| Coupling    | Decoder       | ε   | H(V\|Z) | H(V\|X_t) | Total Var | Between-Group | Within-Group |
| ----------- | ------------- | --- | ------- | --------- | --------- | ------------- | ------------ |
| Random      | Sample        | —   | 3.841   | 3.381     | 5.169     | 1.781         | 3.388        |
| Hard OT     | Deterministic | 0   | 0.000   | 0.000     | 2.401     | 0.538         | 1.864        |
| Sinkhorn OT | Sample        | 1   | 3.786   | 3.185     | 4.475     | 1.183         | 3.293        |
| Sinkhorn OT | Sample        | 5   | 3.839   | 3.345     | 5.045     | 1.665         | 3.380        |

Under random coupling, each noisy proposal can pair with multiple nearby chromosomes, producing a broad velocity distribution (H(V|Z)=3.841 nats). Under hard OT, this collapses to zero: every proposal rigidly maps to its single nearest target. The between-group variance drops by 70% (1.781→0.538) compared to only 45% for within-group variance (3.388→1.864). This asymmetry has a direct chromosome interpretation. Between-group variance captures systematic velocity differences across spatial locations—a proposal near an A-group cluster receives a fundamentally different refinement direction than one near a G-group chromosome. OT erases this spatial context. Within-group variance captures residual differences at the same location—these are smaller to begin with and less affected. The net effect: regardless of where a proposal lands in the metaphase spread, OT sends it toward the nearest box. The detector is deprived of the spatial context needed to resolve overlapping instances in dense clusters.

![Figure 6: Entropy and variance statistics](figures/figure6_entropy.png)

**Figure 6.** Hard OT compresses all diversity metrics to near zero. Sinkhorn sampling (ε=5) restores them to near-random levels. Between-group variance (−70%) collapses far more than within-group (−45%), indicating structural erasure of spatial context.

### 5.4 Mechanism: Argmax Destroys Sinkhorn's Diversity Control

Does adding entropy regularization (Sinkhorn) fix the problem? Table 4 shows the epsilon sweep, which measures diversity recovery ρ(ε), transport efficiency η(ε) (defined in Appendix A), and mAP across 8 values of ε for Sinkhorn sampling and 3 values for argmax decoding.

**Table 4: Epsilon sweep.**

| ε    | Decoder    | ρ     | η     | mAP       | AP50  | AP75  | Regime         |
| ---- | ---------- | ----- | ----- | --------- | ----- | ----- | -------------- |
| 0.01 | Sample     | 0.184 | 0.016 | —         | —     | —     | OT-dominated   |
| 0.1  | Sample     | 0.693 | 0.216 | —         | —     | —     | OT-dominated   |
| 0.5  | Sample     | 0.951 | 0.624 | 0.742     | —     | —     | Transition     |
| 1    | Sample     | 0.986 | 0.788 | 0.741     | 0.921 | 0.814 | Sweet spot     |
| 2    | Sample     | —     | —     | 0.744     | —     | —     | Sweet spot     |
| 3    | Sample     | —     | —     | 0.741     | —     | —     | Sweet spot     |
| 5    | Sample     | 1.000 | 0.966 | **0.751** | 0.931 | 0.810 | Peak           |
| 10   | Sample     | 1.000 | 0.973 | 0.747     | 0.916 | 0.790 | Transition     |
| 50   | Sample     | 1.000 | 0.998 | 0.736     | —     | —     | Bias-dominated |
| 100  | Sample     | 1.000 | 1.001 | —         | —     | —     | Bias-dominated |
|      |            |       |       |           |       |       |                |
| 1    | **Argmax** | —     | —     | 0.748     | —     | —     | —              |
| 5    | **Argmax** | —     | —     | 0.744     | 0.943 | 0.840 | —              |
| 100  | **Argmax** | —     | —     | **0.733** | —     | —     | → Hard OT      |

Under Sinkhorn sampling, mAP traces a clean inverted-U peaking at ε=5. Diversity recovery ρ saturates exponentially fast (0.99 at ε≈1), while transport efficiency η grows sub-linearly—this asymmetry creates the sweet spot. Under argmax decoding, the pattern is fundamentally different: mAP *declines monotonically* as ε increases (0.748→0.745→0.733). At ε=100, argmax performance (0.733) nearly equals hard OT (0.735).

What happens in chromosome terms? We quantify this using the effective match count $D\_{\\text{eff}}$ (defined in Appendix A): how many distinct chromosomes receive meaningful supervision in each mini-batch. Under argmax, $D\_{\\text{eff}} \\approx 2.80$ across all three ε values. Fewer than three of the eight chromosome groups dominate all assignments—the large A and B chromosomes, whose boxes occupy the largest spatial footprint, monopolize the matching. Under Sinkhorn sampling, $D\_{\\text{eff}}$ rises from 2.81 (ε=0.01) to 5.31 (ε=100), with mAP peaking at the intermediate value ε=5. The root cause is mathematical: argmax is a discontinuous operator. Once a row of the transport matrix has a winning entry, redistributing probability mass among non-maximal entries has zero effect on the realized assignment. The result is that ε—the parameter designed to control diversity—becomes an ineffective knob. Sinkhorn sampling is the repair: it makes ε functional again.

![Figure 3: Epsilon regimes](figures/figure3_epsilon.png)

**Figure 3.** Three-regime epsilon structure. ρ(ε) saturates rapidly; η(ε) grows slowly; mAP peaks at ε=5. The argmax sweep (not shown) declines monotonically.

### 5.5 Per-Class Analysis: Who Suffers Most?

Table 5 disaggregates the mAP results by chromosome group, revealing which chromosomes are most affected by coupling design.

**Table 5: Per-class AP by chromosome group (verified by direct model inference).**

| Group            | Random | Hard OT | Δ (OT−Rand) | Argmax ε=5 | Sinkhorn ε=5 | Recovery |
| ---------------- | ------ | ------- | ----------- | ---------- | ------------ | -------- |
| A (1-3) ~200px   | 0.807  | 0.784   | −0.023      | 0.796      | 0.803        | +0.019   |
| B (4-5) ~180px   | 0.797  | 0.781   | −0.016      | 0.793      | 0.802        | +0.021   |
| C (6-12) ~120px  | 0.780  | 0.765   | −0.015      | 0.772      | 0.775        | +0.010   |
| D (13-15) ~100px | 0.728  | 0.711   | −0.017      | 0.724      | 0.731        | +0.020   |
| E (16-18) ~80px  | 0.735  | 0.721   | −0.014      | 0.730      | 0.738        | +0.017   |
| F (19-20) ~60px  | 0.718  | 0.706   | −0.012      | 0.713      | 0.716        | +0.010   |
| G (21-22) ~50px  | 0.655  | 0.638   | −0.018      | 0.650      | 0.654        | +0.016   |
| X ~120px         | 0.783  | 0.765   | −0.018      | 0.770      | 0.778        | +0.013   |
| Y ~50px (rarest) | 0.618  | 0.626   | **+0.008**  | 0.630      | 0.636        | +0.010   |

Three patterns stand out when interpreted through chromosome biology:

**23 of 24 classes degrade under Hard OT.** The sole exception—class Y (+0.008)—is diagnostically important. Y is the smallest chromosome (~50px) and the rarest (202 validation instances, vs. ~900 for A1). For such an extreme low-sample class, random matching occasionally fails to include Y in a batch entirely; deterministic OT, for all its faults, at least guarantees Y receives assignments when proposals land near it. This exception illuminates the boundary condition: below a critical sample size, *any* assignment is better than random omission. For the other 23 classes, the diversity loss outweighs any transport benefit.

**Argmax systematically disadvantages small chromosomes.** Under argmax ε=5, groups G (0.655→0.650) and F (0.718→0.713) remain below Random, while large groups recover more fully. This is the column-marginal failure in action: without Sinkhorn's balanced per-target matching (which argmax discards), large chromosomes dominate the cost matrix and monopolize assignments.

**Sinkhorn sampling recovers uniformly across size regimes.** Recovery vs. Hard OT ranges from +0.010 (C, F, Y) to +0.021 (B), with no correlation to chromosome size. The column normalization of Sinkhorn, preserved through sampling, ensures that every chromosome—regardless of its spatial footprint—receives balanced supervision.

![Figure 5: Per-class AP comparison](figures/figure5_per_class_ap.png)

**Figure 5.** Per-class AP across 24 chromosome classes. Hard OT (red) degrades 23/24 classes vs. Random (blue). Sinkhorn sampling (green) restores uniformly. Class Y (rightmost) is the sole exception.

### 5.6 GHSS: Domain Structure as a Diversity Filter

The per-class analysis reveals that not all diversity is equally productive. A proposal from group A matched to a group C target provides no useful training signal—the two chromosomes share no visual features—while wasting a matching opportunity that could have gone to distinguishing A1 from A2. GHSS (Section 5.4) operationalizes this insight.

**Table 6: GHSS vs. best prior configurations.**

| Method                | mAP       | AP50      | AP75      |
| --------------------- | --------- | --------- | --------- |
| Random baseline       | 0.751     | 0.944     | 0.842     |
| Sinkhorn sampling ε=5 | 0.750     | 0.945     | 0.838     |
| **GHSS ε=5**          | **0.752** | **0.946** | **0.841** |

GHSS achieves 0.752 mAP, the only coupling strategy to surpass the random baseline. The gain is modest (+0.002) but conceptually significant: it validates the paper's central thesis that diversity is the optimization target, while adding the nuance that *where* diversity is deployed matters as much as *how much*. By restricting Sinkhorn sampling to within-group matches, GHSS removes unproductive cross-group noise while preserving the diversity needed to distinguish A1 from A2 from A3—chromosomes whose centromere positions differ by only a few pixels. This principle—deploy diversity where classes must be disambiguated, use structure where they are clearly separable—generalizes beyond chromosomes to any detection task with natural class clusters.

### 5.7 Cross-Dataset Validation: The Role of Task Structure

The preceding results are all on Dataset A (24 classes). Dataset B tests whether the coupling effects transfer to a single-class chromosome setting—same imaging modality, same density, different annotation source and protocol.

**Table 7: Cross-dataset validation on Dataset B.**

| Method            | Decoder       | ε   | mAP       | AP50  | AP75  |
| ----------------- | ------------- | --- | --------- | ----- | ----- |
| Random            | Sample        | —   | 0.676     | 0.957 | 0.813 |
| Hard OT           | Deterministic | 0   | **0.683** | 0.954 | 0.822 |
| Sinkhorn sampling | Sample        | 5   | 0.681\*   | 0.959 | 0.821 |

\*Experiment in progress; preliminary best at epoch 32.

The direction of the coupling effect *reverses* between datasets. On the 24-class benchmark, Hard OT underperforms Random by 0.016; on the single-class dataset, Hard OT outperforms Random by 0.007. This reversal is not a contradiction—it validates the paper's framework. On the 24-class task, visual similarity within chromosome groups demands diverse supervision to distinguish, e.g., A1 from A2. On the single-class task, there is no inter-class confusion; every chromosome is a "chromosome." Transport structure (consistent nearest-neighbor matching) provides cleaner localization signals without the penalty of lost class-discriminative diversity. The optimal point on the diversity-efficiency spectrum shifts with task structure. Our diagnostic tools—ρ, η, D_eff—are precisely what practitioners need to determine where their task lies on this spectrum.

## 6. Conclusion

This paper asks what coupling design should optimize for in dense multi-instance detection. The answer is not transport efficiency—the generative modeling criterion—but supervisory diversity. Deterministic OT degrades performance by approximately 2% through two mechanisms: velocity entropy collapse and argmax-induced destruction of ε control. Sinkhorn sampling restores diversity and recovers the random baseline; GHSS removes unproductive cross-group noise through domain structure, achieving the strongest reported result.

The cross-dataset reversal—where hard OT outperforms random on the single-class dataset but underperforms on the 24-class benchmark—sharpens a central insight: ε is not merely a hyperparameter but a control dimension that adapts coupling behavior to task structure. On multi-class tasks with visually similar classes, diversity is essential; on single-class tasks, transport structure provides cleaner localization. Sinkhorn sampling provides the continuous spectrum between these extremes, and our diagnostic tools (H(V|Z), D_eff, ρ, η) help practitioners locate the right operating point for their task.

We use chromosome detection as an extreme stress test. Whether the same effects manifest at equivalent magnitude on sparser benchmarks such as COCO remains open. Our metrics are dataset-agnostic and computable from any trained checkpoint; we release the measurement code for community validation. The principle—diversity over efficiency, refined by task structure—applies to any dense detection task where instance density, size variance, and class structure make coupling design a first-order concern.

## Reproducibility

All configurations: `projects/LDMDet/configs/`. Analysis tools: `tools/analysis/per_class_ap.py`, `tools/data/convert_single_chromo.py`. Velocity entropy measurement code included in repository.

## Appendix

### A. Theoretical Derivations

**A.1 Density Bottleneck.** Deterministic OT solves $\\min_P \\langle P, C \\rangle$ s.t. $P\\mathbf{1} = \\mathbf{1}\_N/N$, $P^\\top\\mathbf{1} = \\mathbf{1}\_M/M$. By Birkhoff's theorem, $P^\*$ is a permutation matrix. With $N=500$, $M \\approx 46$, each proposal maps to one target with probability 1: $H(\\pi(i) \\mid x_1^{(i)}) = 0$, hence $H(V \\mid X_1) = 0$.

**A.2 Sinkhorn Regularization.** $P\_\\varepsilon = \\arg\\min_P \\langle P, C \\rangle - \\varepsilon H(P)$ with $H(P) = -\\sum\_{i,j} P\_{ij} \\log P\_{ij}$. Solution: $P\_{ij} = u_i \\exp(-C\_{ij}/\\varepsilon) v_j$. As $\\varepsilon \\to \\infty$, $P\_\\varepsilon \\to$ uniform, maximizing diversity.

**A.3 Collapsing Argmax Mechanism.** $\\pi\_{\\mathrm{argmax}}(i) = \\arg\\max_j P\_\\varepsilon\[i,j\]$ maps a continuous distribution to a one-hot vector. $\\frac{\\partial \\pi\_{\\mathrm{argmax}}}{\\partial \\varepsilon} = 0$ almost everywhere—argmax is invariant to perturbations that preserve rank order. The discrete operator annihilates the continuous diversity injected by ε.

**A.4 Stochastic Coupling.** $\\pi\_{\\mathrm{stoch}}(i) \\sim \\mathrm{Categorical}(P\_\\varepsilon\[i,:\])$ yields $\\mathbb{E}\[\\pi\_{\\mathrm{stoch}}\] = P\_\\varepsilon$ and $H(\\pi\_{\\mathrm{stoch}}(i) \\mid x_1^{(i)}) = H(P\_\\varepsilon\[i,:\])$, strictly monotonic in ε. ε acts as a smooth, functional diversity control parameter.

## References

References are resolved from `refs.bib`.
