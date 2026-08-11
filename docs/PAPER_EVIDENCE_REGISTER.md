# Paper evidence register

> Auto-generated from `tools/experiment_db/experiments.db`. Do not edit
> numerical values here; update the manifest/database and regenerate.

## Paper-eligible controlled results

| ID | Family / variant | Data | Metric | Value | Delta | Source |
|---|---|---|---:|---:|---:|---|
| `gacs-d1-adaptive-map` | GACS / adaptive_1_or_2_renewal_on | D1 val, mean(42,123,789) | mAP (absolute) | 0.7383 | -0.0007 | `ross:work_dirs/diagnosis/gacs_chr2024_summary.json` |
| `gacs-d1-fixed2-map` | GACS / fixed2_renewal_on | D1 val, mean(42,123,789) | mAP (absolute) | 0.739 |  | `ross:work_dirs/diagnosis/gacs_chr2024_summary.json` |
| `gacs-d1-fourstep-reference` | GACS / fixed4_reference | D1 val, mean(42,123,789) | mAP (absolute) | 0.747 |  | `ross:work_dirs/diagnosis/gacs_chr2024_summary.json` |
| `gacs-d1-latency-reduction` | GACS / adaptive_1_or_2_renewal_on | D1 benchmark, mean(42,123,789) | latency_reduction (percent) | 13.79 |  | `ross:work_dirs/diagnosis/gacs_chr2024_summary.json` |
| `lqcr-d2-ap90-gain` | LQCR / final_only_beta2 | D2 val, 42 | AP90_delta (absolute_delta) | 0.03041 |  | `ross:docs/paper/latex/figures/v2/data/source_lqcr_final_only_seed42.json` |
| `lqcr-d2-ap95-gain` | LQCR / final_only_beta2 | D2 val, 42 | AP95_delta (absolute_delta) | 0.03251 |  | `ross:docs/paper/latex/figures/v2/data/source_lqcr_final_only_seed42.json` |
| `lqcr-d2-baseline-map` | LQCR / disabled | D2 val, 42 | mAP (absolute) | 0.8630128 |  | `ross:docs/paper/latex/figures/v2/data/source_lqcr_final_only_seed42.json` |
| `lqcr-d2-beta0-map` | LQCR / learned_q_beta0 | D2 val, 42 | mAP (absolute) | 0.8630128 |  | `ross:tools/experiment_db/evidence_sources/lqcr_learned_beta_sweep_seed42.json` |
| `lqcr-d2-beta025-map` | LQCR / learned_q_beta0.25 | D2 val, 42 | mAP (absolute) | 0.8657292 | 0.002716436 | `ross:tools/experiment_db/evidence_sources/lqcr_learned_beta_sweep_seed42.json` |
| `lqcr-d2-beta05-map` | LQCR / learned_q_beta0.5 | D2 val, 42 | mAP (absolute) | 0.8673472 | 0.004334369 | `ross:tools/experiment_db/evidence_sources/lqcr_learned_beta_sweep_seed42.json` |
| `lqcr-d2-beta1-map` | LQCR / learned_q_beta1 | D2 val, 42 | mAP (absolute) | 0.8689229 | 0.005910119 | `ross:tools/experiment_db/evidence_sources/lqcr_learned_beta_sweep_seed42.json` |
| `lqcr-d2-beta2-map` | LQCR / learned_q_beta2 | D2 val, 42 | mAP (absolute) | 0.8704443 | 0.007431458 | `ross:tools/experiment_db/evidence_sources/lqcr_learned_beta_sweep_seed42.json` |
| `lqcr-d2-enabled-map` | LQCR / final_only_beta2 | D2 val, 42 | mAP (absolute) | 0.8704443 | 0.0074315 | `ross:docs/paper/latex/figures/v2/data/source_lqcr_final_only_seed42.json` |
| `head-h3s4-d1-map` | head_compression_control / H3S4_no_distill | D1 val, 42 | mAP (absolute) | 0.746 |  | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `distill-d2-student-latency` | head_distillation / H3_student | D2 benchmark, 42 | latency (ms) | 44.72 | -32.85 | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `distill-d2-student-map` | head_distillation / H3_student | D2 val, 42 | mAP (absolute) | 0.859 | -0.004 | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `distill-d2-teacher-latency` | head_distillation / H6_teacher | D2 benchmark, 42 | latency (ms) | 77.57 |  | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `distill-d2-teacher-map` | head_distillation / H6_teacher | D2 val, 42 | mAP (absolute) | 0.863 |  | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `vpred-d1-val` | prediction_parameterization / v | D1 val, mean(42,123,789) | mAP (absolute) | 0.745 | -0.002 | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `x0-d1-val` | prediction_parameterization / x0 | D1 val, mean(42,123,789) | mAP (absolute) | 0.747 |  | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `vpred-d2-val` | prediction_parameterization / v | D2 val, mean(42,123,789) | mAP (absolute) | 0.857 | -0.002 | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `x0-d2-val` | prediction_parameterization / x0 | D2 val, mean(42,123,789) | mAP (absolute) | 0.859 |  | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `renew-d1-off-map` | renewal_consistency / renewal_off_K500 | D1 val, mean(42,123,789) | mAP (absolute) | 0.7433333 | -0.003 | `ross:work_dirs/diagnosis/d1_topk_validation_seed42.json` |
| `renew-d1-on-map` | renewal_consistency / renewal_on_K500 | D1 val, mean(42,123,789) | mAP (absolute) | 0.7463333 |  | `ross:work_dirs/diagnosis/d1_topk_validation_seed42.json` |
| `renew-d2-off-map` | renewal_consistency / renewal_off_K500 | D2 val, mean(42,123,789) | mAP (absolute) | 0.8583333 | -0.0003334 | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `renew-d2-on-map` | renewal_consistency / renewal_on_K500 | D2 val, mean(42,123,789) | mAP (absolute) | 0.8586667 |  | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `renew-k100-off-map` | renewal_consistency / renewal_off_K100 | D2 val, mean(42,123,789) | mAP (absolute) | 0.808 | -0.0313 | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `renew-k100-on-map` | renewal_consistency / renewal_on_K100 | D2 val, mean(42,123,789) | mAP (absolute) | 0.8393 |  | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `renew-latency-saving` | renewal_consistency / renewal_off_K500 | D2 benchmark, mean(42,123,789) | latency_reduction (percent) | 2.427621 |  | `ross:work_dirs/diagnosis/renewal_latency_3seed.json` |

## Findings and claim boundaries

### F-COCO: General object-detection validation boundary [pending]

Detector-generic mechanisms have theoretical support but have not been empirically established on COCO.

- Scope: Current manuscript
- Generality basis: Proof assumptions are stated at proposal/solver level.
- Caveat: Only two distribution-shifted chromosome datasets are complete; mini-COCO is planned under resource constraints.

### F-DISTILL: Cascade-head distillation [supported]

A six-head teacher can supervise a three-head student while solver NFE remains four.

- Scope: D2 distillation; D1 architecture-only H3S4 control
- Generality basis: Repeated detector heads are compressible when intermediate behavior is supervised.
- Caveat: D1 result is not a distillation run; call 24-to-12 CHE, not NFE.
- Primary metric: mAP -0.004
- Benefit: 1.73x speedup, 77.57 to 44.72 ms

### F-GACS: Geometry-aware conditional stepping [supported]

Frozen geometry/confidence gates skip the second step for 38.64% of D1 images with near-neutral accuracy versus fixed two-step inference.

- Scope: D1, three checkpoint seeds
- Generality basis: Per-image computation can be allocated by proposal convergence statistics.
- Caveat: 0.7383 remains below the four-step reference 0.747; not a lossless production replacement.
- Primary metric: mAP -0.0007 vs fixed2
- Benefit: 13.79% latency reduction

### F-LQCR: Localization-quality calibrated final ranking [supported]

A final-only score p*q^2 improves high-IoU ranking without changing candidate coordinates, class logits, renewal, or solver states before score-dependent post-processing.

- Scope: D2 fixed checkpoint, seed42
- Generality basis: The true-positive survival factorization applies to proposal detectors that rank localized candidates.
- Caveat: D1 clean paired replication is still running; no COCO empirical validation.
- Primary metric: mAP +0.0074315 under the single-image precision protocol
- Benefit: AP90 +0.03041; AP95 +0.03251; monotone learned-beta sweep

### F-RENEW: History-consistent multistep inference [supported]

Disabling box renewal avoids mixing proposal identities in DPM++ history and removes renewal overhead.

- Scope: D1 and D2, K=500; D2 latency on RTX A6000
- Generality basis: Any multistep proposal solver using historical states requires state identity across steps.
- Caveat: Redundancy is required: at K=100, renewal removal loses 0.0313 mAP.
- Primary metric: D2 mAP -0.0003; D1 -0.0030
- Benefit: 2.43% mean latency reduction

### F-X0: Stable data prediction in low-dimensional shifted RF [supported]

x0 prediction is at least as accurate as v prediction and avoids inverse-t squared loss amplification.

- Scope: D1 and D2, three-seed validation
- Generality basis: The parameterization identity and loss weighting depend on path/schedule, not chromosome semantics.
- Caveat: Observed gains are small (+0.002 val on each dataset); treat as optimization evidence, not a universal ranking.
- Primary metric: +0.002 mAP on D1 and D2 val
- Benefit: more stable low-t optimization

## Theory statements

### T-LQCR

The ideal ranking score for IoU-thresholded detection factors class correctness and localization survival.

- Assumptions: A candidate is a true positive at threshold tau iff its class is correct and IoU>=tau; using conditional mean IoU as a cross-threshold statistic additionally assumes a family stochastically ordered by its mean.
- Derivation: P(TP_tau|h)=P(C|h) P(IoU>=tau|C,h). The implemented q is a practical surrogate trained toward E[M*IoU|h], with unmatched indicator M=0; p*q^beta is therefore empirically calibrated rather than asserted to equal the posterior.
- Prediction: Quality-aware re-ranking should mainly improve strict-IoU AP without changing candidate coordinates.
- Empirical status: Supported by a learned-quality beta sweep and D2 AP90/AP95 gains; D1 clean replication pending.
- Source: `docs/paper/latex/main.tex`

### T-RENEW

A multistep DPM++ correction is inconsistent on rows whose proposal identity is replaced between steps.

- Assumptions: The update uses a finite-difference/history term across adjacent states; renewal replaces selected rows with independent proposals.
- Derivation: For D1=(x0_hat_n-x0_hat_{n-1})/(t_n-t_{n-1}), a renewed row compares predictions belonging to different proposal identities, so D1 is not a derivative estimate along one trajectory.
- Prediction: Disabling renewal should preserve or improve solver coherence when K provides enough redundant proposals, and should fail when K is too small.
- Empirical status: Supported at K>=200; low-K counterexample observed at K=100.
- Source: `docs/EXPERIMENT_LINEAGE.md`

### T-X0

x0 and velocity prediction are information-equivalent but not optimization-equivalent under the chosen objective.

- Assumptions: Rectified path x_t=(1-t)x0+t z and v=z-x0.
- Derivation: x0=x_t-t v; converting an x0 squared error to velocity error gives L_v=t^{-2} L_x0.
- Prediction: A shifted schedule emphasizing small t makes v loss amplify low-t gradients; direct x0 prediction should be no worse in four-dimensional box space.
- Empirical status: Directionally supported by D1 and D2 three-seed validation.
- Source: `docs/EXPERIMENT_LINEAGE.md`
