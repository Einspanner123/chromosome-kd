# Dataset 2 SOTA, distillation, and speed migration audit

## Decision

Four fixed-checkpoint SOTA baselines can be migrated as archived `EXACT` v2
identities; DiffusionDet remains `STRUCTURAL_ONLY`. They must not use the
generic `detector_adamw_150e.py`: DINO has a lower learning rate and backbone
multiplier, YOLOX has EMA and mode-switch hooks, and each historical data
pipeline retains its own filtering and batching policy.

The H6 teacher is already represented by the verified archived OT KaryoFlow
identity. The H3 student requires two identities: its training identity and
its inference identity. The historical training configuration used argmax OT,
whereas the paper evaluation loaded the student with the multinomial-OT H3
inference structure. Collapsing these would create a false `EXACT` claim.

The four standard baselines are now materialized as self-contained v2 model,
schedule, and historical dataset bases. The compatibility suites ran in the
project environment: DINO, RTMDet-L, Cascade R-CNN, and YOLOX-S all returned
`EXACT`, with zero configuration differences, strict checkpoint loading, no
hard issues, and canonical import enabled.

## Evidence recovered

| Identity | Checkpoint seed | Tensors | Evidence status |
|---|---:|---:|---|
| DINO R50 | 1769925607 | 627 | sufficient for exact config/checkpoint/test verification |
| RTMDet-L | 1769925607 | 872 | sufficient for exact config/checkpoint/test verification |
| Cascade R-CNN R50 | 1769925607 | 364 | sufficient for exact config/checkpoint/test verification |
| DiffusionDet DDPM | 628372310 | 598 | structural only: seven obsolete schedule buffers prevent strict load |
| YOLOX-S | 1769925607 | 462 | sufficient for exact config/checkpoint/test verification |
| H6 teacher | 335778785 | 590 | already verified in the D2 historical suite |
| H3 distilled student | 1858611995 | 464 | exact inference identity; historical distillation training remains archived and non-executable |

The checkpoint seeds above come from checkpoint metadata. The paper's
42/123/789 rows are deterministic inference repeats of the same checkpoint.
They are useful for inference stability but are not training replications.

## Implemented migration controls

1. Extend `check_legacy_compatibility.py` so `normalized_model` supports
   detectors without a top-level `bbox_head` (Cascade R-CNN uses
   `roi_head`, and DINO/RTMDet/YOLOX have different default semantics).
   Default injection must be limited to LDMDet heads.
2. Materialize one frozen model base and one historical schedule per SOTA
   family. Do not make the v2 files inherit from the soon-to-be-deleted legacy
   configuration tree.
3. Add a dedicated historical baseline matrix with training seeds
   `[1769925607, 628372310]`; enable each method only for its recorded seed.
   This requires per-run seed support in the suite manifest rather than a
   Cartesian training matrix.
4. Add a distillation parent-checkpoint injection path. `load_from` is wrong
   for H6-to-H3 distillation: the checkpoint belongs at
   `model.teacher_checkpoint`, while the student starts from its own state.
5. Keep speed protocols outside scientific training hashes, but bind every
   latency row to resolved-config and checkpoint hashes.

## Compatibility classification

- DINO, RTMDet-L, Cascade R-CNN, and YOLOX-S: verified `EXACT` in the candidate
  v2 historical matrices.
- DiffusionDet is currently `STRUCTURAL_ONLY`: the checkpoint contains seven
  historical diffusion-schedule buffers (`alphas_cumprod_prev`, posterior
  variance/log variance, and four square-root schedules) that the cleaned head
  no longer registers. Evaluation remains executable, but it does not pass the
  strict state-dict gate. Either restore the harmless buffers in a legacy-only
  model or retain the structural classification.
- H6 teacher: `EXACT` through `v2.d2_history.legacy_karyoflow_ot_r50.train`.
- H3 inference: verified `EXACT` through
  `v2.d2_history.legacy_karyoflow_ot_h3_student_r50.train`. The isolated
  historical dataset snapshot preserves the effective category mapping, while
  the compatibility suite binds evaluation to the held-out test annotation.
  The checkpoint strict-loads into the three-head model and all six persisted
  COCO metrics are reproduced exactly.
- H3 training: `STRUCTURAL_ONLY` relative to the paper inference target; retain
  a separate archived argmax-OT distillation target if training reproducibility
  is required.
- The cleaned current `ldmdet` implementation no longer exposes
  `teacher_checkpoint`, `use_distillation`, `distill_head_map`, or
  `distill_lambda`. Therefore historical H3 training is not currently
  executable and must not be advertised as reproducible until the archived
  distillation implementation is recovered and separately audited. The H3
  inference checkpoint remains executable with the three-head structure.
- Historical A6000 latency: valid controlled evidence under its recorded
  protocol, but only `PARTIAL` against the stricter new protocol because clock,
  environment hash, and timing rerun evidence are absent.

## Integrity gates

An archived identity may be mapped to a canonical v2 ID only when all of the
following pass: source hash, checkpoint hash, strict state-dict load, dataset
annotation hash, effective category mapping, scientific-section equality, and
independent recomputation of all six COCO metrics from persisted predictions.
Speed results additionally require the same resolved config and checkpoint for
the paired accuracy point.

Independent COCO recomputation was repeated for the seed-42 persisted
predictions of all five baselines and the H3 student. Every one of the six
metrics (mAP, AP50, AP75, AP_S, AP_M, AP_L) had an absolute delta of exactly
zero from its stored summary.
