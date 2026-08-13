# D1-INHOUSE1700-V1: Dataset Freeze and D2-Aligned Rerun Plan

## Frozen cohort

- Source composite: 2,200 images.
- Exclusion: 500 transformed Dataset 2 images identified by normalized-basename agreement and grayscale resize ZNCC >= 0.99. The minimum accepted score is separated from the maximum rejected score by more than 0.16.
- Included: 1,700 in-house images and their complete Dataset 1 annotations.
- Group proxy: filename stem after removal of the Roboflow export suffix. This gives 1,350 acquisition groups; no group crosses a split.
- Split: train/validation/test = 1,190/170/340 (70%/10%/20%), aligned with Dataset 2's 3,500/500/1,000 ratio.
- Patient identifiers are unavailable. The acquisition-stem group is a conservative proxy and this limitation must be stated in the paper.
- Canonical manifest: `dataset_self1700_v1_provenance_57bc9516b11a.json` (SHA-256 `57bc9516b11aa642d50c32124777f70abdc2074ee18e3250abf1665940eea43c`).
- Annotation SHA-256: train `318120afe81184557cd1c13c985c63f5404ae4ce9e97e5cec6e5edb1d6ec534c`; validation `57466f1fe99201b091b65fbbcf687c0bebca86b82fbed86ba48d8b8d68b6dfaf`; test `52e8868d2f43c609d841d138bd495a410690b399c0904782bc44d5aba7564efd`.

The maximum per-class split-fraction deviation from the image target is below 0.0020. Both servers reproduce byte-identical annotation files and the canonical provenance manifest.

## Primary D2-aligned matrix

All accuracy claims use three independently trained models with training seeds 42, 123, and 789. Validation randomness is fixed at 42. One best-validation checkpoint per run is evaluated once on the held-out test set with inference seed 42.

| Method | Full training runs | Paired short runs | Role |
|---|---:|---:|---|
| KaryoFlow (RF + DPM++) | 3 | 0 | generation baseline |
| KaryoFlow + LQCR | 0 | 3 | final-stage paired decision intervention |
| DiffusionDet | 3 | 0 | diffusion baseline |
| DINO-R50 | 3 | 0 | transformer detector baseline |
| RTMDet-L | 3 | 0 | one-stage detector baseline |
| Cascade R-CNN-R50 | 3 | 0 | multi-stage detector baseline |
| YOLOX-S | 3 | 0 | efficient detector baseline |

Total: 18 full detector trainings, 3 paired LQCR trainings, and 21 final test evaluations. The six COCO metrics (mAP, AP50, AP75, AP_S, AP_M, AP_L) are mandatory for every test record.

## Inference-only analysis

Using frozen KaryoFlow checkpoints:

1. Euler, Heun, and DPM++ at 1--4 steps; this is a same-checkpoint solver comparison and is not a cumulative training ablation.
2. Candidate Top-K and renewal on/off.
3. LQCR beta selected on validation; any test sweep is labeled post-selection sensitivity analysis.
4. AP90/AP95, small/medium/large strata, overlap strata, quality-IoU correlation, and image-cluster paired bootstrap.
5. Speed, peak memory, and throughput are measured only on the Ross A6000 under an exclusive, fixed protocol. Accuracy training may use any of the three GPUs.

## Three-GPU execution order

1. KaryoFlow seeds 42/123/789 start first on A6000/A5000/A4000. The A4000 configuration is not altered; OOM causes queue reassignment rather than a batch-size change.
2. Each completed KaryoFlow parent immediately triggers its seed-matched LQCR short run and final paired test.
3. A6000 prioritizes DINO and KaryoFlow-family jobs. A5000 prioritizes DiffusionDet and remaining heavy jobs. A4000 runs RTMDet, YOLOX, Cascade, and short LQCR jobs.
4. Baseline jobs are dispatched from the preregistered queue; no test result may alter a configuration or checkpoint-selection rule.
5. Deployment distillation is a separate extension. It is not mixed into the primary detector comparison until a paired three-parent protocol is registered.

The first KaryoFlow three-seed result is expected in roughly 18--30 hours under current two-GPU workstation contention. The complete 21-run matrix is expected to take about 4--6 days. ETA is recalibrated from observed epoch time after the first validation epoch of each architecture.

## Evidence lifecycle

1. Pre-register stable train and evaluation IDs in `train_run_registry` and `eval_run_registry`.
2. Record dataset manifest SHA, annotation SHA, resolved config SHA, training seed, code commit, parent run (for LQCR), execution device, and work directory before launch.
3. Preserve failed attempts; mark them failed/superseded and never reuse their metrics.
4. After training, hash the selected checkpoint and source log. Evaluate only the selected checkpoint on test.
5. Write a schema-valid immutable run-evidence JSON, independently recompute COCO metrics from saved predictions, then import idempotently into the database.
6. Only test records with all six metrics, `test_tuned=false`, verified files, and the correct replication unit may become paper eligible.
7. A database audit blocks duplicate active method/seed registrations and active config-SHA drift.

Dataset 2 comparisons must distinguish independent training seeds from fixed-checkpoint inference seeds. Existing Dataset 2 KaryoFlow/LQCR results have three independent parent-child pairs; other Dataset 2 baselines remain point estimates unless two additional training runs per baseline are completed.
