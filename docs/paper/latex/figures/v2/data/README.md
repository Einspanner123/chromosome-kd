# Figure data exports

Place machine-readable, reviewable exports used by empirical figures here.
Each file must be referenced by `figure_manifest.yaml` with its originating
configuration, checkpoint hash, training seed, validation seed, hardware and
evaluation date.  Generated figures must never silently fall back to values
embedded in plotting code.

Current auditable source exports:

- `source_precision_a4_seed42.json`: Dataset 2 A4, 500 validation images.
- `source_lqcr_final_only_seed42.json`: Dataset 2 strict final-only LQCR.
- `source_lqcr_solver_coupled_seed42.json`: solver-coupled counterpart, kept
  for causal audit and not used as the main LQCR result.
- `source_precision_chr2024_seed42_100.json`: Dataset 1, 100-image diagnostic;
  it is not a full multi-seed test result.

The `quality_beta_*` entries use ground-truth IoU for oracle re-ranking. They
are diagnostic upper bounds, not trained LQCR outputs.

- `source_fps_a6000_karyoflow.json` and `source_fps_a6000_dino.json` retain
  the same-hardware RTX A6000 CUDA-event benchmark records (512 x 512,
  batch 1, 10 warmup, 500 measured iterations).
- `source_small_object_cross_dataset.json` records complete-split AP_S values
  for the cross-dataset scale analysis. Dataset 1 uses the 220-image test
  export (seed-42 checkpoints); Dataset 2 uses the 500-image validation table
  (KaryoFlow three-seed mean, other detectors seed 42). Values are interpreted
  only within each dataset.
