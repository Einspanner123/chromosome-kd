# KaryoFlow research scope

This note defines the active scientific boundary of the repository. Numerical
claims are not copied here as an alternative source of truth; their canonical
records are the result IDs in `tools/experiment_db/experiments.db` and the
immutable evidence files under `tools/experiment_db/evidence_sources/`.

## Generation

KaryoFlow adapts few-step box transport to crowded chromosome detection. The
active generation path contains DDPM and rectified-flow training, random or
optimal-transport coupling, Euler/Heun/DPM-Solver++ inference, persistent
proposal identity across multistep history, and Top-K/renewal controls.

Rectified flow and DPM-Solver++ are adopted foundations, not claimed as new
algorithms. The task-level contribution is their detector integration and the
analysis that a multistep history is only semantically valid when each history
entry refers to the same proposal identity.

## Decision

LQCR is the active independent accuracy method. It trains a final-stage scalar
localization-quality branch and ranks candidates with a validation-selected
combination of class confidence and predicted quality. It does not modify box
coordinates, class logits, proposal survival, or solver states before final
score-dependent selection.

The branch is a localization-quality ranking surrogate. It is not a calibrated
estimate of a threshold-specific probability and should not be described as
such without Brier/ECE/reliability evidence. Main claims must use paired
parent-child detector runs and held-out test evaluations.

## Deployment

Deployment studies are separate from the core accuracy contribution:

- DPM-Solver++ reduces network evaluations relative to Heun.
- Removing renewal preserves proposal identity and removes a small amount of
  overhead when the candidate budget is sufficiently redundant.
- Top-K pruning exposes a speed/accuracy operating range.
- Parent-matched head distillation reduces cascade depth.
- GACS is a historical conditional-computation result, not active model code.

Latency claims require the same GPU, batch/input shape, precision, warm-up,
timed iterations, and timing boundary. Accuracy runs may use different GPU
models, but training-seed and fixed-checkpoint inference-seed replication must
never be aggregated as the same statistical unit.

## Claim discipline

- The system addresses the chromosome-detection stage of automated
  karyotyping, not segmentation, straightening, pairing, or diagnosis.
- Small/crowded/overlapping chromosomes motivate the method. Subgroup benefits
  are claimed only where a registered difficult-subset analysis supports them.
- Dataset 2 KaryoFlow versus DINO is an essentially tied point estimate because
  the paired image-bootstrap interval crosses zero.
- GACS and renewal/Top-K are deployment findings, not independent accuracy
  innovations.
- Validation selects checkpoints and hyperparameters. Paper-facing accuracy
  values come from held-out test evaluation and must be imported through the
  run-evidence contract.

Historical alternatives, falsified directions, and their original prose are
recoverable from `refs/tags/archive/pre-cleanup-20260815`.
