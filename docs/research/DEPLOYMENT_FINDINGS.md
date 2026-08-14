# Deployment findings

This document is a compact index of retained deployment evidence. It is not a
replacement for the experiment database.

## Renewal and Top-K

For multistep DPM-Solver++ inference, proposal renewal can mix object identities
inside the solver history. With a redundant candidate budget (`K=500`), turning
renewal off changes Dataset 2 mAP by -0.0003 and reduces mean latency by 2.43%
on the registered same-hardware protocol. At `K=100`, however, removing renewal
loses 0.0313 mAP. The supported conclusion is therefore conditional: preserve
identity when redundancy is sufficient; do not generalize renewal-off to a
small candidate budget.

Canonical database findings and result IDs:

- `F-RENEW`
- `renew-d2-off-map`
- `renew-d1-off-map`
- `renew-latency-saving`
- `renew-k100-off-map`

The complete Dataset 2 inference grid is declared by
`experiments/configs/ablations/d2_topk_renewal.yaml`. Latency is re-evaluated
with `tools/benchmarks/benchmark_latency.py` using a resolved MMEngine config
and an explicit checkpoint.

## Head distillation

The retained student uses three cascade heads and is paired with its six-head
teacher. Training identity and inference identity are recorded separately in
the experiment manifests because the historical training recipe and the paper
evaluation configuration are not interchangeable. The student is a deployment
extension; it is not part of the definition of LQCR.

## GACS historical result

GACS allocates one or two solver steps from proposal-convergence statistics.
The registered Dataset 1 result averages three checkpoint runs: relative to a
fixed two-step reference it changes mAP from 0.7390 to 0.7383 (-0.0007), skips
the second step for 38.64% of images, and reports a 13.79% latency reduction.
The four-step reference remains higher at 0.7470. Consequently GACS is a
conditional-computation study, not a lossless replacement for the main
four-step detector and not a precision contribution.

Canonical database finding and result IDs:

- `F-GACS`
- `gacs-d1-fixed2-map`
- `gacs-d1-adaptive-map`
- `gacs-d1-latency-reduction`
- `gacs-d1-fourstep-reference`

The GACS implementation was deliberately removed from the active model during
cleanup. Its source, thresholds, detailed ablations, and tests remain
recoverable from `refs/tags/archive/pre-cleanup-20260815`. The immutable source
artifact remains `work_dirs/diagnosis/gacs_chr2024_summary.json` as registered
by `tools/experiment_db/evidence_manifest.json`.

## Reporting constraints

GACS validation results must not be inserted into a held-out test SOTA table.
Short-run latency is vulnerable to system jitter, so solver steps and early-exit
rate accompany latency. Final speed comparisons are repeated on the designated
single-GPU benchmark protocol; machine hostnames are never paper-facing.
