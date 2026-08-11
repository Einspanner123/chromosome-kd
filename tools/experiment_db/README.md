# KaryoFlow experiment database

`experiments.db` is the canonical numerical source for the project. Paper
tables and claims must be traceable to this directory; prose documents are
explanations, not competing sources of truth.

## Data layers

1. `experiment`, `evaluation`, `stability`: automatically scanned raw runs.
2. `evidence_artifact`: immutable source files identified by path and SHA-256.
3. `controlled_result`: protocol-checked measurements. Only rows with
   `paper_eligible=1` may be quoted as completed experimental evidence.
4. `finding` and `theory_statement`: bounded claims, assumptions, derivations,
   negative results, engineering lessons, and explicit limitations.
5. `run_snapshot`: server-qualified snapshots (`server:work_dir`) for ongoing
   runs. A snapshot is never paper evidence.

This separation prevents three recurring errors: treating a best checkpoint
as proof that training completed, double-counting evaluations after a refresh,
and promoting a diagnostic/oracle result to a method result.

## Refresh and audit

```bash
python tools/experiment_db/build_experiment_db.py --server ross
python tools/experiment_db/import_evidence_manifest.py
python tools/experiment_db/audit_evidence_db.py
```

Run the raw scanner on both servers. The legacy `experiment` rows have logical
`work_dir` grain. When adding a second server, pass `--namespace-server` so
physical runs use `server:work_dir` and cannot overwrite the ross row; volatile
progress observations may additionally use `run_snapshot`. The curated
`evidence_manifest.json` contains paper results and original source addresses.

## Unified Dataset 2 test evaluation

Use the resumable evaluator for checkpoint-linked comparisons on the 1,000-image
Dataset 2 test split:

```bash
python tools/experiment_db/d2_test_unified.py --gpu-id 0
```

The evaluator explicitly fixes COCO `maxDets` to `(1, 10, 100)`, persists raw
predictions, independently recomputes metrics with `pycocotools`, and registers
results only when the framework and independent values agree. Its run key covers
the resolved config, checkpoint, annotation, inference code, seed, and protocol.
A rerun verifies stored hashes and reuses matching evidence without invoking the
GPU. Large predictions remain under `results/d2_test_unified/`; the immutable,
Git-sized aggregate in `evidence_sources/` and `experiments.db` are the canonical
record.

## Claim policy

- “Validated on both datasets” requires controlled rows for D1 and D2.
- “Detector-generic” may describe a mathematical mechanism whose assumptions
  are detector-level, but must not be rewritten as empirical COCO validation.
- COCO/mini-COCO remains future validation until a completed row is imported.
- Head compression changes cascade-head evaluations (CHE), not solver NFE.
- GACS is compared with the fixed two-step policy; it is not lossless versus
  the four-step production reference.
- Renewal removal is effectively accuracy-neutral at the retained proposal
  count, but the low-`K` failure is part of the finding, not an exception to hide.
- A learned quality-exponent sweep must store checkpoint identity and raw-export
  hashes. A sweep that re-ranks with ground-truth IoU is an oracle bound and
  must never be labeled as an LQCR hyperparameter ablation.
