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

The mandatory run identity, evidence record, replication-unit definitions,
aggregation rules, and paper export gate are specified in
[`EXPERIMENT_EVIDENCE_STANDARD.md`](EXPERIMENT_EVIDENCE_STANDARD.md). New
paper-facing evidence must pass `validate_run_evidence.py`; legacy rows are
being migrated to the same contract and must not be promoted merely because
they already exist in the database.

## Authoritative write path

```bash
python tools/experiment_db/validate_run_evidence.py \
  tools/experiment_db/evidence_sources/runs/<record>.json --verify-files
python tools/experiment_db/import_run_evidence.py \
  tools/experiment_db/evidence_sources/runs/<record>.json
python tools/experiment_db/audit_evidence_db.py
```

Every new accuracy evaluation enters through a schema-validated, hash-verified
run-evidence JSON. The importer is transactional and idempotent: reimporting an
identical record is a no-op, while reusing an identity with different content
fails. Training runs must be preregistered before evaluation. Paper exports may
only reference the resulting immutable artifact and controlled-result IDs.

The curated manifest importers and specialized registration scripts in the
recovery tag exist for evidence created before this contract. They are not
valid templates for new experiments.

## Legacy raw-run inventory

`build_experiment_db.py` scans historical work directories into the raw
`experiment`/`evaluation` layer. This layer is useful for discovery and progress
recovery, but does not make a result paper-eligible. When scanning another
machine, pass `--namespace-server` so equal relative paths cannot alias.

The existing `evidence_manifest.json`, compatibility reports, and source JSONs
are immutable legacy evidence. Do not regenerate or rewrite them during source
cleanup.

## Unified held-out test evaluation

Use the resumable evaluator for checkpoint-linked comparisons on either held-out
test split (Dataset 1: 220 images; Dataset 2: 1,000 images):

```bash
python tools/experiment_db/d2_test_unified.py --dataset d1 --gpu-id 0
python tools/experiment_db/d2_test_unified.py --dataset d2 --gpu-id 0
```

The evaluator explicitly fixes COCO `maxDets` to `(1, 10, 100)`, persists raw
predictions, independently recomputes metrics with `pycocotools`, and registers
results only when the framework and independent values agree. Its run key covers
the resolved config, checkpoint, annotation, inference code, seed, and protocol.
A rerun verifies stored hashes and reuses matching evidence without invoking the
GPU. Large predictions remain under `results/d1_test_unified/` or
`results/d2_test_unified/`; the immutable,
Git-sized aggregate in `evidence_sources/` and `experiments.db` are the canonical
record.

Accuracy evaluations may use checkpoints trained or evaluated on different GPU
models, provided that the random seed and full evaluation protocol are recorded.
Hardware consistency is required only for latency, throughput, memory, and other
efficiency comparisons; those measurements must use one declared device and
protocol. Machine hostnames are operational metadata and must not appear in
paper-facing exports.

## Known legacy quality boundaries

- The raw scanner layer contains mixed historical grain and is not a paper
  source by itself.
- Some legacy diagnostic and dataset-inventory rows use `paper_eligible=1` for
  bounded non-accuracy claims. Final detector accuracy exports still require a
  held-out test split and controlled evidence.
- Fixed-checkpoint inference repetitions and independent training repetitions
  remain distinct aggregation units.

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
