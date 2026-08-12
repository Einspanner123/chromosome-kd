# Experiment evidence standard

This document defines the mandatory path from a model run to a paper number.
The database is the numerical source of truth, but a database row is valid only
when it points to immutable, independently verifiable evidence.

## 1. Stable identities

Every training run receives a `train_run_id` before launch:

`<dataset>__<method>__trainseed-<seed>__<config-hash-12>`

Every evaluation receives an `eval_run_id`:

`<train-run-id>__<split>__inferseed-<seed-or-deterministic>__<protocol-hash-12>`

Names, timestamps, hosts, GPU indices, and working-directory locations are not
identities. Moving a run must not change either identifier.

## 2. Required run record

Each completed evaluation must have one JSON record conforming to
`schemas/run_evidence.schema.json`. It records:

- dataset and exact split, annotation path and SHA-256;
- training seed and, separately, inference seed;
- source config, checkpoint, and framework log with SHA-256;
- code revision and dirty-tree state;
- the replication unit and parent training run;
- all six COCO box metrics; and
- lifecycle status and exclusion reason, if any.

The evaluation protocol also records the number of test images, COCO metric
definition and `maxDets`, evaluation-code revision, proposal count, solver,
steps/NFE, and the validation record used for checkpoint or hyperparameter
selection. These fields distinguish a numerically identical metric produced
under a different scientific protocol.

LQCR and any other final-stage intervention must point to its immutable parent
detector through `parent_train_run_id` and `parent_checkpoint_sha256`.

## 3. Replication units

Only these values are permitted:

- `independent_training_seed`: separately optimized detector weights;
- `fixed_checkpoint_inference_seed`: one checkpoint with stochastic inference;
- `paired_final_stage_intervention`: a frozen parent detector with only the
  declared final-stage component fitted or enabled;
- `hardware_efficiency_repeat`: repeated latency/throughput measurement under
  one hardware protocol;
- `deterministic_evaluation`: one deterministic evaluation of one checkpoint.

These units must never be pooled into a shared standard deviation. In
particular, inference-seed variance is not training variance.

## 4. Lifecycle

The state transition is:

`planned -> running -> trained -> test_evaluated -> verified -> paper_eligible`

`failed`, `invalid`, and `superseded` are terminal evidence states. A best
checkpoint alone does not prove completion. `paper_eligible` requires a held-out
test result, successful hash verification, complete metrics, and an approved
replication unit.

Validation results may select checkpoints or hyperparameters but are never
paper-eligible final detector results. Test evaluation must not alter model or
hyperparameter selection.

## 5. Aggregation rules

An independent-training aggregate requires:

1. three distinct `training_seed` values;
2. three distinct checkpoint hashes;
3. the same dataset annotation hash and evaluation protocol hash;
4. one test evaluation per trained model, unless a prespecified inference-seed
   average is first computed within each model;
5. arithmetic mean and sample standard deviation across trained models; and
6. complete mAP, AP50, AP75, AP_S, AP_M, and AP_L values for every model.

A paired final-stage effect requires equal training seeds and one-to-one parent
checkpoint links. Deltas are computed within seed before their mean is reported.
The baseline mean must never be subtracted from an unmatched intervention mean
and described as a paired effect.

A fixed-checkpoint inference aggregate may quantify stochastic proposal
variation, but must be labelled as such and reported as a point estimate in a
table that also contains independent-training aggregates.

Three training seeds support descriptive mean and sample standard deviation,
not a strong normal-theory significance claim. A conditional confidence
interval for one checkpoint must use paired image-level bootstrap resampling
and must be labelled as conditional on that trained model.

## 6. Paper export gate

Paper tables and figures must be generated from a versioned export, never by
copying values from logs or prose. The export command must fail when:

- `split` is not `test` for a final accuracy claim;
- a required artifact or SHA-256 is missing or mismatched;
- any of the six COCO metrics is absent;
- a training aggregate contains fewer than three distinct checkpoints;
- a paired intervention lacks its parent link;
- replication units are silently mixed;
- an invalid or superseded artifact is referenced; or
- the same physical run is counted twice through host/path aliases.

Every paper-facing claim receives one audit status:
`PASS`, `CONDITIONAL_FIXED_CKPT`, `DESCRIPTIVE_ONLY`, `VALIDATION_ONLY`,
`BLOCKED_MIXED_UNIT`, `BLOCKED_TEST_TUNING`, or
`BLOCKED_MISSING_PROVENANCE`.

Efficiency claims additionally require the same device model, batch size,
image size, warm-up, measurement iterations, precision mode, and software
environment. Accuracy checkpoints may originate on different GPU models.

## 7. Operational checklist

Before training:

1. freeze the dataset split and store its annotation SHA-256;
2. generate `train_run_id` from dataset, method, training seed, and config hash;
3. record the code revision and dirty-tree state;
4. write the resolved configuration into the run directory.

After training:

1. identify completion from terminal logs or the declared stopping rule;
2. hash the selected checkpoint, resolved config, and training log;
3. register validation only as selection evidence;
4. schedule one standardized held-out-test evaluation.

After evaluation:

1. emit the run-evidence JSON atomically;
2. independently recompute COCO metrics and compare with framework output;
3. validate the record with `validate_run_evidence.py`;
4. import idempotently and run the database audit;
5. export paper-facing tables from verified records.
