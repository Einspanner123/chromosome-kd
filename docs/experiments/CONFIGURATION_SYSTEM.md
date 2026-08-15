# Canonical experiment configuration system

## Status

The canonical configuration tree is `experiments/configs/`. The former `v2`
namespace and all earlier parallel configuration trees were retired after the
paper route matrix, D2 checkpoint compatibility reports, and head-distillation
identity were completed. Their exact source remains available from
`archive/pre-cleanup-20260815`.

## Composition model

An experiment is composed from four independent layers:

1. `_base_/datasets/` defines dataset identity, split hashes, transforms, and
   evaluators.
2. `_base_/models/` and `_base_/schedules/` define reusable implementation and
   optimization components.
3. `methods/` defines one dataset-independent scientific method.
4. `matrices/` binds a dataset, methods, training seeds, and tracking namespace.

Inference grids live in `ablations/`; hardware and dynamic-compute protocols
live in `deployment/`. Generated catalogs, paper routes, and historical
compatibility declarations live in `experiments/manifests/` so they cannot be
mistaken for MMEngine configurations.

The matrix resolver produces a flattened `resolved_config.py`. MMEngine loads
that file normally through `Config.fromfile` and `Runner.from_cfg`. The YAML
matrix is orchestration metadata, not a replacement configuration language.

## Stable scientific identity

The resolver records:

- dataset manifest and split annotation hashes;
- matrix, dataset, method, and runtime source hashes;
- resolved and scientific configuration hashes;
- training seed and replication unit;
- parent checkpoint identity for paired LQCR or head-distillation runs;
- Git commit and dirty-diff hash;
- validation selection rule and `test_tuned` status.

The scientific hash excludes machine-local output and checkpoint paths.
Checkpoint SHA-256 is recorded independently.

## Current method boundary

- KaryoFlow generation: rectified flow, shifted time, AdaLN-Zero, DPM-Solver++
  four-step integration, random coupling, and persistent proposal identity.
- LQCR decision: frozen parent, final-stage continuous-IoU quality branch, and
  final-only ranking.
- Head distillation: parent-matched H6 teacher to H3 student using the restored
  distillation implementation and explicit head mapping.
- Conventional baselines: DINO R50, RTMDet-L, Cascade R-CNN R50, YOLOX-S, and
  DiffusionDet under dataset-specific matrices.

Files suffixed `_legacy` are executable compatibility identities required to
load existing D2 paper checkpoints. They are not templates for new training.
Older source configs referenced by historical evaluations resolve from the
recovery Git tag instead of remaining in the active tree.

## Commands

Preview a matrix:

```bash
python tools/experiments/launch.py \
  --matrix experiments/configs/matrices/d1_inhouse1700.yaml --plan
```

Resolve one configuration:

```bash
python tools/experiments/launch.py \
  --matrix experiments/configs/matrices/d1_inhouse1700.yaml \
  --method karyoflow --seed 42 --resolve-only
```

A parent-matched child additionally requires `--parent-checkpoint <path>`.

## Required validation gates

```bash
python tools/configs/audit_configs.py --roundtrip
python tools/configs/audit_configs.py --build-models
python tools/configs/audit_generation_ablation.py
python tools/configs/audit_inference_ablations.py
python tools/configs/audit_protocols.py
python tools/experiment_db/audit_config_compatibility.py
python tools/experiment_db/audit_experiment_route_matrix.py
python tools/experiment_db/audit_paper_claim_coverage.py
```

The active tree contains 42 base files, 22 methods, 14 matrices, 8 ablation
protocols, and 4 deployment protocols. Every method is referenced by at least
one audited matrix or a paper compatibility route.
