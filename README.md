# KaryoFlow

KaryoFlow is a chromosome detector built around few-step box transport. The
active research line has three layers:

1. **Generation** — DDPM and rectified-flow box generation with Euler, Heun,
   or DPM-Solver++ integration and identity-consistent proposal refinement.
2. **Decision** — final-stage localization-quality-aware ranking (LQCR).
3. **Deployment** — Top-K proposal pruning, renewal analysis, and
   parent-matched cascade-head distillation.

The repository implements the chromosome-detection stage of automated
karyotyping. It does not claim to perform chromosome segmentation,
straightening, pairing, or abnormality diagnosis.

## Repository map

- `ldmdet/`: framework-independent KaryoFlow model core.
- `experiments/configs/`: canonical compositional configurations.
- `experiments/mmdet_bridge/`: the minimal MMDetection adapter.
- `experiments/runners/`: train and test entry points.
- `experiments/manifests/`: experiment identities and paper routes.
- `tools/experiment_db/`: immutable evidence, SQLite source of truth, and
  evidence validation/import/export tools.
- `tests/`: focused tests for the active method and evidence contract.
- `docs/paper/`: manuscript sources and editable figures.
- `docs/experiments/`: experiment route and compatibility documentation.

`work_dirs/` and `results/` are local runtime roots. They are intentionally not
versioned or bulk-moved because checkpoints and predictions are large and
evidence records bind them by path and SHA-256.

MMDetection is an external pinned dependency (`mmdet==3.3.0`), not a vendored
source tree. Project-specific registry adapters live only in
`experiments/mmdet_bridge/`.

## Installation

Create an environment with a CUDA-compatible PyTorch build, then install the
tested OpenMMLab runtime and KaryoFlow:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

For CUDA builds where `mmcv` is not available from the default index, install
the matching wheel with OpenMIM before the remaining requirements.

## Configuration and launch

Audit every canonical combination before launching:

```bash
python tools/configs/audit_configs.py --build-models
```

Inspect or resolve an experiment:

```bash
python tools/experiments/launch.py \
  --matrix experiments/configs/matrices/d1_inhouse1700.yaml \
  --method karyoflow --seed 42 --plan

python tools/experiments/launch.py \
  --matrix experiments/configs/matrices/d1_inhouse1700.yaml \
  --method karyoflow --seed 42 --resolve-only
```

Use `--launch --gpu-id <id>` only after checking the resolved configuration,
dataset manifest, seed, tracker project, and output identity.

## Verification

```bash
pytest -q
python tools/configs/audit_generation_ablation.py
python tools/configs/audit_inference_ablations.py
python tools/configs/audit_protocols.py
python tools/experiment_db/audit_evidence_db.py
python tools/experiment_db/audit_experiment_route_matrix.py
```

New accuracy results must conform to
`tools/experiment_db/EXPERIMENT_EVIDENCE_STANDARD.md` and enter the database
through `validate_run_evidence.py` followed by `import_run_evidence.py`.

## Recovery

The 2026-08-15 repository cleanup is fully recoverable. The untouched starting
point is `refs/tags/archive/pre-cleanup-20260815`; each completed phase has a
`cleanup/p*-...` annotated tag and a gate record under
`docs/maintenance/baselines/`.
