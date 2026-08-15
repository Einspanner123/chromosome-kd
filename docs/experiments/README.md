# Experiment documentation

This directory contains only current experiment-system documents:

- `CONFIGURATION_SYSTEM.md`: composition, inheritance, and validation rules;
- `PAPER_EXPERIMENT_ROUTE_MATRIX.md`: every paper claim and required run;
- `D2_HISTORY_COMPATIBILITY.md`: historical checkpoint mapping policy; and
- `D2_SOTA_DISTILL_SPEED_MIGRATION_AUDIT.md`: D2 baseline/deployment evidence.

The machine-readable authority is in `experiments/manifests/` and
`tools/experiment_db/experiments.db`. Documents must not introduce numerical
results that lack controlled-result IDs.

`PAPER_EXPERIMENT_ROUTE_MATRIX.md` is generated from the canonical route
manifest and may contain internal scheduling labels and runtime paths. Those
fields are operational metadata and must never be copied into the manuscript.

The canonical manuscript contract is
`experiments/manifests/paper_claim_manifest.yaml`. It maps every planned
dataset, accuracy, ablation, decision, diagnostic, and deployment claim to one
or more route IDs. Coverage and submission readiness are separate checks:

```bash
python tools/experiment_db/audit_paper_claim_coverage.py
python tools/experiment_db/audit_paper_claim_coverage.py --submission-ready
```

The first command verifies complete route coverage. The second is expected to
fail while required experiments are planned or blocked. Manuscript numbers
must not be refreshed until the submission-ready gate passes.
