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
