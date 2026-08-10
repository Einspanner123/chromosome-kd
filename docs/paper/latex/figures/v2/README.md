# Rebuilt paper figures

This directory is the vector-first source of truth for the revised paper
figures.  Legacy assets remain in the parent directory until every LaTeX
reference has been migrated and verified.

## Layout

- `sources/`: editable Python or TikZ sources;
- `data/`: machine-readable JSON/CSV inputs for empirical figures;
- `generated/`: PDF for LaTeX, SVG for editing, PNG for review;
- `figure_manifest.yaml`: evidence and provenance boundary for each figure;
- `build_all.py`: deterministic build entry point.

## Build

Run from this directory with the project environment:

```bash
python build_all.py
```

Empirical values must be loaded from files under `data/` or from a documented
experiment export.  Do not hard-code display-only metrics, ratios, p-values,
or confidence intervals in plotting scripts.

`fig03_lqcr_clean_replication.py` and `fig05_lqcr_stratified_effect.py` are
evidence-gated builders. They intentionally fail until their clean paired JSON
exports exist and pass provenance checks; they are therefore listed as pending
rather than executed by `build_all.py`.
