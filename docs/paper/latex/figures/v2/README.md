# Rebuilt paper figures

This directory is the vector-first source of truth for all figures used by the
current manuscript. Superseded figure assets have been removed from the
submission tree and remain recoverable from version control.

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

Only figures referenced by the manuscript are built. Exploratory analyses are
kept outside the submission figure directory until their evidence and wording
have been finalized.
