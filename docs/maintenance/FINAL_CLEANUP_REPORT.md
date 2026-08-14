# Repository Cleanup Completion Report

Date: 2026-08-15

Status: complete

Recovery ref: `refs/tags/archive/pre-cleanup-20260815`

Final ref: `refs/tags/cleanup/final-20260815`

## Outcome

The active branch now presents one KaryoFlow implementation and one experiment
configuration system. The tracked surface decreased from 2,812 files at the
recovery ref to 476 files in the final index, an 83.1% reduction. Removed source
remains available through Git; generated material was moved to ignored quarantine
instead of being permanently deleted.

The five primary working roots are:

- `ldmdet/`: the detector runtime, including rectified box transport, DPM++
  sampling, identity-consistent refinement, LQCR, and head distillation;
- `experiments/`: inherited MMEngine configurations, matrices, manifests, and the
  minimal train/test/diagnostic runners;
- `tools/`: stable configuration, experiment, evidence, data, analysis, and
  benchmark commands;
- `tests/`: active runtime, configuration, evidence, and reproducibility tests;
- `docs/`: current research boundaries, maintenance records, experiment lineage,
  and paper material.

## Major dispositions

- Replaced 1,777 tracked vendored MMDetection files with the pinned official
  `mmdet==3.3.0` dependency and a small project adapter.
- Converged experimental configuration on the parent/child MMEngine hierarchy
  under `experiments/configs` and retained the 261-run paper route matrix.
- Retired inactive project forks, obsolete research directions, stale generated
  figures, upstream repository boilerplate, broken Git links, host-specific queue
  scripts, and duplicate diagnostic runners.
- Preserved registered experiment evidence, checkpoint identities, seed semantics,
  six historical ETA JSON records, paper sources, and editable figure sources.
- Moved 734 MB of generated caches and local residues to
  `results/cleanup_quarantine_20260815/`. This directory is ignored and may be
  archived externally after a separate artifact-retention decision.

## Verification

The final gate passed:

- Ruff lint and format checks over 72 active Python files;
- 92 unit tests;
- construction of all 38 canonical model/config combinations;
- 4 generation-ablation stages, 76 inference variants, and 8 protocols;
- evidence database audit with 1,318 controlled and 166 diagnostic results;
- paper route audit with 46 groups and 261 expanded runs;
- unchanged experiment database SHA-256
  `df0a547629a4f3c004c140ed12a3b77cf5679da13ef24b03f039892f2d51a63d`;
- official MMDetection 3.3.0 import from `site-packages`;
- successful build of `karyoflow-0.1.0-py3-none-any.whl`;
- bootstrap shell syntax and Git whitespace checks.

The complete machine-readable gate is in
`docs/maintenance/baselines/p8_final_audit.json`.

## Recovery

To inspect the repository before cleanup without modifying the current branch:

```bash
git worktree add ../chromosome-kd-pre-cleanup refs/tags/archive/pre-cleanup-20260815
```

To compare the final state with the frozen baseline:

```bash
git diff --stat refs/tags/archive/pre-cleanup-20260815..refs/tags/cleanup/final-20260815
```

Do not reset the active branch merely to retrieve a historical file. Restore a
specific path from the recovery ref or use a separate worktree.

## Bounded exceptions and release checklist

- `.build_tmp/` and `.hf_cache/` contain root-owned regenerable cache files. They
  are ignored and do not participate in imports, configuration, tests, or evidence.
- Immutable internal evidence may retain historical runtime locations to preserve
  provenance. Paper-facing narrative and public source documentation do not rely on
  those locations. A public evidence export should use the existing neutral IDs.
- The Ross environment does not currently expose the `pre-commit` executable. The
  equivalent lint, format, test, and audit commands passed directly, and CI is
  configured to run them on a clean environment.
- The repository has no owner-selected `LICENSE`. Public release must wait until
  the owner selects and adds an appropriate license; this cleanup does not infer
  legal intent.
- Runtime artifacts under `work_dirs/` and `results/` are intentionally outside
  the public source boundary. Archive or publish only the evidence/checkpoint subset
  selected by the paper release policy.
