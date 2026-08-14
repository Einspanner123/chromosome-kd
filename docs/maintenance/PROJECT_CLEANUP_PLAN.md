# Repository Cleanup and Recovery Plan

Status: completed on 2026-08-15

Plan ID: `repo-cleanup-20260815`

Baseline commit: `323da0beaf1a93441915e81dd90f8039012c7f3c`

Recovery tag: `archive/pre-cleanup-20260815`

Recovery branch: `archive/pre-cleanup-20260815`

## Objective

Reduce the active repository to the smallest understandable and reproducible
KaryoFlow codebase while preserving every paper-facing result, checkpoint
identity, configuration lineage, and experiment record needed for audit or
reproduction.

The active branch represents the current scientific truth. Historical research
directions remain recoverable through Git and the external artifact archive;
they are not kept beside active implementation merely for convenience.

## Non-negotiable invariants

1. No paper-facing metric may lose its controlled-result ID or evidence source.
2. No paper-facing checkpoint may lose its SHA-256 identity or parent/child lineage.
3. Training-seed and fixed-checkpoint inference-seed replication remain distinct.
4. Historical configurations leave the active tree only after compatibility is
   classified as `EXACT`, `STRUCTURAL_ONLY`, or explicitly non-reproducible.
5. Tracked material is removed only after the recovery tag and branch exist.
6. Ignored runtime material is inventoried and quarantined before physical deletion.
7. `work_dirs/` and `results/` are not physically deleted during source cleanup.
8. Permanent artifact deletion requires a separate explicit decision after audits.
9. No active file may contain a personal absolute path or server name.
10. Every phase ends with a commit and state-file update.

## Target repository boundary

The final repository exposes five primary working directories:

- `ldmdet/`: production runtime implementation only.
- `experiments/`: canonical configurations, matrices, manifests, and runners.
- `tools/`: stable configuration, experiment, evidence, analysis, data, and
  benchmark CLIs.
- `tests/`: tests for active runtime, configurations, evidence, and data protocols.
- `docs/`: current architecture, experiment protocol, evidence, and paper.

Infrastructure may additionally include `third_party/` if a local MMDetection
fork remains necessary and an ignored `artifacts` link outside the repository.
Root-level `analysis/`, `projects/`, `results/`, and `work_dirs/` are not part of
the final source layout.

## Classification vocabulary

Every relevant path receives exactly one disposition:

- `KEEP`: necessary in its current active role.
- `MOVE`: necessary, but belongs under a different active boundary.
- `GIT_DELETE`: tracked history not required by the active branch and recoverable
  from the recovery tag.
- `ARTIFACT_ARCHIVE`: generated evidence that must be moved to the external store.
- `REGENERATE`: cache or derived output whose generating command is known.
- `BLOCKED`: a runtime, paper, or evidence dependency remains unresolved.

## Required evidence for deletion

A tracked path becomes `GIT_DELETE` only when all applicable checks pass:

- unreachable from active configurations and public imports;
- not required to strict-load a paper checkpoint;
- not required at its active path by paper source, evidence, or route matrices;
- any scientific conclusion is captured by canonical evidence or the falsified
  direction summary;
- associated tests are obsolete or migrated to an active equivalent.

Ignored artifacts become `ARTIFACT_ARCHIVE` only after recording path, size,
identity, experiment relationship, and destination.

## Execution phases and gates

### P0 — Freeze and baseline

- Create recovery tag and branch.
- Record tracked blobs, artifact summaries, Git state, route-matrix identity,
  evidence audit, and paper claim references.
- Confirm no active jobs would be disturbed.

Gate: recovery refs resolve and the baseline manifest is reproducible.

### P1 — Reproducible caches and empty paths

- Remove Python, pytest, Ruff, build, model-cache, LaTeX, and other generated caches.
- Remove empty directories and broken local-development links where safe.
- Do not move training/evaluation artifacts.

Gate: imports, config audit, evidence audit, and selected tests pass.

### P2 — Canonical experiment configuration

- Promote `experiments/configs` to the only active configuration system.
- Remove the temporary `v2` namespace after legacy siblings are retired.
- Keep bases, methods, matrices, ablations, deployment protocols, and manifests.
- Keep historical checkpoint identities under compatibility manifests, not runnable
  methods.
- Restore and validate head-distillation training identity before retiring its source.

Gate: every paper experiment is in the route matrix; runnable configs parse with
MMEngine; paper checkpoints have explicit compatibility classifications.

### P3 — Runtime and test convergence

- Retain only RF/DPM++ generation, identity-consistent refinement, LQCR, head
  distillation, shared heads, matching, losses, data structures, and integration.
- Move package-local tests and benchmark tools to canonical roots.
- Remove implementation and parameters used solely by disproved directions.

Gate: configs build, checkpoint compatibility passes, and mainline tests pass.

### P4 — Tools and evidence convergence

- Replace one-off `add_*`, `fix_*`, `migrate_*`, and host-specific scripts with
  stable validate/import/aggregate/audit/export commands.
- Separate immutable evidence from the mutable SQLite registry.
- Prove the paper export can be rebuilt from registered evidence.

Gate: a clean database rebuild produces the same controlled paper results.

### P5 — Runtime artifact relocation

- Move `work_dirs/`, `results/`, and generated analysis output to an external,
  content-addressed artifact root on the same filesystem.
- Preserve checkpoint, prediction, log, config, dataset, and evaluation identities.
- Deduplicate only by verified content hash.

Gate: paper results/checkpoints resolve by artifact URI and recorded hash; quarantine
remains intact.

### P6 — Research and documentation convergence

- Remove `projects/` from the active branch.
- Move reusable analysis code under `tools/`; archive generated data.
- Replace the research-note forest with current architecture/protocol/evidence
  documents and one falsified-direction index.
- Keep paper source, official template, editable figure sources, and required output.

Gate: docs contain no stale metric, internal path, server name, or ambiguous status.

### P7 — MMDetection dependency boundary

- Diff vendored `mmdet/` against the pinned upstream release.
- Prefer local adapters plus a pinned package.
- If a fork is unavoidable, use an independently pinned dependency.

Gate: clean install, training/inference smoke tests, and checkpoint loading pass.

### P8 — Final audit and release checkpoint

- Run structural, configuration, evidence, dataset, test, and paper audits.
- Generate before/after inventory and final report.
- Create a final cleanup tag only after all gates pass.

## Commit discipline

Each phase receives a dedicated commit using one of these prefixes:

- `chore(cleanup-baseline):`
- `chore(cleanup-cache):`
- `refactor(configs):`
- `refactor(ldmdet):`
- `refactor(evidence):`
- `chore(artifacts):`
- `docs(cleanup):`
- `refactor(deps):`
- `chore(cleanup-final):`

The state file is updated in the same commit. Failed gates do not advance phase
status. Unrelated changes are never folded into cleanup commits.

## Rollback

- Complete source rollback: checkout `archive/pre-cleanup-20260815`.
- Phase rollback: revert that phase commit.
- File rollback: restore it from the recovery tag.
- Artifact rollback: move its quarantine path to the recorded original location and
  verify size/hash.

The archive branch and tag must never be deleted by this cleanup.
