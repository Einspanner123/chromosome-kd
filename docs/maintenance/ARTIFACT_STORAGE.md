# Runtime artifact storage policy

The repository keeps source code and small immutable evidence in Git. Large
runtime artifacts remain local and are referenced by content hashes.

## Stable local roots

| Root | Role | Git policy | Relocation policy |
|---|---|---|---|
| `work_dirs/` | training checkpoints, resolved configs, framework logs, tracker exports | ignored | keep in place while any evidence or compatibility record references it |
| `results/` | predictions, evaluation logs, benchmark outputs, archives | ignored except explicitly force-added summaries | keep in place; promote only small immutable summaries into `tools/experiment_db/evidence_sources/` |
| `analysis/` | generated visual diagnostics and legacy analysis outputs | generated binaries ignored | source scripts move to `tools/analysis/`; outputs remain local |
| `experiments/analysis/*_cache/` | reproducible intermediate caches | ignored | may be regenerated, but do not delete during evidence cleanup without a separate cache manifest |

The roots are intentionally not renamed to a new `artifacts/` directory. The
current evidence database contains path-qualified provenance, and moving about
835 GB of training artifacts would add I/O risk without improving scientific
identity. Stable content hashes, not directory names, are the authority.

## Promotion rule

A large output becomes paper evidence only through this sequence:

1. keep the original prediction/log/checkpoint in its runtime root;
2. hash it in a run-evidence record;
3. independently recompute the metric when applicable;
4. promote a Git-sized immutable JSON summary to
   `tools/experiment_db/evidence_sources/`;
5. transactionally import the run record and pass the database audit.

Moving or copying a file must not create a new experimental replicate. Physical
aliases with equal SHA-256 values share one scientific identity.

## Recovery and deletion

- Tracked sources are recoverable from the phase tags.
- `work_dirs/` and `results/` are not recoverable from Git and are never
  permanently deleted by repository cleanup.
- Regenerable caches may be removed only after recording their paths and sizes.
- Root-owned `.build_tmp/` and `.hf_cache/` residues are ignored and have no
  source or evidence role.

The frozen inventory for this cleanup is
`docs/maintenance/baselines/p5_artifact_inventory.json`.
