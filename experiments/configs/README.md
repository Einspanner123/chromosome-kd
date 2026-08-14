# Canonical experiment configuration

The configuration system separates reusable scientific methods from dataset-specific experiment
matrices while preserving native MMEngine execution.

- `_base_/datasets`: dataset identity, split hashes, pipelines, and evaluators.
- `_base_/models`: reusable detector structures.
- `_base_/schedules`: optimization and model-selection policy.
- `_base_/runtime`: credential-free MMEngine runtime.
- `methods`: dataset-independent model plus schedule definitions.
- `matrices`: dataset, methods, training seeds, and tracking namespace.
- `../manifests`: generated catalog, compatibility records, and paper route matrix.
- `ablations`: split-aware inference or selection protocols; these never
  masquerade as independently trained methods.
- `deployment`: hardware-locked efficiency and dynamic-compute protocols.

The active configuration layers are:

1. `methods/*.py`: dataset-independent trainable scientific identities.
2. `matrices/*.yaml`: dataset, tracking namespace, methods, and training seeds.
3. `ablations/*.yaml`: fixed-checkpoint inference grids and validation-only
   selection grids.
4. `../manifests/paper_experiment_route_matrix.yaml`: the authoritative mapping
   from every required experiment to its method, matrix, protocol, output, and
   database family.

Files containing `_legacy` are immutable compatibility snapshots for existing
checkpoints. New training must use a canonical method without that suffix.

The YAML matrix is an orchestration document, not an MMEngine config. The
resolver combines one dataset, one method, and the shared runtime, then writes
a flattened `resolved_config.py`. MMEngine consumes that file normally through
`Config.fromfile` and `Runner.from_cfg`.

Preview a complete matrix without writing files:

```bash
python tools/experiments/launch.py \
  --matrix experiments/configs/matrices/d1_inhouse1700.yaml --plan
```

Materialize one native MMEngine configuration:

```bash
python tools/experiments/launch.py \
  --matrix experiments/configs/matrices/d1_inhouse1700.yaml \
  --method karyoflow --seed 42 --resolve-only
```

Add `--launch --gpu-id 0` to train. A paired child such as LQCR additionally
requires `--parent-checkpoint` from the same training seed.

Seeds, executors, work directories, parent checkpoint paths, and tracker
credentials are runtime inputs. They are never embedded in method definitions.
The scientific hash is computed after matrix resolution but excludes the seed
and machine-specific runtime state.
