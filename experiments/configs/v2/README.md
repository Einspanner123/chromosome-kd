# Experiment configuration v2

V2 separates reusable scientific methods from dataset-specific experiment
matrices while preserving native MMEngine execution.

- `_base_/datasets`: dataset identity, split hashes, pipelines, and evaluators.
- `_base_/models`: reusable detector structures.
- `_base_/schedules`: optimization and model-selection policy.
- `_base_/runtime`: credential-free MMEngine runtime.
- `methods`: dataset-independent model plus schedule definitions.
- `matrices`: dataset, methods, training seeds, and tracking namespace.
- `manifests`: generated catalog of resolved matrix-method combinations.
- `ablations`: registered single-variable interventions (future).
- `deployment`: compression and dynamic inference definitions (future).

The YAML matrix is an orchestration document, not an MMEngine config. The
resolver combines one dataset, one method, and the shared runtime, then writes
a flattened `resolved_config.py`. MMEngine consumes that file normally through
`Config.fromfile` and `Runner.from_cfg`.

Preview a complete matrix without writing files:

```bash
python tools/experiments/launch_v2.py \
  --matrix experiments/configs/v2/matrices/d1_inhouse1700.yaml --plan
```

Materialize one native MMEngine configuration:

```bash
python tools/experiments/launch_v2.py \
  --matrix experiments/configs/v2/matrices/d1_inhouse1700.yaml \
  --method karyoflow --seed 42 --resolve-only
```

Add `--launch --gpu-id 0` to train. A paired child such as LQCR additionally
requires `--parent-checkpoint` from the same training seed.

Seeds, executors, work directories, parent checkpoint paths, and tracker
credentials are runtime inputs. They are never embedded in method definitions.
The scientific hash is computed after matrix resolution but excludes the seed
and machine-specific runtime state.
