# Experiment configuration v2

This tree is the only configuration source for new experiments. It does not
inherit from legacy experiment files.

- `_base_/datasets`: dataset identity, split, pipeline, loader, evaluator.
- `_base_/models`: scientific model structure only.
- `_base_/schedules`: optimizer, scheduler, loop, selection hooks.
- `_base_/runtime`: credential-free runtime and module registration.
- `recipes`: executable scientific combinations with stable `config_id`.
- `ablations`: inference/training deltas added after the canonical recipes.
- `deployment`: compression recipes added after their training code is restored.

Training seeds, executors, work directories, parent checkpoint paths, and
tracker credentials are runtime inputs and must not be hard-coded here.

The current canonical KaryoFlow explicitly preserves the running D1 semantics:
random coupling and renewal enabled. Any future change to renewal is a new
versioned scientific configuration, never a silent default change.
