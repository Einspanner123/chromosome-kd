# Ross experiment scheduler

`tools/experiments/scheduler.py` is the single coordinator for long-running
training and evaluation jobs.  Its state is persisted under
`work_dirs/v2/scheduler/state.json` and contains exactly three ordered arrays:

1. `pending`: tasks waiting for dependencies and an assigned GPU resource;
2. `running`: tasks that have been launched or adopted from an existing run;
3. `completed`: process-finished tasks, including their asynchronous
   synchronization and evidence-import status.

The running queue is not treated as FIFO.  Every task is polled independently
by task ID, so any task can finish first.  State updates and worker receipts are
written atomically.  Restarting the coordinator does not restart a live worker
process; incomplete post-processing is changed to `retry` and resumed.

## Evidence pipeline

Training and evaluation do not write ad-hoc database rows through the
scheduler.  The scheduler reuses the repository evidence contracts:

- training: `training_completion.json` →
  `tools/experiments/import_training_completion.py`;
- evaluation: `run_evidence.json` → schema/file validation →
  `tools/experiment_db/import_run_evidence.py`.

After import, the scheduler queries the relevant registry row and artifact or
checkpoint identity.  It then runs `audit_evidence_db.py`.  Imports are
idempotent.  Remote result transfers and evidence preparation may run in
parallel, while SQLite import and the global audit are serialized.

Externally launched legacy runs can be adopted with a process-pattern monitor.
After their work directory is synchronized,
`finalize_external_training.py` emits the same standard completion artifact;
the normal importer then handles registration.

## Commands

```bash
PYTHONPATH=. python tools/experiments/scheduler.py init
PYTHONPATH=. python tools/experiments/scheduler.py load-plan experiments/configs/scheduler/d1_generation_g0_g3.yaml
PYTHONPATH=. python tools/experiments/scheduler.py enqueue task1.json task2.json
PYTHONPATH=. python tools/experiments/scheduler.py adopt active_task.json
PYTHONPATH=. python tools/experiments/scheduler.py status
PYTHONPATH=. python tools/experiments/scheduler.py tick
PYTHONPATH=. python tools/experiments/scheduler.py daemon
```

The coordinator is deployed as a detached process in the project `chromo`
environment.  Worker control uses `scheduler_worker.py`; remote copies live in
`/tmp` so existing clean training worktrees are not made dirty.

## Task record

Each JSON task contains:

- immutable `task_id`, `kind`, and `resource_id`;
- argv-form `command` (never a shell string) or an `external_monitor`;
- explicit `dependencies`;
- optional `postprocess.sync`, `prepare_command`, and `evidence` sections;
- template variables for work directory, train-run ID, method, and seed.

`resource_id` is exclusive while a task is in the running queue.  Accuracy
jobs may use any registered GPU; latency jobs must be assigned to the Ross
A6000 resource under the paper timing protocol.
