#!/usr/bin/env python3
"""Replace the blocked A4000 accuracy queue with a serial Ross queue."""

from __future__ import annotations
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiments.scheduler import StateStore, utc_now, validate_task

STATE = ROOT / 'work_dirs/v2/scheduler/state.json'
OUTPUT = (
    ROOT / 'experiments/configs/scheduler/d1_test_completion_serial_ross.yaml'
)


def ready_order(tasks: dict[str, dict]) -> list[str]:
    standard = [
        'd1-test-g0-seed42-a4000',
        'd1-test-g0-seed123-a4000',
        'd1-test-g1-seed42-a4000',
        'd1-test-g1-seed123-a4000',
        'd1-test-g1-seed789-a4000',
        'd1-test-g2-seed42-a4000',
        'd1-test-g2-seed123-a4000',
        'd1-test-g2-seed789-a4000',
        'd1-test-g3-seed42-a4000',
        'd1-test-g3-seed123-a4000',
    ]
    solver = [key for key in tasks if key.startswith('d1-test-solver-')]
    topk = [key for key in tasks if key.startswith('d1-test-topk-')]
    return standard + solver + topk


def convert(task: dict, previous: str | None, final: bool = False) -> dict:
    record = copy.deepcopy(task)
    old_id = record['task_id']
    record['task_id'] = old_id.removesuffix('-a4000') + '-ross'
    record['resource_id'] = 'ross:a6000:0'
    record['dependencies'] = [previous] if previous else []
    if final:
        record['dependencies'].append('d1-g3-seed789')
    record['attempt'] = 0
    record['created_at'] = utc_now()
    record['working_directory'] = '.'
    for key in (
        'worker',
        'process_status',
        'started_at',
        'finished_at',
        'launch_guard',
        'launch_probe',
    ):
        record.pop(key, None)
    variables = record.setdefault('variables', {})
    if 'result_root' in variables:
        variables['result_root'] = variables['result_root'].replace(
            '-a4000', '-ross'
        )
    sync = record.setdefault('postprocess', {}).setdefault('sync', {})
    sync.clear()
    sync['required'] = False
    return validate_task(record)


def main() -> None:
    store = StateStore(STATE)
    with store.locked() as state:
        old = {
            task['task_id']: task
            for task in state['pending']
            if task['task_id'].startswith('d1-test-')
        }
        final_id = 'd1-test-g3-seed789-a4000'
        required = set(ready_order(old)) | {final_id}
        missing = sorted(required - set(old))
        if missing:
            raise RuntimeError(f'missing pending test tasks: {missing}')

        state['pending'] = [
            task
            for task in state['pending']
            if not task['task_id'].startswith('d1-test-')
        ]
        converted: list[dict] = []
        previous = None
        for task_id in ready_order(old):
            record = convert(old[task_id], previous)
            converted.append(record)
            previous = record['task_id']
        final = convert(old[final_id], previous, final=True)
        converted.append(final)
        state['pending'].extend(converted)
        state.setdefault('migrations', []).append(
            {
                'migration_id': 'd1-test-a4000-to-ross-v1',
                'created_at': utc_now(),
                'reason': (
                    'workstation connectivity repeatedly failed; accuracy '
                    'tests are hardware-independent and Ross is idle'
                ),
                'removed_pending_tasks': sorted(old),
                'created_tasks': [task['task_id'] for task in converted],
                'serial_resource': 'ross:a6000:0',
            }
        )

    OUTPUT.write_text(
        yaml.safe_dump(
            {'schema_version': 1, 'tasks': converted}, sort_keys=False
        ),
        encoding='utf-8',
    )
    print(
        json.dumps(
            {
                'status': 'MIGRATED',
                'tasks': len(converted),
                'ready_now': len(converted) - 1,
                'g3_seed789_dependency': final['dependencies'],
                'output': str(OUTPUT.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
