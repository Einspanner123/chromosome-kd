import json
import sys
import time
from pathlib import Path

import yaml

from tools.experiments.scheduler import Scheduler, initial_state, validate_task


def write_config(tmp_path: Path) -> Path:
    config = {
        'schema_version': 1,
        'state_path': str(tmp_path / 'state.json'),
        'poll_interval_seconds': 0.01,
        'postprocess_workers': 1,
        'worker_script': str(
            Path(__file__).resolve().parents[2]
            / 'tools/experiments/scheduler_worker.py'
        ),
        'hosts': {
            'local': {
                'ssh_target': None,
                'project_root': str(tmp_path),
                'python': sys.executable,
                'worker_root': str(tmp_path / 'workers'),
            }
        },
        'resources': {'local:gpu0': {'host': 'local', 'gpu_id': 0}},
    }
    path = tmp_path / 'scheduler.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    return path


def task(task_id: str, command: list[str], dependencies=None) -> dict:
    return validate_task(
        {
            'task_id': task_id,
            'kind': 'test',
            'resource_id': 'local:gpu0',
            'command': command,
            'dependencies': dependencies or [],
        }
    )


def test_initial_state_has_exactly_three_queues():
    state = initial_state()
    assert set(state) == {
        'schema_version',
        'updated_at',
        'pending',
        'running',
        'completed',
    }


def test_scheduler_persists_and_recovers_task(tmp_path, monkeypatch):
    scheduler = Scheduler(write_config(tmp_path))
    monkeypatch.setattr(
        scheduler,
        'postprocess',
        lambda _task: {'status': 'succeeded', 'finished_at': 'now'},
    )
    marker = tmp_path / 'done.txt'
    first = task(
        'first',
        [
            sys.executable,
            '-c',
            f'from pathlib import Path; Path({str(marker)!r}).write_text("ok")',
        ],
    )
    second = task('second', [sys.executable, '-c', 'pass'], ['first'])
    with scheduler.store.locked() as state:
        state['pending'].extend([first, second])

    output = scheduler.tick()
    assert output['counts']['running'] == 1
    assert output['pending'] == ['second']
    for _ in range(100):
        time.sleep(0.02)
        output = scheduler.tick()
        if any(item['task_id'] == 'first' for item in output['completed']):
            break
    assert marker.read_text() == 'ok'
    assert output['counts']['completed'] == 1

    for _ in range(100):
        time.sleep(0.01)
        output = scheduler.tick()
        if output['completed'][0]['postprocess_status'] == 'succeeded':
            break
    output = scheduler.tick()
    assert output['running'][0]['task_id'] == 'second'

    reloaded = json.loads((tmp_path / 'state.json').read_text())
    assert len(reloaded['pending']) == 0
    assert len(reloaded['running']) == 1
    assert len(reloaded['completed']) == 1


def test_external_task_does_not_require_command():
    record = validate_task(
        {
            'task_id': 'adopted',
            'kind': 'external_training',
            'resource_id': 'local:gpu0',
            'external_monitor': {'process_pattern': 'resolved_config.py'},
        }
    )
    assert 'command' not in record
