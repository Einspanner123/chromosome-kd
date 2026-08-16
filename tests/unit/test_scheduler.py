import json
import sys
import time
from pathlib import Path

import yaml

import tools.experiments.scheduler as scheduler_module
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


def test_remote_sync_renders_destination_in_coordinator_namespace(
    tmp_path, monkeypatch
):
    config_path = write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config['hosts']['local']['ssh_target'] = 'worker@example'
    config['hosts']['local']['project_root'] = '/remote/project'
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(scheduler_module, 'ROOT', tmp_path)
    scheduler = Scheduler(config_path)
    record = task('remote-sync', [sys.executable, '-c', 'pass'])
    record['variables'] = {'work_dir': 'work_dirs/example'}
    record['postprocess'] = {
        'sync': {
            'required': True,
            'source': '{project_root}/{work_dir}',
            'destination': '{project_root}/{work_dir}',
        }
    }
    calls = []

    def capture(command, **_kwargs):
        calls.append(command)

    monkeypatch.setattr(scheduler_module.subprocess, 'run', capture)
    scheduler.sync_artifacts(record)
    assert calls[0][-2] == 'worker@example:/remote/project/work_dirs/example/'
    assert calls[0][-1] == str(tmp_path / 'work_dirs/example') + '/'


def test_disk_guard_blocks_launch_and_persists_reason(tmp_path):
    config_path = write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config['hosts']['local']['disk_check_path'] = str(tmp_path)
    config['hosts']['local']['min_free_disk_gib'] = 10**9
    config_path.write_text(yaml.safe_dump(config))
    scheduler = Scheduler(config_path)
    blocked = task('blocked', [sys.executable, '-c', 'pass'])
    with scheduler.store.locked() as state:
        state['pending'].append(blocked)

    output = scheduler.tick()

    assert output['pending'] == ['blocked']
    assert output['counts']['running'] == 0
    state = json.loads((tmp_path / 'state.json').read_text())
    guard = state['pending'][0]['launch_guard']
    assert guard['status'] == 'blocked_insufficient_space'
    assert guard['free_bytes'] < guard['required_bytes']


def test_disk_guard_can_be_disabled(tmp_path):
    scheduler = Scheduler(write_config(tmp_path))
    record = task('unguarded', [sys.executable, '-c', 'pass'])

    assert scheduler.disk_launch_guard(record) == {'status': 'disabled'}
