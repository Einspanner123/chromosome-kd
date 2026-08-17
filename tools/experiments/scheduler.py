#!/usr/bin/env python3
"""Persistent three-queue scheduler for KaryoFlow experiments.

The coordinator owns three JSON arrays: ``pending``, ``running`` and
``completed``.  Worker processes are detached and independently observable,
so a coordinator restart does not lose or duplicate training.  Completed
tasks enter an asynchronous, idempotent post-processing pipeline that can pull
remote artifacts, import standard evidence, verify database registration and
run the global evidence audit.
"""

from __future__ import annotations
import argparse
import base64
import concurrent.futures
import datetime as dt
import fcntl
import json
import os
import shlex
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiments.registry import DB, sha256_file


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    temporary.replace(path)


def initial_state() -> dict:
    return {
        'schema_version': 1,
        'updated_at': utc_now(),
        'pending': [],
        'running': [],
        'completed': [],
    }


def validate_task(task: dict) -> dict:
    required = ('task_id', 'kind', 'resource_id')
    missing = [key for key in required if not task.get(key)]
    if missing:
        raise ValueError(f'task missing required fields: {missing}')
    if 'command' in task:
        command = task['command']
        if not isinstance(command, list) or not command:
            raise ValueError('task command must be a non-empty list')
        if not all(isinstance(item, str) and item for item in command):
            raise ValueError('task command items must be non-empty strings')
    elif 'external_monitor' not in task:
        raise ValueError('task requires command or external_monitor')
    task = json.loads(json.dumps(task))
    task.setdefault('dependencies', [])
    task.setdefault('working_directory', '.')
    task.setdefault('postprocess', {})
    task.setdefault('attempt', 0)
    task.setdefault('created_at', utc_now())
    return task


class StateStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + '.lock')

    @contextmanager
    def locked(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state = (
                json.loads(self.path.read_text(encoding='utf-8'))
                if self.path.is_file()
                else initial_state()
            )
            yield state
            state['updated_at'] = utc_now()
            atomic_json(self.path, state)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@contextmanager
def daemon_guard(path: Path):
    """Hold a process-lifetime lock and publish the active daemon PID."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + '.lock')
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                'another scheduler daemon is already active'
            ) from error
        atomic_json(path, {'pid': os.getpid(), 'started_at': utc_now()})
        try:
            yield
        finally:
            path.unlink(missing_ok=True)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class Scheduler:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(
            self.config_path.read_text(encoding='utf-8')
        )
        state_path = Path(self.config['state_path'])
        self.store = StateStore(
            state_path if state_path.is_absolute() else ROOT / state_path
        )
        daemon_path = Path(
            self.config.get(
                'daemon_pid_path', 'work_dirs/v2/scheduler/daemon.json'
            )
        )
        self.daemon_pid_path = (
            daemon_path if daemon_path.is_absolute() else ROOT / daemon_path
        )
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(self.config.get('postprocess_workers', 2))
        )
        self.futures: dict[str, concurrent.futures.Future] = {}
        self.database_lock = threading.Lock()

    def host_for(self, task: dict) -> dict:
        resource = self.config['resources'][task['resource_id']]
        return self.config['hosts'][resource['host']]

    def resource_for(self, task: dict) -> dict:
        return self.config['resources'][task['resource_id']]

    def host_command(
        self, host: dict, command: list[str], *, check: bool = False
    ) -> subprocess.CompletedProcess:
        if host.get('ssh_target'):
            final = [
                'ssh',
                '-o',
                'BatchMode=yes',
                '-o',
                'ConnectTimeout=10',
                host['ssh_target'],
                shlex.join(command),
            ]
        else:
            final = command
        return subprocess.run(
            final,
            cwd=ROOT,
            check=check,
            capture_output=True,
            text=True,
        )

    def render(self, value: str, task: dict, host: dict) -> str:
        resource = self.resource_for(task)
        values = {
            'project_root': host['project_root'],
            'python': host['python'],
            'gpu_id': resource['gpu_id'],
            'task_id': task['task_id'],
        }
        values.update(task.get('variables', {}))
        # Replace only scheduler-owned placeholders.  Command arguments may
        # legitimately contain JSON objects whose braces must remain literal.
        rendered = value
        for key, replacement in values.items():
            rendered = rendered.replace('{' + key + '}', str(replacement))
        return rendered

    def rendered_command(self, task: dict, host: dict) -> list[str]:
        return [self.render(item, task, host) for item in task['command']]

    def worker_command(self, host: dict, *items: str) -> list[str]:
        worker = host.get('worker_script', self.config['worker_script'])
        return [host['python'], worker, *items]

    def disk_launch_guard(self, task: dict) -> dict:
        """Return a persisted preflight result for the worker filesystem."""
        host = self.host_for(task)
        required_gib = float(host.get('min_free_disk_gib', 0))
        check_path = host.get('disk_check_path', host['project_root'])
        if required_gib <= 0:
            return {'status': 'disabled'}
        result = self.host_command(
            host,
            [
                host['python'],
                '-c',
                (
                    'import json, shutil; '
                    f'u=shutil.disk_usage({check_path!r}); '
                    'print(json.dumps({"free_bytes": u.free, '
                    '"total_bytes": u.total}))'
                ),
            ],
            check=False,
        )
        if result.returncode != 0:
            return {
                'status': 'probe_failed',
                'checked_at': utc_now(),
                'path': check_path,
                'error': result.stderr.strip(),
            }
        usage = json.loads(result.stdout)
        required_bytes = int(required_gib * 2**30)
        return {
            'status': (
                'passed'
                if usage['free_bytes'] >= required_bytes
                else 'blocked_insufficient_space'
            ),
            'checked_at': utc_now(),
            'path': check_path,
            'free_bytes': usage['free_bytes'],
            'required_bytes': required_bytes,
        }

    def start_task(self, task: dict) -> dict:
        host = self.host_for(task)
        spec = {
            'task_id': task['task_id'],
            'command': self.rendered_command(task, host),
            'cwd': str(Path(host['project_root']) / task['working_directory']),
            'env': task.get('env', {}),
        }
        encoded = base64.b64encode(
            json.dumps(spec, sort_keys=True).encode('utf-8')
        ).decode('ascii')
        result = self.host_command(
            host,
            self.worker_command(
                host,
                'start',
                '--root',
                host['worker_root'],
                '--spec-base64',
                encoded,
            ),
            check=False,
        )
        if result.returncode != 0:
            task['launch_probe'] = {
                'status': 'worker_start_failed_retryable',
                'returncode': result.returncode,
                'error': result.stderr.strip(),
                'checked_at': utc_now(),
            }
            return None
        payload = json.loads(result.stdout)
        if payload['status'] not in {'running', 'succeeded'}:
            raise RuntimeError(f'worker start failed: {payload}')
        task['attempt'] = int(task.get('attempt', 0)) + 1
        task['started_at'] = payload.get('started_at', utc_now())
        task['worker'] = payload
        task['process_status'] = payload['status']
        return task

    def worker_status(self, task: dict) -> dict:
        host = self.host_for(task)
        if task.get('external_monitor'):
            pattern = task['external_monitor']['process_pattern']
            result = self.host_command(
                host, ['pgrep', '-f', '--', pattern], check=False
            )
            return {
                'status': 'running' if result.returncode == 0 else 'succeeded',
                'task_id': task['task_id'],
                'returncode': 0,
                'finished_at': None if result.returncode == 0 else utc_now(),
            }
        result = self.host_command(
            host,
            self.worker_command(
                host,
                'status',
                '--root',
                host['worker_root'],
                '--task-id',
                task['task_id'],
            ),
            check=False,
        )
        if result.returncode != 0:
            return {
                'status': task.get('process_status', 'running'),
                'task_id': task['task_id'],
                'probe_status': 'ssh_status_failed_retryable',
                'probe_returncode': result.returncode,
                'probe_error': result.stderr.strip(),
                'checked_at': utc_now(),
            }
        return json.loads(result.stdout)

    def sync_artifacts(self, task: dict) -> None:
        sync = task.get('postprocess', {}).get('sync')
        if not sync or not sync.get('required', False):
            return
        host = self.host_for(task)
        source = self.render(sync['source'], task, host).rstrip('/') + '/'
        # The source template is rendered in the worker-host namespace, while
        # the destination always belongs to the Ross coordinator namespace.
        destination = Path(self.local_render(sync['destination'], task))
        if not destination.is_absolute():
            destination = ROOT / destination
        destination.mkdir(parents=True, exist_ok=True)
        if host.get('ssh_target'):
            source = f'{host["ssh_target"]}:{source}'
        command = ['rsync', '-a', '--partial']
        for pattern in sync.get('exclude', []):
            command.extend(['--exclude', pattern])
        command.extend([source, str(destination) + '/'])
        subprocess.run(command, cwd=ROOT, check=True)

    def local_render(self, value: str, task: dict) -> str:
        host = self.host_for(task)
        values = {
            'project_root': str(ROOT),
            'python': sys.executable,
            'gpu_id': self.resource_for(task)['gpu_id'],
            'task_id': task['task_id'],
        }
        values.update(task.get('variables', {}))
        return value.format(**values)

    def run_prepare(self, task: dict) -> None:
        command = task.get('postprocess', {}).get('prepare_command')
        if not command:
            return
        rendered = [self.local_render(item, task) for item in command]
        subprocess.run(rendered, cwd=ROOT, check=True)

    def import_evidence(self, task: dict) -> dict:
        evidence = task.get('postprocess', {}).get('evidence')
        if not evidence:
            return {'kind': 'none', 'registered': True}
        kind = evidence['kind']
        if evidence.get('path'):
            path = Path(self.local_render(evidence['path'], task))
            if not path.is_absolute():
                path = ROOT / path
        elif evidence.get('path_glob'):
            pattern = self.local_render(evidence['path_glob'], task)
            matches = sorted(ROOT.glob(pattern))
            if len(matches) != 1:
                raise RuntimeError(
                    'evidence path_glob must resolve exactly one file: '
                    f'{pattern!r} -> {len(matches)} matches'
                )
            path = matches[0]
        else:
            raise ValueError('evidence requires path or path_glob')
        if kind == 'training_completion':
            command = [
                sys.executable,
                str(ROOT / 'tools/experiments/import_training_completion.py'),
                str(path),
                '--register-if-missing',
            ]
            for name, flag in (
                ('parent_train_run_id', '--parent-train-run-id'),
                ('executor', '--executor'),
                ('tracker_run_name', '--tracker-run-name'),
            ):
                value = evidence.get(name)
                if value:
                    command.extend([flag, self.local_render(str(value), task)])
            if evidence.get('allow_restored_canonical_config'):
                command.append('--allow-restored-canonical-config')
            if evidence.get('allow_registered_provenance_rebind'):
                command.append('--allow-registered-provenance-rebind')
            subprocess.run(command, cwd=ROOT, check=True)
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.verify_training_registration(payload, path)
            return {
                'kind': kind,
                'registered': True,
                'train_run_id': payload['train_run_id'],
            }
        if kind == 'run_evidence':
            subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'tools.experiment_db.validate_run_evidence',
                    str(path),
                    '--verify-files',
                ],
                cwd=ROOT,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'tools.experiment_db.import_run_evidence',
                    str(path),
                ],
                cwd=ROOT,
                check=True,
            )
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.verify_eval_registration(payload)
            return {
                'kind': kind,
                'registered': True,
                'record_id': payload['record_id'],
            }
        raise ValueError(f'unsupported evidence kind: {kind}')

    def verify_training_registration(self, payload: dict, path: Path) -> None:
        connection = sqlite3.connect(DB)
        try:
            row = connection.execute(
                'SELECT status FROM train_run_registry WHERE train_run_id=?',
                (payload['train_run_id'],),
            ).fetchone()
            selected = connection.execute(
                'SELECT checkpoint_sha256 FROM selected_checkpoint '
                'WHERE train_run_id=?',
                (payload['train_run_id'],),
            ).fetchone()
            artifact = connection.execute(
                'SELECT 1 FROM evidence_artifact WHERE path=? AND sha256=?',
                (relative_to_root(path), sha256_file(path)),
            ).fetchone()
        finally:
            connection.close()
        expected_sha = payload['checkpoint']['sha256']
        if row != ('trained',) or selected != (expected_sha,) or not artifact:
            raise RuntimeError('training evidence was not fully registered')

    def verify_eval_registration(self, payload: dict) -> None:
        connection = sqlite3.connect(DB)
        try:
            row = connection.execute(
                'SELECT status FROM eval_run_registry WHERE eval_run_id=?',
                (payload['record_id'],),
            ).fetchone()
        finally:
            connection.close()
        if row != (payload['status'],):
            raise RuntimeError('evaluation evidence was not fully registered')

    def refresh_and_commit_evidence(self, task: dict) -> dict:
        """Refresh matrix derivatives and commit only controlled evidence files."""
        commands = (
            [
                sys.executable,
                '-m',
                'tools.experiment_db.build_experiment_route_matrix',
            ],
            [
                sys.executable,
                '-m',
                'tools.experiment_db.audit_experiment_route_matrix',
            ],
        )
        environment = os.environ.copy()
        environment['KARYOFLOW_COMPLETING_TASK_ID'] = task['task_id']
        for command in commands:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)

        allowed = {
            'docs/experiments/PAPER_EXPERIMENT_ROUTE_MATRIX.md',
            'experiments/manifests/paper_experiment_route_matrix.yaml',
            'tools/experiment_db/experiments.db',
            'tools/experiment_db/exports/paper_experiment_route_matrix.csv',
        }
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--untracked-files=no'],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        changed = {
            line[3:].strip()
            for line in status.stdout.splitlines()
            if line.strip()
        }
        unexpected = sorted(changed - allowed)
        if unexpected:
            raise RuntimeError(
                'refusing evidence commit with unexpected tracked changes: '
                + ', '.join(unexpected)
            )
        if not changed:
            return {'status': 'clean', 'commit': self.git_head()}
        subprocess.run(
            ['git', 'add', '--', *sorted(changed)], cwd=ROOT, check=True
        )
        staged = subprocess.run(
            ['git', 'diff', '--cached', '--quiet'], cwd=ROOT, check=False
        )
        if staged.returncode == 1:
            subprocess.run(
                [
                    'git',
                    'commit',
                    '-m',
                    f'record evidence for {task["task_id"]}',
                ],
                cwd=ROOT,
                check=True,
            )
        elif staged.returncode != 0:
            raise RuntimeError('git diff --cached failed')
        return {'status': 'committed', 'commit': self.git_head()}

    @staticmethod
    def git_head() -> str:
        return subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def postprocess(self, task: dict) -> dict:
        try:
            self.sync_artifacts(task)
            self.run_prepare(task)
            # Transfers and evidence preparation run concurrently.  SQLite
            # import plus the global audit are serialized to avoid lock races.
            with self.database_lock:
                evidence = self.import_evidence(task)
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / 'tools/experiment_db/audit_evidence_db.py'),
                    ],
                    cwd=ROOT,
                    check=True,
                )
                matrix_commit = self.refresh_and_commit_evidence(task)
            return {
                'status': 'succeeded',
                'finished_at': utc_now(),
                'evidence': evidence,
                'matrix_commit': matrix_commit,
            }
        except Exception as error:  # persisted for operator retry
            return {
                'status': 'failed',
                'finished_at': utc_now(),
                'error': f'{type(error).__name__}: {error}',
            }

    def dependencies_satisfied(self, task: dict, state: dict) -> bool:
        completed = {item['task_id']: item for item in state['completed']}
        for dependency in task.get('dependencies', []):
            record = completed.get(dependency)
            if not record or record.get('process_status') != 'succeeded':
                return False
            if record.get('postprocess', {}).get('status') not in {
                None,
                'succeeded',
            }:
                return False
        return True

    def tick(self) -> dict:
        with self.store.locked() as state:
            for task in list(state['running']):
                status = self.worker_status(task)
                task['worker'] = status
                task['process_status'] = status['status']
                if status['status'] in {'succeeded', 'failed', 'lost'}:
                    task['finished_at'] = status.get('finished_at', utc_now())
                    task['postprocess'] = {
                        **task.get('postprocess', {}),
                        'status': 'pending',
                    }
                    state['running'].remove(task)
                    state['completed'].append(task)

            for task_id, future in list(self.futures.items()):
                if not future.done():
                    continue
                record = next(
                    item
                    for item in state['completed']
                    if item['task_id'] == task_id
                )
                record['postprocess'] = {
                    **record.get('postprocess', {}),
                    **future.result(),
                }
                del self.futures[task_id]

            for task in state['completed']:
                post = task.get('postprocess', {})
                if (
                    task.get('process_status') == 'succeeded'
                    and post.get('status') in {'pending', 'retry'}
                    and task['task_id'] not in self.futures
                ):
                    post['status'] = 'running'
                    post['started_at'] = utc_now()
                    self.futures[task['task_id']] = self.pool.submit(
                        self.postprocess, json.loads(json.dumps(task))
                    )

            used = {task['resource_id'] for task in state['running']}
            for task in list(state['pending']):
                if task['resource_id'] in used:
                    continue
                if not self.dependencies_satisfied(task, state):
                    continue
                guard = self.disk_launch_guard(task)
                task['launch_guard'] = guard
                if guard['status'] in {
                    'blocked_insufficient_space',
                    'probe_failed',
                }:
                    continue
                started = self.start_task(task)
                if started is None:
                    continue
                state['pending'].remove(task)
                state['running'].append(started)
                used.add(task['resource_id'])
            return summary(state)

    def recover_postprocess(self) -> None:
        with self.store.locked() as state:
            for task in state['completed']:
                if task.get('postprocess', {}).get('status') == 'running':
                    task['postprocess']['status'] = 'retry'


def relative_to_root(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def all_ids(state: dict) -> set[str]:
    return {
        task['task_id']
        for queue in ('pending', 'running', 'completed')
        for task in state[queue]
    }


def summary(state: dict) -> dict:
    return {
        'updated_at': state['updated_at'],
        'counts': {
            queue: len(state[queue])
            for queue in ('pending', 'running', 'completed')
        },
        'pending': [task['task_id'] for task in state['pending']],
        'running': [
            {
                'task_id': task['task_id'],
                'resource_id': task['resource_id'],
                'status': task.get('process_status'),
            }
            for task in state['running']
        ],
        'completed': [
            {
                'task_id': task['task_id'],
                'process_status': task.get('process_status'),
                'postprocess_status': task.get('postprocess', {}).get(
                    'status'
                ),
            }
            for task in state['completed']
        ],
    }


def load_tasks(path: Path) -> list[dict]:
    text = path.read_text(encoding='utf-8')
    payload = (
        yaml.safe_load(text)
        if path.suffix.lower() in {'.yaml', '.yml'}
        else json.loads(text)
    )
    if isinstance(payload, dict) and 'tasks' in payload:
        payload = payload['tasks']
    records = payload if isinstance(payload, list) else [payload]
    return [validate_task(record) for record in records]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=Path,
        default=ROOT / 'experiments/configs/scheduler/ross_scheduler.yaml',
    )
    subparsers = parser.add_subparsers(dest='action', required=True)
    subparsers.add_parser('init')
    enqueue = subparsers.add_parser('enqueue')
    enqueue.add_argument('tasks', type=Path, nargs='+')
    adopt = subparsers.add_parser('adopt')
    adopt.add_argument('tasks', type=Path, nargs='+')
    load_plan = subparsers.add_parser('load-plan')
    load_plan.add_argument('tasks', type=Path, nargs='+')
    subparsers.add_parser('tick')
    daemon = subparsers.add_parser('daemon')
    daemon.add_argument('--once', action='store_true')
    subparsers.add_parser('status')
    retry = subparsers.add_parser('retry-postprocess')
    retry.add_argument('task_id')
    retry_task = subparsers.add_parser('retry-task')
    retry_task.add_argument('task_id')
    args = parser.parse_args()
    scheduler = Scheduler(args.config)

    if args.action == 'init':
        with scheduler.store.locked() as state:
            output = summary(state)
    elif args.action in {'enqueue', 'adopt', 'load-plan'}:
        tasks = [task for path in args.tasks for task in load_tasks(path)]
        with scheduler.store.locked() as state:
            existing = all_ids(state)
            duplicates = [
                task['task_id']
                for task in tasks
                if task['task_id'] in existing
            ]
            if duplicates:
                raise ValueError(f'duplicate task IDs: {duplicates}')
            for task in tasks:
                queue = (
                    'running'
                    if args.action == 'adopt'
                    or (
                        args.action == 'load-plan'
                        and 'external_monitor' in task
                    )
                    else 'pending'
                )
                if queue == 'running':
                    if 'external_monitor' not in task:
                        raise ValueError(
                            'adopted task requires external_monitor'
                        )
                    task['started_at'] = utc_now()
                    task['process_status'] = 'running'
                state[queue].append(task)
            output = summary(state)
    elif args.action == 'tick':
        output = scheduler.tick()
    elif args.action == 'status':
        with scheduler.store.locked() as state:
            output = summary(state)
    elif args.action == 'retry-postprocess':
        with scheduler.store.locked() as state:
            record = next(
                task
                for task in state['completed']
                if task['task_id'] == args.task_id
            )
            record['postprocess']['status'] = 'retry'
            output = summary(state)
    elif args.action == 'retry-task':
        with scheduler.store.locked() as state:
            record = next(
                task
                for task in state['completed']
                if task['task_id'] == args.task_id
            )
            if record.get('process_status') != 'failed':
                raise ValueError('retry-task requires a failed process')
            host = scheduler.host_for(record)
            prepared = scheduler.host_command(
                host,
                scheduler.worker_command(
                    host,
                    'prepare-retry',
                    '--root',
                    host['worker_root'],
                    '--task-id',
                    record['task_id'],
                ),
                check=True,
            )
            retry_history = record.setdefault('retry_history', [])
            retry_history.append(
                {
                    'prepared_at': utc_now(),
                    'previous_attempt': record.get('attempt', 0),
                    'worker_archive': json.loads(prepared.stdout),
                }
            )
            for key in (
                'worker',
                'process_status',
                'started_at',
                'finished_at',
                'launch_guard',
                'launch_probe',
            ):
                record.pop(key, None)
            record.setdefault('postprocess', {})['status'] = 'pending'
            state['completed'].remove(record)
            state['pending'].insert(0, record)
            output = summary(state)
    else:
        with daemon_guard(scheduler.daemon_pid_path):
            scheduler.recover_postprocess()
            stop = False

            def request_stop(_signum, _frame):
                nonlocal stop
                stop = True

            output = scheduler.tick()
            signal.signal(signal.SIGTERM, request_stop)
            signal.signal(signal.SIGINT, request_stop)
            while not args.once and not stop:
                time.sleep(
                    float(scheduler.config.get('poll_interval_seconds', 30))
                )
                output = scheduler.tick()
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
