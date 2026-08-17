#!/usr/bin/env python3
"""Refresh the paper route matrix after verified scheduler state changes."""

from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / 'work_dirs/v2/scheduler/state.json'
RUNTIME = ROOT / 'work_dirs/v2/scheduler'
LOCK = RUNTIME / 'route_progress_daemon.lock'
PID = RUNTIME / 'route_progress_daemon.pid'


def relevant_snapshot() -> dict:
    try:
        state = json.loads(STATE.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {'state': 'unavailable'}
    snapshot: dict[str, list[dict]] = {}
    for queue in ('pending', 'running', 'completed'):
        records = []
        for task in state.get(queue, []):
            task_id = task.get('task_id', '')
            if (
                task_id.startswith('d1-test-') and task_id.endswith('-ross')
            ) or task_id == 'd1-g3-seed789':
                records.append(
                    {
                        'task_id': task_id,
                        'process_status': task.get('process_status'),
                        'postprocess_status': task.get('postprocess_status'),
                    }
                )
        snapshot[queue] = sorted(records, key=lambda item: item['task_id'])
    return snapshot


def signature(snapshot: dict) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def refresh() -> None:
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
    for command in commands:
        last = None
        for attempt in range(5):
            last = subprocess.run(command, cwd=ROOT, text=True)
            if last.returncode == 0:
                break
            time.sleep(2**attempt)
        if last is None or last.returncode != 0:
            raise RuntimeError(f'matrix refresh command failed: {command}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=float, default=30.0)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()

    RUNTIME.mkdir(parents=True, exist_ok=True)
    with LOCK.open('w', encoding='utf-8') as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('route progress daemon is already running')
        PID.write_text(f'{os.getpid()}\n', encoding='utf-8')
        previous = ''
        try:
            while True:
                snapshot = relevant_snapshot()
                current = signature(snapshot)
                if current != previous:
                    refresh()
                    previous = current
                    print(
                        json.dumps(
                            {
                                'event': 'matrix_refreshed',
                                'signature': current,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if args.once:
                    break
                active = snapshot.get('pending', []) or snapshot.get(
                    'running', []
                )
                if not active:
                    print(json.dumps({'event': 'queue_complete'}), flush=True)
                    break
                time.sleep(args.interval)
        finally:
            PID.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
