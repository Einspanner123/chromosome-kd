#!/usr/bin/env python3
"""Minimal detached task worker used by the Ross experiment scheduler."""

from __future__ import annotations
import argparse
import base64
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    temporary.replace(path)


def task_dir(root: Path, task_id: str) -> Path:
    if not task_id or any(
        char
        not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._'
        for char in task_id
    ):
        raise ValueError(f'invalid task ID: {task_id!r}')
    return root.resolve() / task_id


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def decode_spec(encoded: str) -> dict:
    payload = json.loads(base64.b64decode(encoded).decode('utf-8'))
    if not isinstance(payload.get('command'), list) or not payload['command']:
        raise ValueError('worker spec requires a non-empty command list')
    if not all(isinstance(item, str) and item for item in payload['command']):
        raise ValueError('worker command items must be non-empty strings')
    if not isinstance(payload.get('cwd'), str) or not payload['cwd']:
        raise ValueError('worker spec requires cwd')
    return payload


def start(root: Path, encoded: str) -> dict:
    spec = decode_spec(encoded)
    directory = task_dir(root, spec['task_id'])
    directory.mkdir(parents=True, exist_ok=True)
    spec_path = directory / 'spec.json'
    receipt_path = directory / 'receipt.json'
    result_path = directory / 'result.json'
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding='utf-8'))
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if alive(receipt.get('worker_pid')):
            return {'status': 'running', **receipt}
    atomic_json(spec_path, spec)
    log = (directory / 'worker.log').open('ab', buffering=0)
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            'run',
            '--root',
            str(root.resolve()),
            '--task-id',
            spec['task_id'],
        ],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    receipt = {
        'status': 'running',
        'task_id': spec['task_id'],
        'worker_pid': process.pid,
        'started_at': utc_now(),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def run(root: Path, task_id: str) -> int:
    directory = task_dir(root, task_id)
    spec = json.loads((directory / 'spec.json').read_text(encoding='utf-8'))
    environment = os.environ.copy()
    environment.update(
        {str(k): str(v) for k, v in spec.get('env', {}).items()}
    )
    started_at = utc_now()
    process = subprocess.Popen(
        spec['command'],
        cwd=spec['cwd'],
        env=environment,
        stdin=subprocess.DEVNULL,
    )
    atomic_json(
        directory / 'child.json',
        {'pid': process.pid, 'started_at': started_at},
    )
    returncode = process.wait()
    result = {
        'status': 'succeeded' if returncode == 0 else 'failed',
        'task_id': task_id,
        'returncode': returncode,
        'started_at': started_at,
        'finished_at': utc_now(),
        'worker_pid': os.getpid(),
        'child_pid': process.pid,
    }
    atomic_json(directory / 'result.json', result)
    return returncode


def status(root: Path, task_id: str) -> dict:
    directory = task_dir(root, task_id)
    result_path = directory / 'result.json'
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding='utf-8'))
    receipt_path = directory / 'receipt.json'
    if not receipt_path.is_file():
        return {'status': 'unknown', 'task_id': task_id}
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if alive(receipt.get('worker_pid')):
        return {'status': 'running', **receipt}
    return {
        'status': 'lost',
        'task_id': task_id,
        'worker_pid': receipt.get('worker_pid'),
        'started_at': receipt.get('started_at'),
    }


def stop(root: Path, task_id: str) -> dict:
    directory = task_dir(root, task_id)
    child_path = directory / 'child.json'
    receipt_path = directory / 'receipt.json'
    pids = []
    if child_path.is_file():
        pids.append(json.loads(child_path.read_text())['pid'])
    if receipt_path.is_file():
        pids.append(json.loads(receipt_path.read_text())['worker_pid'])
    for pid in pids:
        if alive(pid):
            os.kill(pid, signal.SIGTERM)
    return {'status': 'stop_requested', 'task_id': task_id, 'pids': pids}


def prepare_retry(root: Path, task_id: str) -> dict:
    """Archive a terminal worker attempt so the same task ID can run again."""
    directory = task_dir(root, task_id)
    receipt_path = directory / 'receipt.json'
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if alive(receipt.get('worker_pid')):
            raise RuntimeError(f'worker is still alive for {task_id}')
    attempts = directory / 'attempts'
    attempts.mkdir(parents=True, exist_ok=True)
    number = 1
    while (attempts / f'attempt-{number}').exists():
        number += 1
    archive = attempts / f'attempt-{number}'
    archive.mkdir()
    for name in (
        'spec.json',
        'receipt.json',
        'result.json',
        'child.json',
        'worker.log',
    ):
        source = directory / name
        if source.exists():
            shutil.move(str(source), archive / name)
    return {
        'status': 'prepared',
        'task_id': task_id,
        'archived_attempt': str(archive),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='action', required=True)
    for action in ('start', 'status', 'stop', 'run', 'prepare-retry'):
        child = subparsers.add_parser(action)
        child.add_argument('--root', required=True, type=Path)
        if action == 'start':
            child.add_argument('--spec-base64', required=True)
        else:
            child.add_argument('--task-id', required=True)
    args = parser.parse_args()
    if args.action == 'start':
        output = start(args.root, args.spec_base64)
    elif args.action == 'status':
        output = status(args.root, args.task_id)
    elif args.action == 'stop':
        output = stop(args.root, args.task_id)
    elif args.action == 'prepare-retry':
        output = prepare_retry(args.root, args.task_id)
    else:
        return run(args.root, args.task_id)
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
