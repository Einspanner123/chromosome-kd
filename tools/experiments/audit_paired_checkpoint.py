#!/usr/bin/env python3
"""Verify that a paired child changes no shared checkpoint tensors."""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiments.registry import connect, sha256_file


def _state_dict(path: Path) -> dict:
    import torch

    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError(f'checkpoint has no state_dict: {path}')
    return state


def _selected(connection: sqlite3.Connection, train_run_id: str) -> tuple:
    row = connection.execute(
        'SELECT r.parent_train_run_id,c.checkpoint_path,c.checkpoint_sha256 '
        'FROM train_run_registry r JOIN selected_checkpoint c '
        'ON c.train_run_id=r.train_run_id WHERE r.train_run_id=?',
        (train_run_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f'missing selected checkpoint: {train_run_id}')
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-train-run-id', required=True)
    parser.add_argument('--child-train-run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-child-only', type=int, default=5)
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()

    connection = connect()
    try:
        _, parent_relpath, parent_sha = _selected(
            connection, args.parent_train_run_id
        )
        child_parent, child_relpath, child_sha = _selected(
            connection, args.child_train_run_id
        )
    finally:
        connection.close()
    if child_parent != args.parent_train_run_id:
        raise ValueError(
            'child registry row does not reference the given parent'
        )

    parent_path = ROOT / parent_relpath
    child_path = ROOT / child_relpath
    if sha256_file(parent_path) != parent_sha:
        raise ValueError('parent checkpoint SHA mismatch')
    if sha256_file(child_path) != child_sha:
        raise ValueError('child checkpoint SHA mismatch')

    parent = _state_dict(parent_path)
    child = _state_dict(child_path)
    parent_keys = set(parent)
    child_keys = set(child)
    shared = sorted(parent_keys & child_keys)
    parent_only = sorted(parent_keys - child_keys)
    child_only = sorted(child_keys - parent_keys)
    changed = []
    shape_or_dtype_changed = []
    for key in shared:
        left, right = parent[key], child[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            shape_or_dtype_changed.append(key)
        elif not left.equal(right):
            changed.append(key)

    status = 'PASS'
    errors = []
    if parent_only:
        errors.append(
            f'{len(parent_only)} parent tensors are absent from child'
        )
    if shape_or_dtype_changed:
        errors.append(
            f'{len(shape_or_dtype_changed)} shared tensors changed shape/dtype'
        )
    if changed:
        errors.append(f'{len(changed)} shared tensors changed values')
    if len(child_only) != args.expected_child_only:
        errors.append(
            f'expected {args.expected_child_only} child-only tensors, '
            f'found {len(child_only)}'
        )
    if errors:
        status = 'FAIL'

    record = {
        'schema_version': 1,
        'evidence_type': 'paired_checkpoint_identity',
        'status': status,
        'parent_train_run_id': args.parent_train_run_id,
        'child_train_run_id': args.child_train_run_id,
        'parent_checkpoint': {'path': parent_relpath, 'sha256': parent_sha},
        'child_checkpoint': {'path': child_relpath, 'sha256': child_sha},
        'state_dict': {
            'parent_tensors': len(parent_keys),
            'child_tensors': len(child_keys),
            'shared_tensors': len(shared),
            'unchanged_shared_tensors': len(shared)
            - len(changed)
            - len(shape_or_dtype_changed),
            'changed_shared_tensors': changed,
            'shape_or_dtype_changed': shape_or_dtype_changed,
            'parent_only_tensors': parent_only,
            'child_only_tensors': child_only,
        },
        'errors': errors,
    }
    payload = json.dumps(record, indent=2, sort_keys=True) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(payload, encoding='utf-8')
    temporary.replace(args.output)
    artifact_sha = hashlib.sha256(payload.encode()).hexdigest()

    if args.register:
        if status != 'PASS':
            raise ValueError('refusing to register a failed identity audit')
        artifact_id = f'paired-checkpoint-identity-{artifact_sha[:12]}'
        connection = connect()
        try:
            with connection:
                existing = connection.execute(
                    'SELECT path,sha256,kind,status FROM evidence_artifact '
                    'WHERE artifact_id=?',
                    (artifact_id,),
                ).fetchone()
                values = (
                    str(args.output.resolve().relative_to(ROOT)),
                    artifact_sha,
                    'paired_checkpoint_identity_v1',
                    'verified',
                )
                if existing is not None and tuple(existing) != values:
                    raise ValueError(f'conflicting artifact: {artifact_id}')
                if existing is None:
                    connection.execute(
                        'INSERT INTO evidence_artifact '
                        '(artifact_id,server,path,sha256,kind,status,notes) '
                        'VALUES (?,?,?,?,?,?,?)',
                        (
                            artifact_id,
                            'repository',
                            *values,
                            'All shared model tensors are elementwise identical; '
                            'only the child quality branch is added.',
                        ),
                    )
        finally:
            connection.close()
        record['artifact_id'] = artifact_id

    print(json.dumps({**record, 'sha256': artifact_sha}, indent=2))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
