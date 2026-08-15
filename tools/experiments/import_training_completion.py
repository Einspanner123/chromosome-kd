#!/usr/bin/env python3
"""Import a worker training-completion artifact into the central registry."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiments.registry import (
    connect,
    register_selected_checkpoint,
    set_train_run_status,
    sha256_file,
)


def verified_artifact(record: dict) -> Path:
    path = ROOT / record['path']
    if sha256_file(path) != record['sha256']:
        raise ValueError(f'artifact hash mismatch: {record["path"]}')
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('completion', type=Path)
    args = parser.parse_args()
    completion_path = args.completion.resolve()
    payload = json.loads(completion_path.read_text(encoding='utf-8'))
    if payload.get('evidence_type') != 'registered_training_completion':
        raise ValueError('not a registered training completion artifact')
    train_run_id = payload['train_run_id']
    connection = connect()
    try:
        row = connection.execute(
            'SELECT git_commit,config_sha256,status FROM train_run_registry '
            'WHERE train_run_id=?',
            (train_run_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError(f'unknown train_run_id: {train_run_id}')
    if row[0] != payload['git_commit']:
        raise ValueError('completion Git commit differs from registration')
    if row[1] != payload['resolved_config']['sha256']:
        raise ValueError('completion config differs from registration')
    verified_artifact(payload['manifest'])
    verified_artifact(payload['resolved_config'])
    for log in payload['logs']:
        verified_artifact(log)

    if payload['status'] != 'trained':
        set_train_run_status(train_run_id, 'failed')
        print(f'train_run_id={train_run_id} status=failed')
        return 1
    checkpoint = payload.get('checkpoint')
    if payload.get('checkpoint_candidate_count') != 1 or not checkpoint:
        raise ValueError('completion does not identify exactly one checkpoint')
    checkpoint_path = verified_artifact(checkpoint)
    digest = register_selected_checkpoint(
        train_run_id,
        checkpoint_path,
        selection_source=checkpoint['selection_source'],
        selection_metric=checkpoint['selection_metric'],
    )
    print(
        f'train_run_id={train_run_id} status=trained '
        f'checkpoint_sha256={digest}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
