#!/usr/bin/env python3
"""Import a worker training-completion artifact into the central registry."""

from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiments.registry import (
    connect,
    register_selected_checkpoint,
    register_train_run,
    set_train_run_status,
    sha256_file,
)

MUTABLE_RUNTIME_CONFIG_COMMITS = {
    '2be6fbdfe9cd601b5d055d8d70d12b6ddeef3cb8',
}


def verified_artifact(record: dict) -> Path:
    path = ROOT / record['path']
    if sha256_file(path) != record['sha256']:
        raise ValueError(f'artifact hash mismatch: {record["path"]}')
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('completion', type=Path)
    parser.add_argument('--register-if-missing', action='store_true')
    parser.add_argument('--parent-train-run-id')
    parser.add_argument('--executor')
    parser.add_argument('--tracker-run-name')
    parser.add_argument(
        '--allow-restored-canonical-config', action='store_true'
    )
    parser.add_argument(
        '--allow-registered-provenance-rebind', action='store_true'
    )
    args = parser.parse_args()
    completion_path = args.completion.resolve()
    payload = json.loads(completion_path.read_text(encoding='utf-8'))
    if payload.get('evidence_type') != 'registered_training_completion':
        raise ValueError('not a registered training completion artifact')
    train_run_id = payload['train_run_id']
    manifest_path = verified_artifact(payload['manifest'])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    connection = connect()
    try:
        row = connection.execute(
            'SELECT git_commit,config_sha256,status,dataset_id,method,'
            'training_seed,work_dir,scientific_config_sha256 '
            'FROM train_run_registry '
            'WHERE train_run_id=?',
            (train_run_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None and args.register_if_missing:
        if not args.executor or not args.tracker_run_name:
            raise ValueError(
                '--register-if-missing requires --executor and '
                '--tracker-run-name'
            )
        config_path = ROOT / manifest['resolved_config_path']
        if sha256_file(config_path) != manifest['resolved_config_sha256']:
            raise ValueError('restored canonical config differs from manifest')
        registered_id = register_train_run(
            manifest=manifest,
            config_path=config_path.resolve(),
            work_dir=config_path.parent.resolve(),
            executor=args.executor,
            tracker_run_name=args.tracker_run_name,
            status='planned',
            parent_train_run_id=args.parent_train_run_id,
        )
        if registered_id != train_run_id:
            raise ValueError('completion train_run_id differs from manifest')
        connection = connect()
        try:
            row = connection.execute(
                'SELECT git_commit,config_sha256,status,dataset_id,method,'
                'training_seed,work_dir,scientific_config_sha256 '
                'FROM train_run_registry WHERE train_run_id=?',
                (train_run_id,),
            ).fetchone()
        finally:
            connection.close()
    if row is None:
        raise ValueError(f'unknown train_run_id: {train_run_id}')
    provenance_rebound = False
    if args.allow_registered_provenance_rebind and (
        row[0] != payload['git_commit']
        or row[1] != payload['resolved_config']['sha256']
    ):
        expected = {
            'dataset_id': row[3],
            'method_id': row[4],
            'training_seed': row[5],
            'work_dir': row[6],
            'scientific_config_sha256': row[7],
        }
        observed = {
            'dataset_id': manifest['dataset_id'],
            'method_id': manifest['method_id'],
            'training_seed': manifest['training_seed'],
            'work_dir': str(Path(manifest['resolved_config_path']).parent),
            'scientific_config_sha256': manifest['scientific_config_sha256'],
        }
        if observed != expected:
            raise ValueError(
                'provenance rebind changes scientific run identity: '
                f'expected={expected}, observed={observed}'
            )
        actual_config = verified_artifact(payload['resolved_config'])
        if (
            manifest['resolved_config_sha256']
            != payload['resolved_config']['sha256']
        ):
            raise ValueError('completion config differs from its manifest')
        connection = connect()
        try:
            with connection:
                connection.execute(
                    'UPDATE train_run_registry SET git_commit=?,config_path=?,'
                    'config_sha256=?,updated_at=CURRENT_TIMESTAMP '
                    'WHERE train_run_id=?',
                    (
                        payload['git_commit'],
                        str(actual_config.relative_to(ROOT)),
                        payload['resolved_config']['sha256'],
                        train_run_id,
                    ),
                )
        finally:
            connection.close()
        row = (
            payload['git_commit'],
            payload['resolved_config']['sha256'],
            *row[2:],
        )
        provenance_rebound = True
    if row[0] != payload['git_commit']:
        raise ValueError('completion Git commit differs from registration')
    completion_config_sha = payload['resolved_config']['sha256']
    restored_config = False
    if row[1] != completion_config_sha:
        canonical_path = ROOT / payload['resolved_config']['path']
        restored_config = (
            args.allow_restored_canonical_config
            and payload['git_commit'] in MUTABLE_RUNTIME_CONFIG_COMMITS
            and manifest['resolved_config_sha256'] == row[1]
            and sha256_file(canonical_path) == row[1]
        )
        if not restored_config:
            raise ValueError('completion config differs from registration')
        warnings.warn(
            'accepting restored canonical config: the legacy runtime dumped '
            'its mutated config over resolved_config.py after registration',
            stacklevel=1,
        )
    else:
        verified_artifact(payload['resolved_config'])
    for log in payload['logs']:
        # Schema-v1 workers originally included their own outer launcher log.
        # That file can receive final stdout after training_completion.json is
        # atomically written, making its recorded digest stale by construction.
        # It is not metric evidence; require it to exist for legacy artifacts,
        # but verify every immutable framework/training log normally.
        if Path(log['path']).name == 'worker_launcher.log':
            launcher_log = ROOT / log['path']
            if not launcher_log.is_file():
                raise ValueError(f'missing legacy launcher log: {log["path"]}')
            warnings.warn(
                'legacy worker_launcher.log digest is not verified because '
                'the launcher may append after completion emission',
                stacklevel=1,
            )
            continue
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
    completion_sha = sha256_file(completion_path)
    artifact_id = f'training-completion-{completion_sha[:12]}'
    connection = connect()
    try:
        with connection:
            values = (
                str(completion_path.relative_to(ROOT)),
                completion_sha,
                'distributed_training_completion_v1',
                'verified',
            )
            existing = connection.execute(
                'SELECT path,sha256,kind,status FROM evidence_artifact '
                'WHERE artifact_id=?',
                (artifact_id,),
            ).fetchone()
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
                        (
                            'Pre-registration provenance rebound to the actual '
                            'clean worker manifest; scientific identity unchanged.'
                            if provenance_rebound
                            else 'Canonical pre-run config restored after legacy '
                            'runtime overwrite.'
                            if restored_config
                            else 'Distributed training completion verified.'
                        ),
                    ),
                )
    finally:
        connection.close()
    print(
        f'train_run_id={train_run_id} status=trained '
        f'checkpoint_sha256={digest} artifact_id={artifact_id}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
