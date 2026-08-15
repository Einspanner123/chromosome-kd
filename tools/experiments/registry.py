"""Stable V2 training-run and selected-checkpoint registry helpers."""

from __future__ import annotations
import hashlib
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / 'tools/experiment_db/experiments.db'
SCHEMA = ROOT / 'tools/experiment_db/schema.sql'


def slug(value: str) -> str:
    return value.lower().replace('_', '-').replace('.', '-')


def make_train_run_id(
    dataset_id: str, method_id: str, training_seed: int, science_sha: str
) -> str:
    return (
        f'{slug(dataset_id)}__{slug(method_id)}__trainseed-{training_seed}'
        f'__{science_sha[:12]}'
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def connect(db: Path = DB) -> sqlite3.Connection:
    connection = sqlite3.connect(db)
    connection.execute('PRAGMA foreign_keys=ON')
    connection.executescript(SCHEMA.read_text(encoding='utf-8'))
    columns = {
        row[1]
        for row in connection.execute('PRAGMA table_info(train_run_registry)')
    }
    if 'scientific_config_sha256' not in columns:
        connection.execute(
            'ALTER TABLE train_run_registry ADD COLUMN '
            'scientific_config_sha256 TEXT'
        )
    return connection


def register_train_run(
    *,
    manifest: dict,
    config_path: Path,
    work_dir: Path,
    executor: str,
    tracker_run_name: str,
    status: str,
    parent_train_run_id: str | None = None,
    db: Path = DB,
) -> str:
    train_run_id = make_train_run_id(
        manifest['dataset_id'],
        manifest['method_id'],
        int(manifest['training_seed']),
        manifest['scientific_config_sha256'],
    )
    values = (
        manifest['dataset_id'],
        manifest['method_id'],
        int(manifest['training_seed']),
        str(config_path.relative_to(ROOT)),
        manifest['resolved_config_sha256'],
        manifest['scientific_config_sha256'],
        manifest['dataset_manifest_sha256'],
        manifest['git']['commit'],
        manifest['replication_unit'],
        parent_train_run_id,
        executor,
        manifest.get('tracker_project'),
        tracker_run_name,
        str(work_dir.relative_to(ROOT)),
        status,
    )
    connection = connect(db)
    try:
        with connection:
            parent_artifact = manifest.get('parent_checkpoint')
            if parent_train_run_id:
                parent = connection.execute(
                    'SELECT checkpoint_sha256 FROM selected_checkpoint '
                    'WHERE train_run_id=?',
                    (parent_train_run_id,),
                ).fetchone()
                if parent is None:
                    raise ValueError(
                        'parent train run has no registered selected checkpoint: '
                        f'{parent_train_run_id}'
                    )
                if (
                    not parent_artifact
                    or parent[0] != parent_artifact['sha256']
                ):
                    raise ValueError(
                        'parent checkpoint does not match the selected checkpoint '
                        f'of {parent_train_run_id}'
                    )
            elif parent_artifact:
                raise ValueError(
                    'a resolved parent checkpoint requires parent_train_run_id'
                )
            existing = connection.execute(
                'SELECT dataset_id,method,training_seed,config_path,config_sha256,'
                'scientific_config_sha256,'
                'dataset_manifest_sha256,git_commit,replication_unit,'
                'parent_train_run_id,assigned_executor,tracker_project,'
                'tracker_run_name,work_dir FROM train_run_registry '
                'WHERE train_run_id=?',
                (train_run_id,),
            ).fetchone()
            identity = values[:-1]
            if existing is not None and tuple(existing) != identity:
                raise ValueError(
                    f'conflicting train_run_registry row: {train_run_id}'
                )
            if existing is None:
                connection.execute(
                    'INSERT INTO train_run_registry '
                    '(train_run_id,dataset_id,method,training_seed,config_path,'
                    'config_sha256,scientific_config_sha256,'
                    'dataset_manifest_sha256,git_commit,'
                    'replication_unit,parent_train_run_id,assigned_executor,'
                    'tracker_project,tracker_run_name,work_dir,status) '
                    'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (train_run_id, *values),
                )
            else:
                connection.execute(
                    'UPDATE train_run_registry SET status=?,'
                    'updated_at=CURRENT_TIMESTAMP WHERE train_run_id=?',
                    (status, train_run_id),
                )
    finally:
        connection.close()
    return train_run_id


def register_selected_checkpoint(
    train_run_id: str,
    checkpoint: Path,
    *,
    selection_source: str,
    selection_metric: str,
    selection_value: float | None = None,
    db: Path = DB,
) -> str:
    checkpoint = checkpoint.resolve()
    checkpoint_sha = sha256_file(checkpoint)
    connection = connect(db)
    try:
        with connection:
            if (
                connection.execute(
                    'SELECT 1 FROM train_run_registry WHERE train_run_id=?',
                    (train_run_id,),
                ).fetchone()
                is None
            ):
                raise ValueError(f'unknown train_run_id: {train_run_id}')
            existing = connection.execute(
                'SELECT checkpoint_path,checkpoint_sha256,selection_source,'
                'selection_metric,selection_value FROM selected_checkpoint '
                'WHERE train_run_id=?',
                (train_run_id,),
            ).fetchone()
            values = (
                str(checkpoint.relative_to(ROOT)),
                checkpoint_sha,
                selection_source,
                selection_metric,
                selection_value,
            )
            if existing is not None and tuple(existing) != values:
                raise ValueError(
                    f'conflicting selected checkpoint: {train_run_id}'
                )
            if existing is None:
                connection.execute(
                    'INSERT INTO selected_checkpoint '
                    '(train_run_id,checkpoint_path,checkpoint_sha256,'
                    'selection_source,selection_metric,selection_value) '
                    'VALUES (?,?,?,?,?,?)',
                    (train_run_id, *values),
                )
            connection.execute(
                "UPDATE train_run_registry SET status='trained',"
                'updated_at=CURRENT_TIMESTAMP WHERE train_run_id=?',
                (train_run_id,),
            )
    finally:
        connection.close()
    return checkpoint_sha


def set_train_run_status(
    train_run_id: str, status: str, *, db: Path = DB
) -> None:
    allowed = {
        'planned',
        'running',
        'trained',
        'test_evaluated',
        'verified',
        'paper_eligible',
        'failed',
        'invalid',
        'superseded',
    }
    if status not in allowed:
        raise ValueError(f'invalid train-run status: {status}')
    connection = connect(db)
    try:
        with connection:
            cursor = connection.execute(
                'UPDATE train_run_registry SET status=?,'
                'updated_at=CURRENT_TIMESTAMP WHERE train_run_id=?',
                (status, train_run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f'unknown train_run_id: {train_run_id}')
    finally:
        connection.close()
