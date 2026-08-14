#!/usr/bin/env python3
"""Resolve dataset-independent canonical methods through experiment matrices."""

from __future__ import annotations
import hashlib
import json
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / 'experiments/configs'
RUNTIME = CONFIG_ROOT / '_base_/runtime/default.py'
METHODS = CONFIG_ROOT / 'methods'

SCIENTIFIC_KEYS = (
    'dataset_id',
    'dataset_manifest_sha256',
    'train_annotation_sha256',
    'val_annotation_sha256',
    'test_annotation_sha256',
    'model',
    'train_dataloader',
    'val_dataloader',
    'test_dataloader',
    'val_evaluator',
    'test_evaluator',
    'optim_wrapper',
    'param_scheduler',
    'train_cfg',
    'val_cfg',
    'test_cfg',
    'custom_hooks',
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Prefer a repository-relative path but permit external previews."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _deep_merge(base, update):
    """Merge resolved MMEngine dictionaries while honoring ``_delete_``."""
    if not isinstance(base, dict) or not isinstance(update, dict):
        return deepcopy(update)
    update = deepcopy(update)
    if update.pop('_delete_', False):
        return update
    output = deepcopy(base)
    for key, value in update.items():
        output[key] = (
            _deep_merge(output[key], value) if key in output else value
        )
    return output


def load_matrix(path: str | Path) -> tuple[Path, dict]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    data = yaml.safe_load(path.read_text())
    required = {
        'schema_version',
        'matrix_id',
        'dataset_alias',
        'dataset_config',
        'tracker_project',
        'training_seeds',
        'methods',
    }
    missing = sorted(required - set(data or {}))
    if missing:
        raise ValueError(f'{path}: missing matrix fields {missing}')
    if data['schema_version'] != 1:
        raise ValueError(f'{path}: unsupported schema_version')
    if len(set(data['training_seeds'])) != len(data['training_seeds']):
        raise ValueError(f'{path}: duplicate training seeds')
    if len(set(data['methods'])) != len(data['methods']):
        raise ValueError(f'{path}: duplicate methods')
    return path, data


def method_path(method_name: str) -> Path:
    path = (METHODS / f'{method_name}.py').resolve()
    if path.parent != METHODS.resolve() or not path.is_file():
        raise ValueError(f'unknown v2 method: {method_name}')
    return path


def resolve_config(
    matrix_path: str | Path,
    method_name: str,
    seed: int = 42,
    parent_checkpoint: str | None = None,
):
    matrix_path, matrix = load_matrix(matrix_path)
    if method_name not in matrix['methods']:
        raise ValueError(
            f'{method_name} is not enabled by {matrix["matrix_id"]}'
        )
    if seed not in matrix['training_seeds']:
        raise ValueError(
            f'seed {seed} is not registered by {matrix["matrix_id"]}'
        )

    dataset_path = (ROOT / matrix['dataset_config']).resolve()
    method_config_path = method_path(method_name)
    for path in (dataset_path, method_config_path, RUNTIME):
        if not path.is_file():
            raise FileNotFoundError(path)

    merged = {}
    for path in (RUNTIME, dataset_path, method_config_path):
        merged = _deep_merge(merged, Config.fromfile(path).to_dict())
    method = merged.pop('method')
    config_id = f'v2.{matrix["dataset_alias"]}.{method["method_id"]}.train'
    merged['experiment'] = dict(
        method,
        config_id=config_id,
        matrix_id=matrix['matrix_id'],
        dataset_id=merged['dataset_id'],
        dataset_alias=matrix['dataset_alias'],
        tracker_project=matrix['tracker_project'],
    )
    merged['randomness'] = dict(
        seed=seed, deterministic=False, diff_rank_seed=False
    )
    if parent_checkpoint:
        binding = method.get('parent_checkpoint_binding', 'load_from')
        checkpoint_path = str(Path(parent_checkpoint).resolve())
        if binding == 'load_from':
            merged['load_from'] = checkpoint_path
        elif binding == 'teacher_checkpoint':
            merged['model']['teacher_checkpoint'] = checkpoint_path
        else:
            raise ValueError(
                f'unsupported parent checkpoint binding: {binding}'
            )
    cfg = Config(merged)
    return cfg, dict(
        matrix_path=matrix_path,
        matrix=matrix,
        dataset_path=dataset_path,
        method_path=method_config_path,
    )


def scientific_hash(cfg: Config) -> str:
    payload = deepcopy({key: cfg.get(key) for key in SCIENTIFIC_KEYS})
    # Checkpoint identity is recorded independently in the resolution
    # manifest.  A machine-local path must never alter scientific identity.
    model = payload.get('model')
    if isinstance(model, dict):
        model.pop('teacher_checkpoint', None)
    raw = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), default=str
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def default_work_dir(cfg: Config, seed: int) -> Path:
    return (
        ROOT
        / 'work_dirs/v2'
        / cfg.experiment.dataset_alias
        / cfg.experiment.method_id
        / f'trainseed_{seed}'
    )


def _git_state() -> dict:
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ['git', 'diff', '--binary', 'HEAD'],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    status = subprocess.run(
        ['git', 'status', '--porcelain=v1'],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    state = diff + b'\n--STATUS--\n' + status
    return dict(
        commit=commit,
        dirty=bool(status),
        dirty_diff_sha256=(
            hashlib.sha256(state).hexdigest() if status else None
        ),
    )


def write_resolution(
    cfg: Config, sources: dict, seed: int, output_dir: str | Path | None = None
) -> tuple[Path, Path]:
    output_dir = (
        Path(output_dir) if output_dir else default_work_dir(cfg, seed)
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir = display_path(output_dir)
    config_path = output_dir / 'resolved_config.py'
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', dir=output_dir, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        cfg.dump(temporary)
        temporary.replace(config_path)
    finally:
        temporary.unlink(missing_ok=True)
    resolved_sha = sha256_file(config_path)
    science_sha = scientific_hash(cfg)
    source_paths = [
        sources['matrix_path'],
        sources['dataset_path'],
        sources['method_path'],
        RUNTIME,
    ]
    parent_binding = cfg.experiment.get(
        'parent_checkpoint_binding', 'load_from'
    )
    parent_path = (
        cfg.get('load_from')
        if parent_binding == 'load_from'
        else cfg.model.get('teacher_checkpoint')
    )
    manifest = dict(
        schema_version=1,
        evidence_type='resolved_experiment_config',
        config_id=cfg.experiment.config_id,
        matrix_id=cfg.experiment.matrix_id,
        dataset_id=cfg.dataset_id,
        method_id=cfg.experiment.method_id,
        training_seed=seed,
        replication_unit=cfg.experiment.replication_unit,
        selection=dict(
            split=cfg.experiment.selection_split,
            metric=cfg.experiment.selection_metric,
            test_tuned=cfg.experiment.test_tuned,
        ),
        parent_method_id=cfg.experiment.get('parent_method_id'),
        parent_checkpoint_required=cfg.experiment.get(
            'parent_checkpoint_required', False
        ),
        parent_checkpoint_binding=parent_binding,
        scientific_config_sha256=science_sha,
        resolved_config_path=display_path(config_path),
        resolved_config_sha256=resolved_sha,
        annotation_sha256=dict(
            train=cfg.get('train_annotation_sha256'),
            val=cfg.get('val_annotation_sha256'),
            test=cfg.get('test_annotation_sha256'),
        ),
        dataset_manifest_sha256=cfg.get('dataset_manifest_sha256'),
        parent_checkpoint=(
            dict(
                path=display_path(Path(parent_path)),
                sha256=sha256_file(Path(parent_path)),
            )
            if parent_path
            else None
        ),
        sources=[
            dict(path=display_path(path), sha256=sha256_file(path))
            for path in source_paths
        ],
        git=_git_state(),
    )
    manifest_path = output_dir / 'resolution_manifest.json'
    temporary = manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    temporary.replace(manifest_path)
    return config_path, manifest_path


def matrix_plan(matrix_path: str | Path) -> list[dict]:
    _, matrix = load_matrix(matrix_path)
    records = []
    for method_name in matrix['methods']:
        cfg, _ = resolve_config(
            matrix_path, method_name, matrix['training_seeds'][0]
        )
        for seed in matrix['training_seeds']:
            records.append(
                dict(
                    matrix_id=matrix['matrix_id'],
                    dataset_id=cfg.dataset_id,
                    method_name=method_name,
                    method_id=cfg.experiment.method_id,
                    config_id=cfg.experiment.config_id,
                    training_seed=seed,
                    parent_method_id=cfg.experiment.get('parent_method_id'),
                    parent_checkpoint_required=cfg.experiment.get(
                        'parent_checkpoint_required', False
                    ),
                    scientific_config_sha256=scientific_hash(cfg),
                    work_dir=str(
                        default_work_dir(cfg, seed).relative_to(ROOT)
                    ),
                )
            )
    return records
