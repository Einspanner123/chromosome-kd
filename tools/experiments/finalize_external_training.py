#!/usr/bin/env python3
"""Emit standard training-completion evidence for an externally launched run."""

from __future__ import annotations
import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiments.registry import make_train_run_id, sha256_file


def utc_timestamp(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).isoformat()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--train-run-id')
    parser.add_argument('--executor', required=True)
    parser.add_argument('--gpu-id', required=True, type=int)
    parser.add_argument('--gpu-name')
    parser.add_argument('--command-json', default='[]')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    work_dir = args.work_dir
    if not work_dir.is_absolute():
        work_dir = ROOT / work_dir
    work_dir = work_dir.resolve()
    manifest_path = work_dir / 'resolution_manifest.json'
    config_path = work_dir / 'resolved_config.py'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected_id = make_train_run_id(
        manifest['dataset_id'],
        manifest['method_id'],
        int(manifest['training_seed']),
        manifest['scientific_config_sha256'],
    )
    train_run_id = args.train_run_id or expected_id
    if train_run_id != expected_id:
        raise ValueError(f'train-run ID mismatch: expected {expected_id}')
    canonical_config_sha = manifest['resolved_config_sha256']
    actual_config_sha = sha256_file(config_path)
    if actual_config_sha != canonical_config_sha:
        raise ValueError(
            'resolved config differs from its preregistered manifest; restore '
            'the canonical config before finalization'
        )
    for source in manifest['sources']:
        source_path = ROOT / source['path']
        if sha256_file(source_path) != source['sha256']:
            raise ValueError(f'source hash mismatch: {source["path"]}')
    candidates = sorted(work_dir.glob('best_coco_bbox_mAP_epoch_*.pth'))
    if len(candidates) != 1:
        raise ValueError(
            f'expected one validation-best checkpoint, found {len(candidates)}'
        )
    checkpoint = candidates[0]
    logs = sorted(
        path
        for path in work_dir.rglob('*.log')
        if path.name not in {'worker_launcher.log', 'scheduler_worker.log'}
    )
    if not logs:
        raise ValueError('no immutable framework log found')
    completion_path = work_dir / 'training_completion.json'
    if completion_path.exists() and not args.overwrite:
        existing = json.loads(completion_path.read_text(encoding='utf-8'))
        if existing.get('train_run_id') != train_run_id:
            raise ValueError('existing completion belongs to another run')
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0
    started = min(path.stat().st_mtime for path in logs)
    finished = max(path.stat().st_mtime for path in [checkpoint, *logs])
    payload = {
        'schema_version': 1,
        'evidence_type': 'registered_training_completion',
        'train_run_id': train_run_id,
        'status': 'trained',
        'returncode': 0,
        'started_at': utc_timestamp(started),
        'finished_at': utc_timestamp(finished),
        'executor': args.executor,
        'gpu_id': args.gpu_id,
        'gpu_name': args.gpu_name or platform.node(),
        'git_commit': manifest['git']['commit'],
        'manifest': {
            'path': relative(manifest_path),
            'sha256': sha256_file(manifest_path),
        },
        'resolved_config': {
            'path': relative(config_path),
            'sha256': actual_config_sha,
        },
        'checkpoint': {
            'path': relative(checkpoint),
            'sha256': sha256_file(checkpoint),
            'selection_source': (
                'validation:coco/bbox_mAP; MMEngine CheckpointHook save_best'
            ),
            'selection_metric': 'coco/bbox_mAP',
        },
        'checkpoint_candidate_count': 1,
        'logs': [
            {'path': relative(path), 'sha256': sha256_file(path)}
            for path in logs
        ],
        'tracker_project': manifest.get('tracker_project'),
        'tracker_run_name': (
            f'{manifest["dataset_id"]}_{manifest["method_id"]}'
            f'_seed{manifest["training_seed"]}'
        ),
        'command': json.loads(args.command_json),
        'completion_origin': 'scheduler_finalized_external_run',
    }
    atomic_json(completion_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
