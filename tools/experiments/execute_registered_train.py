#!/usr/bin/env python3
"""Execute one centrally registered training run without a local database write.

This entry point is intended for worker hosts.  The authoritative registry is
kept on the coordinator; workers consume an immutable resolved configuration
and resolution manifest, then atomically emit ``training_completion.json``.
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmengine.config import Config

from tools.experiments.matrix import _git_state
from tools.experiments.registry import make_train_run_id, sha256_file


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def annotation_paths(cfg: Config) -> dict[str, Path]:
    paths = {}
    for split, loader_name in (
        ('train', 'train_dataloader'),
        ('val', 'val_dataloader'),
        ('test', 'test_dataloader'),
    ):
        dataset = cfg[loader_name]['dataset']
        root = Path(dataset.get('data_root', cfg.get('data_root', '')))
        ann = Path(dataset['ann_file'])
        paths[split] = ann if ann.is_absolute() else ROOT / root / ann
    return paths


def gpu_name(gpu_id: int) -> str | None:
    result = subprocess.run(
        [
            'nvidia-smi',
            f'--id={gpu_id}',
            '--query-gpu=name',
            '--format=csv,noheader',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def write_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-run-id', required=True)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--gpu-id', type=int, default=0)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    config_path = ROOT / manifest['resolved_config_path']
    if sha256_file(config_path) != manifest['resolved_config_sha256']:
        raise ValueError('resolved config hash does not match manifest')
    expected_id = make_train_run_id(
        manifest['dataset_id'],
        manifest['method_id'],
        int(manifest['training_seed']),
        manifest['scientific_config_sha256'],
    )
    if args.train_run_id != expected_id:
        raise ValueError(f'train-run ID mismatch: expected {expected_id}')

    code = _git_state()
    if code['dirty'] or code['commit'] != manifest['git']['commit']:
        raise ValueError(
            'worker code state differs from the clean preregistered manifest'
        )
    for source in manifest['sources']:
        source_path = ROOT / source['path']
        if sha256_file(source_path) != source['sha256']:
            raise ValueError(f'source hash mismatch: {source["path"]}')

    cfg = Config.fromfile(config_path)
    for split, path in annotation_paths(cfg).items():
        actual = sha256_file(path)
        expected = manifest['annotation_sha256'][split]
        if actual != expected:
            raise ValueError(
                f'{split} annotation hash mismatch: {actual} != {expected}'
            )

    work_dir = config_path.parent
    completion_path = work_dir / 'training_completion.json'
    tracker_name = (
        f'{cfg.experiment.dataset_alias}_{manifest["method_id"]}'
        f'_seed{manifest["training_seed"]}'
    )
    command = [
        sys.executable,
        str(ROOT / 'experiments/runners/train.py'),
        str(config_path),
        '--seed',
        str(manifest['training_seed']),
        '--gpu-id',
        str(args.gpu_id),
        '--work-dir',
        str(work_dir),
        '--exp-name',
        tracker_name,
    ]
    parent = manifest.get('parent_checkpoint')
    if parent:
        parent_path = ROOT / parent['path']
        if sha256_file(parent_path) != parent['sha256']:
            raise ValueError('parent checkpoint hash mismatch')
        command.extend(['--parent-checkpoint', str(parent_path)])

    started_at = utc_now()
    returncode = subprocess.run(command, cwd=ROOT, check=False).returncode
    candidates = sorted(work_dir.glob('best_coco_bbox_mAP_epoch_*.pth'))
    status = (
        'trained' if returncode == 0 and len(candidates) == 1 else 'failed'
    )
    checkpoint = None
    if len(candidates) == 1:
        checkpoint = {
            'path': relative(candidates[0]),
            'sha256': sha256_file(candidates[0]),
            'selection_source': (
                'validation:coco/bbox_mAP; MMEngine CheckpointHook save_best'
            ),
            'selection_metric': 'coco/bbox_mAP',
        }
    logs = [
        {'path': relative(path), 'sha256': sha256_file(path)}
        for path in sorted(work_dir.rglob('*.log'))
        # This process commonly has stdout/stderr redirected to this file by
        # its outer launcher.  The launcher can append after the completion
        # artifact is written, so it is a mutable transport log rather than
        # immutable training evidence.
        if path.name != 'worker_launcher.log'
    ]
    payload = {
        'schema_version': 1,
        'evidence_type': 'registered_training_completion',
        'train_run_id': args.train_run_id,
        'status': status,
        'returncode': returncode,
        'started_at': started_at,
        'finished_at': utc_now(),
        'executor': platform.node(),
        'gpu_id': args.gpu_id,
        'gpu_name': gpu_name(args.gpu_id),
        'git_commit': code['commit'],
        'manifest': {
            'path': relative(manifest_path),
            'sha256': sha256_file(manifest_path),
        },
        'resolved_config': {
            'path': relative(config_path),
            'sha256': sha256_file(config_path),
        },
        'checkpoint': checkpoint,
        'checkpoint_candidate_count': len(candidates),
        'logs': logs,
        'tracker_project': manifest.get('tracker_project'),
        'tracker_run_name': tracker_name,
        'command': command,
    }
    write_atomic(completion_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == 'trained' else (returncode or 2)


if __name__ == '__main__':
    raise SystemExit(main())
