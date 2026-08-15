#!/usr/bin/env python3
"""Plan, resolve, or launch canonical MMEngine experiments."""

from __future__ import annotations
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiments.matrix import (
    load_matrix,
    matrix_plan,
    resolve_config,
    write_resolution,
)
from tools.experiments.registry import (
    register_selected_checkpoint,
    register_train_run,
    set_train_run_status,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', required=True)
    parser.add_argument('--method')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output-dir')
    parser.add_argument('--parent-checkpoint')
    parser.add_argument('--parent-train-run-id')
    parser.add_argument('--executor', default=platform.node())
    parser.add_argument('--gpu-id', type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true')
    mode.add_argument('--resolve-only', action='store_true')
    mode.add_argument('--launch', action='store_true')
    args = parser.parse_args()

    _, matrix = load_matrix(args.matrix)
    if args.plan or not (args.resolve_only or args.launch):
        records = matrix_plan(args.matrix)
        if args.method:
            records = [r for r in records if r['method_name'] == args.method]
        if args.seed is not None:
            records = [r for r in records if r['training_seed'] == args.seed]
        print(
            json.dumps(
                dict(matrix_id=matrix['matrix_id'], runs=records), indent=2
            )
        )
        return 0

    if args.method is None or args.seed is None:
        parser.error('--resolve-only/--launch require --method and --seed')
    cfg, sources = resolve_config(
        args.matrix, args.method, args.seed, args.parent_checkpoint
    )
    parent_required = cfg.experiment.get('parent_checkpoint_required')
    if parent_required and not args.parent_checkpoint:
        parser.error(f'{args.method} requires --parent-checkpoint')
    if parent_required and not args.parent_train_run_id:
        parser.error(f'{args.method} requires --parent-train-run-id')
    if not parent_required and args.parent_train_run_id:
        parser.error('--parent-train-run-id is only valid for a child method')
    config_path, manifest_path = write_resolution(
        cfg, sources, args.seed, args.output_dir
    )
    manifest = json.loads(manifest_path.read_text())
    tracker_run_name = (
        f'{cfg.experiment.dataset_alias}_{args.method}_seed{args.seed}'
    )
    train_run_id = register_train_run(
        manifest=manifest,
        config_path=config_path.resolve(),
        work_dir=config_path.parent.resolve(),
        executor=args.executor,
        tracker_run_name=tracker_run_name,
        status='running' if args.launch else 'planned',
        parent_train_run_id=args.parent_train_run_id,
    )
    print(
        json.dumps(
            dict(
                status='RESOLVED',
                config=str(config_path),
                manifest=str(manifest_path),
                train_run_id=train_run_id,
                scientific_config_sha256=manifest['scientific_config_sha256'],
            ),
            indent=2,
        )
    )
    if not args.launch:
        return 0

    command = [
        sys.executable,
        str(ROOT / 'experiments/runners/train.py'),
        str(config_path),
        '--seed',
        str(args.seed),
        '--gpu-id',
        str(args.gpu_id),
        '--work-dir',
        str(config_path.parent),
        '--exp-name',
        tracker_run_name,
    ]
    if args.parent_checkpoint:
        command.extend(['--parent-checkpoint', args.parent_checkpoint])
    returncode = subprocess.run(command, cwd=ROOT, check=False).returncode
    set_train_run_status(
        train_run_id, 'trained' if returncode == 0 else 'failed'
    )
    if returncode == 0:
        candidates = sorted(
            config_path.parent.glob('best_coco_bbox_mAP_epoch_*.pth')
        )
        if len(candidates) != 1:
            print(
                'checkpoint selection blocked: expected one validation-best '
                f'checkpoint, found {len(candidates)} in {config_path.parent}',
                file=sys.stderr,
            )
            return 2
        digest = register_selected_checkpoint(
            train_run_id,
            candidates[0],
            selection_source=(
                'validation:coco/bbox_mAP; MMEngine CheckpointHook save_best'
            ),
            selection_metric='coco/bbox_mAP',
        )
        print(
            json.dumps(
                dict(
                    status='TRAINED_AND_SELECTED',
                    train_run_id=train_run_id,
                    checkpoint=str(candidates[0]),
                    checkpoint_sha256=digest,
                ),
                indent=2,
            )
        )
    return returncode


if __name__ == '__main__':
    raise SystemExit(main())
