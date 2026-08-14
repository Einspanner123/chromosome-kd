#!/usr/bin/env python3
"""Plan, resolve, or launch canonical MMEngine experiments."""

from __future__ import annotations
import argparse
import json
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', required=True)
    parser.add_argument('--method')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output-dir')
    parser.add_argument('--parent-checkpoint')
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
    if (
        args.launch
        and cfg.experiment.get('parent_checkpoint_required')
        and not args.parent_checkpoint
    ):
        parser.error(f'{args.method} requires --parent-checkpoint')
    config_path, manifest_path = write_resolution(
        cfg, sources, args.seed, args.output_dir
    )
    print(
        json.dumps(
            dict(
                status='RESOLVED',
                config=str(config_path),
                manifest=str(manifest_path),
                scientific_config_sha256=json.loads(manifest_path.read_text())[
                    'scientific_config_sha256'
                ],
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
        f'{cfg.experiment.dataset_alias}_{args.method}_seed{args.seed}',
    ]
    if args.parent_checkpoint:
        command.extend(['--parent-checkpoint', args.parent_checkpoint])
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
