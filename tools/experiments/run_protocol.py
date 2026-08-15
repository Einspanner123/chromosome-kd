#!/usr/bin/env python3
"""Plan or execute a preregistered inference-ablation protocol."""

from __future__ import annotations
import argparse
import itertools
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiments.matrix import resolve_config
from tools.experiments.registry import DB


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def variants(protocol: dict) -> list[dict]:
    factors = protocol['intervention'].get('factors', {})
    fixed = protocol['intervention'].get('fixed_overrides', {})
    keys = list(factors)
    products = itertools.product(*(factors[key] for key in keys))
    output = []
    for values in products:
        factor_values = dict(zip(keys, values))
        merged = {**fixed, **factor_values}
        derived = protocol['intervention'].get('derived_overrides', {})
        for key, expression in derived.items():
            if isinstance(expression, str):
                match = re.fullmatch(
                    r'(\w+)\s*<\s*(\d+(?:\.\d+)?)', expression
                )
                if not match or match.group(1) not in factor_values:
                    raise ValueError(
                        f'unsupported derived expression: {expression}'
                    )
                merged[key] = float(factor_values[match.group(1)]) < float(
                    match.group(2)
                )
            else:
                merged[key] = expression
        output.append({'factors': factor_values, 'values': merged})
    return output


def variant_name(template: str, values: dict) -> str:
    def replace(match):
        key, formatter = match.group(1), match.group(2)
        value = values[key]
        if formatter == 'on|off':
            return 'on' if value else 'off'
        if formatter == 'decimal_p':
            return str(value).replace('.', 'p')
        return str(value).lower() if isinstance(value, bool) else str(value)

    return re.sub(r'{(\w+)(?::([^}]+))?}', replace, template)


def selected_parents(protocol: dict) -> list[dict]:
    matrix = protocol['base']['matrix']
    method_name = protocol['base']['method']
    expected_seeds = protocol['base']['checkpoint_source'].get(
        'training_seeds',
        protocol['base']['checkpoint_source'].get('parent_training_seeds'),
    )
    cfg, _ = resolve_config(matrix, method_name, int(expected_seeds[0]))
    method_id = cfg.experiment.method_id
    connection = sqlite3.connect(DB)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            'SELECT t.train_run_id,t.dataset_id,t.method,t.training_seed,'
            't.replication_unit,s.checkpoint_path,s.checkpoint_sha256 '
            'FROM train_run_registry t JOIN selected_checkpoint s '
            'ON t.train_run_id=s.train_run_id '
            'WHERE t.dataset_id=? AND t.method=? '
            "AND t.status NOT IN ('failed','invalid','superseded') "
            'ORDER BY t.training_seed',
            (protocol['dataset_id'], method_id),
        ).fetchall()
    finally:
        connection.close()
    by_seed = {row['training_seed']: dict(row) for row in rows}
    missing = [seed for seed in expected_seeds if seed not in by_seed]
    if missing:
        raise RuntimeError(
            f'missing selected {method_id} parents for training seeds: {missing}'
        )
    selected = [by_seed[seed] for seed in expected_seeds]
    if protocol['base']['checkpoint_source'].get(
        'require_distinct_checkpoint_sha256', False
    ) and len({row['checkpoint_sha256'] for row in selected}) != len(selected):
        raise RuntimeError('parent checkpoints are not distinct')
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('protocol', type=Path)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--no-import', action='store_true')
    args = parser.parse_args()
    protocol_path = args.protocol
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    protocol = yaml.safe_load(protocol_path.read_text(encoding='utf-8'))
    if protocol.get('schema_version') != 1:
        raise ValueError('unsupported protocol schema')
    expanded = variants(protocol)
    try:
        parents = selected_parents(protocol)
        blocked = None
    except RuntimeError as error:
        parents, blocked = [], str(error)
    planned = []
    scope = protocol['intervention']['scope']
    for parent in parents:
        for variant in expanded:
            overrides = {
                f'{scope}.{key}': value
                for key, value in variant['values'].items()
            }
            name = variant_name(
                protocol['intervention']['variant_id'], variant['factors']
            )
            command = [
                sys.executable,
                str(ROOT / 'tools/experiments/evaluate.py'),
                '--train-run-id',
                parent['train_run_id'],
                '--split',
                protocol['split'],
                '--inference-seed',
                str(protocol['inference_seed']),
                '--gpu-id',
                str(args.gpu_id),
                '--override-json',
                stable_json(overrides),
                '--protocol-source',
                str(protocol_path),
                '--status',
                'test_evaluated'
                if protocol['split'] == 'val'
                else 'paper_eligible',
            ]
            if args.no_import:
                command.append('--no-import')
            planned.append(
                {
                    'training_seed': parent['training_seed'],
                    'train_run_id': parent['train_run_id'],
                    'variant': name,
                    'overrides': overrides,
                    'command': command,
                }
            )
    expected = protocol.get('evaluation', {}).get('expected_evaluations')
    if expected is not None and parents and len(planned) != int(expected):
        raise RuntimeError(
            f'protocol expansion mismatch: {len(planned)} != {expected}'
        )
    summary = {
        'status': 'BLOCKED_PARENT' if blocked else 'READY',
        'protocol_id': protocol['protocol_id'],
        'dataset_id': protocol['dataset_id'],
        'parents': len(parents),
        'variants_per_parent': len(expanded),
        'evaluations': len(planned),
        'blocked_reason': blocked,
        'runs': [
            {
                key: row[key]
                for key in (
                    'training_seed',
                    'train_run_id',
                    'variant',
                    'overrides',
                )
            }
            for row in planned
        ],
    }
    print(json.dumps(summary, indent=2))
    if not args.execute:
        return 0
    if blocked:
        return 2
    for index, row in enumerate(planned, start=1):
        print(
            f'[{index}/{len(planned)}] seed={row["training_seed"]} '
            f'variant={row["variant"]}',
            flush=True,
        )
        completed = subprocess.run(row['command'], cwd=ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
