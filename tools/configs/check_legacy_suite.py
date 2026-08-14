#!/usr/bin/env python3
"""Run or aggregate a declared suite of legacy compatibility checks."""

from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / 'tools/configs/check_legacy_compatibility.py'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True)
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--index-output')
    args = parser.parse_args()
    suite_path = Path(args.suite)
    if not suite_path.is_absolute():
        suite_path = ROOT / suite_path
    suite = yaml.safe_load(suite_path.read_text())
    runs = suite['runs']
    if args.only:
        selected = set(args.only)
        runs = [run for run in runs if run['run_id'] in selected]
        missing = selected - {run['run_id'] for run in runs}
        if missing:
            raise ValueError(f'unknown run IDs: {sorted(missing)}')
    if args.workers < 1:
        raise ValueError('--workers must be positive')

    def execute(run):
        target_matrix = run.get('target_matrix', suite.get('target_matrix'))
        if not target_matrix:
            raise ValueError(
                f'{run["run_id"]}: target_matrix is required on the run or suite'
            )
        command = [
            sys.executable,
            str(CHECKER),
            '--legacy-config',
            run['legacy_config'],
            '--checkpoint',
            run['checkpoint'],
            '--training-log',
            run['training_log'],
            '--result-summary',
            run['result_summary'],
            '--target-matrix',
            target_matrix,
            '--target-method',
            run['target_method'],
            '--target-seed',
            str(run['target_seed']),
            '--output',
            run['output'],
        ]
        if run.get('parent_checkpoint'):
            command.extend(['--parent-checkpoint', run['parent_checkpoint']])
        subprocess.run(command, cwd=ROOT, check=True)
        return run['run_id']

    if not args.aggregate_only:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(execute, runs))

    records = []
    for run in runs:
        path = ROOT / run['output']
        report = json.loads(path.read_text())
        records.append(
            dict(
                run_id=run['run_id'],
                report_path=run['output'],
                report_sha256=digest(path),
                report_id=report['report_id'],
                status=report['status'],
                canonical_import_allowed=report['canonical_import_allowed'],
                training_seed=report['seeds']['authoritative_training_seed'],
                checkpoint_sha256=report['legacy']['checkpoint']['sha256'],
                scientific_difference_paths=[
                    item['path'] for item in report['scientific_differences']
                ],
                hard_issues=report['hard_issues'],
            )
        )
    index = dict(
        schema_version=1,
        suite_id=suite['suite_id'],
        suite_path=str(suite_path.relative_to(ROOT)),
        target_matrix=suite.get('target_matrix'),
        target_matrices=sorted(
            {
                run.get('target_matrix', suite.get('target_matrix'))
                for run in runs
            }
        ),
        reports=records,
        status_counts={
            status: sum(x['status'] == status for x in records)
            for status in (
                'EXACT',
                'STRUCTURAL_ONLY',
                'INCOMPATIBLE',
                'MISSING_EVIDENCE',
            )
        },
    )
    output = (
        ROOT / args.index_output
        if args.index_output
        else ROOT
        / 'tools/experiment_db/compatibility_reports'
        / f'{suite["suite_id"]}_index.json'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
    print(
        json.dumps(
            dict(
                status='PASS',
                reports=len(records),
                index=display(output),
                status_counts=index['status_counts'],
            ),
            indent=2,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
