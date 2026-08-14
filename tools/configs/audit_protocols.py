#!/usr/bin/env python3
"""Static audit for v2 inference/selection protocol manifests."""

from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
ABLATIONS = ROOT / 'experiments/configs/ablations'


def main() -> int:
    errors = []
    protocol_ids = set()
    for path in sorted(ABLATIONS.glob('*.yaml')):
        data = yaml.safe_load(path.read_text())
        required = {'schema_version', 'protocol_id', 'protocol_type',
                    'dataset_id', 'split', 'selection', 'base',
                    'intervention'}
        missing = sorted(required - set(data or {}))
        if missing:
            errors.append(f'{path.name}: missing {missing}')
            continue
        pid = data['protocol_id']
        if pid in protocol_ids:
            errors.append(f'{path.name}: duplicate protocol_id {pid}')
        protocol_ids.add(pid)
        if data['split'] == 'test' and data['selection'].get('test_tuned') is not False:
            errors.append(f'{path.name}: test protocol must declare test_tuned=false')
        base = data['base']
        matrix = ROOT / base['matrix']
        method = ROOT / 'experiments/configs/methods' / f"{base['method']}.py"
        if not matrix.is_file():
            errors.append(f'{path.name}: missing matrix {base["matrix"]}')
        if not method.is_file():
            errors.append(f'{path.name}: missing method {base["method"]}')
        factors = data['intervention'].get('factors', {})
        if not factors or any(not values for values in factors.values()):
            errors.append(f'{path.name}: empty intervention grid')
        checkpoints = base.get('checkpoints')
        if checkpoints:
            seeds = [item['training_seed'] for item in checkpoints]
            hashes = [item['sha256'] for item in checkpoints]
            if len(set(seeds)) != len(seeds) or len(set(hashes)) != len(hashes):
                errors.append(f'{path.name}: checkpoint replication is not independent')
    if errors:
        print('PROTOCOL AUDIT: FAIL')
        for error in errors:
            print('-', error)
        return 1
    print(f'PROTOCOL AUDIT: PASS protocols={len(protocol_ids)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
