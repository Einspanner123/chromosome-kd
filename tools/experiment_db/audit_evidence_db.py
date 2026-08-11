"""Fail-fast quality audit for the canonical experiment evidence layer."""

import argparse
import hashlib
import json
import os
import sqlite3


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.path.join(HERE, 'experiments.db'))
    args = parser.parse_args()
    conn = sqlite3.connect(args.db)
    errors = []

    fk_errors = conn.execute('PRAGMA foreign_key_check').fetchall()
    if fk_errors:
        errors.append(f'foreign-key violations: {len(fk_errors)}')

    duplicate = conn.execute('''SELECT experiment_id, split, source, epoch,
        checkpoint_path, COUNT(*) FROM evaluation GROUP BY 1,2,3,4,5
        HAVING COUNT(*) > 1''').fetchall()
    if duplicate:
        errors.append(f'duplicate evaluations: {len(duplicate)}')

    for result_id, metric, value, unit in conn.execute(
            'SELECT result_id, metric, value, unit FROM controlled_result'):
        if unit == 'absolute' and metric in ('mAP', 'AP50', 'AP75') and not 0 <= value <= 1:
            errors.append(f'{result_id}: out-of-range {metric}={value}')

    rows = conn.execute('''SELECT r.result_id, r.value, r.delta, b.value
        FROM controlled_result r JOIN controlled_result b
        ON r.baseline_result_id=b.result_id WHERE r.delta IS NOT NULL''').fetchall()
    for result_id, value, delta, baseline in rows:
        # Curated rounded summaries may differ in the seventh decimal place.
        if abs((value - baseline) - delta) > 5e-6:
            errors.append(f'{result_id}: delta arithmetic mismatch')

    for artifact_id, path, expected_sha, kind in conn.execute(
            "SELECT artifact_id, path, sha256, kind FROM evidence_artifact "
            "WHERE status='verified'"):
        abs_path = os.path.join(ROOT, path)
        if not os.path.isfile(abs_path):
            errors.append(f'{artifact_id}: missing source {path}')
            continue
        digest = hashlib.sha256()
        with open(abs_path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha:
            errors.append(f'{artifact_id}: SHA-256 mismatch')

        # This evidence class is intentionally stricter because an earlier
        # diagnostic file contained true-IoU oracle beta sweeps that could be
        # mistaken for learned-quality ablations.
        if kind == 'controlled_fixed_checkpoint_sweep':
            try:
                source = json.load(open(abs_path))
                sweep = source['learned_quality_beta_sweep']
                betas = [float(row['beta']) for row in sweep]
                if betas != [0.0, 0.25, 0.5, 1.0, 2.0]:
                    errors.append(f'{artifact_id}: unexpected learned beta grid')
                identity = source['checkpoint']['identity_audit']
                if identity['nonidentical_shared_tensors'] != 0:
                    errors.append(f'{artifact_id}: detector weights are not paired')
                if identity['extra_nonquality_tensors'] != 0:
                    errors.append(f'{artifact_id}: non-quality tensors were added')
                for row in sweep:
                    source_path = row.get('source_path', '')
                    source_file = os.path.join(ROOT, source_path)
                    if not os.path.isfile(source_file):
                        errors.append(
                            f"{artifact_id}: missing raw source at beta={row['beta']}")
                        continue
                    source_digest = hashlib.sha256()
                    with open(source_file, 'rb') as handle:
                        for chunk in iter(
                                lambda: handle.read(1024 * 1024), b''):
                            source_digest.update(chunk)
                    if source_digest.hexdigest() != row.get('source_sha256'):
                        errors.append(
                            f"{artifact_id}: raw source hash mismatch at "
                            f"beta={row['beta']}")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f'{artifact_id}: malformed learned sweep ({exc})')
        elif kind == 'dataset_annotation_audit':
            try:
                source = json.load(open(abs_path))
                for dataset in source['datasets']:
                    annotation = os.path.join(ROOT, dataset['annotation'])
                    if not os.path.isfile(annotation):
                        errors.append(
                            f"{artifact_id}: missing annotation {dataset['annotation']}")
                        continue
                    annotation_digest = hashlib.sha256()
                    with open(annotation, 'rb') as handle:
                        for chunk in iter(
                                lambda: handle.read(1024 * 1024), b''):
                            annotation_digest.update(chunk)
                    if annotation_digest.hexdigest() != dataset['annotation_sha256']:
                        errors.append(
                            f"{artifact_id}: annotation hash mismatch for "
                            f"{dataset['dataset']}")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f'{artifact_id}: malformed dataset audit ({exc})')

    orphan = conn.execute('''SELECT COUNT(*) FROM finding_evidence fe
        LEFT JOIN controlled_result r ON fe.result_id=r.result_id
        WHERE r.result_id IS NULL''').fetchone()[0]
    if orphan:
        errors.append(f'orphan finding evidence: {orphan}')

    counts = dict(conn.execute('''SELECT evidence_level, COUNT(*)
        FROM controlled_result GROUP BY evidence_level''').fetchall())
    report = {'status': 'FAIL' if errors else 'PASS', 'errors': errors,
              'controlled_result_counts': counts,
              'paper_eligible': conn.execute(
                  'SELECT COUNT(*) FROM controlled_result WHERE paper_eligible=1').fetchone()[0]}
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
