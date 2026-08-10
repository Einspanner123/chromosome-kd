"""Fail-fast quality audit for the canonical experiment evidence layer."""

import argparse
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

    for artifact_id, path in conn.execute(
            "SELECT artifact_id, path FROM evidence_artifact WHERE status='verified'"):
        if not os.path.isfile(os.path.join(ROOT, path)):
            errors.append(f'{artifact_id}: missing source {path}')

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
