#!/usr/bin/env python3
"""Remove non-test detector accuracy from the paper-eligible result set.

Dataset inventory and hardware benchmark rows are intentionally unaffected.
The operation is explicit and idempotent so a legacy database rebuild cannot
silently promote validation accuracy into paper claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / 'tools/experiment_db/experiments.db'
DETECTION_METRICS = (
    'mAP', 'AP50', 'AP75', 'AP90', 'AP95', 'AP_S', 'AP_M', 'AP_L',
    'AP90_delta', 'AP95_delta',
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=Path, default=DB)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    db = args.db if args.db.is_absolute() else ROOT / args.db
    conn = sqlite3.connect(db)
    placeholders = ','.join('?' for _ in DETECTION_METRICS)
    query = f'''SELECT result_id, family, split, metric
                FROM controlled_result
                WHERE paper_eligible=1 AND split!='test'
                  AND metric IN ({placeholders})
                ORDER BY result_id'''
    rows = conn.execute(query, DETECTION_METRICS).fetchall()
    if args.check:
        print(json.dumps(dict(
            status='PASS' if not rows else 'FAIL',
            invalid_rows=[dict(zip(
                ('result_id', 'family', 'split', 'metric'), row))
                for row in rows]), indent=2))
        return 0 if not rows else 1
    with conn:
        conn.executemany(
            'UPDATE controlled_result SET paper_eligible=0 WHERE result_id=?',
            [(row[0],) for row in rows],
        )
    print(json.dumps(dict(status='PASS', demoted=len(rows),
                          result_ids=[row[0] for row in rows]), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
