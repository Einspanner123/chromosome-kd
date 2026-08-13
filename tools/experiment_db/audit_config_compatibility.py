#!/usr/bin/env python3
"""Audit compatibility mappings and enforce the EXACT-only canonical gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / 'tools/experiment_db/experiments.db'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    conn = sqlite3.connect(DB)
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='config_compatibility'").fetchone()
    errors = []
    rows = []
    if not exists:
        errors.append('config_compatibility table missing')
    else:
        rows = conn.execute(
            """SELECT c.report_id,c.status,c.canonical_config_id,
                      a.path,a.sha256,c.training_seed
               FROM config_compatibility c
               JOIN evidence_artifact a ON a.artifact_id=c.artifact_id""").fetchall()
        for report_id, status, canonical, path_text, expected, seed in rows:
            path = ROOT / path_text
            if not path.is_file() or digest(path) != expected:
                errors.append(f'{report_id}: report file/hash mismatch')
                continue
            report = json.loads(path.read_text())
            if report['status'] != status:
                errors.append(f'{report_id}: DB/report status mismatch')
            if report['seeds']['authoritative_training_seed'] != seed:
                errors.append(f'{report_id}: DB/report seed mismatch')
            if canonical is not None and status != 'EXACT':
                errors.append(f'{report_id}: non-EXACT canonical alias')
            if bool(canonical) != bool(report['canonical_import_allowed']):
                # An EXACT report may be registered without mapping until the
                # caller explicitly requests --map-canonical.
                if canonical is not None or status != 'EXACT':
                    errors.append(f'{report_id}: canonical gate mismatch')
    errors.extend(f'foreign-key violation: {row}'
                  for row in conn.execute('PRAGMA foreign_key_check'))
    conn.close()
    print(json.dumps(dict(status='PASS' if not errors else 'FAIL',
                          mappings=len(rows), errors=errors), indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
