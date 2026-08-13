#!/usr/bin/env python3
"""Register verified compatibility reports without aliasing non-EXACT runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / 'tools/experiment_db/experiments.db'
SCHEMA = ROOT / 'tools/experiment_db/schema.sql'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify_source(path_text: str, expected: str) -> None:
    path = resolve(path_text)
    if not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f'evidence drift: {path_text}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('reports', nargs='+')
    parser.add_argument(
        '--map-canonical', action='store_true',
        help='Populate canonical_config_id; blocked unless every report is EXACT')
    args = parser.parse_args()
    paths = [resolve(value) for value in args.reports]
    reports = [json.loads(path.read_text()) for path in paths]
    if args.map_canonical and any(x['status'] != 'EXACT' for x in reports):
        blocked = [x['report_id'] for x in reports if x['status'] != 'EXACT']
        raise RuntimeError(f'canonical mapping blocked for non-EXACT reports: {blocked}')

    conn = sqlite3.connect(DB)
    conn.execute('PRAGMA foreign_keys=ON')
    conn.executescript(SCHEMA.read_text())
    for path, report in zip(paths, reports):
        verify_source(report['legacy']['config_path'],
                      report['legacy']['config_sha256'])
        verify_source(report['legacy']['checkpoint']['path'],
                      report['legacy']['checkpoint']['sha256'])
        verify_source(report['legacy']['training_log']['path'],
                      report['legacy']['training_log']['sha256'])
        verify_source(report['result_evidence']['path'],
                      report['result_evidence']['sha256'])
        report_sha = sha256_file(path)
        artifact_id = f'config-compat-{report_sha[:12]}'
        target = report['target']
        coupling = next((d['legacy'] for d in report['scientific_differences']
                         if d['path'] == 'model.bbox_head.coupling.type'),
                        'matched')
        legacy_method_id = (
            f'legacy.{target["method_id"]}.{str(coupling).lower()}')
        relative = str(path.relative_to(ROOT))
        stale = conn.execute(
            """SELECT report_id,artifact_id FROM config_compatibility
               WHERE legacy_config_sha256=? AND checkpoint_sha256=?
                 AND target_config_id=? AND report_id<>?""",
            (report['legacy']['config_sha256'],
             report['legacy']['checkpoint']['sha256'],
             target['config_id'], report['report_id'])).fetchall()
        stale.extend(conn.execute(
            """SELECT c.report_id,c.artifact_id
               FROM config_compatibility c
               JOIN evidence_artifact a ON a.artifact_id=c.artifact_id
               WHERE a.path=? AND c.report_id<>?""",
            (relative, report['report_id'])).fetchall())
        stale = sorted(set(stale))
        for stale_report_id, stale_artifact_id in stale:
            conn.execute('DELETE FROM config_compatibility WHERE report_id=?',
                         (stale_report_id,))
            conn.execute(
                """DELETE FROM evidence_artifact WHERE artifact_id=?
                   AND NOT EXISTS (
                     SELECT 1 FROM config_compatibility WHERE artifact_id=?)""",
                (stale_artifact_id, stale_artifact_id))
        conn.execute(
            """INSERT OR REPLACE INTO evidence_artifact
               (artifact_id,server,path,sha256,kind,generated_at,status,notes)
               VALUES(?,?,?,?,?,?,?,?)""",
            (artifact_id, 'archive', relative, report_sha,
             'legacy_config_compatibility_report', report['generated_at'],
             'verified', f'{report["status"]}; canonical alias guarded'),
        )
        conn.execute(
            """INSERT OR REPLACE INTO config_compatibility
               (report_id,legacy_method_id,legacy_config_path,
                legacy_config_sha256,checkpoint_sha256,target_config_id,
                target_scientific_config_sha256,status,canonical_config_id,
                training_seed,artifact_id,verified_at,notes)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (report['report_id'], legacy_method_id,
             report['legacy']['config_path'], report['legacy']['config_sha256'],
             report['legacy']['checkpoint']['sha256'], target['config_id'],
             target['scientific_config_sha256'], report['status'],
             target['config_id'] if args.map_canonical else None,
             report['seeds']['authoritative_training_seed'], artifact_id,
             report['generated_at'],
             'Historical result preserved under an explicit compatibility identity'),
        )
    violations = conn.execute('PRAGMA foreign_key_check').fetchall()
    if violations:
        raise RuntimeError(f'foreign-key violations: {violations}')
    conn.commit()
    rows = conn.execute(
        """SELECT report_id,status,canonical_config_id
           FROM config_compatibility ORDER BY report_id""").fetchall()
    conn.close()
    print(json.dumps(dict(status='PASS', registered=len(reports), rows=rows),
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
