#!/usr/bin/env python3
"""Register the synchronized D2 solver/Top-K prediction archive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / 'tools/experiment_db/experiments.db'
ARCHIVE = ROOT / 'results/evidence_archives/d2_inference_predictions_20260814.tar.gz'
SOURCE_ROOTS = (
    ROOT / 'results/d2_test_inference_ablations/solver',
    ROOT / 'results/d2_test_inference_ablations/topk_renewal',
)
MANIFEST_DIR = ROOT / 'tools/experiment_db/evidence_sources'
EXPECTED_FILES = 88
EXPECTED_ARCHIVE_SHA256 = (
    'ccaaef876bd05dfb000b1d76f1db3b4a4914bdf25f81dcf75c03f18f716eaa24'
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> int:
    archive_sha = sha256_file(ARCHIVE)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(f'archive SHA mismatch: {archive_sha}')
    files = []
    for source_root in SOURCE_ROOTS:
        for path in sorted(p for p in source_root.rglob('*') if p.is_file()):
            files.append(dict(path=relative(path), size=path.stat().st_size,
                              sha256=sha256_file(path)))
    if len(files) != EXPECTED_FILES:
        raise RuntimeError(f'expected {EXPECTED_FILES} files, found {len(files)}')
    payload = dict(
        schema_version=1,
        evidence_type='synchronized_raw_prediction_archive',
        dataset='D2',
        split='test',
        groups=['solver', 'topk_renewal'],
        archive=dict(path=relative(ARCHIVE), size=ARCHIVE.stat().st_size,
                     sha256=archive_sha),
        file_count=len(files),
        files=files,
    )
    raw = json.dumps(payload, indent=2, sort_keys=True) + '\n'
    manifest_sha = hashlib.sha256(raw.encode()).hexdigest()
    manifest = MANIFEST_DIR / f'd2_inference_raw_predictions_{manifest_sha[:12]}.json'
    temporary = manifest.with_suffix('.json.tmp')
    temporary.write_text(raw)
    temporary.replace(manifest)

    rows = [
        (f'd2-inference-predictions-archive-{archive_sha[:12]}', 'ross',
         relative(ARCHIVE), archive_sha, 'raw_prediction_archive', None,
         'verified', f'File-level manifest: {relative(manifest)}'),
        (f'd2-inference-predictions-manifest-{manifest_sha[:12]}', 'ross',
         relative(manifest), manifest_sha, 'raw_prediction_manifest', None,
         'verified', f'{len(files)} synchronized solver/Top-K evidence files'),
    ]
    connection = sqlite3.connect(DB)
    with connection:
        # This evidence is content-addressed.  A previous interrupted transfer
        # may have produced a manifest with the same paths but partial file
        # contents; retain only the manifest that matches the immutable archive.
        connection.execute(
            '''DELETE FROM evidence_artifact
               WHERE artifact_id LIKE 'd2-inference-predictions-manifest-%'
                 AND artifact_id != ?''',
            (f'd2-inference-predictions-manifest-{manifest_sha[:12]}',),
        )
        connection.executemany(
            '''INSERT OR REPLACE INTO evidence_artifact(
               artifact_id,server,path,sha256,kind,generated_at,status,notes)
               VALUES(?,?,?,?,?,?,?,?)''', rows)
    for stale in MANIFEST_DIR.glob('d2_inference_raw_predictions_*.json'):
        if stale != manifest:
            stale.unlink()
    print(json.dumps(dict(status='PASS', archive_sha256=archive_sha,
                          manifest=relative(manifest),
                          manifest_sha256=manifest_sha,
                          files=len(files)), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
