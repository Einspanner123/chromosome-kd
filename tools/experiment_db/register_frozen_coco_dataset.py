#!/usr/bin/env python3
"""Validate and register a frozen COCO dataset release."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / 'tools/experiment_db/experiments.db'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--db', type=Path, default=DB)
    parser.add_argument('--display-name', required=True)
    parser.add_argument('--notes', default='')
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    if payload.get('evidence_type') != 'frozen_coco_dataset_manifest':
        raise RuntimeError('unexpected manifest evidence type')
    digest = sha256_file(manifest_path)
    dataset_id = str(payload['dataset_id'])
    splits = payload['splits']
    total_images = 0
    for split, record in splits.items():
        annotation = ROOT / record['annotation_path']
        if sha256_file(annotation) != record['annotation_sha256']:
            raise RuntimeError(f'{split}: annotation SHA mismatch')
        coco = json.loads(annotation.read_text(encoding='utf-8'))
        if len(coco['images']) != record['images']:
            raise RuntimeError(f'{split}: image count mismatch')
        if len(coco['annotations']) != record['annotations']:
            raise RuntimeError(f'{split}: annotation count mismatch')
        observed = [
            {'id': int(item['id']), 'name': str(item['name'])}
            for item in coco['categories']
        ]
        if observed != payload['categories']:
            raise RuntimeError(f'{split}: category schema mismatch')
        total_images += int(record['images'])

    relative_manifest = manifest_path.relative_to(ROOT).as_posix()
    artifact_id = f'dataset-{dataset_id.lower()}-{digest[:12]}'
    construction = {
        'dataset_id': dataset_id,
        'split_protocol': payload['split_protocol'],
        'relative_data_root': payload['relative_data_root'],
        'annotation_sha256': {
            split: record['annotation_sha256']
            for split, record in splits.items()
        },
        'image_inventory_sha256': {
            split: record['image_inventory_sha256']
            for split, record in splits.items()
        },
    }
    connection = sqlite3.connect(args.db)
    connection.execute('PRAGMA foreign_keys=ON')
    connection.execute(
        """INSERT OR REPLACE INTO evidence_artifact
           (artifact_id,server,path,sha256,kind,status,notes)
           VALUES (?, 'ross', ?, ?, 'frozen_coco_dataset_manifest',
                   'verified', ?)""",
        (
            artifact_id,
            relative_manifest,
            digest,
            f'Frozen identity for {dataset_id}; image inventories and COCO annotations verified.',
        ),
    )
    connection.execute(
        """INSERT OR REPLACE INTO dataset_release
           (dataset_id,display_name,root_path,version,image_count,
            category_count,construction_protocol_json,manifest_artifact_id,
            status,notes) VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            dataset_id,
            args.display_name,
            payload['relative_data_root'],
            digest[:12],
            total_images,
            len(payload['categories']),
            json.dumps(construction, sort_keys=True),
            artifact_id,
            'verified',
            args.notes,
        ),
    )
    connection.execute('DELETE FROM dataset_split WHERE dataset_id=?', (dataset_id,))
    for split, record in splits.items():
        connection.execute(
            """INSERT INTO dataset_split
               (dataset_id,split,image_count,annotation_count,
                annotation_path,annotation_sha256) VALUES (?,?,?,?,?,?)""",
            (
                dataset_id,
                split,
                record['images'],
                record['annotations'],
                record['annotation_path'],
                record['annotation_sha256'],
            ),
        )
    connection.commit()
    print(
        json.dumps(
            {
                'artifact_id': artifact_id,
                'dataset_id': dataset_id,
                'manifest_sha256': digest,
                'images': total_images,
                'splits': sorted(splits),
            },
            indent=2,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
