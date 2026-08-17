#!/usr/bin/env python3
"""Build a deterministic identity manifest for a frozen COCO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def split_record(root: Path, split: str) -> dict:
    split_dir = root / split
    annotation = split_dir / '_annotations.coco.json'
    coco = json.loads(annotation.read_text(encoding='utf-8'))
    declared = [str(item['file_name']) for item in coco['images']]
    if len(declared) != len(set(declared)):
        raise RuntimeError(f'{split}: duplicate file_name entries')
    missing = [name for name in declared if not (split_dir / name).is_file()]
    if missing:
        raise RuntimeError(f'{split}: {len(missing)} annotation images missing')
    actual = sorted(
        path.name
        for path in split_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if sorted(declared) != actual:
        raise RuntimeError(f'{split}: annotation/directory image sets differ')

    inventory = hashlib.sha256()
    total_bytes = 0
    for name in sorted(declared):
        path = split_dir / name
        size = path.stat().st_size
        digest = sha256_file(path)
        inventory.update(f'{name}\0{size}\0{digest}\n'.encode())
        total_bytes += size
    return {
        'annotation_path': (Path(split) / annotation.name).as_posix(),
        'annotation_sha256': sha256_file(annotation),
        'images': len(coco['images']),
        'annotations': len(coco['annotations']),
        'categories': len(coco['categories']),
        'image_bytes': total_bytes,
        'image_inventory_sha256': inventory.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-id', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'])
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    records = {split: split_record(root, split) for split in args.splits}
    relative_root = args.root.as_posix().rstrip('/')
    for record in records.values():
        record['annotation_path'] = (
            f"{relative_root}/{record['annotation_path']}"
        )
    category_sets = []
    for split in args.splits:
        coco = json.loads(
            (root / split / '_annotations.coco.json').read_text(encoding='utf-8')
        )
        category_sets.append(
            [(int(item['id']), str(item['name'])) for item in coco['categories']]
        )
    if any(items != category_sets[0] for items in category_sets[1:]):
        raise RuntimeError('category id/name ordering differs across splits')

    payload = {
        'schema_version': 1,
        'evidence_type': 'frozen_coco_dataset_manifest',
        'dataset_id': args.dataset_id,
        'split_protocol': args.protocol,
        'relative_data_root': relative_root,
        'categories': [
            {'id': category_id, 'name': name}
            for category_id, name in category_sets[0]
        ],
        'splits': records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / f'dataset_{args.dataset_id.lower()}_manifest.json'
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    digest = sha256_file(temporary)
    final = args.output_dir / (
        f'dataset_{args.dataset_id.lower()}_original_split_manifest_{digest[:12]}.json'
    )
    temporary.replace(final)
    print(json.dumps({'path': str(final), 'sha256': digest}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
