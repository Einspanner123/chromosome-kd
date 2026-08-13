#!/usr/bin/env python3
"""Align Self1700 COCO category IDs to the canonical Dataset2 schema."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

CANONICAL = (
    "A1", "A2", "A3", "B4", "B5", "C6", "C7", "C8", "C9",
    "C10", "C11", "C12", "D13", "D14", "D15", "E16", "E17",
    "E18", "F19", "F20", "G21", "G22", "X", "Y",
)
SPLITS = ("train", "valid", "test")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def semantic_digest(data: dict) -> str:
    id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    records = []
    for ann in data["annotations"]:
        records.append({
            "id": ann["id"],
            "image_id": ann["image_id"],
            "class_name": id_to_name[ann["category_id"]],
            "bbox": ann.get("bbox"),
            "area": ann.get("area"),
            "iscrowd": ann.get("iscrowd"),
            "segmentation": ann.get("segmentation"),
        })
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def align(data: dict) -> tuple[dict, dict]:
    original = copy.deepcopy(data)
    old_by_id = {c["id"]: c for c in original["categories"]}
    old_name_to_id = {c["name"]: c["id"] for c in original["categories"]}
    if len(old_by_id) != 24 or set(old_name_to_id) != set(CANONICAL):
        raise ValueError("annotation categories are not a unique canonical 24-class set")

    new_id_by_name = {name: idx for idx, name in enumerate(CANONICAL, 1)}
    old_to_new = {
        old_id: new_id_by_name[category["name"]]
        for old_id, category in old_by_id.items()
    }
    before_semantic = semantic_digest(original)
    before_counts = Counter(old_by_id[a["category_id"]]["name"] for a in original["annotations"])

    for ann in data["annotations"]:
        ann["category_id"] = old_to_new[ann["category_id"]]
    categories = []
    for new_id, name in enumerate(CANONICAL, 1):
        category = copy.deepcopy(old_by_id[old_name_to_id[name]])
        category["id"] = new_id
        category["name"] = name
        categories.append(category)
    data["categories"] = categories

    after_semantic = semantic_digest(data)
    after_by_id = {c["id"]: c for c in data["categories"]}
    after_counts = Counter(after_by_id[a["category_id"]]["name"] for a in data["annotations"])
    if before_semantic != after_semantic:
        raise AssertionError("semantic annotation digest changed")
    if before_counts != after_counts:
        raise AssertionError("per-class annotation counts changed")
    if [(c["id"], c["name"]) for c in data["categories"]] != list(enumerate(CANONICAL, 1)):
        raise AssertionError("canonical category mapping was not produced")
    return data, {
        "old_to_new_id": {str(k): v for k, v in sorted(old_to_new.items())},
        "semantic_sha256": after_semantic,
        "images": len(data.get("images", [])),
        "annotations": len(data.get("annotations", [])),
        "class_counts": dict(sorted(after_counts.items())),
    }


def atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-root", type=Path, required=True)
    parser.add_argument("--d2-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    report = {
        "schema_version": 1,
        "operation": "self1700_category_id_alignment_to_dataset2",
        "canonical_categories": [{"id": i, "name": n} for i, n in enumerate(CANONICAL, 1)],
        "applied": bool(args.apply),
        "splits": {},
    }
    backup = args.self_root / ".category_schema_backup_v1"
    if args.apply:
        backup.mkdir(exist_ok=True)

    for split in SPLITS:
        source = args.self_root / split / "_annotations.coco.json"
        before_sha = file_sha(source)
        original = json.loads(source.read_text())
        aligned, info = align(copy.deepcopy(original))
        if args.apply:
            backup_path = backup / f"{split}_annotations_{before_sha[:12]}.coco.json"
            if not backup_path.exists():
                shutil.copy2(source, backup_path)
            atomic_json(source, aligned)
        effective = json.loads(source.read_text()) if args.apply else aligned
        info.update({"before_sha256": before_sha, "after_sha256": file_sha(source) if args.apply else None})
        report["splits"][split] = info
        if [(c["id"], c["name"]) for c in effective["categories"]] != list(enumerate(CANONICAL, 1)):
            raise AssertionError(f"{split}: output mapping mismatch")

    for split in SPLITS:
        d2 = json.loads((args.d2_root / split / "_annotations.coco.json").read_text())
        mapping = [(c["id"], c["name"]) for c in d2["categories"]]
        if mapping != list(enumerate(CANONICAL, 1)):
            raise AssertionError(f"Dataset2 {split} is not canonical: {mapping}")
    report["dataset2_mapping_verified"] = True

    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.evidence, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
