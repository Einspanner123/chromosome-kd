#!/usr/bin/env python3
"""Derive the Self1700 V2 provenance manifest after category-schema alignment."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools/experiment_db/evidence_sources/dataset_self1700_v1_provenance_57bc9516b11a.json"
ALIGNMENT = ROOT / "tools/experiment_db/evidence_sources/self1700_category_alignment_v2.json"
DATA_ROOT = ROOT / "data/ChromosomeSelf1700_coco"
CANONICAL = (
    "A1", "A2", "A3", "B4", "B5", "C6", "C7", "C8", "C9",
    "C10", "C11", "C12", "D13", "D14", "D15", "E16", "E17",
    "E18", "F19", "F20", "G21", "G22", "X", "Y",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def main() -> None:
    manifest = json.loads(SOURCE.read_text())
    alignment = json.loads(ALIGNMENT.read_text())
    manifest["schema_version"] = 2
    manifest["dataset_id"] = "D1_INHOUSE1700_V2"
    manifest["construction_protocol"]["dataset_id"] = "D1_INHOUSE1700_V2"
    manifest["construction_protocol"]["label_schema"] = "Dataset2-aligned canonical chromosome IDs"
    manifest["label_schema"] = {
        "version": 2,
        "canonical_reference": "D2",
        "categories": [{"id": i, "name": n} for i, n in enumerate(CANONICAL, 1)],
        "alignment_evidence": str(ALIGNMENT.relative_to(ROOT)),
        "alignment_evidence_sha256": sha(ALIGNMENT),
        "semantic_annotation_digest_preserved": True,
    }
    for split in ("train", "valid", "test"):
        path = DATA_ROOT / split / "_annotations.coco.json"
        data = json.loads(path.read_text())
        mapping = [(c["id"], c["name"]) for c in data["categories"]]
        assert mapping == list(enumerate(CANONICAL, 1))
        counts = Counter(str(a["category_id"]) for a in data["annotations"])
        manifest["splits"][split].update({
            "sha256": sha(path),
            "images": len(data["images"]),
            "annotations": len(data["annotations"]),
            "category_counts": dict(sorted(counts.items(), key=lambda x: int(x[0]))),
            "semantic_sha256": alignment["splits"][split]["semantic_sha256"],
        })
    provisional = ROOT / "tools/experiment_db/evidence_sources/dataset_self1700_v2_provenance.json"
    atomic(provisional, manifest)
    digest = sha(provisional)
    final = provisional.with_name(f"dataset_self1700_v2_provenance_{digest[:12]}.json")
    os.replace(provisional, final)
    print(json.dumps({
        "dataset_id": manifest["dataset_id"],
        "manifest": str(final.relative_to(ROOT)),
        "manifest_sha256": digest,
        "annotation_sha256": {s: manifest["splits"][s]["sha256"] for s in ("train", "valid", "test")},
    }, indent=2))


if __name__ == "__main__":
    main()
