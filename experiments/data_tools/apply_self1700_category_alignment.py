#!/usr/bin/env python3
"""Idempotently align Self1700 category IDs to the Dataset 2 schema."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/ChromosomeSelf1700_coco"
EVIDENCE = (
    ROOT
    / "tools/experiment_db/evidence_sources/self1700_category_alignment_v2.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    categories = evidence["canonical_categories"]
    output = {}
    for split in ("train", "valid", "test"):
        path = DATA_ROOT / split / "_annotations.coco.json"
        expected = evidence["splits"][split]
        current_sha = sha256_file(path)
        if current_sha == expected["after_sha256"]:
            output[split] = {"status": "already_aligned", "sha256": current_sha}
            continue
        if current_sha != expected["before_sha256"]:
            raise RuntimeError(f"{split}: unexpected pre-alignment annotation SHA")
        mapping = {
            int(old): int(new) for old, new in expected["old_to_new_id"].items()
        }
        document = json.loads(path.read_text(encoding="utf-8"))
        for annotation in document["annotations"]:
            annotation["category_id"] = mapping[int(annotation["category_id"])]
        document["categories"] = categories
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        if sha256_file(temporary) != expected["after_sha256"]:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"{split}: aligned annotation SHA mismatch")
        os.replace(temporary, path)
        output[split] = {"status": "aligned", "sha256": sha256_file(path)}
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
