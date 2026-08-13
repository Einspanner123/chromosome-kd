#!/usr/bin/env python3
"""Verify and register a provenance-locked D1_INHOUSE1700 dataset release."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "tools/experiment_db/experiments.db"
ALLOWED_DATASET_IDS = {"D1_INHOUSE1700_V1", "D1_INHOUSE1700_V2"}

DDL = """
CREATE TABLE IF NOT EXISTS dataset_release (
    dataset_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    root_path TEXT NOT NULL,
    version TEXT NOT NULL,
    image_count INTEGER NOT NULL,
    category_count INTEGER NOT NULL,
    construction_protocol_json TEXT NOT NULL,
    manifest_artifact_id TEXT NOT NULL,
    status TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY (manifest_artifact_id) REFERENCES evidence_artifact(artifact_id)
);
CREATE TABLE IF NOT EXISTS dataset_split (
    dataset_id TEXT NOT NULL,
    split TEXT NOT NULL,
    image_count INTEGER NOT NULL,
    annotation_count INTEGER NOT NULL,
    annotation_path TEXT NOT NULL,
    annotation_sha256 TEXT NOT NULL,
    PRIMARY KEY (dataset_id, split),
    FOREIGN KEY (dataset_id) REFERENCES dataset_release(dataset_id)
);
CREATE TABLE IF NOT EXISTS dataset_sample_provenance (
    dataset_id TEXT NOT NULL,
    sample_file_sha256 TEXT NOT NULL,
    source_split TEXT NOT NULL,
    split TEXT NOT NULL,
    file_name TEXT NOT NULL,
    group_id TEXT NOT NULL,
    pixel_sha256 TEXT NOT NULL,
    provenance TEXT NOT NULL,
    included BOOLEAN NOT NULL,
    source_dataset_id TEXT,
    source_file_name TEXT,
    match_method TEXT NOT NULL,
    match_score REAL,
    PRIMARY KEY (dataset_id, sample_file_sha256),
    FOREIGN KEY (dataset_id) REFERENCES dataset_release(dataset_id)
);
CREATE INDEX IF NOT EXISTS idx_dataset_provenance
    ON dataset_sample_provenance(dataset_id, provenance, included);
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--db", type=Path, default=DB)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    dataset_id = manifest["dataset_id"]
    if dataset_id not in ALLOWED_DATASET_IDS:
        raise RuntimeError(f"unexpected dataset: {manifest['dataset_id']}")
    records = manifest["records"]
    included = [row for row in records if row["provenance"] == "in_house"]
    excluded = [row for row in records if row["provenance"] == "public_d2_derived"]
    if (len(records), len(included), len(excluded)) != (2200, 1700, 500):
        raise RuntimeError("manifest count invariant failed")
    if len({row["public_source_file"] for row in excluded}) != 500:
        raise RuntimeError("public source mapping is not one-to-one")
    if any(row["public_match_zncc"] < 0.99 for row in excluded):
        raise RuntimeError("public-derived match falls below ZNCC threshold")
    if len({row["pixel_sha256"] for row in records}) != 2200:
        raise RuntimeError("duplicate decoded pixels in source composite")
    group_splits = {}
    for row in included:
        previous = group_splits.setdefault(row["group_id"], row["output_split"])
        if previous != row["output_split"]:
            raise RuntimeError(f"group leakage detected: {row['group_id']}")
    for split, evidence in manifest["splits"].items():
        path = ROOT / evidence["path"]
        if sha256(path) != evidence["sha256"]:
            raise RuntimeError(f"annotation hash mismatch: {split}")
        document = json.loads(path.read_text())
        if len(document["images"]) != evidence["images"]:
            raise RuntimeError(f"image count mismatch: {split}")
        if len(document["annotations"]) != evidence["annotations"]:
            raise RuntimeError(f"annotation count mismatch: {split}")
        expected_categories = manifest.get("label_schema", {}).get("categories")
        observed_categories = [
            {"id": category["id"], "name": category["name"]}
            for category in document["categories"]
        ]
        if expected_categories and observed_categories != expected_categories:
            raise RuntimeError(f"category schema mismatch: {split}")

    relative = manifest_path.relative_to(ROOT)
    digest = sha256(manifest_path)
    artifact_id = f"dataset-self1700-{digest[:12]}"
    con = sqlite3.connect(args.db)
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(DDL)
    con.execute(
        """INSERT OR REPLACE INTO evidence_artifact
           (artifact_id,server,path,sha256,kind,status,notes)
           VALUES (?, 'ross', ?, ?, 'dataset_provenance_manifest', 'verified', ?)""",
        (artifact_id, str(relative), digest,
         "Immutable 2,200-image source audit: 1,700 in-house retained and 500 public-derived excluded."),
    )
    con.execute(
        """INSERT OR REPLACE INTO dataset_release
           (dataset_id,display_name,root_path,version,image_count,category_count,
            construction_protocol_json,manifest_artifact_id,status,notes)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (dataset_id, "Pure in-house chromosome dataset (1,700 images)",
         "data/ChromosomeSelf1700_coco", digest[:12], 1700, 24,
         json.dumps(manifest["construction_protocol"], sort_keys=True),
         artifact_id, "verified",
         "Group-disjoint 70/10/20 split; no public-derived images included."),
    )
    con.execute("DELETE FROM dataset_split WHERE dataset_id=?", (dataset_id,))
    for split, evidence in manifest["splits"].items():
        con.execute(
            """INSERT INTO dataset_split
               (dataset_id,split,image_count,annotation_count,annotation_path,annotation_sha256)
               VALUES (?,?,?,?,?,?)""",
            (dataset_id, split, evidence["images"], evidence["annotations"],
             evidence["path"], evidence["sha256"]),
        )
    con.execute("DELETE FROM dataset_sample_provenance WHERE dataset_id=?", (dataset_id,))
    for row in records:
        is_included = row["provenance"] == "in_house"
        con.execute(
            """INSERT INTO dataset_sample_provenance
               (dataset_id,sample_file_sha256,source_split,split,file_name,group_id,pixel_sha256,
                provenance,included,source_dataset_id,source_file_name,
                match_method,match_score) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (dataset_id, row["file_sha256"], row["source_split"],
             row["output_split"] or row["source_split"], row["file_name"], row["group_id"],
             row["pixel_sha256"], row["provenance"], int(is_included),
             None if is_included else "D2",
             None if is_included else row["public_source_file"],
             "basename+resize_zncc+one_to_one", row["public_match_zncc"]),
        )
    con.commit()
    print(json.dumps({
        "artifact_id": artifact_id,
        "dataset_id": dataset_id,
        "registered_samples": len(records),
        "included": len(included),
        "excluded": len(excluded),
    }, indent=2))


if __name__ == "__main__":
    main()
