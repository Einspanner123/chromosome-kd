#!/usr/bin/env python3
"""Import cross-server accuracy evidence into the canonical Ross database."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "tools/experiment_db/experiments.db"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_aggregate(source: Path, staging: Path, server: str,
                     gpu_name: str, raw_root: str) -> tuple[Path, str]:
    aggregate = json.loads(source.read_text())
    if aggregate.get("status") != "verified" or len(aggregate.get("groups", [])) != 1:
        raise RuntimeError(f"invalid single-group aggregate: {source}")
    group = aggregate["groups"][0]
    for row in aggregate["results"]:
        original = staging / row["source_summary"]
        if not original.is_file():
            raise FileNotFoundError(original)
        if sha256_file(original) != row["source_summary_sha256"]:
            raise RuntimeError(f"summary hash mismatch: {original}")
        destination_relative = Path(
            "tools/experiment_db/evidence_sources/raw/"
            f"d2_test_{group}_{row['model_id']}_{row['protocol_hash'][:12]}_summary.json"
        )
        destination = ROOT / destination_relative
        if destination.exists() and sha256_file(destination) != row["source_summary_sha256"]:
            raise RuntimeError(f"immutable summary collision: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(original, destination)
        row["source_summary"] = str(destination_relative)
        row["raw_storage"] = (
            f"{server}:{raw_root}/{group}/{row['model_id']}/"
            f"{row['protocol_hash']}"
        )
    aggregate["execution"] = {
        "server": server,
        "gpu": gpu_name,
        "purpose": "accuracy_only",
        "seed": aggregate["seed"],
        "efficiency_eligible": False,
    }
    payload = (json.dumps(aggregate, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    relative_path = Path(
        f"tools/experiment_db/evidence_sources/d2_test_inference_ablations_"
        f"{digest[:12]}.json"
    )
    output = ROOT / relative_path
    if output.exists() and output.read_bytes() != payload:
        raise RuntimeError(f"immutable aggregate collision: {output}")
    output.write_bytes(payload)
    artifact_id = f"d2-test-inference-ablations-{digest[:12]}"

    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """INSERT OR IGNORE INTO evidence_artifact
               (artifact_id, server, path, sha256, kind, generated_at, status, notes)
               VALUES (?, ?, ?, ?, 'inference_ablation_test_aggregate', ?,
                       'verified', ?)""",
            (artifact_id, server, str(relative_path), digest,
             aggregate["completed_at"],
             f"Accuracy-only on {gpu_name}; raw predictions retained at {raw_root}."),
        )
        for row in aggregate["results"]:
            prefix = f"d2-test-{group.replace('_', '-')}"
            protocol_json = json.dumps(row["protocol"], sort_keys=True)
            for metric, value in row["metrics"].items():
                result_id = (
                    f"{prefix}-{row['model_id']}-{metric.lower()}-"
                    f"{row['protocol_hash'][:12]}"
                )
                existing = connection.execute(
                    "SELECT value, protocol_json FROM controlled_result WHERE result_id=?",
                    (result_id,),
                ).fetchone()
                if existing and existing != (value, protocol_json):
                    raise RuntimeError(f"controlled-result collision: {result_id}")
                connection.execute(
                    """INSERT OR REPLACE INTO controlled_result
                       (result_id, family, variant, dataset, split, seed, metric,
                        value, unit, baseline_result_id, delta, protocol_json,
                        artifact_id, evidence_level, paper_eligible, notes)
                       VALUES (?, ?, ?, 'D2', 'test', ?, ?, ?, 'absolute',
                               NULL, NULL, ?, ?, 'controlled', 1, ?)""",
                    (result_id, f"{group}_d2_test", row["model_label"],
                     str(aggregate["seed"]), metric, value, protocol_json,
                     artifact_id,
                     f"Locked test confirmation on {server}/{gpu_name}; "
                     "not used for hyperparameter selection or efficiency."),
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return output, artifact_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate")
    parser.add_argument("--staging", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--raw-root", required=True)
    args = parser.parse_args()
    path, artifact = import_aggregate(
        Path(args.aggregate), Path(args.staging), args.server,
        args.gpu_name, args.raw_root,
    )
    print(f"Imported {path.relative_to(ROOT)} ({artifact})")


if __name__ == "__main__":
    main()
