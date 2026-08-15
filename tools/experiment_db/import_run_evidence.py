#!/usr/bin/env python3
"""Idempotently import validated per-evaluation evidence into SQLite.

This is the only supported write path for new accuracy evaluations. Historical
manifest importers remain reproducibility tools for pre-standard evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from tools.experiment_db.validate_run_evidence import validate

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "tools/experiment_db/experiments.db"
SCHEMA = ROOT / "tools/experiment_db/schema.sql"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def insert_exact(connection: sqlite3.Connection, table: str, key_name: str,
                 key_value: str, values: dict) -> None:
    columns = list(values)
    existing = connection.execute(
        f"SELECT {', '.join(columns)} FROM {table} WHERE {key_name} = ?",
        (key_value,),
    ).fetchone()
    expected = tuple(values[column] for column in columns)
    if existing is not None:
        if tuple(existing) != expected:
            raise ValueError(
                f"conflicting immutable {table} row: {key_value}")
        return
    connection.execute(
        f"INSERT INTO {table} ({key_name}, {', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in range(len(columns) + 1))})",
        (key_value, *expected),
    )


def import_record(connection: sqlite3.Connection, record: dict,
                  record_path: Path, root: Path) -> None:
    errors = validate(record, root, verify_files=True)
    if errors:
        raise ValueError("; ".join(errors))
    if record["record_type"] != "evaluation":
        raise ValueError("import_run_evidence currently accepts evaluation records")
    if not isinstance(record.get("inference_seed"), int):
        raise ValueError("evaluation import requires an integer inference_seed")

    train_row = connection.execute(
        "SELECT method, dataset_id, training_seed FROM train_run_registry "
        "WHERE train_run_id = ?", (record["train_run_id"],),
    ).fetchone()
    if train_row is None:
        raise ValueError(
            f"train_run_id is not preregistered: {record['train_run_id']}")
    method, dataset_id, training_seed = train_row
    if dataset_id != record["dataset_id"]:
        raise ValueError("dataset_id disagrees with train_run_registry")
    if training_seed != record["training_seed"]:
        raise ValueError("training_seed disagrees with train_run_registry")

    relative_record = record_path.resolve().relative_to(root.resolve())
    record_sha = sha256_file(record_path)
    artifact_id = f"run-evidence-{record_sha[:12]}"
    insert_exact(connection, "evidence_artifact", "artifact_id", artifact_id, {
        "server": "repository",
        "path": relative_record.as_posix(),
        "sha256": record_sha,
        "kind": "run_evidence_v1",
        "generated_at": None,
        "status": "verified",
        "notes": f"Validated evaluation record {record['record_id']}",
    })

    eval_values = {
        "train_run_id": record["train_run_id"],
        "split": record["split"],
        "inference_seed": record["inference_seed"],
        "annotation_sha256": record["annotation"]["sha256"],
        "protocol_name": record["evaluation_protocol"]["metric_definition"],
        "protocol_sha256": record["protocol_sha256"],
        "selection_source": record["selection_source"],
        "test_tuned": int(record.get("test_tuned", False)),
        "status": record["status"],
        "evidence_artifact_id": artifact_id,
        "updated_at": "CURRENT_TIMESTAMP",
    }
    existing_eval = connection.execute(
        "SELECT train_run_id,split,inference_seed,annotation_sha256,"
        "protocol_name,protocol_sha256,selection_source,test_tuned "
        "FROM eval_run_registry WHERE eval_run_id=?", (record["record_id"],),
    ).fetchone()
    identity = tuple(eval_values[key] for key in (
        "train_run_id", "split", "inference_seed", "annotation_sha256",
        "protocol_name", "protocol_sha256", "selection_source", "test_tuned"))
    if existing_eval is not None and tuple(existing_eval) != identity:
        raise ValueError(f"conflicting eval_run identity: {record['record_id']}")
    if existing_eval is None:
        connection.execute(
            "INSERT INTO eval_run_registry "
            "(eval_run_id,train_run_id,split,inference_seed,annotation_sha256,"
            "protocol_name,protocol_sha256,selection_source,test_tuned,status,"
            "evidence_artifact_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (record["record_id"], *identity, record["status"], artifact_id),
        )
    else:
        connection.execute(
            "UPDATE eval_run_registry SET status=?,evidence_artifact_id=?,"
            "updated_at=CURRENT_TIMESTAMP WHERE eval_run_id=?",
            (record["status"], artifact_id, record["record_id"]),
        )

    protocol = stable_json({
        "schema_version": record["schema_version"],
        "train_run_id": record["train_run_id"],
        "training_seed": record["training_seed"],
        "inference_seed": record["inference_seed"],
        "replication_unit": record["replication_unit"],
        "protocol_sha256": record["protocol_sha256"],
        "evaluation_protocol": record["evaluation_protocol"],
        "selection_source": record["selection_source"],
        "test_tuned": record.get("test_tuned", False),
        "annotation": record["annotation"],
        "config": record["config"],
        "checkpoint": record["checkpoint"],
        "source_log": record["source_log"],
        "code": record["code"],
        "parent_train_run_id": record.get("parent_train_run_id"),
        "parent_checkpoint_sha256": record.get("parent_checkpoint_sha256"),
    })
    paper_eligible = int(record["status"] == "paper_eligible")
    for metric, metric_value in sorted(record["metrics"].items()):
        result_id = f"{record['record_id']}::{metric.lower()}"
        insert_exact(connection, "controlled_result", "result_id", result_id, {
            "family": "run_evidence",
            "variant": method,
            "dataset": record["dataset_id"],
            "split": record["split"],
            "seed": str(record["training_seed"]),
            "metric": metric,
            "value": float(metric_value),
            "unit": "absolute",
            "baseline_result_id": None,
            "delta": None,
            "protocol_json": protocol,
            "artifact_id": artifact_id,
            "evidence_level": "controlled",
            "paper_eligible": paper_eligible,
            "notes": "Imported from validated run-evidence record",
        })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    connection = sqlite3.connect(args.db)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(SCHEMA.read_text(encoding="utf-8"))
    try:
        with connection:
            for record_path in args.records:
                payload = json.loads(record_path.read_text(encoding="utf-8"))
                records = payload if isinstance(payload, list) else [payload]
                for record in records:
                    import_record(connection, record, record_path, args.root)
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
