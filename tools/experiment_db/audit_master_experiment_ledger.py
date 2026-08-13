#!/usr/bin/env python3
"""Audit the generated experiment ledger against its manifest and SQLite registry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=ROOT / "tools/experiment_db/experiments.db")
    p.add_argument("--manifest", type=Path, default=ROOT / "tools/experiment_db/manifests/master_experiment_ledger_v1.json")
    p.add_argument("--csv", type=Path, default=ROOT / "tools/experiment_db/exports/master_experiment_ledger.csv")
    p.add_argument("--doc", type=Path, default=ROOT / "docs/experiments/MASTER_EXPERIMENT_LEDGER.md")
    args = p.parse_args()

    errors: list[str] = []
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["experiments"]
    ids = [r["ledger_id"] for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ledger_id in manifest")
    if len(rows) != 46:
        errors.append(f"expected 46 ledger rows, got {len(rows)}")
    if sum(int(r["run_count"]) for r in rows) != 261:
        errors.append("expanded run count is not 261")
    if {r["dataset_id"] for r in rows} != {"D1_INHOUSE1700_V1", "D2", "D1_COMPOSITE2200_LEGACY"}:
        errors.append("dataset coverage mismatch")

    idset = set(ids)
    for r in rows:
        parent = r.get("parent_ledger_id")
        if parent and parent not in idset:
            errors.append(f"{r['ledger_id']}: missing parent {parent}")
        if r["status"].startswith(("PLANNED", "BLOCKED", "RUNNING")):
            for key in ("config_path", "executor_plan", "work_dir_template", "acceptance_gate"):
                if not r.get(key):
                    errors.append(f"{r['ledger_id']}: planned row missing {key}")
        if "*" in (r.get("evidence_artifact_ids") or ""):
            errors.append(f"{r['ledger_id']}: wildcard evidence artifact ID")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    db_rows = {r["ledger_id"]: dict(r) for r in conn.execute("SELECT * FROM experiment_ledger")}
    if set(db_rows) != idset:
        errors.append("manifest and experiment_ledger ID sets differ")
    artifacts = {r[0] for r in conn.execute("SELECT artifact_id FROM evidence_artifact")}
    train_runs = {r[0] for r in conn.execute("SELECT train_run_id FROM train_run_registry")}
    for r in rows:
        for aid in filter(None, (r.get("evidence_artifact_ids") or "").split(";")):
            if aid == "multiple" and r["status"] == "ARCHIVED_NONCOMPARABLE":
                continue
            if aid not in artifacts:
                errors.append(f"{r['ledger_id']}: missing evidence artifact {aid}")
        for tid in filter(None, (r.get("train_run_ids") or "").split(";")):
            if tid not in train_runs:
                errors.append(f"{r['ledger_id']}: missing train_run_id {tid}")
    conn.close()

    with args.csv.open(newline="") as f:
        csv_rows = list(csv.DictReader(f))
    if [r["ledger_id"] for r in csv_rows] != ids:
        errors.append("CSV row order/IDs differ from manifest")

    manifest_hash = sha256(args.manifest)
    doc = args.doc.read_text()
    if manifest_hash not in doc:
        errors.append("document does not contain current manifest SHA256")
    for required in ("D1_INHOUSE1700_V1", "D2", "Generation", "Decision", "Deployment"):
        if required not in doc:
            errors.append(f"document missing required token: {required}")

    if errors:
        print("MASTER LEDGER AUDIT: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("MASTER LEDGER AUDIT: PASS")
    print(f"ledger_rows={len(rows)} expanded_runs={sum(int(r['run_count']) for r in rows)}")
    print(f"manifest_sha256={manifest_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
