#!/usr/bin/env python3
"""Pre-register the complete D2-comparable D1_INHOUSE1700_V1 run matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "tools/experiment_db/experiments.db"
DATASET_ID = "D1_INHOUSE1700_V1"
SEEDS = (42, 123, 789)
FULL_MODELS = {
    "karyoflow": "experiments/configs/self1700/karyoflow.py",
    "diffusiondet": "experiments/configs/self1700/diffusiondet.py",
    "dino_r50": "experiments/configs/self1700/dino_r50.py",
    "rtmdet_l": "experiments/configs/self1700/rtmdet_l.py",
    "cascade_rcnn_r50": "experiments/configs/self1700/cascade_rcnn_r50.py",
    "yolox_s": "experiments/configs/self1700/yolox_s.py",
}
SHORT_MODELS = {"karyoflow_lqcr": "experiments/configs/self1700/lqcr.py"}

DDL = """
CREATE TABLE IF NOT EXISTS train_run_registry (
    train_run_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    method TEXT NOT NULL,
    training_seed INTEGER NOT NULL,
    config_path TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    dataset_manifest_sha256 TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    replication_unit TEXT NOT NULL,
    parent_train_run_id TEXT,
    assigned_executor TEXT NOT NULL,
    tracker_project TEXT,
    tracker_run_name TEXT,
    tracker_run_id TEXT,
    work_dir TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES dataset_release(dataset_id),
    FOREIGN KEY (parent_train_run_id) REFERENCES train_run_registry(train_run_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_train_run_identity
ON train_run_registry(dataset_id, method, training_seed, config_sha256);

CREATE TABLE IF NOT EXISTS eval_run_registry (
    eval_run_id TEXT PRIMARY KEY,
    train_run_id TEXT NOT NULL,
    split TEXT NOT NULL,
    inference_seed INTEGER NOT NULL,
    annotation_sha256 TEXT NOT NULL,
    protocol_name TEXT NOT NULL,
    protocol_sha256 TEXT NOT NULL,
    selection_source TEXT NOT NULL,
    test_tuned BOOLEAN NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    evidence_artifact_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (train_run_id) REFERENCES train_run_registry(train_run_id),
    FOREIGN KEY (evidence_artifact_id) REFERENCES evidence_artifact(artifact_id)
);
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(method: str, seed: int, config_sha: str) -> str:
    return f"d1-inhouse1700-v1__{method}__trainseed-{seed}__{config_sha[:12]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--db", type=Path, default=DB)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["dataset_id"] != DATASET_ID:
        raise RuntimeError("wrong dataset manifest")
    manifest_sha = sha256(args.manifest)
    test_sha = manifest["splits"]["test"]["sha256"]
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    protocol = {
        "split": "test", "inference_seed": 42,
        "metrics": ["mAP", "AP50", "AP75", "AP_S", "AP_M", "AP_L"],
        "evaluator": "pycocotools COCOeval", "max_dets": [1, 10, 300],
        "selection": "best validation mAP checkpoint; test evaluated once after selection",
    }
    protocol_sha = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    executors = ("ross:a6000:0", "workstation:a5000:0", "workstation:a4000:1")

    con = sqlite3.connect(args.db)
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(DDL)
    registry_columns = {
        row[1] for row in con.execute("PRAGMA table_info(train_run_registry)")
    }
    if "tracker_project" not in registry_columns:
        con.execute("ALTER TABLE train_run_registry ADD COLUMN tracker_project TEXT")
    if "tracker_run_name" not in registry_columns:
        con.execute("ALTER TABLE train_run_registry ADD COLUMN tracker_run_name TEXT")
    if "tracker_run_id" not in registry_columns:
        con.execute("ALTER TABLE train_run_registry ADD COLUMN tracker_run_id TEXT")
    planned = []
    parent_ids = {}
    for method, config_path in FULL_MODELS.items():
        config_sha = sha256(ROOT / config_path)
        for index, seed in enumerate(SEEDS):
            run_id = stable_id(method, seed, config_sha)
            if method == "karyoflow":
                parent_ids[seed] = run_id
            work_dir = f"work_dirs/self1700/{method}/trainseed_{seed}"
            executor = executors[(list(FULL_MODELS).index(method) + index) % len(executors)]
            values = (run_id, DATASET_ID, method, seed, config_path, config_sha,
                      manifest_sha, git_commit, "independent_training_seed", None,
                      executor, "KaryoFlow-Self1700", f"{method}_seed{seed}", work_dir, "planned")
            old_ids = [row[0] for row in con.execute(
                "SELECT train_run_id FROM train_run_registry "
                "WHERE dataset_id=? AND method=? AND training_seed=? AND train_run_id<>? "
                "AND status NOT IN ('superseded','invalid')",
                (DATASET_ID, method, seed, run_id)).fetchall()]
            for old_id in old_ids:
                con.execute("UPDATE train_run_registry SET status='superseded',updated_at=CURRENT_TIMESTAMP WHERE train_run_id=?", (old_id,))
                con.execute("UPDATE eval_run_registry SET status='superseded',updated_at=CURRENT_TIMESTAMP WHERE train_run_id=?", (old_id,))
            con.execute(
                """INSERT INTO train_run_registry
                (train_run_id,dataset_id,method,training_seed,config_path,config_sha256,
                 dataset_manifest_sha256,git_commit,replication_unit,parent_train_run_id,
                 assigned_executor,tracker_project,tracker_run_name,work_dir,status)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(train_run_id) DO UPDATE SET
                 dataset_manifest_sha256=excluded.dataset_manifest_sha256,
                 git_commit=excluded.git_commit,
                 assigned_executor=excluded.assigned_executor,work_dir=excluded.work_dir,
                 updated_at=CURRENT_TIMESTAMP""", values)
            planned.append(run_id)
    for method, config_path in SHORT_MODELS.items():
        config_sha = sha256(ROOT / config_path)
        for index, seed in enumerate(SEEDS):
            run_id = stable_id(method, seed, config_sha)
            parent_id = parent_ids[seed]
            work_dir = f"work_dirs/self1700/{method}/trainseed_{seed}"
            values = (run_id, DATASET_ID, method, seed, config_path, config_sha,
                      manifest_sha, git_commit, "paired_final_stage_intervention", parent_id,
                      executors[index], "KaryoFlow-Self1700", f"{method}_seed{seed}", work_dir, "planned")
            old_ids = [row[0] for row in con.execute(
                "SELECT train_run_id FROM train_run_registry "
                "WHERE dataset_id=? AND method=? AND training_seed=? AND train_run_id<>? "
                "AND status NOT IN ('superseded','invalid')",
                (DATASET_ID, method, seed, run_id)).fetchall()]
            for old_id in old_ids:
                con.execute("UPDATE train_run_registry SET status='superseded',updated_at=CURRENT_TIMESTAMP WHERE train_run_id=?", (old_id,))
                con.execute("UPDATE eval_run_registry SET status='superseded',updated_at=CURRENT_TIMESTAMP WHERE train_run_id=?", (old_id,))
            con.execute(
                """INSERT INTO train_run_registry
                (train_run_id,dataset_id,method,training_seed,config_path,config_sha256,
                 dataset_manifest_sha256,git_commit,replication_unit,parent_train_run_id,
                 assigned_executor,tracker_project,tracker_run_name,work_dir,status)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(train_run_id) DO UPDATE SET
                 dataset_manifest_sha256=excluded.dataset_manifest_sha256,
                 git_commit=excluded.git_commit,
                 parent_train_run_id=excluded.parent_train_run_id,
                 assigned_executor=excluded.assigned_executor,work_dir=excluded.work_dir,
                 updated_at=CURRENT_TIMESTAMP""", values)
            planned.append(run_id)

    for train_run_id in planned:
        eval_id = f"{train_run_id}__test__inferseed-42__{protocol_sha[:12]}"
        con.execute(
            """INSERT INTO eval_run_registry
            (eval_run_id,train_run_id,split,inference_seed,annotation_sha256,
             protocol_name,protocol_sha256,selection_source,test_tuned,status)
            VALUES (?,?,?,?,?,?,?,?,0,'planned')
            ON CONFLICT(eval_run_id) DO NOTHING""",
            (eval_id, train_run_id, "test", 42, test_sha,
             "coco-six-metrics-v1", protocol_sha, protocol["selection"]),
        )
    con.commit()
    print(json.dumps({"dataset_id": DATASET_ID, "full_training_runs": 18,
                      "paired_lqcr_runs": 3, "planned_test_evaluations": 21,
                      "protocol_sha256": protocol_sha}, indent=2))


if __name__ == "__main__":
    main()
