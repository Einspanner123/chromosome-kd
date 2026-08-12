#!/usr/bin/env python3
"""Idempotently register conditional Dataset-2 diagnostic evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "tools/experiment_db/experiments.db"
BOOTSTRAP = ROOT / "tools/experiment_db/evidence_sources/d2_lqcr_vs_dino_image_bootstrap_a2344f4de3df.json"
DIFFICULT = ROOT / "tools/experiment_db/evidence_sources/d2_difficult_subsets_paired_train3_725aab1ab102.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_artifact(con: sqlite3.Connection, artifact_id: str, path: Path,
                      kind: str, generated_at: str, notes: str) -> None:
    con.execute(
        """INSERT OR REPLACE INTO evidence_artifact
           (artifact_id,server,path,sha256,kind,generated_at,status,notes)
           VALUES (?, 'ross', ?, ?, ?, ?, 'verified', ?)""",
        (artifact_id, str(path.relative_to(ROOT)), sha256(path), kind,
         generated_at, notes),
    )


def put_result(con: sqlite3.Connection, result_id: str, family: str,
               variant: str, metric: str, value: float, unit: str,
               protocol: dict, artifact_id: str, notes: str) -> None:
    con.execute(
        """INSERT OR REPLACE INTO controlled_result
           (result_id,family,variant,dataset,split,seed,metric,value,unit,
            baseline_result_id,delta,protocol_json,artifact_id,evidence_level,
            paper_eligible,notes)
           VALUES (?,?,?,?,?,?,?,?,?,NULL,NULL,?,?, 'diagnostic',0,?)""",
        (result_id, family, variant, "D2", "test", "conditional", metric,
         float(value), unit, json.dumps(protocol, sort_keys=True), artifact_id,
         notes),
    )


def register_bootstrap(con: sqlite3.Connection) -> None:
    doc = json.loads(BOOTSTRAP.read_text())
    if doc["bootstrap"]["iterations"] != 1000:
        raise RuntimeError("only the formal 1,000-iteration bootstrap may be registered")
    if doc["bootstrap"]["cache_equivalence_validation"]["maximum_absolute_error"] > 1e-12:
        raise RuntimeError("cached COCO accumulation failed equivalence validation")
    artifact_id = "d2-lqcr-vs-dino-image-bootstrap-1000-seed20260812"
    register_artifact(
        con, artifact_id, BOOTSTRAP, "conditional_image_cluster_bootstrap",
        doc["generated_at"],
        "Test-image sampling uncertainty conditional on evaluated checkpoints; not a training-run significance test.",
    )
    protocol = {
        "estimand": doc["estimand"],
        "sampling_unit": doc["sampling_unit"],
        "important_boundary": doc["important_boundary"],
        "bootstrap": doc["bootstrap"],
        "annotation_sha256": doc["annotation_sha256"],
        "lqcr_replication_unit": "independent_detector_training_run",
        "dino_replication_unit": "inference_seed_fixed_checkpoint",
    }
    values = {
        "point_difference_mAP": (doc["point_estimate"]["difference"], "absolute"),
        "bootstrap_difference_mean_mAP": (doc["bootstrap_result"]["difference_mean"], "absolute"),
        "bootstrap_difference_sample_std": (doc["bootstrap_result"]["difference_sample_std"], "sample_std"),
        "ci95_lower_mAP": (doc["bootstrap_result"]["ci95_lower"], "absolute"),
        "ci95_upper_mAP": (doc["bootstrap_result"]["ci95_upper"], "absolute"),
        "fraction_difference_gt_zero": (doc["bootstrap_result"]["fraction_difference_gt_zero"], "proportion"),
    }
    for metric, (value, unit) in values.items():
        put_result(
            con, f"d2-lqcr-vs-dino-bootstrap1000-{metric}",
            "conditional_image_bootstrap_d2", "KaryoFlow+LQCR_minus_DINO",
            metric, value, unit, protocol, artifact_id,
            "Conditional image-level diagnostic; DINO has inference-seed, not training-seed, replication.",
        )


def iter_difficult_deltas(node: dict, prefix: tuple[str, ...] = ()):
    for key, value in node.items():
        path = prefix + (key,)
        if isinstance(value, dict) and "paired_delta_mean" in value:
            yield path, value
        elif isinstance(value, dict):
            yield from iter_difficult_deltas(value, path)


def register_difficult(con: sqlite3.Connection) -> None:
    doc = json.loads(DIFFICULT.read_text())
    if (doc["dataset"], doc["split"], len(doc["runs"])) != ("D2", "test", 3):
        raise RuntimeError("unexpected difficult-subset protocol")
    artifact_id = "d2-difficult-subsets-paired-train3-725aab1ab102"
    register_artifact(
        con, artifact_id, DIFFICULT, "conditional_difficult_subset_analysis",
        doc["generated_at"],
        "Conditional GT-instance stratification over three paired detector runs; diagnostic only, with no subset AP claim.",
    )
    protocol = {
        "replication_unit": doc["replication_unit"],
        "definitions": doc["definitions"],
        "quality_controls": doc["quality_controls"],
        "interpretation_guardrails": doc["interpretation_guardrails"],
        "subset_AP_excluded": doc["definitions"]["subset_AP_excluded"],
    }
    for path, leaf in iter_difficult_deltas(doc["aggregates"]):
        descriptor = "/".join(path[:-1])
        metric = path[-1]
        stable = hashlib.sha256("/".join(path).encode()).hexdigest()[:16]
        put_result(
            con, f"d2-difficult-paired-delta-{stable}",
            "conditional_difficult_subset_d2", descriptor, metric,
            leaf["paired_delta_mean"], "absolute", protocol, artifact_id,
            "Mean paired LQCR-minus-parent delta across three detector training runs; conditional descriptive evidence.",
        )


def main() -> None:
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys=ON")
    with con:
        register_bootstrap(con)
        register_difficult(con)
    print(json.dumps({
        "bootstrap_sha256": sha256(BOOTSTRAP),
        "difficult_subset_sha256": sha256(DIFFICULT),
        "diagnostic_rows": con.execute(
            """SELECT COUNT(*) FROM controlled_result
               WHERE artifact_id IN (?,?) AND evidence_level='diagnostic'
                 AND paper_eligible=0""",
            ("d2-lqcr-vs-dino-image-bootstrap-1000-seed20260812",
             "d2-difficult-subsets-paired-train3-725aab1ab102"),
        ).fetchone()[0],
    }, indent=2))


if __name__ == "__main__":
    main()
