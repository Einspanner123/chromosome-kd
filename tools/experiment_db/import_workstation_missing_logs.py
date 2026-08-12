"""Register synchronized workstation logs and complete D1 clean test metrics.

Idempotent: artifact and controlled-result identifiers are stable, while
evaluation rows use the database identity index.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "tools/experiment_db/experiments.db"
LIST = ROOT / "tools/experiment_db/evidence_sources/workstation_logs_sync_20260812.json"
METRICS = ("mAP", "AP50", "AP75", "AP_S", "AP_M", "AP_L")
SEED_VALUES = {
    42: (0.7400, 0.9310, 0.8230, 0.5020, 0.7260, 0.6390),
    123: (0.7450, 0.9320, 0.8320, 0.5200, 0.7250, 0.6560),
    789: (0.7390, 0.9340, 0.8250, 0.5050, 0.7210, 0.6650),
}
LOGS = {
    42: "work_dirs/paired_clean/a4_random_chr2024_seed42/20260811_143917/20260811_143917.log",
    123: "work_dirs/paired_clean/a4_random_chr2024_seed123/20260811_144242/20260811_144242.log",
    789: "work_dirs/paired_clean/a4_random_chr2024_seed789/20260811_144331/20260811_144331.log",
}
LQCR_VALUES = {
    42: (0.7470, 0.9310, 0.8310, 0.5060, 0.7300, 0.6410),
    123: (0.7500, 0.9320, 0.8360, 0.5240, 0.7290, 0.6530),
    789: (0.7470, 0.9340, 0.8320, 0.5140, 0.7270, 0.6680),
}
LQCR_LOGS = {
    42: "results/lqcr_clean_d1_test_seed42_20260811/20260811_190714/20260811_190714.log",
    123: "results/lqcr_clean_d1_test_seed123_20260811/20260811_190842/20260811_190842.log",
    789: "results/lqcr_clean_d1_test_seed789_20260811/20260811_190956/20260811_190956.log",
}
LQCR_CHECKPOINTS = {
    42: "work_dirs/paired_clean/lqcr_final_only_chr2024_seed42/best_coco_bbox_mAP_epoch_5.pth",
    123: "work_dirs/paired_clean/lqcr_final_only_chr2024_seed123/best_coco_bbox_mAP_epoch_2.pth",
    789: "work_dirs/paired_clean/lqcr_final_only_chr2024_seed789/best_coco_bbox_mAP_epoch_2.pth",
}
CHECKPOINTS = {
    42: "work_dirs/paired_clean/a4_random_chr2024_seed42/best_coco_bbox_mAP_epoch_62.pth",
    123: "work_dirs/paired_clean/a4_random_chr2024_seed123/best_coco_bbox_mAP_epoch_68.pth",
    789: "work_dirs/paired_clean/a4_random_chr2024_seed789/best_coco_bbox_mAP_epoch_72.pth",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads(LIST.read_text())
    con = sqlite3.connect(DB)
    now = datetime.now(timezone.utc).isoformat()

    for item in manifest["files"]:
        path = ROOT / item["path"]
        actual = sha(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"SHA mismatch: {item['path']}")
        con.execute(
            """INSERT OR REPLACE INTO evidence_artifact
               (artifact_id,server,path,sha256,kind,generated_at,status,notes)
               VALUES (?,?,?,?,?,?,?,?)""",
            (item["artifact_id"], "workstation->ross", item["path"], actual,
             item["kind"], manifest["generated_at"], item.get("status", "verified"),
             ("Known invalid all-zero evaluation; retained only to prevent accidental reuse."
              if item.get("status") == "invalid" else
              "Synchronized only because Ross lacked this log or lightweight metadata; no existing Ross file was overwritten.")),
        )

    # The seed-42 source already existed on Ross and was therefore absent from
    # the workstation-delta manifest. Register all three metric sources here.
    for seed, log in list(LOGS.items()) + list(LQCR_LOGS.items()):
        path = ROOT / log
        digest = sha(path)
        con.execute(
            """INSERT OR REPLACE INTO evidence_artifact
               (artifact_id,server,path,sha256,kind,generated_at,status,notes)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("wslog_" + digest[:16], "ross" if seed == 42 else "workstation->ross",
             log, digest, "test_log", manifest["generated_at"], "verified",
             "Source log for the independent-training-seed D1 test result."),
        )

    protocol = json.dumps({
        "dataset": "D1", "split": "test", "images": 220,
        "replication_unit": "independent_training_seed",
        "seeds": [42, 123, 789], "metric_family": "COCO bbox",
    }, sort_keys=True)
    for seed, values in SEED_VALUES.items():
        log = LOGS[seed]
        artifact_id = "wslog_" + sha(ROOT / log)[:16]
        exp_id = f"work_dirs/paired_clean/a4_random_chr2024_seed{seed}"
        con.execute(
            """INSERT INTO experiment
               (experiment_id,name,work_dir,server,seed,status,best_checkpoint,notes)
               VALUES (?,?,?,?,?,'completed',?,?)
               ON CONFLICT(experiment_id) DO UPDATE SET
                 server='ross+workstation', seed=excluded.seed,
                 status='completed', best_checkpoint=excluded.best_checkpoint""",
            (exp_id, f"a4_random_chr2024_seed{seed}", exp_id,
             "ross" if seed == 42 else "ross+workstation", seed,
             CHECKPOINTS[seed], "Independent-training-seed D1 clean random-coupling run."),
        )
        con.execute(
            """INSERT OR REPLACE INTO evaluation
               (eval_id,experiment_id,split,mAP,AP50,AP75,AP_small,AP_medium,AP_large,
                epoch,checkpoint_path,evaluated_at,source)
               VALUES ((SELECT eval_id FROM evaluation WHERE experiment_id=? AND split='test'
                       AND source='synchronized_test_log' AND checkpoint_path=?),
                       ?,'test',?,?,?,?,?,?,NULL,?,?, 'synchronized_test_log')""",
            (exp_id, CHECKPOINTS[seed], exp_id, *values, CHECKPOINTS[seed], now),
        )
        for metric, value in zip(METRICS, values):
            rid = f"d1_clean_karyoflow_trainseed{seed}_{metric.lower()}"
            con.execute(
                """INSERT OR REPLACE INTO controlled_result
                   (result_id,family,variant,dataset,split,seed,metric,value,unit,
                    baseline_result_id,delta,protocol_json,artifact_id,evidence_level,
                    paper_eligible,notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rid, "clean_random_mainline", "KaryoFlow", "D1", "test",
                 str(seed), metric, value, "absolute", None, None, protocol,
                 artifact_id, "controlled", 1,
                 "Independent training seed; complete held-out-test metric from synchronized source log."),
            )

    import statistics
    for j, metric in enumerate(METRICS):
        vals = [v[j] for v in SEED_VALUES.values()]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals)
        for suffix, value, unit in (("mean", mean, "absolute"), ("std", std, "sample_std")):
            rid = f"d1_clean_karyoflow_train3_{metric.lower()}_{suffix}"
            con.execute(
                """INSERT OR REPLACE INTO controlled_result
                   (result_id,family,variant,dataset,split,seed,metric,value,unit,
                    baseline_result_id,delta,protocol_json,artifact_id,evidence_level,
                    paper_eligible,notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rid, "clean_random_mainline", "KaryoFlow", "D1", "test",
                 "mean(42,123,789)", metric if suffix == "mean" else metric + "_std",
                 value, unit, None, None, protocol,
                 "wslog_" + sha(ROOT / LOGS[123])[:16], "controlled", 1,
                 "Aggregate over three independently trained models. Seed-level source artifacts are separately registered."),
            )

    # LQCR: all six metrics for each independently trained detector and their
    # aggregate. The base detector was held fixed during quality-branch fitting.
    for seed, values in LQCR_VALUES.items():
        log = LQCR_LOGS[seed]
        artifact_id = "wslog_" + sha(ROOT / log)[:16]
        exp_id = f"results/lqcr_clean_d1_test_seed{seed}_20260811"
        con.execute(
            """INSERT INTO experiment
               (experiment_id,name,work_dir,server,seed,status,best_checkpoint,notes)
               VALUES (?,?,?,?,?,'completed',?,?)
               ON CONFLICT(experiment_id) DO UPDATE SET seed=excluded.seed,
                 status='completed', best_checkpoint=excluded.best_checkpoint""",
            (exp_id, f"lqcr_final_only_chr2024_seed{seed}", exp_id,
             "ross" if seed == 42 else "ross+workstation", seed,
             LQCR_CHECKPOINTS[seed], "Independent-training-seed D1 final-only LQCR test."),
        )
        con.execute(
            """INSERT OR REPLACE INTO evaluation
               (eval_id,experiment_id,split,mAP,AP50,AP75,AP_small,AP_medium,AP_large,
                epoch,checkpoint_path,evaluated_at,source)
               VALUES ((SELECT eval_id FROM evaluation WHERE experiment_id=? AND split='test'
                       AND source='synchronized_test_log' AND checkpoint_path=?),
                       ?,'test',?,?,?,?,?,?,NULL,?,?, 'synchronized_test_log')""",
            (exp_id, LQCR_CHECKPOINTS[seed], exp_id, *values,
             LQCR_CHECKPOINTS[seed], now),
        )
        for metric, value in zip(METRICS, values):
            base_id = f"d1_clean_karyoflow_trainseed{seed}_{metric.lower()}"
            rid = f"d1_clean_lqcr_trainseed{seed}_{metric.lower()}"
            base = SEED_VALUES[seed][METRICS.index(metric)]
            con.execute(
                """INSERT OR REPLACE INTO controlled_result
                   (result_id,family,variant,dataset,split,seed,metric,value,unit,
                    baseline_result_id,delta,protocol_json,artifact_id,evidence_level,
                    paper_eligible,notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rid, "paired_clean_d1_test", "KaryoFlow+LQCR", "D1", "test",
                 str(seed), metric, value, "absolute", base_id, value-base,
                 protocol, artifact_id, "controlled", 1,
                 "Independent training seed; final-only quality ranking; complete held-out-test metric."),
            )

    for j, metric in enumerate(METRICS):
        vals = [v[j] for v in LQCR_VALUES.values()]
        base_vals = [v[j] for v in SEED_VALUES.values()]
        mean, std = statistics.mean(vals), statistics.stdev(vals)
        base_mean = statistics.mean(base_vals)
        base_id = f"d1_clean_karyoflow_train3_{metric.lower()}_mean"
        for suffix, value, unit, baseline, delta in (
            ("mean", mean, "absolute", base_id, mean-base_mean),
            ("std", std, "sample_std", None, None),
        ):
            rid = f"d1_clean_lqcr_train3_{metric.lower()}_{suffix}"
            con.execute(
                """INSERT OR REPLACE INTO controlled_result
                   (result_id,family,variant,dataset,split,seed,metric,value,unit,
                    baseline_result_id,delta,protocol_json,artifact_id,evidence_level,
                    paper_eligible,notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rid, "paired_clean_d1_test", "KaryoFlow+LQCR", "D1", "test",
                 "mean(42,123,789)", metric if suffix == "mean" else metric + "_std",
                 value, unit, baseline, delta, protocol,
                 "wslog_" + sha(ROOT / LQCR_LOGS[123])[:16], "controlled", 1,
                 "Aggregate over three independently trained final-only LQCR models."),
            )
    con.commit()
    print(f"registered {len(manifest['files'])} synchronized artifacts and complete D1 metrics")


if __name__ == "__main__":
    main()
